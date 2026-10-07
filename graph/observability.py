"""관측성 계층: 결정 로그·노드 실행 이벤트를 State 밖(외부 JSONL)으로 적재한다.

State에는 last_decision 1건만 남기고, 전체 이력은 여기서 outputs/traces/{run_id}.jsonl에
append한다. 각 레코드는 {ts, run_id, step, node, event, ...}이며 run_id가 LangSmith 트레이스
(루트 run id·metadata)와 체크포인트(thread_id)를 잇는 상관 키다.

같은 결정은 두 곳에 더 남긴다:
  - LangSmith : supervisor 노드 run에 action 태그·metadata, 그 아래 "decision: action → 대상" span
    (트리 이름만 봐도 라우팅이 보이게), 종료 후 루트 run에 라우팅·재작업·품질 요약 feedback
  - 보고서 부록 : 최종 보고서 끝에 결정 이력 표(render_decision_appendix)
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import langsmith as ls
from langsmith.run_helpers import get_current_run_tree
from langsmith.utils import tracing_is_enabled

from config import settings

_LOCK = threading.Lock()
_NODE_ELAPSED: dict[str, float] = defaultdict(float)
_NODE_RUN_COUNT: dict[str, int] = defaultdict(int)


def trace_path(run_id: str) -> Path:
    return settings.outputs_path / "traces" / f"{run_id}.jsonl"


def now_ts() -> str:
    """결정 로그·State 타임스탬프 공통 형식 (로컬 시각, 초 단위 ISO 8601)."""
    return datetime.now().isoformat(timespec="seconds")


def log_event(run_id: str, node: str, event: str, **fields: Any) -> None:
    """외부 결정/실행 로그 1건을 기록한다 (병렬 노드에서 호출되므로 락으로 직렬화)."""
    record = {
        "ts": now_ts(),
        "run_id": run_id,
        "node": node,
        "event": event,
        **fields,
    }
    path = trace_path(run_id or "unknown")
    with _LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def record_timing(name: str, elapsed: float) -> None:
    with _LOCK:
        _NODE_ELAPSED[name] += elapsed
        _NODE_RUN_COUNT[name] += 1


class Timer:
    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed = time.perf_counter() - self.start


def print_timing_summary() -> None:
    """실행 종료 후 노드별 누적 소요 시간을 표로 출력한다. app.py가 호출한다."""
    if not _NODE_ELAPSED:
        return
    print("\n=== 노드별 실행 시간 요약 ===")
    for name, total in sorted(_NODE_ELAPSED.items(), key=lambda kv: kv[1], reverse=True):
        runs = _NODE_RUN_COUNT[name]
        suffix = f" ({runs}회 실행 합계)" if runs > 1 else ""
        print(f"  {name:<24} {total:7.1f}초{suffix}")


def summarize_decisions(run_id: str) -> list[dict[str, Any]]:
    """외부 로그에서 Supervisor 결정만 순서대로 꺼낸다 (보고서·콘솔 요약용)."""
    path = trace_path(run_id)
    if not path.exists():
        return []
    decisions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("event") == "decision":
            decisions.append(record)
    return decisions


def summarize_routing(run_id: str) -> list[dict[str, Any]]:
    """step별 마지막 결정만 step 순으로 반환한다 (--resume으로 같은 step이 다시 기록돼도 중복 없음)."""
    by_step = {d.get("step"): d for d in summarize_decisions(run_id)}
    return [by_step[step] for step in sorted(by_step, key=lambda s: s or 0)]


def count_reworks(decisions: list[dict[str, Any]]) -> int:
    """재작업(rework_*)·보고서 재작성(revise_report) 결정 수."""
    actions = Counter(d.get("action", "") for d in decisions)
    return sum(v for k, v in actions.items() if k.startswith("rework") or k == "revise_report")


# ── LangSmith ────────────────────────────────────────────────────────────────


def trace_decision(decision: dict[str, Any], sufficiency: dict[str, Any] | None = None) -> None:
    """Supervisor 결정을 LangSmith 트레이스에 남긴다 (트레이싱 비활성이면 아무것도 하지 않음).

    supervisor 노드 run에는 필터용 태그(action:...)·metadata를 붙이고, 그 아래에 이름이
    "decision: {action} → {대상}"인 span을 만들어 트레이스 트리만 펼쳐도 라우팅이 읽히게 한다.
    관측성 실패가 그래프를 멈추면 안 되므로 예외는 삼킨다.
    """
    if not tracing_is_enabled():
        return
    action, targets = decision["action"], decision["targets"]
    tags = [f"action:{action}"]
    metadata = {"step": decision["step"], "action": action, "targets": targets, "reason": decision["reason"]}
    try:
        node_run = get_current_run_tree()
        if node_run is not None:
            node_run.add_tags(tags)
            node_run.add_metadata(metadata)
        with ls.trace(
            name=f"decision: {action} → {', '.join(targets) or 'END'}",
            run_type="chain",
            inputs={"step": decision["step"], "sufficiency": sufficiency},
            tags=tags,
            metadata=metadata,
        ) as span:
            span.end(outputs={"targets": targets, "reason": decision["reason"]})
    except Exception as exc:  # noqa: BLE001
        print(f"[observability] LangSmith 결정 기록 실패(무시): {exc}")


def record_run_feedback(run_id: str, quality_passed: bool | None) -> None:
    """종료 후 LangSmith 루트 run에 라우팅·재작업·품질 요약을 feedback 점수로 남긴다.

    끝난 run은 업데이트(PATCH)를 한 번만 받아 태그를 덧붙일 수 없으므로(409), 사후 기록용인
    feedback을 쓴다. 프로젝트 목록에 열로 표시되고 supervisor_reworks > 0 등으로 필터할 수 있다.
    """
    if not tracing_is_enabled():
        return
    decisions = summarize_routing(run_id)
    reworked = [f"{d['action']}→{','.join(d.get('targets') or [])}" for d in decisions
                if d.get("action", "").startswith("rework") or d.get("action") == "revise_report"]
    scores = [
        ("supervisor_routes", len(decisions), None),
        ("supervisor_reworks", len(reworked), "; ".join(reworked) or None),
        ("quality_passed", 1 if quality_passed else 0, None),
    ]
    try:
        from langchain_core.tracers.langchain import wait_for_all_tracers

        wait_for_all_tracers()  # 루트 run이 먼저 적재돼야 feedback이 연결된다
        client = ls.Client()
        for key, score, comment in scores:
            client.create_feedback(run_id, key=key, score=score, comment=comment)
    except Exception as exc:  # noqa: BLE001
        print(f"[observability] LangSmith 실행 요약 기록 실패(무시): {exc}")


# ── 보고서 부록 ──────────────────────────────────────────────────────────────


def _cell(text: str, limit: int = 90) -> str:
    text = " ".join(str(text).split()).replace("|", "\\|")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def render_decision_appendix(run_id: str) -> str:
    """외부 결정 로그로 보고서 부록(Supervisor 결정 이력 표)을 만든다. 로그가 없으면 빈 문자열."""
    decisions = summarize_routing(run_id)
    if not decisions:
        return ""
    lines = [
        "## 부록. Supervisor 결정 이력",
        "",
        f"- run_id : `{run_id}` — LangSmith 루트 run id·체크포인트 thread_id·결정 로그"
        f"(`outputs/traces/{run_id}.jsonl`)를 잇는 상관 키",
        f"- 라우팅 {len(decisions)}회, 재작업/재작성 {count_reworks(decisions)}회 "
        "(같은 코드라도 State에 따라 경로가 달라짐)",
        "",
        "| step | 결정 | 대상 | 사유 |",
        "|---|---|---|---|",
    ]
    for d in decisions:
        targets = ", ".join(d.get("targets") or []) or "END"
        lines.append(f"| {d.get('step')} | {d.get('action')} | {targets} | {_cell(d.get('reason', ''))} |")
    return "\n".join(lines) + "\n"


def insert_appendix(report: str, appendix: str) -> str:
    """부록을 REFERENCE 절 바로 앞에 넣는다 (가이드 목차: 마지막 챕터는 REFERENCE).

    REFERENCE 헤딩이 없으면 끝에 붙인다. 부록이 비면 보고서를 그대로 돌려준다.
    """
    if not appendix:
        return report
    match = re.search(r"^#{1,2}\s*REFERENCE\b", report, flags=re.MULTILINE)
    if match is None:
        return report.rstrip() + "\n\n" + appendix
    head, tail = report[: match.start()], report[match.start():]
    return head.rstrip() + "\n\n" + appendix.rstrip() + "\n\n" + tail
