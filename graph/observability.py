"""관측성 계층: 결정 로그·노드 실행 이벤트를 State 밖(외부 JSONL)으로 적재한다.

State에는 last_decision 1건만 남기고, 전체 이력은 여기서 outputs/traces/{run_id}.jsonl에
append한다. 각 레코드는 {ts, run_id, step, node, event, ...}이며 run_id가 LangSmith 트레이스
(루트 run id·metadata)와 체크포인트(thread_id)를 잇는 상관 키다.
"""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from config import settings

_LOCK = threading.Lock()
_NODE_ELAPSED: dict[str, float] = defaultdict(float)
_NODE_RUN_COUNT: dict[str, int] = defaultdict(int)


def trace_path(run_id: str) -> Path:
    return settings.outputs_path / "traces" / f"{run_id}.jsonl"


def log_event(run_id: str, node: str, event: str, **fields: Any) -> None:
    """외부 결정/실행 로그 1건을 기록한다 (병렬 노드에서 호출되므로 락으로 직렬화)."""
    record = {
        "ts": datetime.now().isoformat(timespec="seconds"),
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
