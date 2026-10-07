"""Supervisor (조정 계층).

모든 하위 Agent는 Supervisor로만 돌아오고(hub-and-spoke), 하위 Agent 간 직접 엣지는 없다.
Supervisor는 매 스텝 현재 State(수집된 관점, 근거 충분도, 검증/품질 verdict, 예산)를 읽어
다음에 실행할 노드 집합(next_nodes)을 결정하고, graph/workflow.py의 add_conditional_edges가
그 값으로 분기한다. 노드 실행 순서는 하드코딩되어 있지 않다 — 같은 정책이라도 State에 따라
"관점 재작업", "보고서 재작성", "실패 Agent 재시도/제외", "종료"로 서로 다른 경로를 탄다.

라우팅 결정은 LLM이 아니라 이 모듈의 결정론적 정책(decide)이 내린다. LLM 판정이 필요한
부분(claim-근거 대조, 보고서 품질)은 하위 노드(faithfulness_check, quality_evaluation)가
구조화 verdict로 State에 남기고, Supervisor는 그 verdict + 예산으로만 분기한다 →
같은 State면 항상 같은 라우팅 = 재현성, 그리고 모든 결정에 사유(reason)가 붙는다.

정책 우선순위 (decide):
  0. 종료 가드      : step_count > max_steps → (필요 시 보고서 1회 생성) → 품질 평가 1회 → END
  1. 기술 조사      : technical_evidence 미수집 → tech_research
  2. 관점 수집      : 미수집/실패 관점 → 병렬 디스패치 (실패는 재시도 상한 후 '제외')
  3. 근거 충분도    : 관점 × 기술별 evidence 수·출처 다양성·insufficient_evidence 판정 → 부족 관점만 재작업
  4. 종합/검증      : synthesis → faithfulness_check, 검증 실패 claim의 출처 Agent만 재작업
  5. 보고서/품질    : report_writer → quality_evaluation
  6. 품질 미달 루프 : 관점 문제 → 해당 관점 재작업, 서술 문제 → 보고서 재작성, 예산 소진 → END
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from langgraph.graph import END

from config import settings
from graph.observability import log_event
from graph.state import GraphState

END_ACTION = "end"

TECH_NODE = "tech_research"
# 관점 Agent 노드 → (State 출력 키, 관점 이름)
PERSPECTIVE_NODES: dict[str, str] = {
    "trl_evaluation": "기술 성숙도",
    "market_evaluation": "시장성",
    "stakeholder_evaluation": "이해관계자",
    "domain_evaluation": "도메인 적용",
}
SYNTHESIS_NODE = "synthesis"
FAITHFULNESS_NODE = "faithfulness_check"
REPORT_NODE = "report_writer"
QUALITY_NODE = "quality_evaluation"

WORKER_NODES = [TECH_NODE, *PERSPECTIVE_NODES, SYNTHESIS_NODE, FAITHFULNESS_NODE, REPORT_NODE, QUALITY_NODE]
# 관점 결과가 바뀌면 다시 계산해야 하는 하류 노드
DOWNSTREAM_OF_PERSPECTIVES = [SYNTHESIS_NODE, FAITHFULNESS_NODE, REPORT_NODE, QUALITY_NODE]
REWORKABLE = {TECH_NODE, *PERSPECTIVE_NODES}

_NEEDS_RUN = {"pending", "running"}  # running = 직전 실행이 중단됨(재개 시 다시 실행)


def initial_control_state(run_id: str) -> dict[str, Any]:
    """init 노드가 주입하는 제어 메타데이터 초기값."""
    return {
        "run_id": run_id,
        "step_count": 0,
        "max_steps": settings.max_supervisor_steps,
        "next_nodes": [],
        "last_decision": {},
        "node_status": {name: "pending" for name in WORKER_NODES},
        "attempts": {},
        "errors": {},
        "rework_counts": {},
        "retry_hints": {},
        "sufficiency": {},
        "faithfulness_rounds": 0,
        "report_revisions": 0,
        "quality_feedback": "",
    }


# ── 근거 충분도 판정 (결정론적) ──────────────────────────────────────────────


def assess_sufficiency(state: GraphState, agent: str) -> dict[str, Any]:
    """관점 Agent 하나의 근거 충분도를 "관점 × 기술" 단위로, State만으로 판정한다.

    Agent 전체 합계로 세면 한 기술의 근거가 다른 기술을 가려 준다(예: 두 기술 모두 자기 논문
    1편에만 의존해도 합치면 출처 2종). 그래서 선정 기술마다 따로 본다.
    기준: 기술마다 (1) 그 기술로 태깅된 evidence_items 수 ≥ min_evidence_items,
         (2) 서로 다른 출처 수 ≥ min_distinct_sources (단일 출처 의존 방지),
         (3) insufficient_evidence=true가 아님.
    두 기술 공용 자료(technology=None)는 기술별 근거로 세지 않는다.
    """
    output = state.get(agent) or {}
    items = [e for e in state.get("evidence_items", []) if e.get("agent") == agent]

    per_tech: dict[str, dict[str, Any]] = {}
    problems = []
    for tech in state.get("selected_technologies", {}):
        tech_items = [e for e in items if e.get("technology") == tech]
        sources = {e.get("document_id") or e.get("source_url") for e in tech_items} - {None}
        tech_problems = []
        if len(tech_items) < settings.min_evidence_items:
            tech_problems.append(f"근거 {len(tech_items)}건 < {settings.min_evidence_items}")
        if len(sources) < settings.min_distinct_sources:
            tech_problems.append(f"출처 {len(sources)}종 < {settings.min_distinct_sources} (단일 출처 의존)")
        value = output.get(tech)
        if isinstance(value, dict) and value.get("insufficient_evidence"):
            tech_problems.append("Agent가 정보 부족으로 판정")
        per_tech[tech] = {"evidence_count": len(tech_items), "distinct_sources": len(sources)}
        if tech_problems:
            problems.append(f"{tech}: {', '.join(tech_problems)}")

    return {
        "sufficient": not problems,
        "per_technology": per_tech,
        "reason": "; ".join(problems) if problems else "기술별 근거 수·출처 다양성 기준 충족",
    }


# ── 정책 ──────────────────────────────────────────────────────────────────────


class _Decision:
    def __init__(self, state: GraphState) -> None:
        self.state = state
        self.status: dict[str, str] = dict(state.get("node_status", {}))
        self.updates: dict[str, Any] = {}
        self.attempts = dict(state.get("attempts", {}))
        self.rework_counts = dict(state.get("rework_counts", {}))

    def status_of(self, node: str) -> str:
        return self.status.get(node, "pending")

    def set_status(self, nodes: list[str], value: str) -> None:
        for node in nodes:
            self.status[node] = value
        self.updates.setdefault("node_status", {}).update({n: value for n in nodes})

    def can_rework(self, node: str) -> bool:
        return self.rework_counts.get(node, 0) < settings.max_rework_per_agent

    def resolve_failure(self, node: str) -> bool:
        """실패한 노드를 재시도할지(True) 제외할지(False) 결정한다 — Fall-back 정책."""
        self.attempts[node] = self.attempts.get(node, 0) + 1
        self.updates["attempts"] = self.attempts
        if self.attempts[node] <= settings.max_failure_retries:
            return True
        self.set_status([node], "excluded")
        return False

    def dispatch(self, nodes: list[str], action: str, reason: str) -> tuple[list[str], str, str]:
        self.set_status(nodes, "running")
        return nodes, action, reason

    def rework(self, hints: dict[str, str], action: str, reason: str) -> tuple[list[str], str, str]:
        """근거 부족 관점(또는 기술 조사)에 재작업을 요청하고 하류 산출물을 무효화한다."""
        targets = sorted(hints)
        for node in targets:
            self.rework_counts[node] = self.rework_counts.get(node, 0) + 1
        self.updates["rework_counts"] = self.rework_counts
        self.updates["retry_hints"] = hints
        # 기술 조사를 다시 하면 모든 관점이 그 결과에 의존하므로 함께 다시 돈다.
        if TECH_NODE in targets:
            self.set_status(list(PERSPECTIVE_NODES), "pending")
            targets = [TECH_NODE]
        self.set_status(DOWNSTREAM_OF_PERSPECTIVES, "pending")
        self.updates["quality_feedback"] = ""
        return self.dispatch(targets, action, reason)


def decide(state: GraphState) -> tuple[list[str], str, str, dict[str, Any]]:
    """현재 State로 (다음 노드 목록, action, reason, State 갱신분)을 결정한다. 순수 함수."""
    d = _Decision(state)
    step = state.get("step_count", 0) + 1
    d.updates["step_count"] = step
    targets, action, reason = _policy(d, state, step)
    return targets, action, reason, d.updates


def _policy(d: _Decision, state: GraphState, step: int) -> tuple[list[str], str, str]:
    # 0. 종료 가드 ───────────────────────────────────────────────────────────
    max_steps = state.get("max_steps", settings.max_supervisor_steps)
    # 상한 이후에도 품질 평가(필수 게이트)는 건너뛰지 않는다: 보고서가 없거나 낡았으면 1회 생성,
    # 현재 보고서가 미평가면 1회 평가 후 결과(통과/미달)와 무관하게 종료 → 상한 + 최대 2스텝.
    if step > max_steps:
        if d.status_of(REPORT_NODE) in _NEEDS_RUN:
            return d.dispatch([REPORT_NODE], "force_report", f"스텝 상한({max_steps}) 도달 — 보고서 생성 후 품질 평가 1회")
        if (state.get("final_report") and d.status_of(REPORT_NODE) == "done"
                and d.status_of(QUALITY_NODE) in _NEEDS_RUN):
            return d.dispatch([QUALITY_NODE], "force_quality", f"스텝 상한({max_steps}) 도달 — 종료 전 품질 평가 1회(필수 게이트)")
        verdict = state.get("quality_verdict") or {}
        outcome = "통과" if verdict.get("passed") else f"미달 {verdict.get('failed_criteria')}" if verdict else "평가 불가"
        return [], END_ACTION, f"스텝 상한({max_steps}) 도달 — 현재 보고서로 종료 (품질 평가: {outcome})"

    # 1. 기술 조사 ───────────────────────────────────────────────────────────
    tech = d.status_of(TECH_NODE)
    if tech in _NEEDS_RUN:
        return d.dispatch([TECH_NODE], "collect_tech", "technical_evidence 미수집 — 관점 평가의 공통 입력")
    if tech == "failed":
        if d.resolve_failure(TECH_NODE):
            return d.dispatch([TECH_NODE], "retry_failed", f"tech_research 실패 재시도: {state.get('errors', {}).get(TECH_NODE, '')[:120]}")
        # 제외되면 관점 Agent는 기술 조사 요약 없이 각자 RAG/외부검색으로 진행한다.

    # 2. 관점 수집 (병렬 디스패치) ──────────────────────────────────────────
    to_run, retried, excluded = [], [], []
    for node in PERSPECTIVE_NODES:
        status = d.status_of(node)
        if status in _NEEDS_RUN:
            to_run.append(node)
        elif status == "failed":
            (retried if d.resolve_failure(node) else excluded).append(node)
    if excluded:
        log_event(state.get("run_id", ""), "supervisor", "fallback_exclude", nodes=excluded,
                  errors={n: state.get("errors", {}).get(n, "") for n in excluded})
    if to_run or retried:
        missing = [PERSPECTIVE_NODES[n] for n in to_run + retried]
        reason = f"미수집 관점 {missing}" + (f" (실패 재시도: {retried})" if retried else "")
        return d.dispatch(sorted(to_run + retried), "collect_perspectives", reason)

    # 3. 근거 충분도 게이트 ─────────────────────────────────────────────────
    sufficiency = {
        node: assess_sufficiency(state, node) for node in PERSPECTIVE_NODES if d.status_of(node) == "done"
    }
    d.updates["sufficiency"] = sufficiency
    lacking = {
        node: f"[Supervisor 충분도 게이트] {verdict['reason']}. 다른 질의어·다른 출처로 근거를 보강하라."
        for node, verdict in sufficiency.items()
        if not verdict["sufficient"] and d.can_rework(node)
    }
    if lacking:
        detail = {PERSPECTIVE_NODES[n]: sufficiency[n]["reason"] for n in lacking}
        return d.rework(lacking, "rework_insufficient", f"근거 부족 관점 재작업 요청: {detail}")

    # 4. 종합 → 검증 (검증 재작업은 max_faithfulness_rounds 라운드까지) ───────────────────────────────────────────────────────
    for node in (SYNTHESIS_NODE, FAITHFULNESS_NODE):
        status = d.status_of(node)
        if status == "failed" and d.resolve_failure(node):
            return d.dispatch([node], "retry_failed", f"{node} 실패 재시도")
        if status in _NEEDS_RUN:
            reason = (
                "관점 수집·충분도 게이트 통과 — 관점 간 일치/상충 종합"
                if node == SYNTHESIS_NODE
                else "종합 claim이 수집 근거로 뒷받침되는지 검증 필요"
            )
            return d.dispatch([node], "synthesize" if node == SYNTHESIS_NODE else "verify", reason)

    check = state.get("faithfulness_check") or {}
    rounds = state.get("faithfulness_rounds", 0)
    if (d.status_of(FAITHFULNESS_NODE) == "done" and not check.get("passed", True)
            and rounds < settings.max_faithfulness_rounds):
        hints = check.get("retry_hints", {})
        targets = {n: hints.get(n, "검증 실패 claim 근거 보강") for n in check.get("agents_to_retry", [])
                   if n in REWORKABLE and d.can_rework(n)}
        if targets:
            failed_claims = len(check.get("insufficient_evidence_claims", []))
            d.updates["faithfulness_rounds"] = rounds + 1
            return d.rework(targets, "rework_unfaithful",
                            f"검증 실패 claim {failed_claims}건의 출처 Agent 재작업: {sorted(targets)}")
        # Agent 예산 소진/책임 Agent 불명 → 실패 claim은 보고서에 '근거 부족으로 검증되지 않음'으로 표기.

    # 5. 보고서 → 품질 평가 ────────────────────────────────────────────────
    for node in (REPORT_NODE, QUALITY_NODE):
        status = d.status_of(node)
        if status == "failed":
            if d.resolve_failure(node):
                return d.dispatch([node], "retry_failed", f"{node} 실패 재시도")
            if node == REPORT_NODE or not state.get("final_report"):
                return [], END_ACTION, f"{node} 반복 실패로 제외 — 종료"
            return [], END_ACTION, "품질 평가 반복 실패 — 평가 없이 현재 보고서로 종료"
        if status in _NEEDS_RUN:
            if node == REPORT_NODE:
                rev = state.get("report_revisions", 0)
                reason = "근거 충분성 확인 완료 — 보고서 작성" if rev == 0 else f"품질 미달 피드백 반영 재작성 ({rev}회차)"
                return d.dispatch([node], "write_report", reason)
            return d.dispatch([node], "evaluate_quality", "보고서 생성 후 품질 평가(필수 게이트)")

    # 6. 품질 verdict 분기 ──────────────────────────────────────────────────
    verdict = state.get("quality_verdict") or {}
    if verdict.get("passed"):
        return [], END_ACTION, "품질 평가 4개 항목 통과 — 종료"

    failed = verdict.get("failed_criteria", [])
    targets = {
        n: f"[품질 평가 미달: {', '.join(failed)}] {verdict.get('feedback', '')[:300]}"
        for n in verdict.get("rework_targets", [])
        if n in PERSPECTIVE_NODES and d.can_rework(n)
    }
    if targets:
        decision = d.rework(targets, "rework_quality", f"품질 미달({failed}) — 관점 근거 문제로 판정된 Agent 재작업: {sorted(targets)}")
        d.updates["quality_feedback"] = verdict.get("feedback", "")  # 재작성될 보고서에도 피드백 전달
        return decision

    revisions = state.get("report_revisions", 0)
    if revisions < settings.max_report_revisions:
        d.updates["report_revisions"] = revisions + 1
        d.updates["quality_feedback"] = verdict.get("feedback", "")
        d.set_status([QUALITY_NODE], "pending")
        return d.dispatch([REPORT_NODE], "revise_report", f"품질 미달({failed}) — 보고서 재작성 요청 ({revisions + 1}/{settings.max_report_revisions})")

    return [], END_ACTION, f"품질 미달({failed})이나 재작성 예산 소진 — 미달 사항을 verdict로 남기고 종료"


def supervisor_node(state: GraphState) -> dict[str, Any]:
    targets, action, reason, updates = decide(state)
    step = updates["step_count"]
    decision = {"step": step, "action": action, "targets": targets, "reason": reason}
    print(f"[supervisor] step {step:>2} | {action:<22} -> {targets or 'END'} | {reason}", flush=True)
    log_event(
        state.get("run_id", ""),
        "supervisor",
        "decision",
        step=step,
        action=action,
        targets=targets,
        reason=reason,
        sufficiency=updates.get("sufficiency"),
        rework_counts=updates.get("rework_counts", state.get("rework_counts")),
    )
    return {**updates, "next_nodes": targets, "last_decision": decision}


def route_from_supervisor(state: GraphState) -> list[str] | str:
    """조건부 엣지: Supervisor가 State에 남긴 next_nodes로 분기한다 (빈 목록이면 END)."""
    return state.get("next_nodes") or END


def count_actions(decisions: list[dict[str, Any]]) -> Counter:
    return Counter(d.get("action") for d in decisions)
