"""LangGraph 워크플로우 조립 — Supervisor 패턴.

    init -> supervisor ─┬─> tech_research ───────────┐
                        ├─> trl/market/stakeholder/   │ (Supervisor가 고른 노드만, 필요 시 병렬)
                        │   domain_evaluation         │
                        ├─> synthesis                 ├─> supervisor (모든 하위 노드는 여기로만 복귀)
                        ├─> faithfulness_check        │
                        ├─> report_writer             │
                        ├─> quality_evaluation ───────┘
                        └─> END

- 하위 Agent → Supervisor 단방향 엣지만 있고, 하위 Agent 간 엣지는 없다 (직접 통신 금지).
- Supervisor → 하위 Agent는 add_conditional_edges 한 곳뿐이며, 분기 값은 Supervisor가 State를 보고
  결정한 next_nodes다 (graph/supervisor.py의 decide). 순서·스텝 수는 고정되어 있지 않다.
- 하위 노드는 _worker 래퍼로 감싸 예외를 State(node_status=failed, errors)로 바꾼다 → Supervisor가
  재시도/제외(Fall-back)를 결정한다. 병렬 디스패치 중 한 Agent가 죽어도 그래프 전체가 멈추지 않는다.
"""

from __future__ import annotations

import traceback
import uuid
from typing import Callable

from langgraph.graph import END, StateGraph

from agents import (
    domain_evaluation,
    faithfulness_check,
    market_evaluation,
    quality_evaluation,
    report_writer,
    stakeholder_evaluation,
    synthesis as synthesis_agent,
    tech_research,
    trl_evaluation,
)
from graph.observability import Timer, log_event, now_ts, record_timing
from graph.state import GraphState
from graph.supervisor import WORKER_NODES, initial_control_state, route_from_supervisor, supervisor_node
from rubrics import EVALUATION_RUBRIC
from technologies import SELECTED_TECHNOLOGIES

AGENT_RUNNERS: dict[str, Callable[[GraphState], dict]] = {
    "tech_research": tech_research.run,
    "trl_evaluation": trl_evaluation.run,
    "market_evaluation": market_evaluation.run,
    "stakeholder_evaluation": stakeholder_evaluation.run,
    "domain_evaluation": domain_evaluation.run,
    "synthesis": synthesis_agent.run,
    "faithfulness_check": faithfulness_check.run,
    "report_writer": report_writer.run,
    "quality_evaluation": quality_evaluation.run,
}
assert set(AGENT_RUNNERS) == set(WORKER_NODES)


def _worker(name: str, fn: Callable[[GraphState], dict]) -> Callable[[GraphState], dict]:
    """하위 노드 공통 래퍼: 실행 상태·에러를 제어 메타로 남기고, 실행 이벤트는 외부 로그로 보낸다."""

    def wrapper(state: GraphState) -> dict:
        run_id = state.get("run_id", "")
        hint = state.get("retry_hints", {}).get(name, "")
        log_event(run_id, name, "start", rework_hint=hint[:200] or None)
        print(f"[{name}] 시작" + (" (재작업)" if hint else ""), flush=True)
        with Timer() as timer:
            try:
                update = fn(state)
                error = None
            except Exception as exc:  # noqa: BLE001 - 실패는 Supervisor가 처리한다
                update = {}
                error = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
        record_timing(name, timer.elapsed)
        status = "failed" if error else "done"
        log_event(run_id, name, "end", status=status, elapsed_s=round(timer.elapsed, 1), error=error)
        print(f"[{name}] {status} - {timer.elapsed:.1f}초", flush=True)
        update["node_status"] = {name: status}
        if error:
            update["errors"] = {name: error}  # 형식 유지: Supervisor가 문자열로 읽는다
            update["error_times"] = {name: now_ts()}  # 언제 실패했는가 (병렬 실패는 merge_dict로 병합)
        return update

    return wrapper


def init_node(state: GraphState) -> dict:
    """평가 질문 + Human 선정 기술 + Rubric 로드, 제어 메타데이터 초기화."""
    run_id = state.get("run_id") or str(uuid.uuid4())
    log_event(run_id, "init", "start", research_question=state.get("research_question"))
    return {
        "selected_technologies": SELECTED_TECHNOLOGIES,
        "evaluation_rubric": EVALUATION_RUBRIC,
        **initial_control_state(run_id),
    }


def build_graph(checkpointer=None):
    graph = StateGraph(GraphState)
    graph.add_node("init", init_node)
    graph.add_node("supervisor", supervisor_node)
    for name, fn in AGENT_RUNNERS.items():
        graph.add_node(name, _worker(name, fn))
        graph.add_edge(name, "supervisor")  # 하위 노드는 Supervisor로만 복귀

    graph.set_entry_point("init")
    graph.add_edge("init", "supervisor")
    graph.add_conditional_edges("supervisor", route_from_supervisor, [*AGENT_RUNNERS, END])
    return graph.compile(checkpointer=checkpointer)
