"""LangGraph 워크플로우 조립 (sample.pdf D절 Graph 흐름 설계, 9쪽 플로우차트를 반영).

PDF 노드 대응 관계:
  평가 질문 입력 + 기술 정보 및 평가 Rubric 로드   -> init_node
  기술원문 RAG 검색 + 기술 조사 Agent              -> tech_research (자체적으로 RAG 검색 수행)
  기술 성숙도/시장/이해관계자/도메인 평가 Agent     -> trl/market/stakeholder/domain_evaluation
  평가 종합 Agent + synthesis 생성                 -> synthesis
  검증 Agent(Faithfulness Check)                   -> faithfulness_check
  {Faithfulness Check 통과?} 분기                   -> route_after_faithfulness (조건부 엣지)
  보고서 생성 Agent + 최종 평가 보고서               -> report_writer

## 재검색 루프 (9쪽 플로우차트의 핵심 차이점)
기존 v0.0은 실패 시 무조건 tech_research부터 전체를 다시 돌았다. PDF 업데이트본은
"실패 claim의 출처 Agent로 라우팅"하여 실패에 책임 있는 Agent만 재실행하도록 요구한다.
이를 위해 LangGraph의 Send API로 faithfulness_check가 계산한 agents_to_retry 목록에
있는 노드만 동적으로 재호출한다. 재실행된 Agent의 출력은 기존 정적 엣지(예:
market_evaluation -> synthesis)를 그대로 타고 흘러 synthesis가 다시 실행되며, 이때
재실행되지 않은 다른 평가 Agent들의 값은 State에 남아있는 이전 결과가 그대로 쓰인다
(LangGraph는 매 superstep마다 "이번에 갱신된 채널 중 하나라도 자신의 입력과 걸려 있으면"
재실행하는 방식이라 나머지 3개를 다시 돌 필요가 없다).
"""

import time
from collections import defaultdict
from typing import Callable

from langgraph.graph import END, StateGraph
from langgraph.types import Send

from agents import (
    domain_evaluation,
    faithfulness_check,
    market_evaluation,
    report_writer,
    stakeholder_evaluation,
    synthesis as synthesis_agent,
    tech_research,
    trl_evaluation,
)
from config import settings
from graph.state import GraphState
from rubrics import EVALUATION_RUBRIC
from technologies import SELECTED_TECHNOLOGIES

# 노드별 누적 실행 시간 (재시도로 같은 노드가 여러 번 실행되면 합산한다).
_NODE_ELAPSED: dict[str, float] = defaultdict(float)
_NODE_RUN_COUNT: dict[str, int] = defaultdict(int)


def _timed(name: str, fn: Callable[[GraphState], dict]) -> Callable[[GraphState], dict]:
    """Agent 실행 시간을 측정해 터미널에 출력하고, 요약용으로 누적한다."""

    def wrapper(state: GraphState) -> dict:
        start = time.perf_counter()
        print(f"[timing] {name} 시작", flush=True)
        try:
            return fn(state)
        finally:
            elapsed = time.perf_counter() - start
            _NODE_ELAPSED[name] += elapsed
            _NODE_RUN_COUNT[name] += 1
            print(f"[timing] {name} 완료 - {elapsed:.1f}초", flush=True)

    return wrapper


def print_timing_summary() -> None:
    """실행 종료 후 단계별 누적 소요 시간을 표로 출력한다. app.py가 호출한다."""
    if not _NODE_ELAPSED:
        return
    print("\n=== 단계별 실행 시간 요약 ===")
    for name, total in sorted(_NODE_ELAPSED.items(), key=lambda kv: kv[1], reverse=True):
        runs = _NODE_RUN_COUNT[name]
        suffix = f" ({runs}회 실행 합계)" if runs > 1 else ""
        print(f"  {name:<24} {total:7.1f}초{suffix}")
    print(f"  {'합계(노드 실행 시간)':<24} {sum(_NODE_ELAPSED.values()):7.1f}초")

# 재검색 대상이 될 수 있는 Agent 노드 (evidence_items를 만드는 Agent 전체).
# PDF 표는 evidence_items 생성 Agent로 기술조사·시장·이해관계자·도메인만 명시하지만,
# 기술 성숙도 평가 Agent도 자체 RAG 검색을 하므로(PDF Agent 정의 표 RAG=O) 동일하게
# 재검색 대상에 포함한다.
EVALUATION_AGENT_NODES = [
    "tech_research",
    "trl_evaluation",
    "market_evaluation",
    "stakeholder_evaluation",
    "domain_evaluation",
]


def init_node(state: GraphState) -> dict:
    """평가 질문 입력 + 기술 정보 및 평가 Rubric 로드 (Human 선정값 + Rubric을 State에 주입)."""
    return {
        "selected_technologies": SELECTED_TECHNOLOGIES,
        "evaluation_rubric": EVALUATION_RUBRIC,
        "retry_count": 0,
        "max_retries": settings.max_verification_retries,
        "retry_hints": {},
    }


def route_after_faithfulness(state: GraphState):
    """{Faithfulness Check 통과?} 분기.

    - 통과: report_writer로 진행.
    - 실패 + 재시도 소진: '정보 부족 claim 표시 및 한계 기록'은 faithfulness_check가
      이미 State에 남겼으므로 그대로 report_writer로 진행.
    - 실패 + 재시도 가능: 실패 claim의 출처 Agent들만 Send로 재호출.
    - 실패했지만 책임 Agent를 특정할 수 없는 경우(예: 아직 evidence_items를 만들지
      않는 Agent의 주장): 재시도해도 해결되지 않으므로 report_writer로 진행.
    """
    check = state.get("faithfulness_check", {})
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", settings.max_verification_retries)

    if check.get("passed", False):
        return "report_writer"
    if retry_count >= max_retries:
        return "report_writer"

    agents_to_retry = {
        name for name in check.get("agents_to_retry", []) if name in EVALUATION_AGENT_NODES
    }
    if not agents_to_retry:
        return "report_writer"

    # tech_research를 재실행하면 정적 엣지(tech_research -> 4관점 평가 Agent)를 타고
    # 4개 평가 Agent가 어차피 다시 실행된다. 이 상태에서 그 중 하나를 또 Send로 직접
    # 보내면 같은 라운드에 두 번 실행되어(한 번은 tech_research의 팬아웃으로, 한 번은
    # 직접 Send로) synthesis/faithfulness_check/report_writer까지 중복 실행되는 버그가
    # 있었다(2026-09-22 실제 API 실행에서 report_writer가 2번 도는 것으로 발견). 따라서
    # tech_research가 재시도 대상이면 하위 4개는 명시적 Send 목록에서 제외한다.
    if "tech_research" in agents_to_retry:
        downstream = agents_to_retry & {
            "trl_evaluation",
            "market_evaluation",
            "stakeholder_evaluation",
            "domain_evaluation",
        }
        agents_to_retry -= downstream

    return [Send(name, dict(state)) for name in sorted(agents_to_retry)]


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("init", init_node)
    graph.add_node("tech_research", _timed("tech_research", tech_research.run))
    graph.add_node("trl_evaluation", _timed("trl_evaluation", trl_evaluation.run))
    graph.add_node("market_evaluation", _timed("market_evaluation", market_evaluation.run))
    graph.add_node(
        "stakeholder_evaluation", _timed("stakeholder_evaluation", stakeholder_evaluation.run)
    )
    graph.add_node("domain_evaluation", _timed("domain_evaluation", domain_evaluation.run))
    graph.add_node("synthesis", _timed("synthesis", synthesis_agent.run))
    graph.add_node("faithfulness_check", _timed("faithfulness_check", faithfulness_check.run))
    graph.add_node("report_writer", _timed("report_writer", report_writer.run))

    graph.set_entry_point("init")
    graph.add_edge("init", "tech_research")

    # 기술 조사 Agent -> 4관점 평가 Agent (병렬 팬아웃) -> 평가 종합 Agent (팬인)
    for perspective_node in [
        "trl_evaluation",
        "market_evaluation",
        "stakeholder_evaluation",
        "domain_evaluation",
    ]:
        graph.add_edge("tech_research", perspective_node)
        graph.add_edge(perspective_node, "synthesis")

    graph.add_edge("synthesis", "faithfulness_check")

    graph.add_conditional_edges(
        "faithfulness_check",
        route_after_faithfulness,
        EVALUATION_AGENT_NODES + ["report_writer"],
    )

    graph.add_edge("report_writer", END)

    return graph.compile()
