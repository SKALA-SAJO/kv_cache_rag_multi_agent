"""LangGraph 워크플로우 조립 (RAG-Design PDF D.2 절 Graph 흐름 설계를 그대로 구현).

PDF의 노드 A~L 대응 관계:
  A(평가 질문 입력)+B(기술 정보 확인)         -> init_node
  C(기술 문서 RAG 검색)+D(기술 조사 Agent)      -> tech_research (자체적으로 RAG 검색 수행)
  E/F/G/H(4관점 평가 Agent)                    -> trl/market/stakeholder/domain_evaluation
  I(평가 종합 Agent)+S(synthesis 생성)          -> synthesis
  V(검증 Agent)                                -> faithfulness_check
  J(분기: 근거 충분한가?)                       -> route_after_faithfulness (조건부 엣지)
  K(보고서 생성 Agent)+L(최종 보고서)            -> report_writer

C/D, I/S를 하나의 노드로 합친 이유는 각 단계가 순수 함수 하나로 표현 가능해 별도 노드로
쪼개는 이점이 없기 때문이다 (README 'PDF 설계와의 대응' 절 참고).
"""

from langgraph.graph import END, StateGraph

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
from technologies import SELECTED_TECHNOLOGIES


def init_node(state: GraphState) -> dict:
    """A. 평가 질문 입력 + B. 기술 정보 확인 (Human 기반 선정 결과를 State에 주입)."""
    return {
        "selected_technologies": SELECTED_TECHNOLOGIES,
        "verification_retry_count": 0,
    }


def route_after_faithfulness(state: GraphState) -> str:
    """J. 근거와 출처가 충분한가? (Faithfulness Check 통과 기준)."""
    check = state.get("faithfulness_check", {})
    retry_count = state.get("verification_retry_count", 0)

    if check.get("passed", False):
        return "report"
    if retry_count >= settings.max_verification_retries:
        # 재시도 한도 초과: 보고서는 생성하되 6.1 '한계' 절에 검증 미통과 사실이 반영된다.
        return "report"
    return "retry"


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("init", init_node)
    graph.add_node("tech_research", tech_research.run)
    graph.add_node("trl_evaluation", trl_evaluation.run)
    graph.add_node("market_evaluation", market_evaluation.run)
    graph.add_node("stakeholder_evaluation", stakeholder_evaluation.run)
    graph.add_node("domain_evaluation", domain_evaluation.run)
    graph.add_node("synthesis", synthesis_agent.run)
    graph.add_node("faithfulness_check", faithfulness_check.run)
    graph.add_node("report_writer", report_writer.run)

    graph.set_entry_point("init")
    graph.add_edge("init", "tech_research")

    # D -> E, F, G, H (병렬 팬아웃) -> I (팬인)
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
        {"retry": "tech_research", "report": "report_writer"},
    )

    graph.add_edge("report_writer", END)

    return graph.compile()
