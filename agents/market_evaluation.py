"""시장 평가 Agent.

시장 자료 RAG 코퍼스 필터링(Phase 5)은 아직 붙이지 않는다. 대신 팀이 Phase 3에서
등록할 외부 검색 도구를 공통 tool-calling 루프로 호출하고, 실제 검색 결과만
evidence_items와 references에 기록한다.
"""

from agents.base import (
    get_external_search_tool,
    load_prompt,
    search_results_to_evidence_items,
    search_results_to_references,
    structured_call_with_external_search,
)
from agents.schemas import QualitativeAssessment
from graph.state import GraphState

SYSTEM_PROMPT = load_prompt("market_evaluation")
AGENT_NAME = "market_evaluation"


def run(state: GraphState) -> dict:
    technologies = state["selected_technologies"]
    technical_evidence = state.get("technical_evidence", {})
    retry_hint = state.get("retry_hints", {}).get(AGENT_NAME, "")
    search_tool = get_external_search_tool()
    result_by_tech: dict = {}
    evidence_items: list[dict] = []
    references: list[dict] = []

    for tech_name, tech_info in technologies.items():
        user_content = (
            f"## 기술명\n{tech_name}\n\n"
            f"## 분류\n{tech_info['category']}\n\n"
            f"## 기술 조사 Agent 요약\n{technical_evidence.get(tech_name, {})}"
        )
        if retry_hint:
            user_content += f"\n\n## 이전 검증에서 부족했던 부분\n{retry_hint}"

        result, search_results = structured_call_with_external_search(
            QualitativeAssessment,
            SYSTEM_PROMPT,
            user_content,
            search_tool,
        )
        result_by_tech[tech_name] = result.model_dump()
        evidence_items.extend(
            search_results_to_evidence_items(
                search_results,
                agent=AGENT_NAME,
                claim=f"{tech_name}의 시장성 평가에 사용한 외부 검색 근거",
            )
        )
        references.extend(search_results_to_references(search_results))

    return {
        "market_evaluation": result_by_tech,
        "evidence_items": evidence_items,
        "references": references,
    }
