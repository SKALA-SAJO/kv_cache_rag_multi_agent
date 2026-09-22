"""이해관계자 평가 Agent.

설계상 RAG는 사용하지 않는다. 대신 Phase 3에서 등록할 외부 검색 도구를 공통
tool-calling 루프로 호출하고, 실제 검색 결과만 evidence_items와 references에 기록한다.
"""

from concurrent.futures import ThreadPoolExecutor

from agents.base import (
    get_external_search_tool,
    load_prompt,
    search_results_to_evidence_items,
    search_results_to_references,
    structured_call_with_external_search,
)
from agents.schemas import StakeholderAssessment
from graph.state import GraphState

SYSTEM_PROMPT = load_prompt("stakeholder_evaluation")
AGENT_NAME = "stakeholder_evaluation"


def _process_technology(
    tech_name: str, technical_evidence: dict, search_tool, retry_hint: str
) -> tuple[str, dict, list[dict], list[dict]]:
    """기술 하나를 독립적으로 평가한다. 다른 기술의 이해관계자 판단을 참조하지
    않으므로 기술별로 병렬 실행해도 결과는 순차 실행과 동일하다. search_tool은 이미
    그래프 레벨에서 market/stakeholder 두 Agent가 동시에 호출하는 공유 싱글턴이라,
    기술별 동시 호출도 같은 전제 위에 있다."""
    user_content = (
        f"## 기술명\n{tech_name}\n\n"
        f"## 기술 조사 Agent 요약\n{technical_evidence.get(tech_name, {})}"
    )
    if retry_hint:
        user_content += f"\n\n## 이전 검증에서 부족했던 부분\n{retry_hint}"

    result, search_results = structured_call_with_external_search(
        StakeholderAssessment,
        SYSTEM_PROMPT,
        user_content,
        search_tool,
    )
    evidence_items = search_results_to_evidence_items(
        search_results,
        agent=AGENT_NAME,
        claim=f"{tech_name}의 이해관계자 평가에 사용한 외부 검색 근거",
    )
    references = search_results_to_references(search_results)
    return tech_name, result.model_dump(), evidence_items, references


def run(state: GraphState) -> dict:
    technologies = state["selected_technologies"]
    technical_evidence = state.get("technical_evidence", {})
    retry_hint = state.get("retry_hints", {}).get(AGENT_NAME, "")
    search_tool = get_external_search_tool()
    result_by_tech: dict = {}
    evidence_items: list[dict] = []
    references: list[dict] = []

    # 기술 간 참조가 없는 독립 작업이므로 병렬 실행한다(외부검색 tool-calling 포함 —
    # 가장 느린 구간). 제출 순서대로 결과를 모아(완료 순서가 아님) 순차 실행과 동일한
    # 병합 순서를 보장한다.
    with ThreadPoolExecutor(max_workers=len(technologies) or 1) as executor:
        futures = [
            executor.submit(_process_technology, tech_name, technical_evidence, search_tool, retry_hint)
            for tech_name in technologies
        ]
        for future in futures:
            tech_name, result_dump, tech_evidence_items, tech_references = future.result()
            result_by_tech[tech_name] = result_dump
            evidence_items.extend(tech_evidence_items)
            references.extend(tech_references)

    return {
        "stakeholder_evaluation": result_by_tech,
        "evidence_items": evidence_items,
        "references": references,
    }
