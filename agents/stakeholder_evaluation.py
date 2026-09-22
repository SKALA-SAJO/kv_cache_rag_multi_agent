"""이해관계자 평가 Agent (PDF Agent). RAG 여부: X. 도구: 외부 정보 검색 도구.

기술 조사 Agent의 technical_evidence + Tavily 외부 검색 결과("공개 웹 자료")를 입력으로
쓴다. 평가 대상은 경쟁 기술 진영·개발자·도입 기업 및 서비스 운영자·투자·업계 관계자
4유형(prompts/stakeholder_evaluation.md 참고). 검색 결과는 evidence_items로 등록된다.
"""

from agents.base import load_prompt, search_results_to_evidence_items, structured_call
from agents.schemas import StakeholderAssessment
from graph.state import GraphState
from rag.external_search import search as external_search

SYSTEM_PROMPT = load_prompt("stakeholder_evaluation")
AGENT_NAME = "stakeholder_evaluation"


def _format_search_results(results: list[dict]) -> str:
    if not results:
        return "(외부 검색 결과 없음)"
    blocks = []
    for i, r in enumerate(results, start=1):
        blocks.append(f"[외부-{i}] {r.get('title', '')} ({r.get('url', '')})\n{r.get('content', '')}")
    return "\n\n".join(blocks)


def run(state: GraphState) -> dict:
    technologies = state["selected_technologies"]
    technical_evidence = state.get("technical_evidence", {})
    retry_hint = state.get("retry_hints", {}).get(AGENT_NAME, "")
    all_evidence_items: list[dict] = []
    result_by_tech: dict = {}

    for tech_name in technologies:
        query = f"{tech_name} competitors developers adoption cost concerns industry reaction"
        if retry_hint:
            query = f"{query} {retry_hint}"

        search_results = external_search(query)
        all_evidence_items.extend(
            search_results_to_evidence_items(
                search_results, agent=AGENT_NAME, claim=f"{tech_name} 이해관계자 근거(외부 검색)"
            )
        )

        user_content = (
            f"## 기술명\n{tech_name}\n\n"
            f"## 기술 조사 Agent 요약\n{technical_evidence.get(tech_name, {})}\n\n"
            f"## 외부 검색 결과\n{_format_search_results(search_results)}"
        )
        if retry_hint:
            user_content += f"\n\n## 이전 검증에서 부족했던 부분\n{retry_hint}"

        result = structured_call(StakeholderAssessment, SYSTEM_PROMPT, user_content)
        result_by_tech[tech_name] = result.model_dump()

    return {
        "stakeholder_evaluation": result_by_tech,
        "evidence_items": all_evidence_items,
    }
