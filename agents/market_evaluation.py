"""시장 평가 Agent (PDF Agent). RAG 여부: O(시장자료 코퍼스) + 외부 정보 검색 도구.

RAG는 rag/retriever.py의 market_document doc_type 필터로 시장자료(Gemini API Long
Context 문서)만 검색한다. 여기에 Tavily 외부 검색 결과를 더해 최신 채택 현황을 보강한다.
두 출처 모두 evidence_items로 등록되며(source_type="RAG"/"external_search"), 검증
Agent가 claim의 근거를 이 두 출처 중 어디서 찾았는지로 재검색 대상 Agent를 판단한다.
"""

from agents.base import (
    documents_to_evidence_items,
    documents_to_references,
    format_context,
    load_prompt,
    search_results_to_evidence_items,
    structured_call,
)
from agents.schemas import QualitativeAssessment
from graph.state import GraphState
from rag.external_search import search as external_search
from rag.retriever import retrieve

SYSTEM_PROMPT = load_prompt("market_evaluation")
AGENT_NAME = "market_evaluation"
MARKET_DOC_TYPES = frozenset({"market_document"})


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
    all_docs = []
    all_evidence_items: list[dict] = []
    result_by_tech: dict = {}

    for tech_name, tech_info in technologies.items():
        query = f"{tech_name} {tech_info['core_approach']} 시장 채택 생태계 상용화 프레임워크 지원"
        if retry_hint:
            query = f"{query} {retry_hint}"

        docs = retrieve(query, doc_types=MARKET_DOC_TYPES)
        all_docs.extend(docs)
        all_evidence_items.extend(
            documents_to_evidence_items(docs, agent=AGENT_NAME, claim=f"{tech_name} 시장성 근거(문서)")
        )

        search_results = external_search(f"{tech_name} adoption production deployment framework support")
        all_evidence_items.extend(
            search_results_to_evidence_items(
                search_results, agent=AGENT_NAME, claim=f"{tech_name} 시장성 근거(외부 검색)"
            )
        )

        user_content = (
            f"## 기술명\n{tech_name}\n\n"
            f"## 분류\n{tech_info['category']}\n\n"
            f"## 기술 조사 Agent 요약\n{technical_evidence.get(tech_name, {})}\n\n"
            f"## 시장자료 RAG 검색 결과\n{format_context(docs)}\n\n"
            f"## 외부 검색 결과\n{_format_search_results(search_results)}"
        )
        if retry_hint:
            user_content += f"\n\n## 이전 검증에서 부족했던 부분\n{retry_hint}"

        result = structured_call(QualitativeAssessment, SYSTEM_PROMPT, user_content)
        result_by_tech[tech_name] = result.model_dump()

    return {
        "market_evaluation": result_by_tech,
        "retrieved_documents": all_docs,
        "references": documents_to_references(all_docs),
        "evidence_items": all_evidence_items,
    }
