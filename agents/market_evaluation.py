"""시장 평가 Agent. RAG 여부: O(시장 자료 코퍼스) + 외부 정보 검색 도구.

시장 자료(Gemini API Long Context 공식 문서, doc_type=market_document) 코퍼스를
기술과 무관하게 한 번 검색해 장문맥 기능의 실제 활용 사례·비용·지연·확장성 근거로
쓰고, 기술별 판단(상용화·채택 사례 등)은 외부 검색 도구(Phase 3 API 등록 전에는
get_external_search_tool()이 None이라 tool-calling이 자동으로 스킵됨)로 보강한다.
"""

from langchain_core.documents import Document

from agents.base import (
    documents_to_evidence_items,
    documents_to_references,
    format_context,
    get_external_search_tool,
    load_prompt,
    search_results_to_evidence_items,
    search_results_to_references,
    structured_call_with_external_search,
)
from agents.schemas import QualitativeAssessment
from graph.state import GraphState
from rag.retriever import retrieve
from scripts.download_papers import DOC_TYPE_MARKET

SYSTEM_PROMPT = load_prompt("market_evaluation")
AGENT_NAME = "market_evaluation"


def _retrieve_market_corpus(retry_hint: str = "") -> list[Document]:
    """시장 자료(Gemini API Long Context 문서) 코퍼스를 명시적으로 검색한다.

    sample.pdf 'RAG 문서 구성'에서 시장 자료는 기술과 무관하게(technology=None) 장문맥
    기능의 실제 활용 사례·비용·지연·확장성 고려사항을 확인하는 용도다. domain_evaluation의
    _retrieve_domain_corpus(LongBench/RULER)와 동일한 패턴으로 doc_type만 필터링해
    기술별이 아니라 한 번만 검색한다.
    """
    query = "Gemini API 장문맥(long context) 기능 실제 활용 사례 비용 지연 확장성 고려사항"
    if retry_hint:
        query = f"{query} {retry_hint}"
    return retrieve(query, doc_types=DOC_TYPE_MARKET)


def run(state: GraphState) -> dict:
    technologies = state["selected_technologies"]
    technical_evidence = state.get("technical_evidence", {})
    retry_hint = state.get("retry_hints", {}).get(AGENT_NAME, "")
    search_tool = get_external_search_tool()
    result_by_tech: dict = {}
    evidence_items: list[dict] = []
    references: list[dict] = []

    market_docs = _retrieve_market_corpus(retry_hint=retry_hint)
    evidence_items.extend(
        documents_to_evidence_items(
            market_docs,
            agent=AGENT_NAME,
            claim="장문맥 기능 실제 활용 사례·비용·확장성 근거(Gemini API 문서)",
        )
    )
    references.extend(documents_to_references(market_docs))
    market_context = format_context(market_docs)

    for tech_name, tech_info in technologies.items():
        user_content = (
            f"## 기술명\n{tech_name}\n\n"
            f"## 분류\n{tech_info['category']}\n\n"
            f"## 기술 조사 Agent 요약\n{technical_evidence.get(tech_name, {})}\n\n"
            f"## 시장 자료 RAG 근거 (Gemini API Long Context 문서, 기술 비특정)\n{market_context}"
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
