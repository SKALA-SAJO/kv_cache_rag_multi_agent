"""도메인 평가 Agent (PDF Agent). RAG 여부: O — 기술 문서에서 장문맥 관련 근거를 검색한다.

담당(Phase 2-B, 최윤영): 도메인자료(LongBench·RULER 논문) 코퍼스 검색 추가는 아직
반영되지 않았다. 현재는 기존 기술원문(논문) 코퍼스만 검색한다.
"""

from agents.base import (
    documents_to_evidence_items,
    documents_to_references,
    format_context,
    load_prompt,
    structured_call,
)
from agents.schemas import QualitativeAssessment
from graph.state import GraphState
from rag.retriever import retrieve

SYSTEM_PROMPT = load_prompt("domain_evaluation")
AGENT_NAME = "domain_evaluation"


def run(state: GraphState) -> dict:
    technologies = state["selected_technologies"]
    technical_evidence = state.get("technical_evidence", {})
    retry_hint = state.get("retry_hints", {}).get(AGENT_NAME, "")
    all_docs = []
    all_evidence_items: list[dict] = []
    result_by_tech: dict = {}

    for tech_name in technologies:
        query = f"{tech_name} 장문맥 long context 컨텍스트 길이 메모리 효율 정확도 지연"
        if retry_hint:
            query = f"{query} {retry_hint}"
        docs = retrieve(query)
        all_docs.extend(docs)
        all_evidence_items.extend(
            documents_to_evidence_items(docs, agent=AGENT_NAME, claim=f"{tech_name} 도메인 적합성 근거")
        )

        context = format_context(docs)
        user_content = (
            f"## 기술명\n{tech_name}\n\n"
            f"## 기술 조사 Agent 요약\n{technical_evidence.get(tech_name, {})}\n\n"
            f"## Context\n{context}"
        )
        result = structured_call(QualitativeAssessment, SYSTEM_PROMPT, user_content)
        result_by_tech[tech_name] = result.model_dump()

    return {
        "retrieved_documents": all_docs,
        "domain_evaluation": result_by_tech,
        "references": documents_to_references(all_docs),
        "evidence_items": all_evidence_items,
    }
