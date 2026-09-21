"""도메인 평가 Agent (PDF Agent H). RAG 여부: O — 기술 문서에서 장문맥 관련 근거를 검색한다."""

from agents.base import documents_to_references, format_context, load_prompt, structured_call
from agents.schemas import RubricAssessment
from graph.state import GraphState
from rag.retriever import retrieve

SYSTEM_PROMPT = load_prompt("domain_evaluation")


def run(state: GraphState) -> dict:
    technologies = state["selected_technologies"]
    technical_evidence = state.get("technical_evidence", {})
    all_docs = []
    result_by_tech: dict = {}

    for tech_name in technologies:
        query = f"{tech_name} 장문맥 long context 컨텍스트 길이 메모리 효율 정확도 지연"
        docs = retrieve(query)
        all_docs.extend(docs)

        context = format_context(docs)
        user_content = (
            f"## 기술명\n{tech_name}\n\n"
            f"## 기술 조사 Agent 요약\n{technical_evidence.get(tech_name, {})}\n\n"
            f"## Context\n{context}"
        )
        result = structured_call(RubricAssessment, SYSTEM_PROMPT, user_content)
        result_by_tech[tech_name] = result.model_dump()

    return {
        "retrieved_documents": all_docs,
        "domain_evaluation": result_by_tech,
        "references": documents_to_references(all_docs),
    }
