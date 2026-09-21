"""기술 성숙도 평가 Agent (PDF Agent E). RAG 여부: O — 기술 문서에서 TRL 관련 근거를 검색한다."""

from agents.base import documents_to_references, format_context, load_prompt, structured_call
from agents.schemas import RubricAssessment
from graph.state import GraphState
from rag.retriever import retrieve

SYSTEM_PROMPT = load_prompt("trl_evaluation")


def run(state: GraphState) -> dict:
    technologies = state["selected_technologies"]
    technical_evidence = state.get("technical_evidence", {})
    all_docs = []
    result_by_tech: dict = {}

    for tech_name in technologies:
        query = f"{tech_name} 공개 구현 오픈소스 실험 환경 벤치마크 상용 적용 TRL"
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
        "trl_evaluation": result_by_tech,
        "references": documents_to_references(all_docs),
    }
