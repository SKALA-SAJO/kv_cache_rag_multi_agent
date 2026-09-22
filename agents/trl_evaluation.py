"""기술 성숙도 평가 Agent (PDF Agent). RAG 여부: O — 기술 문서에서 TRL 관련 근거를 검색한다.

담당(Phase 2-A, 서지원): 구현자료(GitHub README) 코퍼스 검색 추가는 아직 반영되지 않았다.
"""

from agents.base import documents_to_references, format_context, load_prompt, structured_call
from agents.schemas import TRLAssessment
from graph.state import GraphState
from rag.retriever import retrieve

SYSTEM_PROMPT = load_prompt("trl_evaluation")
AGENT_NAME = "trl_evaluation"


def run(state: GraphState) -> dict:
    technologies = state["selected_technologies"]
    technical_evidence = state.get("technical_evidence", {})
    retry_hint = state.get("retry_hints", {}).get(AGENT_NAME, "")
    all_docs = []
    result_by_tech: dict = {}

    for tech_name in technologies:
        query = f"{tech_name} 공개 구현 오픈소스 실험 환경 벤치마크 상용 적용 TRL"
        if retry_hint:
            query = f"{query} {retry_hint}"
        docs = retrieve(query)
        all_docs.extend(docs)

        context = format_context(docs)
        user_content = (
            f"## 기술명\n{tech_name}\n\n"
            f"## 기술 조사 Agent 요약\n{technical_evidence.get(tech_name, {})}\n\n"
            f"## Context\n{context}"
        )
        result = structured_call(TRLAssessment, SYSTEM_PROMPT, user_content)
        result_by_tech[tech_name] = result.model_dump()

    return {
        "retrieved_documents": all_docs,
        "trl_evaluation": result_by_tech,
        "references": documents_to_references(all_docs),
    }
