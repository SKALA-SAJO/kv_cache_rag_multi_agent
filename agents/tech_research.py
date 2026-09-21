"""기술 조사 Agent (PDF Agent D). RAG 여부: O — 기술 문서 RAG 검색을 직접 수행한다."""

from agents.base import documents_to_references, format_context, load_prompt, structured_call
from agents.schemas import TechEvidence
from graph.state import GraphState
from rag.retriever import retrieve

SYSTEM_PROMPT = load_prompt("tech_research")


def run(state: GraphState) -> dict:
    technologies = state["selected_technologies"]
    all_docs = []
    evidence: dict = {}

    for tech_name, tech_info in technologies.items():
        query = f"{tech_name} {tech_info['core_approach']} 원리 성능 한계 실험 조건"
        docs = retrieve(query)
        all_docs.extend(docs)

        context = format_context(docs)
        user_content = (
            f"## 기술명\n{tech_name}\n\n"
            f"## 핵심 접근 (Human 선정 근거)\n{tech_info['core_approach']}\n\n"
            f"## Context\n{context}"
        )
        result = structured_call(TechEvidence, SYSTEM_PROMPT, user_content)
        evidence[tech_name] = result.model_dump()

    return {
        "retrieved_documents": all_docs,
        "technical_evidence": evidence,
        "references": documents_to_references(all_docs),
    }
