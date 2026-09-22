"""기술 조사 Agent (PDF Agent). RAG 여부: O — 기술 문서 RAG 검색을 직접 수행한다.

담당(Phase 2-A, 서지원): 구현자료(GitHub README) 코퍼스 검색 추가는 아직 반영되지 않았다.
현재는 기존 기술원문(논문) 코퍼스만 검색한다 — rag/ingest.py에 구현자료가 색인되면
query에 반영하면 된다.
"""

from agents.base import (
    documents_to_evidence_items,
    documents_to_references,
    format_context,
    load_prompt,
    structured_call,
)
from agents.schemas import TechEvidence
from graph.state import GraphState
from rag.retriever import retrieve

SYSTEM_PROMPT = load_prompt("tech_research")
AGENT_NAME = "tech_research"


def run(state: GraphState) -> dict:
    technologies = state["selected_technologies"]
    retry_hint = state.get("retry_hints", {}).get(AGENT_NAME, "")
    all_docs = []
    all_evidence_items: list[dict] = []
    evidence: dict = {}

    for tech_name, tech_info in technologies.items():
        query = f"{tech_name} {tech_info['core_approach']} 원리 성능 한계 실험 조건"
        if retry_hint:
            query = f"{query} {retry_hint}"
        docs = retrieve(query)
        all_docs.extend(docs)
        all_evidence_items.extend(
            documents_to_evidence_items(docs, agent=AGENT_NAME, claim=f"{tech_name} 기술 조사 근거")
        )

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
        "evidence_items": all_evidence_items,
    }
