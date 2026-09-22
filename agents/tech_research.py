"""기술 조사 Agent.

sample.pdf B.3/D.1에 따라 기술 원문과 공식 구현자료(GitHub README)를 각각 검색한다.
구현자료가 아직 색인되지 않은 개발 환경에서도 논문 검색은 정상 동작하며, 구현 근거는
빈 Context로 전달되어 LLM이 ``Context 내 근거 없음``으로 명시하게 한다.
"""

from __future__ import annotations

from langchain_core.documents import Document

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

IMPLEMENTATION_MARKERS = ("implementation", "github", "readme")


def _is_implementation_document(doc: Document) -> bool:
    """구현자료 색인 담당자가 붙일 수 있는 대표 메타데이터 형식을 폭넓게 수용한다."""
    metadata = doc.metadata
    metadata_types = {
        str(metadata.get(key, "")).strip().lower()
        for key in ("doc_type", "document_type", "corpus_type", "source_type")
    }
    source = str(metadata.get("source", "")).lower()
    url = str(metadata.get("source_url", metadata.get("url", ""))).lower()
    return (
        any(marker in value for value in metadata_types for marker in IMPLEMENTATION_MARKERS)
        or "readme" in source
        or source.endswith(".md")
        or "github.com" in source
        or "github.com" in url
    )


def _deduplicate_documents(documents: list[Document]) -> list[Document]:
    seen: set[tuple[str, str, str]] = set()
    unique: list[Document] = []
    for doc in documents:
        key = (
            str(doc.metadata.get("chunk_id", "")),
            str(doc.metadata.get("source", "")),
            doc.page_content,
        )
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique


def _retrieve_corpora(
    tech_name: str,
    core_approach: str,
    retry_hint: str,
) -> tuple[list[Document], list[Document]]:
    """논문과 공식 구현자료를 별도 질의한다.

    현재 retriever는 통합 인덱스를 사용하므로 구현자료 검색 결과는 메타데이터로 한 번 더
    제한한다. 이승준 담당 코퍼스가 추가되면 README 청크의 ``document_type``(또는
    ``corpus_type``)만 ``github_readme``로 지정하면 별도 Agent 수정 없이 연결된다.
    """
    paper_query = f"{tech_name} {core_approach} 원리 성능 한계 실험 조건 논문"
    implementation_query = (
        f"{tech_name} official GitHub README 공개 구현 설치 실행 재현 벤치마크 "
        "지원 환경 요구사항 제한"
    )
    if retry_hint:
        paper_query = f"{paper_query} {retry_hint}"
        implementation_query = f"{implementation_query} {retry_hint}"

    paper_candidates = retrieve(paper_query)
    implementation_candidates = retrieve(implementation_query)
    paper_docs = _deduplicate_documents(
        [doc for doc in paper_candidates if not _is_implementation_document(doc)]
    )
    implementation_docs = _deduplicate_documents(
        [doc for doc in implementation_candidates if _is_implementation_document(doc)]
    )
    return paper_docs, implementation_docs


def run(state: GraphState) -> dict:
    technologies = state["selected_technologies"]
    retry_hint = state.get("retry_hints", {}).get(AGENT_NAME, "")
    all_docs = []
    all_evidence_items: list[dict] = []
    evidence: dict = {}

    for tech_name, tech_info in technologies.items():
        paper_docs, implementation_docs = _retrieve_corpora(
            tech_name,
            tech_info["core_approach"],
            retry_hint,
        )
        docs = _deduplicate_documents(paper_docs + implementation_docs)
        all_docs.extend(docs)
        paper_items = documents_to_evidence_items(
            paper_docs,
            agent=AGENT_NAME,
            claim=f"{tech_name} 원리·성능·한계·실험 조건 근거",
        )
        implementation_items = documents_to_evidence_items(
            implementation_docs,
            agent=AGENT_NAME,
            claim=f"{tech_name} 공개 구현·재현 조건 근거",
        )
        for item in implementation_items:
            item["limitation"] = (
                "공식 README의 자체 기술 내용이며 실제 운용·독립 재현을 직접 입증하지 않음"
            )
        all_evidence_items.extend(paper_items + implementation_items)

        paper_context = (
            format_context(paper_docs)
            if paper_docs
            else "Context 내 기술 원문 근거 없음"
        )
        implementation_context = (
            format_context(implementation_docs)
            if implementation_docs
            else "Context 내 공식 구현자료 근거 없음"
        )
        user_content = (
            f"## 기술명\n{tech_name}\n\n"
            f"## 핵심 접근 (Human 선정 근거)\n{tech_info['core_approach']}\n\n"
            f"## 기술 원문 Context\n{paper_context}\n\n"
            f"## 공식 구현자료 Context\n{implementation_context}"
        )
        result = structured_call(TechEvidence, SYSTEM_PROMPT, user_content)
        evidence[tech_name] = result.model_dump()

    # 기술별 검색 결과가 서로 겹칠 수 있으므로 State reducer에 넘기기 전에 다시 제거한다.
    all_docs = _deduplicate_documents(all_docs)
    return {
        "retrieved_documents": all_docs,
        "technical_evidence": evidence,
        "references": documents_to_references(all_docs),
        "evidence_items": all_evidence_items,
    }
