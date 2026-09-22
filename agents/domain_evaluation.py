"""도메인 평가 Agent (PDF Agent). RAG 여부: O.

- 기술 원문(DeepSeek-V2, InfiniGen)에서 장문맥 관련 근거를 검색한다.
- 도메인 자료(LongBench·RULER) 코퍼스를 별도 쿼리로 함께 검색해, 장문맥 평가 관점의
  정의/주의점(예: long-context 벤치마크 과제 구성, "지원 컨텍스트" vs "실효 컨텍스트")도
  컨텍스트로 제공한다.
"""

from langchain_core.documents import Document

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


def _dedup_documents(documents: list[Document]) -> list[Document]:
    seen: set[tuple[str, str]] = set()
    unique: list[Document] = []
    for doc in documents:
        key = (doc.metadata.get("source", "unknown"), str(doc.metadata.get("page", "?")))
        if key in seen:
            continue
        seen.add(key)
        unique.append(doc)
    return unique


def _retrieve_domain_corpus(retry_hint: str = "") -> list[Document]:
    """LongBench·RULER 코퍼스를 명시적으로 검색한다.

    sample.pdf의 'RAG 문서 구성'에는 장문맥 도메인 자료로 LongBench/RULER 논문을
    포함하도록 되어 있다. v0.0의 기본 retriever는 기술 원문만을 전제로 작성됐기 때문에,
    도메인 평가 Agent가 별도 쿼리로 도메인 자료를 반드시 끌어오도록 보강한다.
    """

    queries = [
        "LongBench long context benchmark tasks evaluation",
        "RULER real context size effective context length evaluation",
    ]
    if retry_hint:
        queries = [f"{q} {retry_hint}" for q in queries]

    docs: list[Document] = []
    # 각각에서 일부만 가져오도록 top_k를 작게 둔다.
    for q in queries:
        docs.extend(retrieve(q, top_k=3))
    return _dedup_documents(docs)


def run(state: GraphState) -> dict:
    technologies = state["selected_technologies"]
    technical_evidence = state.get("technical_evidence", {})
    retry_hint = state.get("retry_hints", {}).get(AGENT_NAME, "")
    all_docs = []
    all_evidence_items: list[dict] = []
    result_by_tech: dict = {}

    # Domain corpus: LongBench/RULER (기술과 무관한 '도메인 정의/평가 근거')
    domain_docs = _retrieve_domain_corpus(retry_hint=retry_hint)
    all_docs.extend(domain_docs)
    all_evidence_items.extend(
        documents_to_evidence_items(
            domain_docs,
            agent=AGENT_NAME,
            claim="장문맥 도메인 평가 기준(LongBench/RULER) 근거",
        )
    )

    for tech_name in technologies:
        query = f"{tech_name} 장문맥 long context 컨텍스트 길이 메모리 효율 정확도 지연"
        if retry_hint:
            query = f"{query} {retry_hint}"
        tech_docs = retrieve(query)
        all_docs.extend(tech_docs)
        all_evidence_items.extend(
            documents_to_evidence_items(
                tech_docs,
                agent=AGENT_NAME,
                claim=f"{tech_name} 장문맥 처리 애플리케이션 적합성 근거",
            )
        )

        docs_for_context = _dedup_documents([*tech_docs, *domain_docs])
        context = format_context(docs_for_context)
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
