"""Agent 공통 유틸리티: LLM 호출, 프롬프트 로딩, 문서 -> 컨텍스트/참고문헌 변환."""

from __future__ import annotations

from typing import TypeVar

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from config import PROMPTS_DIR, settings

T = TypeVar("T", bound=BaseModel)


def get_llm(role: str = "generator", temperature: float = 0.0) -> ChatOpenAI:
    """role: "generator" 또는 "judge" (PDF Tech Stack 절 LLM/Generator·LLM/Judge 구분)."""
    model = settings.judge_model if role == "judge" else settings.generator_model
    return ChatOpenAI(model=model, temperature=temperature)


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")


def structured_call(
    schema: type[T],
    system_prompt: str,
    user_content: str,
    role: str = "generator",
) -> T:
    """시스템 프롬프트 + 사용자 컨텐츠로 LLM을 호출하고 pydantic 스키마로 파싱된 결과를 반환한다."""
    llm = get_llm(role).with_structured_output(schema)
    return llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_content)]
    )


def format_context(documents: list[Document]) -> str:
    """검색된 문서를 [n] 인용 태그를 붙여 LLM 컨텍스트 문자열로 변환한다."""
    if not documents:
        return "(검색된 근거 문서 없음)"
    blocks = []
    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        blocks.append(f"[{i}] ({source} p.{page})\n{doc.page_content}")
    return "\n\n".join(blocks)


def documents_to_references(documents: list[Document]) -> list[dict]:
    """중복 제거된 참고문헌 목록 (State: references)."""
    seen: set[tuple] = set()
    refs = []
    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        key = (source, page)
        if key in seen:
            continue
        seen.add(key)
        refs.append(
            {
                "source": source,
                "page": page,
                "technology": doc.metadata.get("technology"),
            }
        )
    return refs


def documents_to_evidence_items(
    documents: list[Document],
    agent: str,
    claim: str,
    quote_length: int = 300,
) -> list[dict]:
    """RAG로 검색된 문서를 evidence_items(State: evidence_items) 형태로 변환한다.

    claim-level(어떤 문장이 어떤 근거를 뒷받침하는지)까지는 세분화하지 않고, 이 Agent가
    이 호출에서 사용한 근거 전체를 claim 하나에 묶어 등록한다 — 세부 claim 단위 귀속은
    평가 종합 Agent가 만드는 synthesis 단계 이후 검증 Agent가 evidence_items를 참조해
    판단한다. RAG로 실제 검색된 문서에서만 만들어지므로 근거를 지어내지 않는다.
    """
    items = []
    for doc in documents:
        items.append(
            {
                "claim": claim,
                "evidence_quote": doc.page_content[:quote_length],
                "source_url": None,
                "document_id": doc.metadata.get("source", "unknown"),
                "page_or_section": str(doc.metadata.get("page", "?")),
                "source_type": "RAG",
                "limitation": "",
                "agent": agent,
            }
        )
    return items


def search_results_to_evidence_items(
    results: list[dict],
    agent: str,
    claim: str,
    quote_length: int = 300,
) -> list[dict]:
    """외부 검색 도구 결과를 evidence_items(State: evidence_items) 형태로 변환한다.

    documents_to_evidence_items의 외부 검색(source_type="external_search") 버전.
    어떤 검색 API를 쓰든(Tavily/Google 등) rag/external_search.py가 결과를
    {"title": str, "url": str, "content": str} 형태로 정규화해서 넘긴다고 가정한다 —
    이 함수 자체는 특정 API에 의존하지 않는다. 실제로 검색한 결과에서만 만들어지므로
    근거를 지어내지 않는다 (documents_to_evidence_items와 동일한 원칙).
    """
    items = []
    for r in results:
        content = r.get("content") or r.get("snippet") or ""
        items.append(
            {
                "claim": claim,
                "evidence_quote": content[:quote_length],
                "source_url": r.get("url"),
                "document_id": None,
                "page_or_section": None,
                "source_type": "external_search",
                "limitation": "",
                "agent": agent,
            }
        )
    return items


def format_evidence_items(evidence_items: list[dict]) -> str:
    """검증 Agent가 참조할 수 있도록 evidence_items를 번호 붙은 컨텍스트 문자열로 변환한다."""
    if not evidence_items:
        return "(등록된 근거 없음)"
    blocks = []
    for i, item in enumerate(evidence_items):
        source = item.get("source_url") or item.get("document_id") or "unknown"
        page = item.get("page_or_section") or "?"
        blocks.append(
            f"[{i}] (agent={item.get('agent')}, source_type={item.get('source_type')}, "
            f"{source} {page})\n{item.get('evidence_quote', '')}"
        )
    return "\n\n".join(blocks)
