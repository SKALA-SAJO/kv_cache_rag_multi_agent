"""Agent 공통 유틸리티: LLM 호출, 프롬프트 로딩, 문서 -> 컨텍스트/참고문헌 변환."""

from __future__ import annotations

import json
from typing import Any, TypeVar

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from config import PROMPTS_DIR, settings

T = TypeVar("T", bound=BaseModel)

# 외부 검색 API는 팀 합의 후 rag/external_search.py에서 구현한다. 이 모듈은
# Tavily, Google Custom Search 등 특정 공급자에 의존하지 않고 BaseTool 계약만 사용한다.
_external_search_tool: BaseTool | None = None


def get_llm(role: str = "generator", temperature: float = 0.0) -> ChatOpenAI:
    """role: "generator" 또는 "judge" (PDF Tech Stack 절 LLM/Generator·LLM/Judge 구분)."""
    model = settings.judge_model if role == "judge" else settings.generator_model
    return ChatOpenAI(model=model, temperature=temperature)


def register_external_search_tool(tool: BaseTool) -> None:
    """선정된 외부 검색 도구를 시장·이해관계자 Agent 공용으로 등록한다.

    Phase 3에서 구현할 ``rag.external_search`` 어댑터가 앱 시작 시 한 번 호출한다.
    BaseTool의 입력은 ``{"query": str}``, 출력은 title/url/content 필드를 가진
    dict 목록(또는 해당 JSON 문자열)으로 정규화해야 한다.
    """
    global _external_search_tool
    _external_search_tool = tool


def get_external_search_tool() -> BaseTool | None:
    """등록된 외부 검색 도구를 반환한다. API 연동 전에는 None을 반환한다."""
    return _external_search_tool


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


def _normalize_search_results(raw_result: Any) -> list[dict[str, str]]:
    """서로 다른 검색 API 응답을 title/url/content 공통 형식으로 정규화한다."""
    if isinstance(raw_result, str):
        try:
            raw_result = json.loads(raw_result)
        except json.JSONDecodeError:
            raw_result = [{"content": raw_result}]
    if isinstance(raw_result, dict):
        raw_result = raw_result.get("results", [raw_result])
    if not isinstance(raw_result, list):
        return []

    normalized: list[dict[str, str]] = []
    for item in raw_result:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or item.get("snippet") or "").strip()
        url = str(item.get("url") or "").strip()
        title = str(item.get("title") or url or "외부 검색 결과").strip()
        if content or url:
            normalized.append({"title": title, "url": url, "content": content})
    return normalized


def _deduplicate_search_results(results: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    unique: list[dict[str, str]] = []
    for result in results:
        key = (result.get("url", ""), result.get("content", ""))
        if key in seen:
            continue
        seen.add(key)
        unique.append(result)
    return unique


def format_search_results(results: list[dict[str, str]]) -> str:
    """외부 검색의 실제 결과만 최종 구조화 평가 호출에 전달한다."""
    if not results:
        return "(외부 검색 결과 없음)"
    blocks = []
    for index, result in enumerate(results, start=1):
        blocks.append(
            f"[{index}] {result.get('title', '외부 검색 결과')}\n"
            f"URL: {result.get('url', '')}\n"
            f"{result.get('content', '')}"
        )
    return "\n\n".join(blocks)


def run_external_search_loop(
    system_prompt: str,
    user_content: str,
    search_tool: BaseTool | None,
    role: str = "generator",
    max_tool_rounds: int = 2,
) -> list[dict[str, str]]:
    """LLM이 검색 질의를 만들고 도구 결과를 읽는 공용 tool-calling 루프.

    검색 API가 아직 등록되지 않은 경우 빈 목록을 반환한다. 이때 호출 Agent는
    evidence_items를 만들지 않으므로, 일반 지식을 외부 검색 근거처럼 기록하지 않는다.
    """
    if search_tool is None:
        return []

    tool_instruction = """
외부 검색 도구를 사용해 평가에 필요한 공개 근거를 찾아라. 검색 질의는 구체적으로 작성하고,
공식 문서·논문·공식 저장소·신뢰할 수 있는 산업 자료를 우선한다. 충분한 결과를 얻으면 도구 호출을
멈춘다. 도구 결과에 없는 사실이나 URL을 만들어내지 않는다.
"""
    messages = [
        SystemMessage(content=f"{system_prompt}\n\n{tool_instruction}"),
        HumanMessage(content=user_content),
    ]
    llm_with_tools = get_llm(role).bind_tools([search_tool])
    collected: list[dict[str, str]] = []

    for _ in range(max_tool_rounds):
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        tool_calls = response.tool_calls
        if not tool_calls:
            break

        for tool_call in tool_calls:
            if tool_call["name"] != search_tool.name:
                messages.append(
                    ToolMessage(
                        content="등록되지 않은 도구 요청입니다.",
                        tool_call_id=tool_call["id"],
                    )
                )
                continue
            raw_result = search_tool.invoke(tool_call["args"])
            normalized = _normalize_search_results(raw_result)
            collected.extend(normalized)
            messages.append(
                ToolMessage(
                    content=json.dumps(normalized, ensure_ascii=False),
                    tool_call_id=tool_call["id"],
                )
            )

    return _deduplicate_search_results(collected)


def structured_call_with_external_search(
    schema: type[T],
    system_prompt: str,
    user_content: str,
    search_tool: BaseTool | None,
    role: str = "generator",
) -> tuple[T, list[dict[str, str]]]:
    """검색 → 실제 검색 결과 주입 → Pydantic 구조화 평가를 수행한다.

    최종 출력은 별도의 ``structured_call``로 생성해 기존 6개 Agent와 같은
    Pydantic 출력 패턴을 유지한다.
    """
    results = run_external_search_loop(system_prompt, user_content, search_tool, role)
    search_context = format_search_results(results)
    final_content = (
        f"{user_content}\n\n## 외부 검색 결과\n{search_context}\n\n"
        "외부 검색 결과가 있다면 그 결과에 포함된 사실과 URL만 외부 출처로 사용하라. "
        "검색 결과가 없으면 외부 검색 근거가 없음을 limitations와 confidence에 명시하라."
    )
    return structured_call(schema, system_prompt, final_content, role), results


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


def search_results_to_references(results: list[dict[str, str]]) -> list[dict]:
    """실제로 검색된 웹 자료만 State ``references`` 형식으로 변환한다."""
    references = []
    for result in _deduplicate_search_results(results):
        references.append(
            {
                "source": result.get("title") or result.get("url") or "외부 검색 결과",
                "url": result.get("url"),
                "source_type": "external_search",
            }
        )
    return references


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
