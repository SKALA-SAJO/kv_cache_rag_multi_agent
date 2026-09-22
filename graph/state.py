"""LangGraph State 정의 (sample.pdf D.1절 State 설계 표를 반영).

retrieved_documents / references / evidence_items는 병렬 Agent가 누적하고 재검색
루프에서도 계속 늘어나야 하므로 list 리듀서를 쓴다. PDF D.3 설계 원칙: "병렬 Agent가
누적하는 retrieved_documents, references, evidence_items에는 list reducer를 적용하고,
URL 또는 document_id 기준으로 중복을 제거한다" — 단순 concat(operator.add)이 아니라
아래 dedupe_* 함수로 매 병합 시 전역 중복 제거까지 수행한다. 그 외 필드는 담당 Agent가
한 번씩만 쓰므로 기본(마지막 쓰기 우선) 동작을 사용한다.
"""

from typing import Annotated, Any, TypedDict

from langchain_core.documents import Document


def dedupe_documents(existing: list[Document], new: list[Document]) -> list[Document]:
    """retrieved_documents 리듀서: chunk_id(없으면 source+page+본문 앞부분) 기준 중복 제거."""
    combined = existing + new
    seen: set = set()
    result: list[Document] = []
    for doc in combined:
        key = doc.metadata.get("chunk_id") or (
            doc.metadata.get("source"),
            doc.metadata.get("page"),
            doc.page_content[:80],
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(doc)
    return result


def dedupe_references(existing: list[dict], new: list[dict]) -> list[dict]:
    """references 리듀서: (source, page) 기준 중복 제거."""
    combined = existing + new
    seen: set = set()
    result: list[dict] = []
    for ref in combined:
        key = (ref.get("source"), ref.get("page"))
        if key in seen:
            continue
        seen.add(key)
        result.append(ref)
    return result


def dedupe_evidence_items(existing: list[dict], new: list[dict]) -> list[dict]:
    """evidence_items 리듀서: agent + document_id/source_url + 페이지·섹션 + 인용문 앞부분

    기준 중복 제거. agent를 키에 넣지 않으면 서로 다른 Agent(예: market_evaluation과
    stakeholder_evaluation)가 같은 URL/내용을 인용했을 때 한쪽 Agent의 근거가 통째로
    사라져서, faithfulness_check의 실패 claim -> 출처 Agent 귀속이 틀어진다(실제 mock
    테스트에서 재현·발견됨).
    """
    combined = existing + new
    seen: set = set()
    result: list[dict] = []
    for item in combined:
        key = (
            item.get("agent"),
            item.get("document_id") or item.get("source_url"),
            item.get("page_or_section"),
            (item.get("evidence_quote") or "")[:80],
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


class GraphState(TypedDict, total=False):
    # 초기화 Node가 생성
    research_question: str
    selected_technologies: dict[str, dict[str, Any]]
    evaluation_rubric: dict[str, Any]
    max_retries: int

    # 기술 문서 RAG 검색 / 기술 조사 Agent
    retrieved_documents: Annotated[list[Document], dedupe_documents]
    technical_evidence: dict[str, Any]

    # 4관점 평가 Agent
    trl_evaluation: dict[str, Any]
    market_evaluation: dict[str, Any]
    stakeholder_evaluation: dict[str, Any]
    domain_evaluation: dict[str, Any]

    # 전체 Agent가 누적, URL/document_id 기준 중복 제거
    references: Annotated[list[dict[str, Any]], dedupe_references]
    # claim, evidence_quote, source_url/document_id, page_or_section, source_type,
    # limitation(+agent)을 담는 공통 근거 목록 (agents/schemas.py의 EvidenceItem)
    evidence_items: Annotated[list[dict[str, Any]], dedupe_evidence_items]

    # 평가 종합 / 검증 / 보고서 생성 Agent
    synthesis: dict[str, Any]
    faithfulness_check: dict[str, Any]
    final_report: str

    # 검증 Agent가 갱신, 종료 판단 Node(route_after_faithfulness)가 사용
    retry_count: int
    # 검증 Agent가 실패 claim의 출처 Agent별로 채우는 재검색 힌트: {agent_name: hint}
    retry_hints: dict[str, str]
