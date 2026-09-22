"""LangGraph State 정의 (sample.pdf D.1절 State 설계 표를 반영).

retrieved_documents / references / evidence_items는 병렬 Agent가 누적하고 재검색
루프에서도 계속 늘어나야 하므로 operator.add 리듀서를 쓴다 (PDF D.3 설계 원칙:
"병렬 Agent가 누적하는 retrieved_documents, references, evidence_items에는
list reducer를 적용"). 그 외 필드는 담당 Agent가 한 번씩만 쓰므로 기본(마지막
쓰기 우선) 동작을 사용한다.
"""

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict, total=False):
    # 초기화 Node가 생성
    research_question: str
    selected_technologies: dict[str, dict[str, Any]]
    evaluation_rubric: dict[str, Any]
    max_retries: int

    # 기술 문서 RAG 검색 / 기술 조사 Agent
    retrieved_documents: Annotated[list[Document], operator.add]
    technical_evidence: dict[str, Any]

    # 4관점 평가 Agent
    trl_evaluation: dict[str, Any]
    market_evaluation: dict[str, Any]
    stakeholder_evaluation: dict[str, Any]
    domain_evaluation: dict[str, Any]

    # 전체 Agent가 누적 (URL/document_id 기준 중복 제거는 agents/base.py에서 수행)
    references: Annotated[list[dict[str, Any]], operator.add]
    # claim, evidence_quote, source_url/document_id, page_or_section, source_type,
    # limitation(+agent)을 담는 공통 근거 목록 (agents/schemas.py의 EvidenceItem)
    evidence_items: Annotated[list[dict[str, Any]], operator.add]

    # 평가 종합 / 검증 / 보고서 생성 Agent
    synthesis: dict[str, Any]
    faithfulness_check: dict[str, Any]
    final_report: str

    # 검증 Agent가 갱신, 종료 판단 Node(route_after_faithfulness)가 사용
    retry_count: int
    # 검증 Agent가 실패 claim의 출처 Agent별로 채우는 재검색 힌트: {agent_name: hint}
    retry_hints: dict[str, str]
