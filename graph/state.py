"""LangGraph State 정의 (RAG-Design PDF D.1 절 State 설계 표 그대로 반영).

retrieved_documents / references는 재검색 루프(Faithfulness Check 실패 시)에서
누적되어야 하므로 operator.add 리듀서를 사용한다. 그 외 필드는 담당 Agent가
한 번씩만 쓰므로 기본(마지막 쓰기 우선) 동작을 사용한다.
"""

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict, total=False):
    # 시작 Node가 생성
    research_question: str
    selected_technologies: dict[str, dict[str, Any]]

    # 기술 문서 RAG 검색 Node / 기술 조사 Agent
    retrieved_documents: Annotated[list[Document], operator.add]
    technical_evidence: dict[str, Any]

    # 4관점 평가 Agent
    trl_evaluation: dict[str, Any]
    market_evaluation: dict[str, Any]
    stakeholder_evaluation: dict[str, Any]
    domain_evaluation: dict[str, Any]

    # 전체 Agent가 누적
    references: Annotated[list[dict[str, Any]], operator.add]

    # 평가 종합 / 검증 / 보고서 생성 Agent
    synthesis: dict[str, Any]
    faithfulness_check: dict[str, Any]
    final_report: str

    # v0.0에서 추가: 재검색 루프 무한 반복 방지용 카운터 (PDF에 명시되지 않은 안전장치)
    verification_retry_count: int
