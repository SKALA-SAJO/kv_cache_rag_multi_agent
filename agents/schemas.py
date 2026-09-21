"""Agent 구조화 출력 스키마.

RAG-Design PDF D.1 State 설계 표에 정의된 각 State의 내부 구조를 정의한다.
LLM 호출은 기술(technology)별로 한 번씩 이루어지므로, 스키마는 기술 1개에 대한
출력 단위로 정의하고 Agent 코드가 이를 모아 {기술명: 결과} dict로 조립한다.
"""

from typing import Literal

from pydantic import BaseModel, Field

Confidence = Literal["낮음", "중간", "높음"]


class TechEvidence(BaseModel):
    """기술 조사 Agent 출력 (technical_evidence[기술명])."""

    principle: str = Field(description="기술의 핵심 원리")
    performance: str = Field(description="보고된 성능·효과")
    limitations: str = Field(description="논문/자료에 명시된 한계")
    experimental_conditions: str = Field(description="실험 환경, 기준 모델, 조건")
    sources: list[str] = Field(description="근거로 사용한 문서·페이지 표기")


class RubricAssessment(BaseModel):
    """TRL / 시장성 / 도메인 적합성 평가 공통 출력.

    점수 스케일은 Agent마다 다르다 (TRL 1~9, 시장성·도메인 1~5).
    근거가 부족하면 점수를 억지로 매기지 않고 insufficient_evidence=True로 표시한다
    (PDF C.5 절 "근거가 부족한 경우 점수를 억지로 부여하지 않고 정보 부족으로 표시").
    """

    score: int | None = Field(default=None, description="Rubric 점수. 근거 부족 시 null")
    insufficient_evidence: bool = Field(default=False, description="정보 부족 여부")
    rationale: str = Field(description="점수 판단 근거 요약")
    evidence: list[str] = Field(description="근거로 사용한 구체적 사실·수치")
    sources: list[str] = Field(description="출처 (문서명·페이지/섹션)")
    limitations: str = Field(description="이 평가의 한계")
    confidence: Confidence = Field(description="평가 신뢰도")


class StakeholderView(BaseModel):
    stakeholder: str = Field(description="이해관계자 유형 (예: 모델 개발자, 서비스 운영자, 개발자, 사용자)")
    benefits: list[str] = Field(description="이 이해관계자 입장에서의 이점")
    concerns: list[str] = Field(description="이 이해관계자 입장에서의 우려·장벽")


class StakeholderAssessment(BaseModel):
    """이해관계자 평가 Agent 출력 (stakeholder_evaluation[기술명]).

    PDF C.5-3 이해관계자 수용성 Rubric(1~5점)에 따른 종합 점수 + 이해관계자별 상세 시각.
    """

    score: int | None = Field(default=None, description="이해관계자 수용성 Rubric 점수(1~5). 근거 부족 시 null")
    insufficient_evidence: bool = Field(default=False, description="정보 부족 여부")
    views: list[StakeholderView] = Field(description="이해관계자 유형별 이점·우려")
    limitations: str = Field(description="이 평가의 한계")
    confidence: Confidence = Field(description="평가 신뢰도")


class FavorableCondition(BaseModel):
    technology: str = Field(description="기술명 (selected_technologies의 key와 일치)")
    condition: str = Field(description="이 기술이 상대적으로 유리한 구체적 조건")


class Synthesis(BaseModel):
    """평가 종합 Agent 출력.

    우열을 선언하지 않고 조건별 차이를 정리한다 (PDF D.3 절 설계 원칙).
    favorable_conditions는 OpenAI Structured Output(json_schema strict 모드)이 임의 키를
    갖는 dict 타입을 지원하지 않아 list[FavorableCondition]으로 표현한다.
    """

    agreements: list[str] = Field(description="관점 간 일치하는 시사점")
    conflicts: list[str] = Field(description="관점 간 상충되는 지점 (제거하지 않고 그대로 보고)")
    favorable_conditions: list[FavorableCondition] = Field(
        description="기술별로 유리한 조건 (기술마다 최소 1개)"
    )


class ClaimCheck(BaseModel):
    claim: str
    status: Literal["pass", "fail"]
    note: str = Field(description="판정 근거 또는 부족한 이유")


class FaithfulnessCheckResult(BaseModel):
    """검증 Agent 출력 (faithfulness_check State).

    claim-evidence 대조 결과. 하나라도 fail이면 passed=False.
    """

    claim_checks: list[ClaimCheck]
    passed: bool
    insufficient_evidence_claims: list[str] = Field(
        description="근거 부족으로 판정된 claim 목록 (재검색 질의 생성에 사용)"
    )
