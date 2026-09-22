"""Agent 구조화 출력 스키마.

sample.pdf(업데이트본) D.1 State 설계 표 + C.5 Rubric 절을 반영한다.
LLM 호출은 기술(technology)별로 한 번씩 이루어지므로, 스키마는 기술 1개에 대한
출력 단위로 정의하고 Agent 코드가 이를 모아 {기술명: 결과} dict로 조립한다.
"""

from typing import Literal

from pydantic import BaseModel, Field

Confidence = Literal["낮음", "중간", "높음"]

# PDF C.5절: 시장성/이해관계자/도메인 적합성은 1~5점이 아니라 3단계 정성 라벨로 표시한다.
# "라벨은 관점별 평가를 구조화하기 위한 수단이며, 기술의 종합적인 우열이나 최종 추천을
# 의미하지 않는다."
EvidenceLevel = Literal["근거 부족", "근거 제한적", "근거 충분"]


class TechEvidence(BaseModel):
    """기술 조사 Agent 출력 (technical_evidence[기술명])."""

    principle: str = Field(description="기술의 핵심 원리")
    performance: str = Field(description="보고된 성능·효과")
    limitations: str = Field(description="논문/자료에 명시된 한계")
    experimental_conditions: str = Field(description="실험 환경, 기준 모델, 조건")
    sources: list[str] = Field(description="근거로 사용한 문서·페이지 표기")


class TRLAssessment(BaseModel):
    """기술 성숙도 평가 Agent 출력 (trl_evaluation[기술명]).

    TRL은 PDF C.5-1절대로 1~9 숫자 척도를 유지한다 (시장성/이해관계자/도메인과 달리
    "공개 정보를 근거로 개발 단계를 절대적으로 위치시키는 목적"이므로 라벨화하지 않음).
    """

    score: int | None = Field(default=None, description="TRL 점수(1~9). 근거 부족 시 null")
    insufficient_evidence: bool = Field(default=False, description="정보 부족 여부")
    rationale: str = Field(description="TRL 판단 근거 요약")
    evidence: list[str] = Field(description="근거로 사용한 구체적 사실·수치")
    sources: list[str] = Field(description="출처 (문서명·페이지/섹션)")
    limitations: str = Field(
        description="이 평가의 한계. TRL 구간별 공개 정보량 차이(1~3 학술논문 풍부 / "
        "4~6 영업비밀로 정보 갭 큼 / 7~9 수율·원가 등 비공개)를 함께 명시한다"
    )
    confidence: Confidence = Field(description="평가 신뢰도")


class QualitativeAssessment(BaseModel):
    """시장성 / 도메인 적합성 평가 공통 출력 (market_evaluation·domain_evaluation[기술명]).

    PDF C.5-2, 5-4절: "근거 부족 / 근거 제한적 / 근거 충분" 3단계 정성 라벨.
    """

    level: EvidenceLevel | None = Field(default=None, description="근거 수준 라벨. 판단 불가 시 null")
    insufficient_evidence: bool = Field(default=False, description="정보 부족 여부")
    rationale: str = Field(description="라벨 판단 근거 요약")
    evidence: list[str] = Field(description="근거로 사용한 구체적 사실·수치")
    sources: list[str] = Field(description="출처 (문서명·페이지/섹션 또는 URL)")
    limitations: str = Field(description="이 평가의 한계")
    confidence: Confidence = Field(description="평가 신뢰도")


class StakeholderView(BaseModel):
    stakeholder: str = Field(
        description="이해관계자 유형 (경쟁 기술 진영 / 개발자 / 도입 기업 및 서비스 운영자 / 투자·업계 관계자)"
    )
    benefits: list[str] = Field(description="이 이해관계자 입장에서의 이점")
    concerns: list[str] = Field(description="이 이해관계자 입장에서의 우려·장벽")


class StakeholderAssessment(BaseModel):
    """이해관계자 평가 Agent 출력 (stakeholder_evaluation[기술명]).

    PDF C.5-3절 이해관계자 수용성 Rubric(3단계 근거 수준)에 따른 종합 라벨 + 유형별 상세 시각.
    """

    level: EvidenceLevel | None = Field(default=None, description="근거 수준 라벨. 판단 불가 시 null")
    insufficient_evidence: bool = Field(default=False, description="정보 부족 여부")
    views: list[StakeholderView] = Field(description="이해관계자 유형별 이점·우려")
    limitations: str = Field(description="이 평가의 한계")
    confidence: Confidence = Field(description="평가 신뢰도")


class FavorableCondition(BaseModel):
    technology: str = Field(description="기술명 (selected_technologies의 key와 일치)")
    condition: str = Field(description="이 기술이 상대적으로 유리한 구체적 조건")


class Synthesis(BaseModel):
    """평가 종합 Agent 출력.

    우열을 선언하지 않고 조건별 차이를 정리한다 (PDF D.3절 설계 원칙).
    favorable_conditions는 OpenAI Structured Output(json_schema strict 모드)이 임의 키를
    갖는 dict 타입을 지원하지 않아 list[FavorableCondition]으로 표현한다.
    """

    agreements: list[str] = Field(description="관점 간 일치하는 시사점")
    conflicts: list[str] = Field(description="관점 간 상충되는 지점 (제거하지 않고 그대로 보고)")
    favorable_conditions: list[FavorableCondition] = Field(
        description="기술별로 유리한 조건 (기술마다 최소 1개)"
    )


class EvidenceItem(BaseModel):
    """공통 근거 항목 (PDF D.1 evidence_items State).

    PDF 표는 claim/evidence_quote/source_url/document_id/page_or_section/
    source_type/limitation만 나열하지만, 검증 Agent가 "실패한 claim의 출처 Agent로
    라우팅"(9쪽 플로우차트)하려면 각 근거가 어느 Agent에서 나왔는지 알아야 한다.
    이를 위해 agent 필드를 추가했다 — PDF 표에는 없지만 표적형 재검색 루프에 필수적인
    엔지니어링 보강이다 (graph/workflow.py의 route_after_faithfulness 참고).
    """

    claim: str = Field(description="이 근거가 뒷받침하는 주장")
    evidence_quote: str = Field(description="근거 원문 인용")
    source_url: str | None = Field(default=None, description="외부 검색으로 얻은 경우의 출처 URL")
    document_id: str | None = Field(default=None, description="RAG로 얻은 경우의 문서 식별자(파일명 등)")
    page_or_section: str | None = Field(default=None, description="페이지 또는 섹션 표기")
    source_type: Literal["RAG", "external_search"] = Field(description="근거 출처 유형")
    limitation: str = Field(default="", description="이 근거의 한계")
    agent: str = Field(description="이 근거를 생성한 Agent 이름 (graph 노드 이름과 일치)")


class ClaimCheck(BaseModel):
    claim: str
    status: Literal["pass", "fail"]
    note: str = Field(description="판정 근거 또는 부족한 이유")
    evidence_refs: list[int] = Field(
        default_factory=list,
        description="이 claim을 판정할 때 사용한 evidence_items의 인덱스 목록. "
        "fail인 claim의 재검색 대상 Agent를 찾는 데 사용된다 (근거 자체가 없으면 빈 리스트)",
    )


class FaithfulnessCheckResult(BaseModel):
    """검증 Agent 출력 (faithfulness_check State).

    claim-evidence 대조 결과. 하나라도 fail이면 passed=False.
    agents_to_retry/retry_hints는 LLM이 아니라 faithfulness_check.py가 evidence_refs를
    이용해 프로그램적으로 계산하므로 이 pydantic 스키마에는 포함하지 않는다.
    """

    claim_checks: list[ClaimCheck]
    passed: bool
    insufficient_evidence_claims: list[str] = Field(
        description="근거 부족으로 판정된 claim 목록"
    )
