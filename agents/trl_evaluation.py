"""기술 성숙도 평가 Agent.

기술 원문과 공식 구현자료를 별도로 검색하고, sample.pdf C.6의 TRL 구간별 공개 정보
비대칭(1~3 학술자료, 4~6 영업비밀, 7~9 수율·원가/운영정보)을 평가 한계에 남긴다.
"""

from agents.base import (
    documents_to_evidence_items,
    documents_to_references,
    format_context,
    load_prompt,
    structured_call,
)
from agents.schemas import TRLAssessment
from agents.tech_research import _deduplicate_documents, _retrieve_corpora
from graph.state import GraphState

SYSTEM_PROMPT = load_prompt("trl_evaluation")
AGENT_NAME = "trl_evaluation"


def _trl_information_gap(score: int | None) -> str:
    """설계 문서가 요구한 TRL 구간별 정보 갭을 결과에 일관되게 기록한다."""
    if score is None:
        return (
            "TRL 정보 갭: 1~3은 학술 논문 근거를 확인할 수 있으나, 4~6의 프로토타입·유사 "
            "환경 검증 정보와 7~9의 실제 운용·수율·원가 정보가 공개되지 않으면 구간 판정이 어렵다."
        )
    if score <= 3:
        return (
            "TRL 1~3 정보 갭: 학술 논문 중심의 원리·개념검증 근거는 비교적 풍부하지만, "
            "TRL 4 이상을 입증할 프로토타입 구현·환경 검증 근거는 별도로 필요하다."
        )
    if score <= 6:
        return (
            "TRL 4~6 정보 갭: 구현·시연 근거 일부는 확인 가능하지만 기업 내부 검증과 "
            "영업비밀 정보가 공개되지 않는 경우가 많아 실제 환경 유사성을 제한적으로만 판단할 수 있다."
        )
    return (
        "TRL 7~9 정보 갭: 실제 운용 단계의 수율·원가·장기 안정성·장애율 등 핵심 지표가 "
        "비공개인 경우가 많아 공개 출시나 저장소 존재만으로 최고 성숙도를 확정할 수 없다."
    )


def run(state: GraphState) -> dict:
    technologies = state["selected_technologies"]
    technical_evidence = state.get("technical_evidence", {})
    retry_hint = state.get("retry_hints", {}).get(AGENT_NAME, "")
    all_docs = []
    all_evidence_items: list[dict] = []
    result_by_tech: dict = {}

    for tech_name, tech_info in technologies.items():
        paper_docs, implementation_docs = _retrieve_corpora(
            tech_name,
            tech_info["core_approach"],
            retry_hint,
        )
        docs = _deduplicate_documents(paper_docs + implementation_docs)
        all_docs.extend(docs)

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
            f"## 기술 조사 Agent 요약\n{technical_evidence.get(tech_name, {})}\n\n"
            f"## 기술 원문 Context\n{paper_context}\n\n"
            f"## 공식 구현자료 Context\n{implementation_context}"
        )
        result = structured_call(TRLAssessment, SYSTEM_PROMPT, user_content)
        information_gap = _trl_information_gap(result.score)
        existing_limitations = (result.limitations or "").strip()
        if information_gap not in existing_limitations:
            result.limitations = f"{existing_limitations} {information_gap}".strip()
        result_by_tech[tech_name] = result.model_dump()

        evidence_items = documents_to_evidence_items(
            docs,
            agent=AGENT_NAME,
            claim=(
                f"{tech_name} TRL {result.score} 판단 근거"
                if result.score is not None
                else f"{tech_name} TRL 정보 부족 판단 근거"
            ),
        )
        for item in evidence_items:
            item["limitation"] = information_gap
        all_evidence_items.extend(evidence_items)

    # 기술별 검색 결과가 서로 겹칠 수 있으므로 State reducer에 넘기기 전에 다시 제거한다.
    all_docs = _deduplicate_documents(all_docs)
    return {
        "retrieved_documents": all_docs,
        "trl_evaluation": result_by_tech,
        "references": documents_to_references(all_docs),
        "evidence_items": all_evidence_items,
    }
