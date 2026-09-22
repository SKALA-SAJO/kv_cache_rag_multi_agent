"""이해관계자 평가 Agent (PDF Agent). RAG 여부: X. 도구: 외부 정보 검색 도구(PDF 업데이트본).

입력은 기술 조사 Agent의 technical_evidence를 기본으로 하되, PDF 업데이트본은 여기에
"공개 웹 자료"(외부 검색 결과)가 추가되어야 한다고 명시한다. 평가 대상도 "모델 개발자·
서비스 운영자·개발자·사용자"에서 "경쟁 기술 진영·개발자·도입 기업 및 서비스 운영자·
투자·업계 관계자"로 바뀌었다 (담당: Phase 2-D, 박인애). 외부 검색 도구가 아직 없어 현재는
technical_evidence만으로 판단하고, evidence_items도 만들지 않는다 — 하지 않은 검색을
한 것처럼 근거를 지어내지 않기 위해서다. 검증 Agent는 이 Agent의 claim에 대해 근거
출처를 찾지 못하므로 재시도로 해결되지 않고 '정보 부족'으로만 표시된다.

TODO(박인애, Phase 2-D):
- 외부 정보 검색 도구(Tavily 등) 연동, stakeholder_evaluation을 tool-calling Agent로 재작성
- prompts/stakeholder_evaluation.md의 이해관계자 4유형 문구를 새 정의로 갱신
- 검색 결과로 evidence_items(source_type="external_search") 채우기
"""

from agents.base import load_prompt, structured_call
from agents.schemas import StakeholderAssessment
from graph.state import GraphState

SYSTEM_PROMPT = load_prompt("stakeholder_evaluation")
AGENT_NAME = "stakeholder_evaluation"


def run(state: GraphState) -> dict:
    technologies = state["selected_technologies"]
    technical_evidence = state.get("technical_evidence", {})
    retry_hint = state.get("retry_hints", {}).get(AGENT_NAME, "")
    result_by_tech: dict = {}

    for tech_name in technologies:
        user_content = (
            f"## 기술명\n{tech_name}\n\n"
            f"## 기술 조사 Agent 요약\n{technical_evidence.get(tech_name, {})}"
        )
        if retry_hint:
            user_content += f"\n\n## 이전 검증에서 부족했던 부분\n{retry_hint}"
        result = structured_call(StakeholderAssessment, SYSTEM_PROMPT, user_content)
        result_by_tech[tech_name] = result.model_dump()

    return {"stakeholder_evaluation": result_by_tech}
