"""시장 평가 Agent (PDF Agent).

PDF 업데이트본(sample.pdf) 기준 RAG=O(시장자료 코퍼스) + 외부 정보 검색 도구가 필요하지만,
둘 다 아직 구현되어 있지 않다 (담당: Phase 2-C, 박인애). 현재는 기술 조사 Agent 결과 +
LLM의 일반 지식으로만 판단하며, 그 한계를 출력의 limitations/confidence에 명시하도록
프롬프트에서 강제한다. evidence_items도 아직 만들지 않는다 — 실제 검색 없이 근거를
지어내는 것을 막기 위해서다(하지 않은 검색을 한 것처럼 evidence_items를 채우지 않는다).
그 결과 이 Agent의 claim은 검증 Agent가 근거 출처를 찾지 못해 재시도로도 해결되지 않고
'정보 부족'으로만 표시된다 — Phase 2-C가 시장자료 RAG + 외부 검색 도구를 붙이고 나서
documents_to_evidence_items(또는 동등한 도구 결과 변환)로 evidence_items를 채워야 한다.

TODO(박인애, Phase 2-C):
- rag/retriever.py에 시장자료(Gemini API 문서) 코퍼스 필터링 검색 추가
- 외부 정보 검색 도구(Tavily 등) 연동, market_evaluation을 tool-calling Agent로 재작성
- 위 두 출처에서 얻은 근거로 evidence_items(source_type="RAG"/"external_search") 채우기
"""

from agents.base import load_prompt, structured_call
from agents.schemas import QualitativeAssessment
from graph.state import GraphState

SYSTEM_PROMPT = load_prompt("market_evaluation")
AGENT_NAME = "market_evaluation"


def run(state: GraphState) -> dict:
    technologies = state["selected_technologies"]
    technical_evidence = state.get("technical_evidence", {})
    retry_hint = state.get("retry_hints", {}).get(AGENT_NAME, "")
    result_by_tech: dict = {}

    for tech_name, tech_info in technologies.items():
        user_content = (
            f"## 기술명\n{tech_name}\n\n"
            f"## 분류\n{tech_info['category']}\n\n"
            f"## 기술 조사 Agent 요약\n{technical_evidence.get(tech_name, {})}"
        )
        if retry_hint:
            user_content += f"\n\n## 이전 검증에서 부족했던 부분\n{retry_hint}"
        result = structured_call(QualitativeAssessment, SYSTEM_PROMPT, user_content)
        result_by_tech[tech_name] = result.model_dump()

    return {"market_evaluation": result_by_tech}
