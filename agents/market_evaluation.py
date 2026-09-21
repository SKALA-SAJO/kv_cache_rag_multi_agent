"""시장 평가 Agent (PDF Agent F).

v0.0 구현 범위: RAG는 기술 문서 검색에만 사용하기로 하여, 시장·산업 자료 코퍼스는 색인하지
않았다. 따라서 이 Agent는 검색을 수행하지 않고 기술 조사 Agent 결과 + LLM의 일반 지식으로
판단하며, 그 한계를 출력의 limitations/confidence에 명시하도록 프롬프트에서 강제한다.
(README '실제 구현 범위 및 한계' 참고)
"""

from agents.base import load_prompt, structured_call
from agents.schemas import RubricAssessment
from graph.state import GraphState

SYSTEM_PROMPT = load_prompt("market_evaluation")


def run(state: GraphState) -> dict:
    technologies = state["selected_technologies"]
    technical_evidence = state.get("technical_evidence", {})
    result_by_tech: dict = {}

    for tech_name, tech_info in technologies.items():
        user_content = (
            f"## 기술명\n{tech_name}\n\n"
            f"## 분류\n{tech_info['category']}\n\n"
            f"## 기술 조사 Agent 요약\n{technical_evidence.get(tech_name, {})}"
        )
        result = structured_call(RubricAssessment, SYSTEM_PROMPT, user_content)
        result_by_tech[tech_name] = result.model_dump()

    return {"market_evaluation": result_by_tech}
