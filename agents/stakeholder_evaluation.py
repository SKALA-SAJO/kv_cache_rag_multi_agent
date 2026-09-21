"""이해관계자 평가 Agent (PDF Agent G). RAG 여부: X (PDF 설계상 비-RAG).

입력은 기술 조사 Agent의 technical_evidence만 사용한다 (PDF Graph 흐름 설계상 D -> G 직접 연결).
"""

from agents.base import load_prompt, structured_call
from agents.schemas import StakeholderAssessment
from graph.state import GraphState

SYSTEM_PROMPT = load_prompt("stakeholder_evaluation")


def run(state: GraphState) -> dict:
    technologies = state["selected_technologies"]
    technical_evidence = state.get("technical_evidence", {})
    result_by_tech: dict = {}

    for tech_name in technologies:
        user_content = (
            f"## 기술명\n{tech_name}\n\n"
            f"## 기술 조사 Agent 요약\n{technical_evidence.get(tech_name, {})}"
        )
        result = structured_call(StakeholderAssessment, SYSTEM_PROMPT, user_content)
        result_by_tech[tech_name] = result.model_dump()

    return {"stakeholder_evaluation": result_by_tech}
