"""평가 종합 Agent (PDF Agent I). RAG 여부: X. 4관점 평가 결과를 종합한다."""

import json

from agents.base import load_prompt, structured_call
from agents.schemas import Synthesis
from graph.state import GraphState

SYSTEM_PROMPT = load_prompt("synthesis")


def run(state: GraphState) -> dict:
    payload = {
        "selected_technologies": list(state["selected_technologies"].keys()),
        "trl_evaluation": state.get("trl_evaluation", {}),
        "market_evaluation": state.get("market_evaluation", {}),
        "stakeholder_evaluation": state.get("stakeholder_evaluation", {}),
        "domain_evaluation": state.get("domain_evaluation", {}),
    }
    user_content = json.dumps(payload, ensure_ascii=False, indent=2)
    result = structured_call(Synthesis, SYSTEM_PROMPT, user_content)
    return {"synthesis": result.model_dump()}
