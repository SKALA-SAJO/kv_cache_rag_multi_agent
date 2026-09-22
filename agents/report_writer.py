"""보고서 생성 Agent (PDF Agent K). RAG 여부: X. 종합 결과를 최종 Markdown 보고서로 구성한다."""

import json
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage

from agents.base import get_llm, load_prompt
from config import settings
from graph.state import GraphState

SYSTEM_PROMPT = load_prompt("report_writer")


def run(state: GraphState) -> dict:
    payload = {
        "research_question": state.get("research_question"),
        "selected_technologies": state.get("selected_technologies"),
        "technical_evidence": state.get("technical_evidence"),
        "trl_evaluation": state.get("trl_evaluation"),
        "market_evaluation": state.get("market_evaluation"),
        "stakeholder_evaluation": state.get("stakeholder_evaluation"),
        "domain_evaluation": state.get("domain_evaluation"),
        "synthesis": state.get("synthesis"),
        "faithfulness_check": state.get("faithfulness_check"),
        "references": state.get("references"),
        "retry_count": state.get("retry_count", 0),
    }
    user_content = json.dumps(payload, ensure_ascii=False, indent=2)

    llm = get_llm("generator")
    response = llm.invoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_content)]
    )
    report_markdown = response.content

    settings.outputs_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = settings.outputs_path / f"report_{timestamp}.md"
    report_path.write_text(report_markdown, encoding="utf-8")
    print(f"[report_writer] 보고서 저장: {report_path}")

    return {"final_report": report_markdown}
