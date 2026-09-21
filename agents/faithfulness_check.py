"""검증 Agent (PDF Agent, Faithfulness Check). RAG 여부: X.

synthesis의 각 claim을 retrieved_documents 원문과 대조한다. 실패 시 재검색 루프를 위해
verification_retry_count를 증가시킨다 (PDF에 명시되지 않은 무한 루프 방지 장치, config의
MAX_VERIFICATION_RETRIES로 상한을 둔다. graph/workflow.py 참고).
"""

import json

from agents.base import format_context, load_prompt, structured_call
from agents.schemas import FaithfulnessCheckResult
from graph.state import GraphState

SYSTEM_PROMPT = load_prompt("faithfulness_check")


def run(state: GraphState) -> dict:
    synthesis = state.get("synthesis", {})
    documents = state.get("retrieved_documents", [])
    context = format_context(documents)

    user_content = (
        f"## Synthesis (검증 대상 claim)\n{json.dumps(synthesis, ensure_ascii=False, indent=2)}\n\n"
        f"## 검색된 근거 문서 (Context)\n{context}"
    )
    result = structured_call(FaithfulnessCheckResult, SYSTEM_PROMPT, user_content, role="judge")

    retry_count = state.get("verification_retry_count", 0)
    if not result.passed:
        retry_count += 1

    return {
        "faithfulness_check": result.model_dump(),
        "verification_retry_count": retry_count,
    }
