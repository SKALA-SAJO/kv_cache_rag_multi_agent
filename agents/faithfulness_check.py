"""검증 Agent (PDF Agent, Faithfulness Check). RAG 여부: X. 도구: 근거 대조 도구
(Claim-Evidence 매칭).

synthesis의 각 claim을 evidence_items(State)와 대조한다. 실패한 claim은
evidence_refs를 통해 evidence_items의 agent 필드로 역추적되어, "어느 Agent를
재실행해야 하는지"(agents_to_retry)와 "그 Agent에게 줄 재검색 힌트"(retry_hints)를
이 파일이 프로그램적으로 계산한다 — LLM은 claim-evidence 일치 여부만 판정한다.

Supervisor 패턴의 통신 제약에 따라 이 Agent는 다른 Agent의 입력(retry_hints)을 직접
쓰지 않는다. 재작업 후보와 힌트는 자기 verdict(faithfulness_check) 안에만 담고, 재작업
여부·대상은 Supervisor(graph/supervisor.py)가 재작업 상한과 함께 결정한다.
"""

import json
from collections import defaultdict

from agents.base import format_evidence_items, load_prompt, structured_call
from agents.schemas import FaithfulnessCheckResult
from graph.state import GraphState

SYSTEM_PROMPT = load_prompt("faithfulness_check")


def run(state: GraphState) -> dict:
    synthesis = state.get("synthesis", {})
    evidence_items = state.get("evidence_items", [])
    context = format_evidence_items(evidence_items)

    user_content = (
        f"## Synthesis (검증 대상 claim)\n{json.dumps(synthesis, ensure_ascii=False, indent=2)}\n\n"
        f"## Evidence Items (번호 붙은 근거 목록)\n{context}"
    )
    result = structured_call(FaithfulnessCheckResult, SYSTEM_PROMPT, user_content, role="judge")

    agents_to_retry: set[str] = set()
    hints_by_agent: dict[str, list[str]] = defaultdict(list)

    for check in result.claim_checks:
        if check.status != "fail":
            continue
        attributed = False
        for idx in check.evidence_refs:
            if 0 <= idx < len(evidence_items):
                agent_name = evidence_items[idx].get("agent")
                if agent_name:
                    agents_to_retry.add(agent_name)
                    hints_by_agent[agent_name].append(f"'{check.claim}' 근거 부족: {check.note}")
                    attributed = True
        if not attributed:
            # evidence_refs가 비어있다 = 애초에 이 claim을 뒷받침할 근거가 하나도 없었다는 뜻.
            # 어느 Agent도 이 claim의 근거를 만들지 않았으므로 재시도로 해결되지 않고
            # '정보 부족'으로만 표시된다 (report_writer가 faithfulness_check 결과를 그대로 반영).
            pass

    retry_hints = {agent: " / ".join(notes) for agent, notes in hints_by_agent.items()}

    return {
        "faithfulness_check": {
            **result.model_dump(),
            "agents_to_retry": sorted(agents_to_retry),
            "retry_hints": retry_hints,
        },
    }
