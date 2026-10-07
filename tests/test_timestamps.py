"""결과 키 타임스탬프(error_times, last_decision.ts)가 State에 남는지 외부 API 없이 점검한다.

"에러가 났다"가 아니라 "언제, 누가" 실패했는지를 State만으로 알 수 있어야 한다.
"""

from __future__ import annotations

from datetime import datetime
import unittest
from unittest.mock import patch

from graph import workflow
from graph.state import ControlState
from graph.supervisor import initial_control_state, supervisor_node
from tests.test_state_workflow import _fake_runners


def _is_iso_seconds(value: str) -> bool:
    # observability.log_event의 ts와 같은 형식: 2026-10-07T16:45:12
    return datetime.fromisoformat(value).isoformat(timespec="seconds") == value


class TimestampTest(unittest.TestCase):
    def test_schema_and_initial_state(self) -> None:
        self.assertIn("error_times", ControlState.__annotations__)
        self.assertEqual(initial_control_state("t")["error_times"], {})

    def test_failed_worker_records_error_time_without_changing_error_format(self) -> None:
        def boom(state):
            raise RuntimeError("down")

        update = workflow._worker("market_evaluation", boom)({"run_id": "t"})
        self.assertEqual(update["errors"], {"market_evaluation": "RuntimeError: down"})  # 문자열 그대로
        self.assertTrue(_is_iso_seconds(update["error_times"]["market_evaluation"]))
        self.assertNotIn("error_times", workflow._worker("x", lambda s: {})({"run_id": "t"}))  # 성공 시 없음

    def test_last_decision_has_timestamp(self) -> None:
        state = {"research_question": "q", "selected_technologies": {"A": {}}, **initial_control_state("t")}
        decision = supervisor_node(state)["last_decision"]
        self.assertEqual(decision["action"], "collect_tech")
        self.assertTrue(_is_iso_seconds(decision["ts"]))

    def test_graph_run_keeps_error_time_of_failed_parallel_agent(self) -> None:
        calls: list[str] = []
        with patch.dict(workflow.AGENT_RUNNERS, _fake_runners(calls, fail_once={"domain_evaluation"})):
            result = workflow.build_graph().invoke(
                {"research_question": "q", "run_id": "ts-test"}, config={"recursion_limit": 80})
        self.assertEqual(list(result["error_times"]), ["domain_evaluation"])  # 병렬 실패도 병합되어 남음
        self.assertTrue(result["errors"]["domain_evaluation"].startswith("RuntimeError"))
        self.assertTrue(_is_iso_seconds(result["last_decision"]["ts"]))


if __name__ == "__main__":
    unittest.main()
