"""관측성 계층(결정 이력 부록·LangSmith 결정 기록)을 외부 API 없이 점검한다."""

from __future__ import annotations

import tempfile
import unittest
from unittest.mock import MagicMock, patch

from config import settings
from graph import observability as obs


class DecisionAppendixTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.patcher = patch.object(settings, "outputs_dir", self.tmp.name)
        self.patcher.start()

    def tearDown(self) -> None:
        self.patcher.stop()
        self.tmp.cleanup()

    def _decide(self, run_id: str, step: int, action: str, targets: list[str], reason: str) -> None:
        obs.log_event(run_id, "supervisor", "decision", step=step, action=action, targets=targets, reason=reason)

    def test_appendix_lists_decisions_with_counts(self) -> None:
        self._decide("r1", 1, "collect_tech", ["tech_research"], "미수집")
        self._decide("r1", 2, "rework_insufficient", ["domain_evaluation"], "출처 1종 | 단일 출처")
        obs.log_event("r1", "domain_evaluation", "end", status="done")  # 결정이 아닌 이벤트는 제외
        self._decide("r1", 3, "end", [], "품질 통과")

        appendix = obs.render_decision_appendix("r1")
        self.assertIn("## 부록. Supervisor 결정 이력", appendix)
        self.assertIn("`r1`", appendix)
        self.assertIn("라우팅 3회, 재작업/재작성 1회", appendix)
        self.assertIn("| 2 | rework_insufficient | domain_evaluation | 출처 1종 \\| 단일 출처 |", appendix)
        self.assertIn("| 3 | end | END |", appendix)

    def test_resumed_step_is_listed_once(self) -> None:
        # --resume으로 중단된 step이 다시 기록돼도 step별 마지막 결정만 남는다
        self._decide("r2", 1, "collect_tech", ["tech_research"], "첫 시도")
        self._decide("r2", 1, "collect_tech", ["tech_research"], "재개 후")
        decisions = obs.summarize_routing("r2")
        self.assertEqual([d["reason"] for d in decisions], ["재개 후"])

    def test_missing_log_gives_empty_appendix(self) -> None:
        self.assertEqual(obs.render_decision_appendix("no-such-run"), "")


class TraceDecisionTest(unittest.TestCase):
    DECISION = {"step": 3, "action": "rework_insufficient", "targets": ["domain_evaluation"], "reason": "r"}

    def test_noop_when_tracing_disabled(self) -> None:
        with patch.object(obs, "tracing_is_enabled", return_value=False), \
                patch.object(obs.ls, "trace") as trace:
            obs.trace_decision(self.DECISION)
        trace.assert_not_called()

    def test_tags_node_run_and_opens_named_span(self) -> None:
        node_run = MagicMock()
        with patch.object(obs, "tracing_is_enabled", return_value=True), \
                patch.object(obs, "get_current_run_tree", return_value=node_run), \
                patch.object(obs.ls, "trace") as trace:
            obs.trace_decision(self.DECISION, {"domain_evaluation": {"sufficient": False}})
        node_run.add_tags.assert_called_once_with(["action:rework_insufficient"])
        self.assertEqual(trace.call_args.kwargs["name"], "decision: rework_insufficient → domain_evaluation")

    def test_langsmith_failure_does_not_break_graph(self) -> None:
        with patch.object(obs, "tracing_is_enabled", return_value=True), \
                patch.object(obs, "get_current_run_tree", side_effect=RuntimeError("down")):
            obs.trace_decision(self.DECISION)  # 예외가 밖으로 나오지 않아야 한다


class RunFeedbackTest(unittest.TestCase):
    def test_feedback_summarizes_routes_reworks_and_quality(self) -> None:
        decisions = [
            {"step": 1, "action": "collect_tech", "targets": ["tech_research"]},
            {"step": 2, "action": "rework_insufficient", "targets": ["domain_evaluation"]},
            {"step": 3, "action": "end", "targets": []},
        ]
        client = MagicMock()
        with patch.object(obs, "tracing_is_enabled", return_value=True), \
                patch.object(obs, "summarize_routing", return_value=decisions), \
                patch.object(obs.ls, "Client", return_value=client), \
                patch("langchain_core.tracers.langchain.wait_for_all_tracers"):
            obs.record_run_feedback("r", quality_passed=True)
        scores = {c.kwargs["key"]: c.kwargs["score"] for c in client.create_feedback.call_args_list}
        self.assertEqual(scores, {"supervisor_routes": 3, "supervisor_reworks": 1, "quality_passed": 1})


if __name__ == "__main__":
    unittest.main()
