"""Supervisor 패턴 State·라우팅 계약을 외부 API 없이 점검한다.

- 정책 단위 테스트: decide()가 State에 따라 다른 경로를 고르는지 (순서 하드코딩 아님)
- 그래프 통합 테스트: 하위 Agent를 가짜 함수로 바꿔 실제 LangGraph를 끝까지 돌려,
  재작업 루프·Fall-back·품질 미달 재작성·종료 보장을 확인한다.
- 체크포인트 재개: 관점 노드 중단 후 SQLite DB를 다시 열고 같은 thread_id로 재개한다.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langgraph.checkpoint.sqlite import SqliteSaver

from config import settings
from graph import workflow
from graph.state import GraphState, dedupe_evidence_items, dedupe_references, merge_dict
from graph.supervisor import PERSPECTIVE_NODES, assess_sufficiency, decide, initial_control_state

TECHS = {"A": {"category": "SW", "core_approach": "a"}, "B": {"category": "HW", "core_approach": "b"}}


def _evidence(agent: str, n: int, sources: int, techs=tuple(TECHS)) -> list[dict]:
    """기술마다 n건, 출처 sources종의 근거 (충분도 게이트는 관점 × 기술 단위로 센다)."""
    return [
        {"agent": agent, "technology": tech, "document_id": f"{agent}-{tech}-src{i % sources}",
         "evidence_quote": f"q{i}", "page_or_section": str(i)}
        for tech in techs
        for i in range(n)
    ]


def _state(**overrides) -> dict:
    state = {"research_question": "q", "selected_technologies": TECHS, **initial_control_state("test-run")}
    state.update(overrides)
    return state


class StateSchemaTest(unittest.TestCase):
    def test_control_and_payload_keys_exist(self) -> None:
        keys = set(GraphState.__annotations__)
        payload = {"technical_evidence", "trl_evaluation", "market_evaluation", "stakeholder_evaluation",
                   "domain_evaluation", "evidence_items", "references", "synthesis", "final_report", "quality_verdict"}
        control = {"run_id", "step_count", "max_steps", "next_nodes", "last_decision", "node_status",
                   "attempts", "errors", "rework_counts", "retry_hints", "sufficiency", "report_revisions"}
        self.assertTrue(payload <= keys)
        self.assertTrue(control <= keys)
        self.assertNotIn("retrieved_documents", keys)  # 대용량 원문은 State에 두지 않는다

    def test_reducers(self) -> None:
        self.assertEqual(merge_dict({"a": "done"}, {"b": "failed"}), {"a": "done", "b": "failed"})
        ref = {"source": "paper.pdf", "page": 3}
        self.assertEqual(len(dedupe_references([ref], [ref])), 1)
        ev = {"agent": "x", "document_id": "p", "page_or_section": "3", "evidence_quote": "same"}
        self.assertEqual(len(dedupe_evidence_items([ev], [ev])), 1)


class SupervisorPolicyTest(unittest.TestCase):
    def test_first_step_routes_to_tech_research(self) -> None:
        targets, action, _, updates = decide(_state())
        self.assertEqual(targets, ["tech_research"])
        self.assertEqual(updates["step_count"], 1)

    def test_missing_perspectives_dispatched_in_parallel(self) -> None:
        status = {**initial_control_state("t")["node_status"], "tech_research": "done", "trl_evaluation": "done"}
        targets, action, _, _ = decide(_state(node_status=status))
        self.assertEqual(action, "collect_perspectives")
        self.assertEqual(set(targets), set(PERSPECTIVE_NODES) - {"trl_evaluation"})

    def test_insufficient_perspective_gets_rework(self) -> None:
        status = {**initial_control_state("t")["node_status"], "tech_research": "done",
                  **{n: "done" for n in PERSPECTIVE_NODES}}
        evidence = [e for n in PERSPECTIVE_NODES if n != "market_evaluation" for e in _evidence(n, 4, 2)]
        evidence += _evidence("market_evaluation", 4, 1)  # 단일 출처
        targets, action, reason, updates = decide(_state(node_status=status, evidence_items=evidence,
                                                         **{n: {"A": {}, "B": {}} for n in PERSPECTIVE_NODES}))
        self.assertEqual((targets, action), (["market_evaluation"], "rework_insufficient"))
        self.assertIn("market_evaluation", updates["retry_hints"])
        self.assertEqual(updates["node_status"]["synthesis"], "pending")
        self.assertIn("단일 출처", reason)

    def test_rework_budget_exhausted_moves_on(self) -> None:
        status = {**initial_control_state("t")["node_status"], "tech_research": "done",
                  **{n: "done" for n in PERSPECTIVE_NODES}}
        targets, action, _, _ = decide(_state(
            node_status=status, rework_counts={n: settings.max_rework_per_agent for n in PERSPECTIVE_NODES}))
        self.assertEqual((targets, action), (["synthesis"], "synthesize"))

    def test_failed_agent_retried_then_excluded(self) -> None:
        status = {**initial_control_state("t")["node_status"], "tech_research": "done",
                  **{n: "done" for n in PERSPECTIVE_NODES}, "stakeholder_evaluation": "failed"}
        targets, action, _, updates = decide(_state(node_status=status))
        self.assertEqual(targets, ["stakeholder_evaluation"])
        _, _, _, updates = decide(_state(node_status=status, attempts=updates["attempts"]))
        self.assertEqual(updates["node_status"]["stakeholder_evaluation"], "excluded")

    def test_quality_failure_routes_to_revision_and_terminates(self) -> None:
        status = {n: "done" for n in initial_control_state("t")["node_status"]}
        verdict = {"passed": False, "failed_criteria": ["neutrality"], "rework_targets": [], "feedback": "fix"}
        common = dict(node_status=status, quality_verdict=verdict, final_report="r",
                      rework_counts={n: 9 for n in PERSPECTIVE_NODES})
        targets, action, _, updates = decide(_state(**common))
        self.assertEqual((targets, action), (["report_writer"], "revise_report"))
        self.assertEqual(updates["quality_feedback"], "fix")
        targets, action, _, _ = decide(_state(**common, report_revisions=settings.max_report_revisions))
        self.assertEqual((targets, action), ([], "end"))

    def test_faithfulness_rework_rounds_are_capped(self) -> None:
        status = {**initial_control_state("t")["node_status"], **{n: "done" for n in PERSPECTIVE_NODES},
                  "tech_research": "done", "synthesis": "done", "faithfulness_check": "done"}
        check = {"passed": False, "agents_to_retry": ["trl_evaluation"], "retry_hints": {}}
        sufficient = {"evidence_items": [e for n in PERSPECTIVE_NODES for e in _evidence(n, 4, 2)],
                      **{n: {"A": {}, "B": {}} for n in PERSPECTIVE_NODES}}
        targets, action, _, updates = decide(_state(node_status=status, faithfulness_check=check, **sufficient))
        self.assertEqual((targets, action), (["trl_evaluation"], "rework_unfaithful"))
        self.assertEqual(updates["faithfulness_rounds"], 1)
        targets, action, _, _ = decide(_state(node_status=status, faithfulness_check=check, **sufficient,
                                              faithfulness_rounds=settings.max_faithfulness_rounds))
        self.assertEqual((targets, action), (["report_writer"], "write_report"))

    def test_tech_rework_reruns_only_co_targeted_perspectives(self) -> None:
        status = {**initial_control_state("t")["node_status"], **{n: "done" for n in PERSPECTIVE_NODES},
                  "tech_research": "done", "synthesis": "done", "faithfulness_check": "done"}
        check = {"passed": False, "agents_to_retry": ["stakeholder_evaluation", "tech_research"],
                 "retry_hints": {"tech_research": "t", "stakeholder_evaluation": "s"},
                 "insufficient_evidence_claims": ["c"]}
        sufficient = {"evidence_items": [e for n in PERSPECTIVE_NODES for e in _evidence(n, 4, 2)],
                      **{n: {"A": {}, "B": {}} for n in PERSPECTIVE_NODES}}
        targets, action, reason, updates = decide(_state(node_status=status, faithfulness_check=check, **sufficient))
        self.assertEqual((targets, action), (["tech_research"], "rework_unfaithful"))
        self.assertEqual(updates["node_status"]["stakeholder_evaluation"], "pending")
        for kept in ("trl_evaluation", "market_evaluation", "domain_evaluation"):
            self.assertNotIn(kept, updates["node_status"])  # 지목되지 않은 관점은 done 유지
        self.assertIn("결과 유지", reason)

        # 기술 조사 완료 후: 재작업 지시받은 관점만 디스패치되고 사유가 '재작업'으로 구분된다
        status2 = {**status, **updates["node_status"], "tech_research": "done"}
        targets, action, reason, _ = decide(_state(node_status=status2, retry_hints=updates["retry_hints"],
                                                   rework_counts=updates["rework_counts"], **sufficient))
        self.assertEqual((targets, action), (["stakeholder_evaluation"], "collect_perspectives"))
        self.assertIn("재작업 지시 관점", reason)
        self.assertNotIn("미수집", reason)

    def test_decision_reasons_distinguish_first_run_from_rework(self) -> None:
        done = {n: "done" for n in initial_control_state("t")["node_status"]}
        sufficient = {"evidence_items": [e for n in PERSPECTIVE_NODES for e in _evidence(n, 4, 2)],
                      **{n: {"A": {}, "B": {}} for n in PERSPECTIVE_NODES}}
        pending_report = {**done, "report_writer": "pending", "quality_evaluation": "pending"}
        _, _, first, _ = decide(_state(node_status=pending_report, **sufficient))
        self.assertIn("근거 충분성 확인 완료", first)
        _, _, again, _ = decide(_state(node_status=pending_report, final_report="old",
                                       rework_counts={"domain_evaluation": 1}, **sufficient))
        self.assertIn("재작업 결과 반영", again)
        _, _, quality, _ = decide(_state(node_status=pending_report, final_report="old",
                                         quality_feedback="fix", **sufficient))
        self.assertIn("품질 미달 원인 관점 재작업 반영", quality)

        pending_synth = {**done, **{n: "pending" for n in ("synthesis", "faithfulness_check",
                                                            "report_writer", "quality_evaluation")}}
        _, _, synth_first, _ = decide(_state(node_status=pending_synth, **sufficient))
        _, _, synth_again, _ = decide(_state(node_status=pending_synth, synthesis={"agreements": []},
                                             rework_counts={"trl_evaluation": 1}, **sufficient))
        self.assertIn("충분도 게이트 통과", synth_first)
        self.assertIn("재종합", synth_again)

    def test_step_cap_still_runs_quality_gate_then_terminates(self) -> None:
        cap = settings.max_supervisor_steps
        done = {n: "done" for n in initial_control_state("t")["node_status"]}
        # 보고서 미작성 → 1회 생성
        targets, action, _, _ = decide(_state(step_count=cap))
        self.assertEqual((targets, action), (["report_writer"], "force_report"))
        # 보고서는 있지만 미평가 → 품질 평가 1회 (상한이어도 필수 게이트는 건너뛰지 않음)
        status = {**done, "quality_evaluation": "pending"}
        targets, action, _, _ = decide(_state(step_count=cap + 1, node_status=status, final_report="r"))
        self.assertEqual((targets, action), (["quality_evaluation"], "force_quality"))
        # 평가 끝 → 미달이어도 종료
        verdict = {"passed": False, "failed_criteria": ["neutrality"]}
        targets, action, reason, _ = decide(_state(step_count=cap + 2, node_status=done, final_report="r",
                                                   quality_verdict=verdict))
        self.assertEqual((targets, action), ([], "end"))
        self.assertIn("미달", reason)

    def test_sufficiency_is_judged_per_technology(self) -> None:
        # Agent 합계로는 출처 2종이지만 B 기술은 단일 출처 → 미달 (합계 판정이 가리던 경우)
        evidence = _evidence("domain_evaluation", 4, 2, techs=("A",)) + _evidence("domain_evaluation", 4, 1, techs=("B",))
        verdict = assess_sufficiency(_state(evidence_items=evidence), "domain_evaluation")
        self.assertFalse(verdict["sufficient"])
        self.assertIn("B: ", verdict["reason"])
        self.assertNotIn("A: ", verdict["reason"])
        self.assertEqual(verdict["per_technology"]["B"]["distinct_sources"], 1)
        # 공용 자료(technology=None)는 기술별 근거로 세지 않는다
        shared = [{**e, "technology": None} for e in _evidence("domain_evaluation", 4, 3, techs=("A",))]
        self.assertFalse(assess_sufficiency(_state(evidence_items=shared), "domain_evaluation")["sufficient"])

    def test_sufficiency_flags_insufficient_technology(self) -> None:
        state = _state(trl_evaluation={"A": {"insufficient_evidence": True}, "B": {}},
                       evidence_items=_evidence("trl_evaluation", 5, 3))
        self.assertFalse(assess_sufficiency(state, "trl_evaluation")["sufficient"])


def _fake_runners(calls: list[str], fail_once: set[str]):
    """하위 Agent 대역. market은 첫 실행에서 단일 출처 근거만 내 충분도 게이트에 걸리고,
    품질 평가는 첫 판정에서 미달 → 보고서 재작성 → 두 번째 판정 통과."""
    counts: dict[str, int] = {}

    def make(name):
        def run(state):
            counts[name] = counts.get(name, 0) + 1
            calls.append(name)
            if name in fail_once and counts[name] == 1:
                raise RuntimeError("simulated failure")
            if name == "tech_research":
                return {"technical_evidence": {"A": {}, "B": {}}}
            if name in PERSPECTIVE_NODES:
                sources = 1 if (name == "market_evaluation" and counts[name] == 1) else 3
                techs = tuple(state["selected_technologies"])
                return {name: {t: {} for t in techs}, "evidence_items": _evidence(name, 4, sources, techs)}
            if name == "synthesis":
                return {"synthesis": {"agreements": []}}
            if name == "faithfulness_check":
                return {"faithfulness_check": {"passed": True, "agents_to_retry": []}}
            if name == "report_writer":
                return {"final_report": f"report v{state.get('report_revisions', 0)}"}
            if name == "quality_evaluation":
                ok = counts[name] >= 2
                return {"quality_verdict": {"passed": ok, "failed_criteria": [] if ok else ["groundedness"],
                                            "rework_targets": [], "feedback": "add citations"}}
            raise AssertionError(name)
        return run

    return {name: make(name) for name in workflow.AGENT_RUNNERS}


class SupervisorGraphTest(unittest.TestCase):
    def test_dynamic_run_with_rework_fallback_and_revision(self) -> None:
        calls: list[str] = []
        with tempfile.TemporaryDirectory() as tmp, patch.object(settings, "outputs_dir", tmp), \
                patch.dict(workflow.AGENT_RUNNERS, _fake_runners(calls, fail_once={"domain_evaluation"})):
            result = workflow.build_graph().invoke(
                {"research_question": "q", "run_id": "graph-test"}, config={"recursion_limit": 80})

        self.assertEqual(result["final_report"], "report v1")  # 품질 미달 → 재작성본
        self.assertTrue(result["quality_verdict"]["passed"])
        self.assertEqual(calls.count("market_evaluation"), 2)  # 충분도 게이트 재작업
        self.assertEqual(calls.count("domain_evaluation"), 2)  # 실패 → 재시도
        self.assertEqual(calls.count("report_writer"), 2)
        self.assertEqual(calls.count("quality_evaluation"), 2)
        self.assertEqual(result["rework_counts"], {"market_evaluation": 1})
        self.assertEqual(result["next_nodes"], [])
        self.assertLessEqual(result["step_count"], settings.max_supervisor_steps)

    def test_graph_always_terminates_when_quality_never_passes(self) -> None:
        calls: list[str] = []
        runners = _fake_runners(calls, fail_once=set())

        def never_pass(state):
            calls.append("quality_evaluation")
            return {"quality_verdict": {"passed": False, "failed_criteria": ["bias_control"],
                                        "rework_targets": ["market_evaluation"], "feedback": "x"}}

        runners["quality_evaluation"] = never_pass
        with tempfile.TemporaryDirectory() as tmp, patch.object(settings, "outputs_dir", tmp), \
                patch.dict(workflow.AGENT_RUNNERS, runners):
            result = workflow.build_graph().invoke(
                {"research_question": "q", "run_id": "never-pass"}, config={"recursion_limit": 120})

        self.assertFalse(result["quality_verdict"]["passed"])
        self.assertEqual(result["next_nodes"], [])
        self.assertLessEqual(result["rework_counts"]["market_evaluation"], settings.max_rework_per_agent)
        self.assertLessEqual(result["report_revisions"], settings.max_report_revisions)


class TechReworkScopeGraphTest(unittest.TestCase):
    def test_tech_rework_does_not_rerun_untargeted_perspectives(self) -> None:
        calls: list[str] = []
        runners = _fake_runners(calls, fail_once=set())
        faith_calls = 0

        def faith_fails_once(state):
            nonlocal faith_calls
            faith_calls += 1
            calls.append("faithfulness_check")
            if faith_calls == 1:
                return {"faithfulness_check": {
                    "passed": False, "agents_to_retry": ["tech_research", "stakeholder_evaluation"],
                    "retry_hints": {}, "insufficient_evidence_claims": ["c"]}}
            return {"faithfulness_check": {"passed": True, "agents_to_retry": []}}

        runners["faithfulness_check"] = faith_fails_once
        with tempfile.TemporaryDirectory() as tmp, patch.object(settings, "outputs_dir", tmp), \
                patch.dict(workflow.AGENT_RUNNERS, runners):
            result = workflow.build_graph().invoke(
                {"research_question": "q", "run_id": "tech-scope"}, config={"recursion_limit": 80})

        self.assertEqual(calls.count("tech_research"), 2)
        self.assertEqual(calls.count("stakeholder_evaluation"), 2)  # 함께 지목 → 재실행
        self.assertEqual(calls.count("trl_evaluation"), 1)  # 미지목 → 결과 유지
        self.assertEqual(calls.count("domain_evaluation"), 1)
        # market은 첫 실행 단일 출처로 충분도 게이트 재작업 1회 → 2회 (tech 재작업과 무관)
        self.assertEqual(calls.count("market_evaluation"), 2)
        self.assertEqual(result["next_nodes"], [])


class CheckpointResumeTest(unittest.TestCase):
    def test_resume_after_interrupt_with_reopened_sqlite_checkpoint(self) -> None:
        calls: list[str] = []
        runners = _fake_runners(calls, fail_once=set())
        domain_runner = runners["domain_evaluation"]
        domain_attempts = 0

        def interrupt_once(state):
            nonlocal domain_attempts
            domain_attempts += 1
            if domain_attempts == 1:
                # RuntimeError는 _worker가 처리하므로 실제 Ctrl+C처럼 실행을 끊는다.
                raise KeyboardInterrupt("simulated checkpoint interruption")
            return domain_runner(state)

        runners["domain_evaluation"] = interrupt_once
        run_id = "checkpoint-resume-test"
        config = {
            "configurable": {"thread_id": run_id},
            "recursion_limit": settings.graph_recursion_limit,
        }

        with tempfile.TemporaryDirectory() as tmp, patch.object(settings, "outputs_dir", tmp), \
                patch.dict(workflow.AGENT_RUNNERS, runners):
            database = str(Path(tmp) / "checkpoints.sqlite")
            with SqliteSaver.from_conn_string(database) as checkpointer:
                graph = workflow.build_graph(checkpointer=checkpointer)
                with self.assertRaisesRegex(KeyboardInterrupt, "simulated checkpoint interruption"):
                    graph.invoke({"research_question": "q", "run_id": run_id}, config=config)

                snapshot = graph.get_state(config)
                self.assertIn("domain_evaluation", snapshot.next)
                self.assertEqual(snapshot.values["node_status"]["tech_research"], "done")
                self.assertEqual(snapshot.values["node_status"]["domain_evaluation"], "running")
                self.assertNotIn("final_report", snapshot.values)
                self.assertEqual(calls.count("tech_research"), 1)
                self.assertEqual(calls.count("report_writer"), 0)

            # 새 연결·새 그래프로 복구하여 메모리만으로 이어가는 경우를 배제한다.
            with SqliteSaver.from_conn_string(database) as checkpointer:
                resumed_graph = workflow.build_graph(checkpointer=checkpointer)
                self.assertEqual(resumed_graph.get_state(config).values["run_id"], run_id)
                other_config = {"configurable": {"thread_id": "other-thread"}}
                self.assertEqual(resumed_graph.get_state(other_config).values, {})
                result = resumed_graph.invoke(None, config=config)
                self.assertEqual(resumed_graph.get_state(config).next, ())

        self.assertEqual(domain_attempts, 2)
        self.assertEqual(calls.count("tech_research"), 1)  # 초기 단계부터 재시작하지 않음
        self.assertEqual(result["run_id"], run_id)
        self.assertEqual(result["node_status"]["domain_evaluation"], "done")
        self.assertEqual(result["final_report"], "report v1")
        self.assertTrue(result["quality_verdict"]["passed"])
        self.assertEqual(result["next_nodes"], [])


if __name__ == "__main__":
    unittest.main()
