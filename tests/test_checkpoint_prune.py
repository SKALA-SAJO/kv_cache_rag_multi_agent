"""정상 종료 run의 체크포인트 정리 테스트 (API 호출 없음, 임시 SQLite DB 사용).

재개 자체의 검증은 test_state_workflow.CheckpointResumeTest가 맡는다. 여기서는 "정리해도 최종 State가
보존되는가 / 다른 run을 건드리지 않는가 / 끊긴 run은 정리 대상이 아닌가"만 본다.
"""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from langgraph.checkpoint.sqlite import SqliteSaver

from config import settings
from graph import workflow
from graph.checkpoint_maintenance import is_run_finished, prune_checkpoints
from tests.test_state_workflow import _fake_runners


def _config(run_id: str) -> dict:
    return {"configurable": {"thread_id": run_id}, "recursion_limit": settings.graph_recursion_limit}


def _count(db: str, thread_id: str) -> tuple[int, int]:
    # sqlite3의 `with conn`은 연결을 닫지 않고 커밋만 하므로 closing으로 명시적으로 닫는다(열린 연결이 남으면
    # WAL이 비워지지 않아 정리 후 파일 크기가 줄지 않는다).
    with closing(sqlite3.connect(db)) as conn:
        return (
            conn.execute("SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?", (thread_id,)).fetchone()[0],
            conn.execute("SELECT COUNT(*) FROM writes WHERE thread_id = ?", (thread_id,)).fetchone()[0],
        )


def _run_to_end(db: str, run_id: str) -> dict:
    """대역 Agent로 그래프를 끝까지 돌리고 종료 시점의 State 값을 돌려준다."""
    with SqliteSaver.from_conn_string(db) as checkpointer:
        graph = workflow.build_graph(checkpointer=checkpointer)
        graph.invoke({"research_question": "q", "run_id": run_id}, config=_config(run_id))
        return graph.get_state(_config(run_id)).values


def _state_values(db: str, run_id: str) -> dict:
    with SqliteSaver.from_conn_string(db) as checkpointer:
        return workflow.build_graph(checkpointer=checkpointer).get_state(_config(run_id)).values


class CheckpointPruneTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name
        self.db = str(Path(self.tmp) / "checkpoints.sqlite")
        patches = [
            patch.object(settings, "outputs_dir", self.tmp),
            patch.dict(workflow.AGENT_RUNNERS, _fake_runners([], fail_once=set())),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_prune_keeps_only_last_checkpoint_and_preserves_final_state(self) -> None:
        before_values = _run_to_end(self.db, "run-a")
        self.assertGreater(_count(self.db, "run-a")[0], 1)

        before_bytes, after_bytes = prune_checkpoints(self.db, "run-a")

        self.assertEqual(_count(self.db, "run-a")[0], 1)
        self.assertLess(after_bytes, before_bytes)
        after_values = _state_values(self.db, "run-a")
        self.assertEqual(after_values, before_values)  # 근거·보고서·verdict·제어 메타 모두 동일
        self.assertEqual(after_values["final_report"], before_values["final_report"])

    def test_prune_does_not_touch_other_threads_or_missing_thread(self) -> None:
        _run_to_end(self.db, "run-a")
        other_values = _run_to_end(self.db, "run-b")
        other_count = _count(self.db, "run-b")

        prune_checkpoints(self.db, "run-a")
        self.assertEqual(_count(self.db, "run-b"), other_count)
        self.assertEqual(_state_values(self.db, "run-b"), other_values)

        sizes = prune_checkpoints(self.db, "no-such-run")  # 예외 없이 그대로 반환
        self.assertEqual(sizes[0], sizes[1])
        self.assertEqual(_count(self.db, "run-b"), other_count)

    def test_interrupted_run_is_not_finished_and_resumes_after_pruning_is_skipped(self) -> None:
        runners = _fake_runners([], fail_once=set())
        domain_runner = runners["domain_evaluation"]
        attempts = 0

        def interrupt_once(state):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise KeyboardInterrupt("simulated Ctrl+C")  # _worker가 잡지 않아 실행이 끊긴다
            return domain_runner(state)

        runners["domain_evaluation"] = interrupt_once
        with patch.dict(workflow.AGENT_RUNNERS, runners):
            with SqliteSaver.from_conn_string(self.db) as checkpointer:
                graph = workflow.build_graph(checkpointer=checkpointer)
                with self.assertRaises(KeyboardInterrupt):
                    graph.invoke({"research_question": "q", "run_id": "run-c"}, config=_config("run-c"))
                self.assertFalse(is_run_finished(graph, _config("run-c")))  # app.py는 이 run을 정리하지 않는다
            interrupted_count = _count(self.db, "run-c")

            with SqliteSaver.from_conn_string(self.db) as checkpointer:
                graph = workflow.build_graph(checkpointer=checkpointer)
                graph.invoke(None, config=_config("run-c"))  # 정리하지 않았으므로 재개가 끝까지 간다
                self.assertTrue(is_run_finished(graph, _config("run-c")))
        self.assertGreater(_count(self.db, "run-c")[0], interrupted_count[0])

    def test_finished_run_is_detected_and_prune_keeps_it_finished(self) -> None:
        _run_to_end(self.db, "run-d")
        with SqliteSaver.from_conn_string(self.db) as checkpointer:
            self.assertTrue(is_run_finished(workflow.build_graph(checkpointer=checkpointer), _config("run-d")))
        prune_checkpoints(self.db, "run-d")
        with SqliteSaver.from_conn_string(self.db) as checkpointer:
            self.assertTrue(is_run_finished(workflow.build_graph(checkpointer=checkpointer), _config("run-d")))


if __name__ == "__main__":
    unittest.main()
