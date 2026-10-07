"""State 레이어드 구조 테스트 (API 호출 없음).

graph/state.py가 PayloadState(작업 결과)와 ControlState(제어 메타)를 별도 TypedDict로 선언하고
GraphState가 둘을 합성한다는 설계(README "State Schema" 첫 항목)가 코드에서 유지되는지 고정한다.
"""

from __future__ import annotations

import contextlib
import io
import tempfile
import typing
import unittest
from unittest.mock import patch

from config import settings
from graph import workflow
from graph.state import (
    ControlState,
    GraphState,
    PayloadState,
    dedupe_evidence_items,
    dedupe_references,
    merge_dict,
)


class StateLayerTest(unittest.TestCase):
    def test_layers_are_disjoint_and_compose_the_graph_state(self) -> None:
        payload, control = set(PayloadState.__annotations__), set(ControlState.__annotations__)
        self.assertFalse(payload & control)  # 한 키는 한 층에만 속한다
        self.assertEqual(payload | control, set(GraphState.__annotations__))

    def test_key_placement_follows_the_design(self) -> None:
        self.assertTrue({"evidence_items", "references", "synthesis", "quality_verdict"} <= set(PayloadState.__annotations__))
        self.assertTrue({"run_id", "step_count", "max_steps", "next_nodes", "node_status", "attempts", "errors",
                         "rework_counts", "sufficiency"} <= set(ControlState.__annotations__))
        self.assertNotIn("retrieved_documents", GraphState.__annotations__)  # 원문 청크는 State에 두지 않는다

    def test_partial_updates_allowed_on_every_layer(self) -> None:
        # total=False가 아니면 노드가 일부 키만 갱신할 수 없다 (강의: 부분 업데이트 허용 옵션)
        for state_type in (PayloadState, ControlState, GraphState):
            self.assertFalse(state_type.__total__, state_type.__name__)

    def test_reducers_survive_the_split(self) -> None:
        hints = typing.get_type_hints(GraphState, include_extras=True)
        expected = {"node_status": merge_dict, "errors": merge_dict,
                    "references": dedupe_references, "evidence_items": dedupe_evidence_items}
        for key, reducer in expected.items():
            self.assertIn(reducer, typing.get_args(hints[key])[1:], key)

    def test_graph_still_compiles_with_the_layered_schema(self) -> None:
        self.assertIsNotNone(workflow.build_graph())


class WorkerWritesControlKeysTest(unittest.TestCase):
    """제어 키는 Supervisor와 _worker 래퍼만 쓴다 — Agent 본체가 반환한 페이로드에 래퍼가 더하는 키 확인."""

    def _run_worker(self, fn) -> dict:
        with tempfile.TemporaryDirectory() as tmp, patch.object(settings, "outputs_dir", tmp), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return workflow._worker("synthesis", fn)({"run_id": "layer-test"})

    def test_success_adds_only_node_status(self) -> None:
        update = self._run_worker(lambda state: {"synthesis": {"agreements": []}})
        self.assertEqual(set(update), {"synthesis", "node_status"})
        self.assertEqual(update["node_status"], {"synthesis": "done"})

    def test_failure_adds_node_status_and_errors_only(self) -> None:
        def boom(state):
            raise RuntimeError("simulated")

        update = self._run_worker(boom)
        self.assertEqual(set(update), {"node_status", "errors"})
        self.assertEqual(update["node_status"], {"synthesis": "failed"})
        self.assertIn("simulated", update["errors"]["synthesis"])


if __name__ == "__main__":
    unittest.main()
