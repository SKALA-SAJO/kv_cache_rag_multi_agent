"""라이브 테스트의 결과 판정을 실제 API·색인 없이 검증한다."""

from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tests import test_integration_live as live_tests


class LiveResultContractTest(unittest.TestCase):
    def _check_result(self, *, passed: bool, quality_status: str) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            chunks = directory / "chunks.jsonl"
            chunks.touch()
            (directory / "index.faiss").touch()
            settings = SimpleNamespace(
                openai_api_key="fake-key", tavily_api_key="fake-key",
                chunks_file=chunks, vectorstore_path=directory,
                outputs_path=directory, graph_recursion_limit=80,
            )
            result = {
                "evidence_items": [{"source_type": "external_search", "source_url": "https://example.test"}],
                "final_report": "SUMMARY\nreport\nREFERENCE",
                "quality_verdict": {"passed": passed},
                "node_status": {"quality_evaluation": quality_status},
                "next_nodes": [],
            }

            def invoke(*args, **kwargs):
                (directory / "report_fake.md").write_text(result["final_report"], encoding="utf-8")
                return result

            graph = Mock()
            graph.invoke.side_effect = invoke
            with patch("tests.test_integration_live.load_dotenv"), \
                    patch("config.settings", settings), \
                    patch("graph.workflow.build_graph", return_value=graph), \
                    patch("rag.external_search.register"):
                # 본문을 직접 호출하여 실행 조건과 독립적으로 결과 판정만 검증한다.
                test = live_tests.LiveIntegrationTest("test_full_workflow_writes_report_and_external_evidence")
                test.test_full_workflow_writes_report_and_external_evidence()

    def test_completed_quality_pass_is_accepted(self) -> None:
        self._check_result(passed=True, quality_status="done")

    def test_completed_quality_failure_is_accepted(self) -> None:
        # 품질 미달이어도 최종 보고서의 평가가 완료되면 정상 종료로 인정한다.
        self._check_result(passed=False, quality_status="done")

    def test_stale_verdict_after_quality_error_is_rejected(self) -> None:
        for status in ("failed", "excluded"):
            with self.subTest(status=status), self.assertRaisesRegex(AssertionError, "최종 보고서 품질 평가"):
                self._check_result(passed=False, quality_status=status)


if __name__ == "__main__":
    unittest.main()
