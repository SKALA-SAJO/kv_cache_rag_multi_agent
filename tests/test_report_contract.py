"""보고서 계약과 저장 동작을 LLM·검색 모델 호출 없이 점검한다."""

from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage
from langchain_core.documents import Document

from agents import report_writer


class _FakeLLM:
    def invoke(self, messages: list) -> AIMessage:
        self.payload = json.loads(messages[-1].content)
        return AIMessage(
            content=(
                "SUMMARY\n핵심 결과\n\n"
                "## 1. 분석 배경\n내용\n\n"
                "REFERENCE\n1. Example(2026). Source. https://example.test"
            )
        )


class ReportContractTest(unittest.TestCase):
    def test_prompt_declares_summary_first_and_reference_last(self) -> None:
        self.assertLess(report_writer.SYSTEM_PROMPT.index("SUMMARY"), report_writer.SYSTEM_PROMPT.index("REFERENCE"))
        self.assertIn("최종 승자", report_writer.SYSTEM_PROMPT)
        self.assertIn("정보 부족", report_writer.SYSTEM_PROMPT)

    def test_report_is_saved_as_markdown(self) -> None:
        llm = _FakeLLM()
        document = Document(
            page_content="검증용 근거 문장",
            metadata={"source": "fixture.pdf", "chunk_id": "fixture-p1-text-0"},
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            with (
                patch("agents.report_writer.get_llm", return_value=llm),
                patch("agents.report_writer.retrieve", return_value=[document]) as retrieve,
                patch.object(report_writer.settings, "outputs_dir", temporary_dir),
            ):
                result = report_writer.run(
                    {
                        "research_question": "test",
                        "selected_technologies": {},
                        "references": [],
                    }
                )

            reports = list(Path(temporary_dir).glob("report_*.md"))
            self.assertEqual(len(reports), 1)
            self.assertTrue(result["final_report"].startswith("SUMMARY"))
            self.assertIn("REFERENCE", result["final_report"])
            self.assertEqual(reports[0].read_text(encoding="utf-8"), result["final_report"])
            samples = llm.payload["retrieval_sample_results"]
            self.assertEqual(len(samples), len(report_writer.RETRIEVAL_REPORT_SAMPLES))
            self.assertEqual(retrieve.call_count, len(samples))
            for sample in samples:
                self.assertEqual(sample["top_results"][0]["chunk_id"], "fixture-p1-text-0")
                self.assertEqual(sample["top_results"][0]["excerpt"], "검증용 근거 문장")


if __name__ == "__main__":
    unittest.main()
