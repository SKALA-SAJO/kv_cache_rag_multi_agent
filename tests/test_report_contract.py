"""보고서의 SUMMARY/REFERENCE 계약과 저장 동작을 LLM 호출 없이 점검한다."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage

from agents import report_writer


class _FakeLLM:
    def invoke(self, messages: list) -> AIMessage:
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
        with tempfile.TemporaryDirectory() as temporary_dir:
            with patch("agents.report_writer.get_llm", return_value=_FakeLLM()):
                with patch.object(report_writer.settings, "outputs_dir", temporary_dir):
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


if __name__ == "__main__":
    unittest.main()
