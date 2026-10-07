"""보고서의 SUMMARY/REFERENCE 계약과 저장 동작을 LLM 호출 없이 점검한다."""

from __future__ import annotations

from pathlib import Path
import re
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

    def test_clean_report_strips_fake_citations_and_reference_notes(self) -> None:
        cleaned = report_writer.clean_report(
            "본문 [R1] 근거 [orchestration].\n\n## REFERENCE\n- [R1] A (2024). B. [원문: a.pdf]\n\n(참고) 메타"
        )
        self.assertIn("[R1]", cleaned)
        self.assertNotIn("[orchestration]", cleaned)
        self.assertNotIn("원문:", cleaned)
        self.assertNotIn("(참고)", cleaned)

    def test_clean_report_normalizes_reference_id_prefix(self) -> None:
        cleaned = report_writer.clean_report("본문 [R1][R3]\n\n## REFERENCE\n- R1 A (2024).\n- (R3) B (2024).")
        self.assertIn("- [R1] A (2024).", cleaned)
        self.assertIn("- [R3] B (2024).", cleaned)

    def test_complete_references_fills_cited_ids_missing_from_reference(self) -> None:
        catalog = [{"ref_id": "R1", "source": "a.pdf", "url": None}, {"ref_id": "R2", "source": "Blog", "url": "https://x.test"}]
        fixed = report_writer.complete_references("본문 [R1][R2]\n\n## REFERENCE\n- [R1] A (2024).", catalog)
        self.assertIn("[R2] Blog, https://x.test", fixed)
        self.assertEqual(fixed.count("[R1]"), 2)  # 이미 있는 항목은 중복 추가하지 않음

    def test_report_is_saved_as_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            with patch("agents.report_writer.get_llm", return_value=_FakeLLM()), \
                    patch("agents.report_writer.retrieve", return_value=[]):
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


class _ScriptedLLM:
    """호출 순서대로 미리 정한 응답을 돌려주는 대역 (보정 요청 여부를 호출 수로 확인)."""

    def __init__(self, *contents: str) -> None:
        self.contents = list(contents)
        self.calls: list[list] = []

    def invoke(self, messages: list) -> AIMessage:
        self.calls.append(messages)
        return AIMessage(content=self.contents[len(self.calls) - 1])


class CitationSelfCheckTest(unittest.TestCase):
    def setUp(self) -> None:
        from tests.test_quality_evaluation import GOOD_REPORT, REFS
        from agents.base import build_reference_catalog

        self.good = GOOD_REPORT
        self.catalog = build_reference_catalog(REFS)
        # 재작성 중 인용이 빠진 보고서 (실제 사례: 관점 절 인용 0개, 본문 인용 24개 → 4개)
        self.dropped = re.sub(r"\[R\d+\]", "", GOOD_REPORT.split("## REFERENCE")[0]) + "## REFERENCE\n"
        self.log = patch.object(report_writer, "log_event").start()
        self.addCleanup(patch.stopall)

    def test_detects_dropped_citations(self) -> None:
        self.assertEqual(report_writer.citation_issues(self.good, self.catalog), [])
        issues = report_writer.citation_issues(self.dropped, self.catalog)
        self.assertTrue(any("관점 절에 인용 없음" in i for i in issues))

    def test_good_draft_is_used_without_repair(self) -> None:
        llm = _ScriptedLLM(self.good)
        self.assertEqual(report_writer.generate_report(llm, "{}", self.catalog), self.good)
        self.assertEqual(len(llm.calls), 1)

    def test_dropped_citations_are_repaired_once(self) -> None:
        llm = _ScriptedLLM(self.dropped, self.good)
        report = report_writer.generate_report(llm, "{}", self.catalog, run_id="r")
        self.assertEqual(report_writer.citation_issues(report, self.catalog), [])
        self.assertEqual(len(llm.calls), 2)
        self.assertIn("관점 절에 인용 없음", llm.calls[1][-1].content)  # 보정 요청에 미달 사항 전달
        self.assertTrue(self.log.call_args.kwargs["adopted"])

    def test_worse_repair_keeps_original_draft(self) -> None:
        partial = self.good.replace("[R4]", "")  # 일부만 빠진 초안
        llm = _ScriptedLLM(partial, self.dropped)
        report = report_writer.generate_report(llm, "{}", self.catalog)
        self.assertNotIn("[R4]", report.split("## REFERENCE")[0])
        self.assertIn("[R1]", report)  # 더 나빠진 보정본 대신 초안 유지
        self.assertFalse(self.log.call_args.kwargs["adopted"])


if __name__ == "__main__":
    unittest.main()
