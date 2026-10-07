"""보고서 생성 직후 인용 자체 점검·1회 보정(report_writer.generate_report)을 LLM 호출 없이 점검한다."""

from __future__ import annotations

import re
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage

from agents import report_writer


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


class ReferenceSuffixIdTest(unittest.TestCase):
    def test_trailing_reference_id_moves_to_front(self) -> None:
        # 실제 사례: "- DeepSeek-AI (2024). DeepSeek-V2 ... arXiv:2405.04434. [R1]"
        cleaned = report_writer.clean_report(
            "본문 [R1][R2]\n\n## REFERENCE\n- DeepSeek-AI (2024). DeepSeek-V2. arXiv:2405.04434. [R1]\n"
            "- [R2] Lee (2024). InfiniGen.\n"
        )
        self.assertIn("- [R1] DeepSeek-AI (2024). DeepSeek-V2. arXiv:2405.04434.", cleaned)
        self.assertIn("- [R2] Lee (2024). InfiniGen.", cleaned)  # 이미 앞에 있으면 그대로
        self.assertEqual(cleaned.count("[R1]"), 2)


if __name__ == "__main__":
    unittest.main()
