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
        expected = report_writer.ensure_trl_estimate_notice(self.good)  # 4.1 TRL 추정 고지만 추가
        self.assertEqual(report_writer.generate_report(llm, "{}", self.catalog), expected)
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


class BoldCitationTest(unittest.TestCase):
    def test_bold_citations_are_normalized_and_counted(self) -> None:
        # 실제 최종 실행 사례: 4.1~4.4의 인용 36개가 전부 [**R1**] 표기 → 규칙이 "관점 절 인용 없음"으로 오판
        from tests.test_quality_evaluation import GOOD_REPORT, REFS
        from agents.base import build_reference_catalog

        catalog = build_reference_catalog(REFS)
        bold = re.sub(r"\[(R\d+)\]", r"[**\1**]", GOOD_REPORT.split("## REFERENCE")[0]) + \
            "## REFERENCE" + GOOD_REPORT.split("## REFERENCE")[1]
        self.assertTrue(report_writer.citation_issues(bold, catalog))  # 정규화 전엔 미달
        cleaned = report_writer.clean_report(bold)
        self.assertNotIn("**", cleaned.split("## REFERENCE")[0].replace("**평가", ""))
        self.assertEqual(report_writer.citation_issues(cleaned, catalog), [])
        self.assertEqual(report_writer.clean_report("근거 **[R2]** 이다"), "근거 [R2] 이다")


class TrlEstimateNoticeTest(unittest.TestCase):
    REPORT = "## SUMMARY\n요약\n\n## 4. 다관점 평가\n### 4.1 기술 성숙도 관점\n두 기술 모두 TRL 6 [R1].\n\n### 4.2 시장성 관점\n내용\n"

    def test_notice_is_inserted_into_trl_section(self) -> None:
        # 과제 필수: TRL은 공개 정보 기반 추정임을 반드시 명시
        fixed = report_writer.ensure_trl_estimate_notice(self.REPORT)
        section = fixed.split("### 4.1")[1].split("### 4.2")[0]
        self.assertIn("공개 정보 기반 추정", section)
        self.assertEqual(report_writer.ensure_trl_estimate_notice(fixed), fixed)  # 중복 삽입 없음

    def test_existing_notice_or_missing_section_is_left_alone(self) -> None:
        stated = self.REPORT.replace("두 기술 모두", "공개 정보 기반 추정으로 두 기술 모두")
        self.assertEqual(report_writer.ensure_trl_estimate_notice(stated), stated)
        self.assertEqual(report_writer.ensure_trl_estimate_notice("## SUMMARY\n요약\n"), "## SUMMARY\n요약\n")

    def test_generated_report_always_carries_notice(self) -> None:
        with patch.object(report_writer, "log_event"):
            report = report_writer.generate_report(_ScriptedLLM(self.REPORT), "{}", [])
        self.assertIn("공개 정보 기반 추정", report)


if __name__ == "__main__":
    unittest.main()
