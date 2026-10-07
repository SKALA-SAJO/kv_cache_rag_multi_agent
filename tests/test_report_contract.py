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
        self.assertIn("- Blog. https://x.test [R2]", fixed)
        self.assertEqual(fixed.count("[R1]"), 2)  # 이미 있는 항목은 중복 추가하지 않음

    def test_uncited_reference_entries_are_removed(self) -> None:
        catalog = [{"ref_id": f"R{i}", "source": f"s{i}", "url": None} for i in (1, 2, 3)]
        fixed = report_writer.complete_references(
            "본문 [R1][R3]\n\n## REFERENCE\n논문\n- A. [R1]\n- B(본문 미인용). [R2]\n- C. [R3]", catalog)
        self.assertNotIn("[R2]", fixed)
        self.assertIn("[R1]", fixed.split("## REFERENCE")[1])
        self.assertIn("[R3]", fixed.split("## REFERENCE")[1])
        self.assertIn("논문", fixed)  # ID 없는 분류 제목은 유지

    def test_title_is_prepended_once_and_llm_h1_is_replaced(self) -> None:
        state = {"selected_technologies": {"DeepSeek-V2 MLA": {"category": "SW"}, "InfiniGen": {"category": "HW·인프라"}}}
        titled = report_writer.add_title("# LLM이 쓴 제목\n\n## SUMMARY\n요약", state)
        self.assertTrue(titled.startswith(f"# {report_writer.REPORT_TITLE}\n"))
        self.assertEqual(titled.count("\n# "), 0)  # H1은 하나뿐
        self.assertNotIn("LLM이 쓴 제목", titled)
        self.assertIn("DeepSeek-V2 MLA(SW) · InfiniGen(HW·인프라)", titled)
        self.assertLess(titled.index(report_writer.REPORT_TITLE), titled.index("## SUMMARY"))

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
            # State에는 본문이 아니라 보고서 파일 URI만 남는다
            self.assertNotIn("final_report", result)
            self.assertEqual(Path(result["report_path"]), reports[0])
            body = reports[0].read_text(encoding="utf-8")
            self.assertTrue(body.startswith(f"# {report_writer.REPORT_TITLE}"))  # 제목이 맨 앞
            self.assertIn("SUMMARY", body)
            self.assertIn("REFERENCE", body)


    def test_rewrite_points_to_new_file_and_load_report_reads_latest(self) -> None:
        from agents.base import load_report

        state = {"research_question": "t", "selected_technologies": {}, "references": [], "run_id": "abcd1234-x"}
        with tempfile.TemporaryDirectory() as temporary_dir:
            with patch("agents.report_writer.get_llm", return_value=_FakeLLM()), \
                    patch("agents.report_writer.retrieve", return_value=[]), \
                    patch.object(report_writer.settings, "outputs_dir", temporary_dir):
                first = report_writer.run(state)["report_path"]
                second = report_writer.run(state)["report_path"]  # 같은 초·같은 rev로 다시 작성
            self.assertNotEqual(first, second)  # 이전 파일을 덮어쓰지 않고 새 파일을 가리킴
            self.assertTrue(Path(first).exists() and Path(second).exists())
            Path(first).write_text("OLD", encoding="utf-8")
            self.assertIn("SUMMARY", load_report({"report_path": second}))  # 최신 경로만 읽음
            self.assertNotEqual(load_report({"report_path": second}), "OLD")
        self.assertEqual(load_report({}), "")
        self.assertEqual(load_report({"report_path": "/nonexistent/report.md"}), "")

if __name__ == "__main__":
    unittest.main()
