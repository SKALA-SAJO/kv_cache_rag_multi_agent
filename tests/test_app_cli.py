"""실제 API·PDF 생성 없이 app 진입점과 체크포인트 정리 로그를 검증한다."""

from contextlib import redirect_stderr, redirect_stdout
import importlib
from io import StringIO
from pathlib import Path
import re
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from config import settings
from graph import workflow
from tests.test_state_workflow import _fake_runners


class AppCliTest(unittest.TestCase):
    def _run_app(self, *, keep: bool) -> tuple[str, int, str]:
        # app은 import 시 .env를 읽으므로 개인 키·트레이싱 설정을 테스트에 유입시키지 않는다.
        with patch("dotenv.load_dotenv"), patch.dict("os.environ", {"LANGSMITH_TRACING": "false"}):
            app = importlib.import_module("app")
            with tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                db = directory / "checkpoints.sqlite"
                runners = _fake_runners([], fail_once=set())
                writer = runners["report_writer"]

                def write_report(state):
                    result = writer(state)  # 대역도 본문은 파일로 쓰고 report_path(URI)만 돌려준다
                    # 보고서 본문은 가이드에 맞는 목차를 제공한다. app 후처리에서 순서가
                    # 달라지는지를 검사해야 하므로 대역 자체의 형식 오류를 배제한다.
                    body = (
                        f"## SUMMARY\n{Path(result['report_path']).read_text(encoding='utf-8')}\n\n"
                        "## 1. 분석 배경\n대역 분석 결과\n\n"
                        "## REFERENCE\n- [R1] 대역 출처\n"
                    )
                    path = directory / "report_fake.md"
                    path.write_text(body, encoding="utf-8")
                    return {**result, "report_path": str(path)}

                runners["report_writer"] = write_report
                output = StringIO()
                argv = ["app.py", "--keep-checkpoints"] if keep else ["app.py"]
                with patch("sys.argv", argv), \
                        patch.object(settings, "outputs_dir", tmp), \
                        patch.object(settings, "vectorstore_dir", tmp), \
                        patch.object(app, "OUTPUTS_DIR", directory), \
                        patch.object(app, "CHECKPOINT_DB", db), \
                        patch.dict(workflow.AGENT_RUNNERS, runners), \
                        patch.object(app, "register_external_search"), \
                        patch.object(app, "record_run_feedback"), \
                        patch.object(app, "convert_report_to_pdf") as convert_pdf, \
                        patch.object(app, "PdfReader", return_value=SimpleNamespace(pages=[None])), \
                        redirect_stdout(output), redirect_stderr(output):
                    app.main()

                with sqlite3.connect(db) as connection:
                    count = connection.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
                final_paths = list(directory.glob("report_*_final.md"))
                self.assertEqual(len(final_paths), 1)
                convert_pdf.assert_called_once()
                self.assertEqual(convert_pdf.call_args.args[0], final_paths[0])
                final_report = final_paths[0].read_text(encoding="utf-8")
                return output.getvalue(), count, final_report

    def test_finished_app_prints_one_cleanup_line_and_keeps_final_checkpoint(self) -> None:
        output, count, _ = self._run_app(keep=False)
        lines = [line for line in output.splitlines() if line.startswith("[app] 체크포인트 정리:")]
        self.assertEqual(len(lines), 1, output)
        self.assertNotIn("체크포인트 정리 실패", output)
        self.assertEqual(count, 1)

    def test_keep_checkpoints_preserves_history_without_cleanup_log(self) -> None:
        output, count, _ = self._run_app(keep=True)
        self.assertNotIn("[app] 체크포인트 정리", output)
        self.assertGreater(count, 1)

    def test_submission_report_starts_with_summary_and_ends_with_reference(self) -> None:
        # 품질 평가 전 본문이 아니라 PDF 변환에 실제 전달되는 최종 Markdown을 검사한다.
        _, _, final_report = self._run_app(keep=False)
        chapters = re.findall(r"^#{1,2}\s+(.+?)\s*#*\s*$", final_report, flags=re.MULTILINE)
        self.assertTrue(chapters, "최종 제출본에 챕터 목차가 있어야 합니다.")
        self.assertEqual(chapters[0].upper(), "SUMMARY", "첫 챕터는 SUMMARY여야 합니다.")
        self.assertEqual(
            chapters[-1].upper(), "REFERENCE",
            "마지막 챕터는 REFERENCE여야 합니다. 결정 이력 부록은 REFERENCE 앞에 배치해야 합니다.",
        )


if __name__ == "__main__":
    unittest.main()
