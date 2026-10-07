"""실제 API·PDF 생성 없이 app 진입점과 체크포인트 정리 로그를 검증한다."""

from contextlib import redirect_stderr, redirect_stdout
import importlib
from io import StringIO
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from config import settings
from graph import workflow
from tests.test_state_workflow import _fake_runners


class AppCliTest(unittest.TestCase):
    def _run_app(self, *, keep: bool) -> tuple[str, int]:
        # app은 import 시 .env를 읽으므로 개인 키·트레이싱 설정을 테스트에 유입시키지 않는다.
        with patch("dotenv.load_dotenv"), patch.dict("os.environ", {"LANGSMITH_TRACING": "false"}):
            app = importlib.import_module("app")
            with tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                db = directory / "checkpoints.sqlite"
                runners = _fake_runners([], fail_once=set())
                writer = runners["report_writer"]

                def write_report(state):
                    result = writer(state)
                    path = directory / "report_fake.md"
                    path.write_text(result["final_report"], encoding="utf-8")
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
                        patch.object(app, "convert_report_to_pdf"), \
                        patch.object(app, "PdfReader", return_value=SimpleNamespace(pages=[None])), \
                        redirect_stdout(output), redirect_stderr(output):
                    app.main()

                with sqlite3.connect(db) as connection:
                    count = connection.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
                self.assertTrue(list(directory.glob("report_*_final.md")))
                return output.getvalue(), count

    def test_finished_app_prints_one_cleanup_line_and_keeps_final_checkpoint(self) -> None:
        output, count = self._run_app(keep=False)
        lines = [line for line in output.splitlines() if line.startswith("[app] 체크포인트 정리:")]
        self.assertEqual(len(lines), 1, output)
        self.assertNotIn("체크포인트 정리 실패", output)
        self.assertEqual(count, 1)

    def test_keep_checkpoints_preserves_history_without_cleanup_log(self) -> None:
        output, count = self._run_app(keep=True)
        self.assertNotIn("[app] 체크포인트 정리", output)
        self.assertGreater(count, 1)


if __name__ == "__main__":
    unittest.main()
