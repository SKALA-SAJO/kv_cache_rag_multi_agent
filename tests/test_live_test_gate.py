"""실제 API 호출 없이 라이브 테스트의 opt-in 실행 조건을 검증한다."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import Mock, patch


LIVE_TEST_PATH = Path(__file__).with_name("test_integration_live.py")
LIVE_TEST_METHOD = "test_full_workflow_writes_report_and_external_evidence"


class LiveTestGateTest(unittest.TestCase):
    def _assert_gate(self, value: str | None, *, enabled: bool) -> None:
        # skipUnless는 import 시 평가된다. discovery가 이미 읽은 모듈을 바꾸지 않고
        # 별도 모듈 객체로 로드하여 각 환경 조건과 기존 테스트를 서로 격리한다.
        with patch.dict(os.environ):
            if value is None:
                os.environ.pop("RUN_LIVE_TESTS", None)
            else:
                os.environ["RUN_LIVE_TESTS"] = value

            spec = importlib.util.spec_from_file_location("live_gate_fixture", LIVE_TEST_PATH)
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # 활성화 케이스도 원래 테스트 본문(API·색인·파일 작업)은 실행하지 않는다.
            body = Mock(return_value=None)
            with patch.object(module.LiveIntegrationTest, LIVE_TEST_METHOD, body):
                suite = unittest.defaultTestLoader.loadTestsFromTestCase(module.LiveIntegrationTest)
                result = unittest.TestResult()
                suite.run(result)

        self.assertEqual(result.testsRun, 1)
        self.assertTrue(result.wasSuccessful(), (result.errors, result.failures))
        self.assertEqual(len(result.skipped), 0 if enabled else 1)
        if enabled:
            body.assert_called_once_with()
        else:
            body.assert_not_called()

    def test_unset_environment_skips_live_test(self) -> None:
        self._assert_gate(None, enabled=False)

    def test_zero_environment_skips_live_test(self) -> None:
        self._assert_gate("0", enabled=False)

    def test_one_environment_enables_live_test(self) -> None:
        self._assert_gate("1", enabled=True)


if __name__ == "__main__":
    unittest.main()
