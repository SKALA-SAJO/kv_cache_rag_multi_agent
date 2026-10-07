"""테스트 격리(LangSmith 전송 차단·산출물 임시 경로)가 다른 테스트보다 먼저 적용됐는지 점검한다.

파일명이 `test__`로 시작해 discovery 정렬에서 가장 먼저 로드된다. 이 import가 tests 패키지
(`tests/__init__.py`)를 실행해, 이후 로드되는 테스트 모듈이 app/config를 import하기 전에 격리를 건다.
"""

from __future__ import annotations

import unittest

import tests  # noqa: F401  격리 적용 (tests/__init__.py)


class TestIsolationTest(unittest.TestCase):
    def test_app_import_does_not_reenable_langsmith(self) -> None:
        # app.py는 import 시 .env(LANGSMITH_TRACING=true)를 override로 다시 읽는다
        import app  # noqa: F401
        from graph import observability

        self.assertFalse(observability.tracing_is_enabled())

    def test_outputs_go_to_temporary_folder(self) -> None:
        from config import BASE_DIR, settings

        self.assertNotEqual(settings.outputs_path.resolve(), (BASE_DIR / "outputs").resolve())


if __name__ == "__main__":
    unittest.main()
