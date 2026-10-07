"""공개 환경 설정 예시의 이름·기본값과 기본 데이터 경로를 검증한다."""

import unittest
from unittest.mock import patch

from dotenv import dotenv_values

from config import BASE_DIR, Settings


class EnvironmentContractTest(unittest.TestCase):
    def test_example_settings_names_and_values_match_config(self) -> None:
        example = BASE_DIR / ".env.example"
        values = dotenv_values(example)
        external_keys = {"LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT", "LANGSMITH_ENDPOINT"}
        for key in values:
            if key not in external_keys:
                self.assertIn(key.lower(), Settings.model_fields, key)
        with patch.dict("os.environ", {}, clear=True):
            configured = Settings(_env_file=example)
            defaults = Settings(_env_file=None)
        for key in values:
            if key not in external_keys:
                self.assertEqual(getattr(configured, key.lower()), getattr(defaults, key.lower()), key)

    def test_default_paths_match_download_and_report_locations(self) -> None:
        from scripts.download_papers import RAW_DIR
        from scripts.report_to_pdf import OUTPUTS_DIR

        with patch.dict("os.environ", {}, clear=True):
            settings = Settings(_env_file=BASE_DIR / ".env.example")
        self.assertEqual(settings.raw_data_path, RAW_DIR)
        self.assertEqual(settings.outputs_path, OUTPUTS_DIR)
        self.assertEqual(settings.vectorstore_path, BASE_DIR / "vectorstore")
        self.assertEqual(settings.chunks_file, BASE_DIR / "data/processed/chunks.jsonl")


if __name__ == "__main__":
    unittest.main()
