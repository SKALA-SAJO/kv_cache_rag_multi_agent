"""검색 평가용 Gold 질문셋의 형식을 검증한다."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = ROOT / "data" / "eval" / "golden_questions.json"
REQUIRED_KEYS = {
    "id",
    "question",
    "expected_chunk_id",
    "expected_source",
    "doc_types",
    "technology",
}


class EvaluationDatasetTests(unittest.TestCase):
    def test_golden_questions_have_required_retrieval_fields(self) -> None:
        records = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))

        self.assertGreaterEqual(len(records), 5)
        self.assertEqual(len({record["id"] for record in records}), len(records))

        for record in records:
            self.assertEqual(set(record), REQUIRED_KEYS)
            self.assertTrue(record["question"].strip())
            self.assertTrue(record["expected_chunk_id"].strip())
            self.assertTrue(record["expected_source"].strip())
            self.assertTrue(record["doc_types"])


if __name__ == "__main__":
    unittest.main()
