"""검색 평가용 Gold 질문셋의 형식을 검증한다."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATHS = {
    "smoke": ROOT / "data" / "eval" / "smoke_questions.json",
    "heldout": ROOT / "data" / "eval" / "heldout_questions.json",
}
REQUIRED_KEYS = {
    "id",
    "question",
    "expected_chunk_id",
    "expected_source",
    "doc_types",
    "technology",
}


class EvaluationDatasetTests(unittest.TestCase):
    def test_question_datasets_have_required_retrieval_fields(self) -> None:
        for dataset, path in DATASET_PATHS.items():
            with self.subTest(dataset=dataset):
                records = json.loads(path.read_text(encoding="utf-8"))
                minimum_size = 5 if dataset == "smoke" else 10

                self.assertGreaterEqual(len(records), minimum_size)
                self.assertEqual(len({record["id"] for record in records}), len(records))

                for record in records:
                    self.assertEqual(set(record), REQUIRED_KEYS)
                    self.assertTrue(record["question"].strip())
                    self.assertTrue(record["expected_chunk_id"].strip())
                    self.assertTrue(record["expected_source"].strip())
                    self.assertTrue(record["doc_types"])


if __name__ == "__main__":
    unittest.main()
