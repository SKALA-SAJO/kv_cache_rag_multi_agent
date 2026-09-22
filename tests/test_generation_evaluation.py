"""Generation 평가 데이터와 비용 없는 집계 로직을 검증한다."""

from __future__ import annotations

import unittest

from tests.evaluate_generation import (
    CASES_PATH,
    GenerationCase,
    aggregate_metrics,
    faithfulness_metrics,
    load_cases,
)


class GenerationEvaluationTests(unittest.TestCase):
    def test_generation_cases_are_valid_and_cover_multiple_questions(self) -> None:
        cases = load_cases()

        self.assertGreaterEqual(len(cases), 3)
        self.assertEqual(len({case.id for case in cases}), len(cases))
        self.assertTrue(all(case.evaluation_scope == "final_report" for case in cases))
        self.assertTrue(all(case.required_aspects for case in cases))
        self.assertTrue(CASES_PATH.exists())

    def test_faithfulness_metrics_use_claim_check_statuses(self) -> None:
        metrics = faithfulness_metrics(
            {"claim_checks": [{"status": "pass"}, {"status": "fail"}, {"status": "pass"}]}
        )

        self.assertEqual(metrics["claim_count"], 3)
        self.assertEqual(metrics["passed_claim_count"], 2)
        self.assertEqual(metrics["failed_claim_count"], 1)
        self.assertAlmostEqual(metrics["faithfulness_rate"], 2 / 3)

    def test_aggregate_metrics_computes_case_means(self) -> None:
        results = [
            {
                "faithfulness": {"faithfulness_rate": 1.0},
                "answer_relevance": {"score": 4},
                "retry_count": 0,
            },
            {
                "faithfulness": {"faithfulness_rate": 0.5},
                "answer_relevance": {"score": 2},
                "retry_count": 2,
            },
        ]

        metrics = aggregate_metrics(results)
        self.assertEqual(metrics["case_count"], 2)
        self.assertAlmostEqual(metrics["mean_faithfulness_rate"], 0.75)
        self.assertAlmostEqual(metrics["mean_answer_relevance"], 3.0)
        self.assertAlmostEqual(metrics["mean_retry_count"], 1.0)

    def test_generation_case_rejects_empty_required_aspects(self) -> None:
        with self.assertRaises(ValueError):
            GenerationCase.model_validate(
                {
                    "id": "invalid",
                    "question": "질문",
                    "required_aspects": [],
                    "evaluation_scope": "final_report",
                }
            )


if __name__ == "__main__":
    unittest.main()
