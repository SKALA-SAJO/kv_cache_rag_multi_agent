"""품질 평가 노드의 규칙 기반 판정과 Hybrid 결합을 LLM 호출 없이 점검한다."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from agents import quality_evaluation as qe
from agents.base import build_reference_catalog
from agents.schemas import CriterionJudgement, QualityJudgement

REFS = [
    {"source": "deepseek_v2_mla.pdf", "page": 1},
    {"source": "deepseek_v2_mla.pdf", "page": 7},
    {"source": "infinigen.pdf", "page": 3},
    {"source": "longbench.pdf", "page": 2},
    {"source": "Blog", "url": "https://example.test/a", "source_type": "external_search"},
]
FILLER = "근거 기반 서술 문장이다. " * 12

GOOD_REPORT = f"""## SUMMARY
두 기술은 접근 층위가 다르다 [R1][R2].

## 4. 다관점 평가
### 4.1 기술 성숙도 관점
MLA는 KV Cache를 줄인다 [R1]. InfiniGen은 프리페치한다 [R2]. {FILLER}
### 4.2 시장성 관점
외부 사례가 있다 [R4]. {FILLER}
### 4.3 이해관계자 관점
운영자 관점 우려가 있다 [R3][R4]. {FILLER}
### 4.4 장문맥 처리 애플리케이션 관점
LongBench 기준 평가가 필요하다 [R3]. 특정 기술을 승자로 선정하지 않는다. {FILLER}

## REFERENCE
- [R1] DeepSeek-AI (2024). DeepSeek-V2.
- [R2] Lee (2024). InfiniGen.
- [R3] Bai (2024). LongBench.
- [R4] Blog. https://example.test/a
"""


class QualityRuleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = build_reference_catalog(REFS)

    def test_catalog_groups_pages_and_assigns_stable_ids(self) -> None:
        self.assertEqual([c["ref_id"] for c in self.catalog], ["R1", "R2", "R3", "R4"])
        self.assertEqual(self.catalog[0]["pages"], [1, 7])

    def test_good_report_passes_rules(self) -> None:
        self.assertTrue(qe.rule_groundedness(GOOD_REPORT, self.catalog)["passed"])
        self.assertTrue(qe.rule_neutrality(GOOD_REPORT)["passed"])  # 부정문 속 '승자'는 허용
        self.assertTrue(qe.rule_coverage(GOOD_REPORT)["passed"])
        evidence = [{"document_id": f"d{i % 4}"} for i in range(8)]
        self.assertTrue(qe.rule_bias_control(GOOD_REPORT, self.catalog, evidence)["passed"])

    def test_detects_unknown_citation_and_missing_section(self) -> None:
        bad = GOOD_REPORT.replace("[R3]. 특정", "[R9]. 특정").replace("### 4.2 시장성 관점", "### 4.2 기타")
        self.assertFalse(qe.rule_groundedness(bad, self.catalog)["passed"])
        self.assertFalse(qe.rule_coverage(bad)["passed"])

    def test_detects_superiority_claim_and_single_source(self) -> None:
        biased = GOOD_REPORT.replace("MLA는 KV Cache를 줄인다", "MLA가 InfiniGen보다 더 우수하다")
        self.assertFalse(qe.rule_neutrality(biased)["passed"])
        evidence = [{"document_id": "only.pdf"}] * 9 + [{"document_id": "other.pdf"}]
        self.assertFalse(qe.rule_bias_control(GOOD_REPORT, self.catalog, evidence)["passed"])

    def test_hybrid_requires_both_rule_and_judge(self) -> None:
        rules = {name: {"passed": True, "issues": []} for name in qe.CRITERIA + ["format"]}
        judgement = QualityJudgement(
            criteria=[
                CriterionJudgement(criterion=c, passed=(c != "bias_control"), score=4 if c != "bias_control" else 2,
                                   issues=[] if c != "bias_control" else ["시장성이 한 출처에 의존"],
                                   rework_targets=[] if c != "bias_control" else ["market_evaluation"])
                for c in qe.CRITERIA
            ],
            feedback="시장성 근거 보강",
        )
        verdict = qe.combine(rules, judgement)
        self.assertFalse(verdict["passed"])
        self.assertEqual(verdict["failed_criteria"], ["bias_control"])
        self.assertEqual(verdict["rework_targets"], ["market_evaluation"])

    def test_page_limit_triggers_format_failure(self) -> None:
        with patch.object(qe, "count_pdf_pages", return_value=14):
            verdict = qe.rule_format(GOOD_REPORT)
        self.assertFalse(verdict["passed"])
        self.assertIn("14쪽", verdict["issues"][0])


if __name__ == "__main__":
    unittest.main()
