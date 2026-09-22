"""Retriever의 doc_type·technology 범위 제한 규칙을 네트워크 없이 점검한다."""

from __future__ import annotations

import unittest

from rag.retriever import _make_filter, _normalize_doc_types


class RetrieverFilterTest(unittest.TestCase):
    def test_normalize_doc_types(self) -> None:
        self.assertIsNone(_normalize_doc_types(None))
        self.assertEqual(_normalize_doc_types("domain_benchmark"), ("domain_benchmark",))
        self.assertEqual(
            _normalize_doc_types(("technical_paper", "implementation_document")),
            ("technical_paper", "implementation_document"),
        )

    def test_filter_limits_document_group_and_technology(self) -> None:
        predicate = _make_filter(("technical_paper",), "DeepSeek-V2 MLA")
        assert predicate is not None
        self.assertTrue(predicate({"doc_type": "technical_paper", "technology": "DeepSeek-V2 MLA"}))
        self.assertFalse(predicate({"doc_type": "technical_paper", "technology": "InfiniGen"}))
        self.assertFalse(predicate({"doc_type": "domain_benchmark", "technology": "all"}))

    def test_domain_filter_accepts_shared_documents(self) -> None:
        predicate = _make_filter(("domain_benchmark",), None)
        assert predicate is not None
        self.assertTrue(predicate({"doc_type": "domain_benchmark", "technology": "all"}))


if __name__ == "__main__":
    unittest.main()
