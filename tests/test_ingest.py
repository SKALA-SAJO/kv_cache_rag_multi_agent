"""설계서 B.2/B.3의 코퍼스·청킹·메타데이터 계약을 점검한다."""

from __future__ import annotations

import unittest

from langchain_core.documents import Document

from rag.ingest import CHUNK_OVERLAP_TOKENS, CHUNK_SIZE_TOKENS, _infer_section, _split_documents
from scripts.download_papers import CORPUS_SOURCES


class IngestTest(unittest.TestCase):
    def test_corpus_plan_has_all_document_groups(self) -> None:
        doc_types = {source.doc_type for source in CORPUS_SOURCES}
        self.assertEqual(len(CORPUS_SOURCES), 7)
        self.assertEqual(
            doc_types,
            {
                "technical_paper",
                "implementation_document",
                "domain_benchmark",
                "market_document",
            },
        )

    def test_token_chunking_matches_design(self) -> None:
        self.assertEqual(CHUNK_SIZE_TOKENS, 500)
        self.assertEqual(CHUNK_OVERLAP_TOKENS, 50)

    def test_section_is_inferred_or_has_fallback(self) -> None:
        self.assertEqual(_infer_section("Introduction\n본문", "page 1"), "Introduction")
        self.assertEqual(_infer_section("본문만 있음", "page 1"), "page 1")

    def test_caption_is_separated_and_metadata_is_preserved(self) -> None:
        document = Document(
            page_content=(
                "본문 설명입니다.\n\n"
                "Table 1: KV Cache comparison\n표 설명입니다.\n\n"
                "Figure 2: Retrieval flow\n그림 설명입니다."
            ),
            metadata={
                "source": "sample.pdf",
                "source_url": "https://example.test/paper",
                "source_title": "Sample",
                "doc_type": "technical_paper",
                "technology": "DeepSeek-V2 MLA",
                "page": 1,
                "section": "Introduction",
            },
        )
        chunks = _split_documents([document])
        content_types = {chunk.metadata["content_type"] for chunk in chunks}
        self.assertEqual(content_types, {"text", "table_caption", "figure_caption"})
        for chunk in chunks:
            self.assertTrue(chunk.metadata["chunk_id"])
            self.assertEqual(chunk.metadata["doc_type"], "technical_paper")
            self.assertEqual(chunk.metadata["section"], "Introduction")
            self.assertEqual(chunk.metadata["source_url"], "https://example.test/paper")


if __name__ == "__main__":
    unittest.main()
