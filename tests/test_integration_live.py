"""실제 OpenAI·Tavily API를 쓰는 통합 테스트.

기본 테스트에는 비용이 발생하지 않도록 비활성화한다.
``RUN_LIVE_TESTS=1``일 때만 실행한다.
"""

from __future__ import annotations

import os
from pathlib import Path
import unittest

from dotenv import load_dotenv


@unittest.skipUnless(os.getenv("RUN_LIVE_TESTS") == "1", "RUN_LIVE_TESTS=1일 때만 실제 API를 호출합니다.")
class LiveIntegrationTest(unittest.TestCase):
    def test_full_workflow_writes_report_and_external_evidence(self) -> None:
        load_dotenv(override=True)
        from config import settings
        from graph.workflow import build_graph
        from rag.external_search import register

        self.assertTrue(settings.openai_api_key, "OPENAI_API_KEY가 필요합니다.")
        self.assertTrue(settings.tavily_api_key, "TAVILY_API_KEY가 필요합니다.")
        self.assertTrue(settings.chunks_file.exists(), "먼저 rag.ingest를 실행하세요.")
        self.assertTrue((settings.vectorstore_path / "index.faiss").exists(), "FAISS 색인이 필요합니다.")

        before = set(Path(settings.outputs_path).glob("report_*.md"))
        register()
        result = build_graph().invoke(
            {
                "research_question": (
                    "DeepSeek-V2 MLA와 InfiniGen을 시장성과 이해관계자 관점에서 근거 기반으로 비교해줘."
                )
            }
        )
        after = set(Path(settings.outputs_path).glob("report_*.md"))
        external = [
            item
            for item in result.get("evidence_items", [])
            if item.get("source_type") == "external_search"
        ]

        self.assertTrue(external, "실제 외부 검색 근거가 저장되어야 합니다.")
        self.assertTrue(all(item.get("source_url") for item in external))
        self.assertIn("SUMMARY", result["final_report"])
        self.assertIn("REFERENCE", result["final_report"])
        self.assertTrue(after - before, "outputs/에 Markdown 보고서가 생성되어야 합니다.")


if __name__ == "__main__":
    unittest.main()
