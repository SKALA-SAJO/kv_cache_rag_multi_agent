"""외부 검색 tool-calling과 evidence/reference 변환을 API 호출 없이 점검한다."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage

import agents.base as base


class _FakeSearchTool:
    name = "external_search"

    def invoke(self, arguments: dict) -> list[dict]:
        if not arguments.get("query"):
            raise ValueError("query is required")
        return [
            {
                "title": "Official source",
                "url": "https://example.test/source",
                "content": "Verified public evidence.",
            }
        ]


class _FakeBoundLLM:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, messages: list) -> AIMessage:
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "external_search",
                        "args": {"query": "KV Cache public adoption"},
                        "id": "call-1",
                    }
                ],
            )
        return AIMessage(content="검색 결과가 충분합니다.")


class _FakeLLM:
    def __init__(self) -> None:
        self.bound = _FakeBoundLLM()

    def bind_tools(self, tools: list) -> _FakeBoundLLM:
        if tools[0].name != "external_search":
            raise ValueError("unexpected tool")
        return self.bound


class ExternalSearchTest(unittest.TestCase):
    def test_search_loop_collects_only_tool_results(self) -> None:
        with patch("agents.base.get_llm", return_value=_FakeLLM()):
            results = base.run_external_search_loop(
                "system prompt",
                "user prompt",
                _FakeSearchTool(),
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["url"], "https://example.test/source")

    def test_search_results_become_traceable_evidence_and_references(self) -> None:
        results = [
            {
                "title": "Official source",
                "url": "https://example.test/source",
                "content": "Verified public evidence.",
            }
        ]
        evidence = base.search_results_to_evidence_items(
            results,
            agent="market_evaluation",
            claim="시장성 평가 근거",
        )
        references = base.search_results_to_references(results)

        self.assertEqual(evidence[0]["source_type"], "external_search")
        self.assertEqual(evidence[0]["source_url"], "https://example.test/source")
        self.assertEqual(references[0]["url"], "https://example.test/source")


if __name__ == "__main__":
    unittest.main()
