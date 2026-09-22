"""설계서 D절의 State·Graph 계약을 외부 API 없이 점검한다."""

from __future__ import annotations

import unittest

from langchain_core.documents import Document

from graph.state import GraphState, dedupe_documents, dedupe_evidence_items, dedupe_references
from graph.workflow import build_graph, init_node, route_after_faithfulness


class StateAndWorkflowTest(unittest.TestCase):
    def test_state_has_required_design_keys(self) -> None:
        required = {
            "research_question",
            "selected_technologies",
            "evaluation_rubric",
            "max_retries",
            "retrieved_documents",
            "technical_evidence",
            "trl_evaluation",
            "market_evaluation",
            "stakeholder_evaluation",
            "domain_evaluation",
            "references",
            "evidence_items",
            "synthesis",
            "faithfulness_check",
            "final_report",
            "retry_count",
            "retry_hints",
        }
        self.assertTrue(required <= set(GraphState.__annotations__))

    def test_reducers_remove_duplicate_evidence(self) -> None:
        document = Document("same", metadata={"chunk_id": "chunk-1"})
        self.assertEqual(len(dedupe_documents([document], [document])), 1)

        reference = {"source": "paper.pdf", "page": 3}
        self.assertEqual(len(dedupe_references([reference], [reference])), 1)

        evidence = {
            "document_id": "paper.pdf",
            "page_or_section": "3",
            "evidence_quote": "same evidence",
        }
        self.assertEqual(len(dedupe_evidence_items([evidence], [evidence])), 1)

    def test_init_node_loads_human_selection_and_rubric(self) -> None:
        initial = init_node({})
        self.assertIn("DeepSeek-V2 MLA", initial["selected_technologies"])
        self.assertIn("InfiniGen", initial["selected_technologies"])
        self.assertEqual(initial["retry_count"], 0)
        self.assertGreaterEqual(initial["max_retries"], 0)
        self.assertIn("trl", initial["evaluation_rubric"])

    def test_faithfulness_routing(self) -> None:
        self.assertEqual(
            route_after_faithfulness({"faithfulness_check": {"passed": True}}),
            "report_writer",
        )
        self.assertEqual(
            route_after_faithfulness(
                {
                    "faithfulness_check": {"passed": False},
                    "retry_count": 2,
                    "max_retries": 2,
                }
            ),
            "report_writer",
        )

        routes = route_after_faithfulness(
            {
                "faithfulness_check": {
                    "passed": False,
                    "agents_to_retry": ["market_evaluation"],
                },
                "retry_count": 0,
                "max_retries": 2,
            }
        )
        self.assertEqual([route.node for route in routes], ["market_evaluation"])

    def test_graph_compiles(self) -> None:
        graph = build_graph()
        self.assertIsNotNone(graph)


if __name__ == "__main__":
    unittest.main()
