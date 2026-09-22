"""보고서 생성 Agent (PDF Agent K). RAG 여부: X. 종합 결과를 최종 Markdown 보고서로 구성한다."""

import json
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage

from agents.base import get_llm, load_prompt
from config import settings
from graph.state import GraphState
from rag.retriever import retrieve

SYSTEM_PROMPT = load_prompt("report_writer")

RETRIEVAL_REPORT_SAMPLES = (
    {
        "question": "MLA는 Key와 Value를 어떤 방식으로 압축해 KV Cache 저장량을 줄이는가?",
        "doc_types": ("technical_paper",),
        "technology": "DeepSeek-V2 MLA",
        "purpose": "한국어 의미 질의가 DeepSeek-V2 MLA 원문의 핵심 구조 설명을 찾는지 확인",
    },
    {
        "question": "InfiniGen dynamic KV cache host memory prefetch",
        "doc_types": ("technical_paper",),
        "technology": "InfiniGen",
        "purpose": "논문 고유 영문 용어가 InfiniGen의 KV Cache 관리·prefetch 설명으로 연결되는지 확인",
    },
    {
        "question": "LongBench의 bilingual multi-task long context benchmark는 무엇을 평가하는가?",
        "doc_types": ("domain_benchmark",),
        "technology": None,
        "purpose": "한국어·영문이 섞인 장문맥 벤치마크 질의에서 관련 원문을 찾는지 확인",
    },
)


def _sample_retrieval_results() -> list[dict]:
    """보고서 작성 시점의 실제 Top-3 검색 결과를 JSON-직렬화 가능한 형태로 만든다."""
    results = []
    for sample in RETRIEVAL_REPORT_SAMPLES:
        documents = retrieve(
            sample["question"],
            top_k=3,
            doc_types=sample["doc_types"],
            technology=sample["technology"],
        )
        results.append(
            {
                "query": sample["question"],
                "purpose": sample["purpose"],
                "doc_types": sample["doc_types"],
                "technology": sample["technology"],
                "top_results": [
                    {
                        "rank": rank,
                        "source": document.metadata.get("source")
                        or document.metadata.get("source_file"),
                        "chunk_id": document.metadata.get("chunk_id"),
                        "excerpt": " ".join(document.page_content.split())[:240],
                    }
                    for rank, document in enumerate(documents, start=1)
                ],
            }
        )
    return results


def run(state: GraphState) -> dict:
    payload = {
        "research_question": state.get("research_question"),
        "selected_technologies": state.get("selected_technologies"),
        "technical_evidence": state.get("technical_evidence"),
        "trl_evaluation": state.get("trl_evaluation"),
        "market_evaluation": state.get("market_evaluation"),
        "stakeholder_evaluation": state.get("stakeholder_evaluation"),
        "domain_evaluation": state.get("domain_evaluation"),
        "synthesis": state.get("synthesis"),
        "faithfulness_check": state.get("faithfulness_check"),
        "references": state.get("references"),
        "retry_count": state.get("retry_count", 0),
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
        "retrieval_sample_results": _sample_retrieval_results(),
    }
    user_content = json.dumps(payload, ensure_ascii=False, indent=2)

    llm = get_llm("generator")
    response = llm.invoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_content)]
    )
    report_markdown = response.content

    settings.outputs_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = settings.outputs_path / f"report_{timestamp}.md"
    report_path.write_text(report_markdown, encoding="utf-8")
    print(f"[report_writer] 보고서 저장: {report_path}")

    return {"final_report": report_markdown}
