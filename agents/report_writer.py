"""보고서 생성 Agent (PDF Agent K). RAG 여부: X. 종합 결과를 최종 Markdown 보고서로 구성한다."""

import json
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage

from agents.base import get_llm, load_prompt
from config import settings
from graph.state import GraphState
from scripts.download_papers import CORPUS_SOURCES

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


# 3.3/3.4/3.5절은 실행마다 바뀌는 평가 결과가 아니라 고정된 시스템 설계 사실이라(sample.pdf
# B.4/D.1/D.2절), payload에 하드코딩된 설명을 함께 넘긴다 — LLM이 실행 메타데이터에 없다는
# 이유로 이 절을 얼버무리지 않도록 하기 위함. config.py/graph/state.py/graph/workflow.py가
# 바뀌면 이 텍스트도 함께 갱신해야 한다.
EMBEDDING_CANDIDATES_NOTE = (
    "후보 비교 - BAAI/bge-m3: 한국어·영어 지원, 긴 입력 처리, Dense·Sparse 검색 지원 "
    "(한계: 상대적으로 무겁고 Hybrid 검색 구현이 복잡함) / intfloat/multilingual-e5-base: "
    "비교적 가볍고 구현 단순 (한계: 전문 기술 용어·긴 문서 검색 성능 별도 검증 필요) / "
    "all-MiniLM-L6-v2: 빠르고 가벼움 (한계: 한국어·영문 기술 논문 혼합 검색에 불리). "
    "최종 선정: BAAI/bge-m3 — 영문 기술 논문과 한국어 질의를 함께 처리해야 하고, "
    "최대 입력 길이 8,192 토큰·Dense 임베딩 차원 1024를 지원하기 때문."
)
GRAPH_DESIGN_NOTE = (
    "State 설계 - research_question/selected_technologies/evaluation_rubric(초기화 Node가 "
    "주입) -> retrieved_documents/technical_evidence(기술 조사 Agent) -> "
    "trl/market/stakeholder/domain_evaluation(4관점 평가, 병렬 실행) -> "
    "evidence_items/references(전체 Agent가 누적, source_url/document_id 기준 중복 제거) -> "
    "synthesis(관점 간 agreements/conflicts/favorable_conditions) -> "
    "faithfulness_check(claim-evidence 대조, pass/fail) -> retry_count/retry_hints(재검색 제어) "
    "-> final_report. "
    "Graph 흐름 설계 - init -> tech_research -> [trl_evaluation/market_evaluation/"
    "stakeholder_evaluation/domain_evaluation 병렬 팬아웃] -> synthesis -> faithfulness_check -> "
    "(검증 실패 시 LangGraph Send API로 근거 부족 claim의 출처 Agent만 표적 재실행, 무한루프 "
    "방지를 위해 최대 retry_count회) -> report_writer."
)


def run(state: GraphState) -> dict:
    payload = {
        "agent_definitions": [
            {
                "node": "tech_research",
                "name": "기술 조사 Agent",
                "rag": True,
                "output": "technical_evidence",
            },
            {
                "node": "trl_evaluation",
                "name": "기술 성숙도 평가 Agent",
                "rag": True,
                "output": "trl_evaluation",
            },
            {
                "node": "market_evaluation",
                "name": "시장 평가 Agent",
                "rag": True,
                "output": "market_evaluation",
            },
            {
                "node": "stakeholder_evaluation",
                "name": "이해관계자 평가 Agent",
                "rag": False,
                "output": "stakeholder_evaluation",
            },
            {
                "node": "domain_evaluation",
                "name": "도메인 평가 Agent",
                "rag": True,
                "output": "domain_evaluation",
            },
            {
                "node": "synthesis",
                "name": "평가 종합 Agent",
                "rag": False,
                "output": "synthesis",
            },
            {
                "node": "faithfulness_check",
                "name": "검증 Agent (Faithfulness Check)",
                "rag": False,
                "output": "faithfulness_check",
            },
            {
                "node": "report_writer",
                "name": "보고서 생성 Agent",
                "rag": False,
                "output": "final_report",
            },
        ],
        "run_config": {
            "generator_model": settings.generator_model,
            "judge_model": settings.judge_model,
            "embedding_model": settings.embedding_model,
            "reranker_model": settings.reranker_model,
            "embedding_device": settings.embedding_device,
            "retrieval_top_k_candidates": settings.retrieval_top_k_candidates,
            "retrieval_top_k_final": settings.retrieval_top_k_final,
        },
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
        "retrieval_sample_results": _sample_retrieval_results(),
        "system_design": {
            "embedding_model": settings.embedding_model,
            "reranker_model": settings.reranker_model,
            "retrieval_top_k_candidates": settings.retrieval_top_k_candidates,
            "retrieval_top_k_final": settings.retrieval_top_k_final,
            "corpus_sources": [
                {"title": s.title, "doc_type": s.doc_type, "technology": s.technology}
                for s in CORPUS_SOURCES
            ],
            "embedding_candidates_note": EMBEDDING_CANDIDATES_NOTE,
            "graph_design_note": GRAPH_DESIGN_NOTE,
        },
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
