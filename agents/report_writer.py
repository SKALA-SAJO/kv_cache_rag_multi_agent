"""보고서 생성 Agent (PDF Agent K). RAG 여부: X. 종합 결과를 최종 Markdown 보고서로 구성한다."""

import json
import re
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage

from agents.base import build_reference_catalog, get_llm, load_prompt
from config import settings
from graph.observability import summarize_decisions
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
    "패턴 - Supervisor(hub-and-spoke). 모든 하위 Agent는 Supervisor로만 복귀하고 Agent 간 직접 "
    "엣지는 없다. Supervisor는 매 스텝 State(수집된 관점, 관점별 근거 충분도, 검증·품질 verdict, "
    "재작업 예산)를 읽어 결정론적 정책으로 next_nodes를 정하고, add_conditional_edges가 그 값으로 "
    "분기한다(실행 순서 하드코딩 없음). "
    "State 설계 - 작업 페이로드(technical_evidence, 4관점 평가, evidence_items/references, synthesis, "
    "faithfulness_check, report_path(보고서 URI), quality_verdict)와 제어 메타데이터(run_id, step_count/max_steps, "
    "next_nodes, last_decision, node_status, attempts, errors, rework_counts, retry_hints, sufficiency, "
    "report_revisions, quality_feedback)를 분리. 결정 로그 전문은 State 밖 JSONL과 LangSmith로 보내고 "
    "run_id로 연결. 병렬 디스패치로 동시에 쓰이는 node_status/errors/evidence_items/references는 리듀서로 병합. "
    "Graph 흐름 - init -> supervisor -> (tech_research | 4관점 Agent 병렬 | synthesis | faithfulness_check | "
    "report_writer | quality_evaluation) -> supervisor ... -> END. 관점별 근거 충분도 게이트(근거 수·출처 "
    "다양성·정보 부족 판정) 미달 관점, Faithfulness 실패 claim의 출처 Agent, 품질 평가에서 근거 부족으로 "
    "지목된 관점만 재작업하고, 서술 문제는 보고서 재작성으로 보낸다. 종료는 품질 평가 통과 또는 "
    "재작업·재작성·스텝 상한 소진 시."
)


_NON_REF_CITATION = re.compile(r"\s*\[(?!R\d)[A-Za-z_]+\]")
# REFERENCE 항목 앞머리의 ID 표기 변형: "- R1 ...", "- (R1) ...", "- R1. ...", "- R1: ..." → "- [R1] ..."
_REF_ID_PREFIX = re.compile(r"^(\s*[-*]\s*)\(?(R\d+)\)?[.:)]?\s+", flags=re.MULTILINE)
_REF_ANNOTATION = re.compile(r"\s*\[(?:원문|external_search|implementation_document|market_document|technical_paper|domain_benchmark)[^\]]*\]")


def clean_report(markdown: str) -> str:
    """프롬프트로 금지했지만 LLM이 종종 남기는 형식 위반을 결정론적으로 제거한다.

    - 본문의 `[R#]`가 아닌 가짜 인용 태그 (예: `[orchestration]`)
    - REFERENCE 항목 뒤의 파일명·doc_type 주석 (예: `[원문: infinigen.pdf]`)
    - REFERENCE 절의 `(참고) ...` 같은 메타 설명 문단
    - REFERENCE 항목 ID 표기 변형(`- R1 ...`)을 `- [R1] ...`로 정규화 (인용 검증 오탐 방지)
    """
    head, sep, tail = markdown.partition("## REFERENCE")
    head = _NON_REF_CITATION.sub("", head)
    if sep:
        lines = [
            _REF_ANNOTATION.sub("", line).rstrip()
            for line in tail.splitlines()
            if not line.strip().startswith(("(참고", "（참고", "※"))
        ]
        tail = _REF_ID_PREFIX.sub(r"\1[\2] ", "\n".join(lines)).rstrip() + "\n"
    return head + sep + tail


def complete_references(markdown: str, catalog: list[dict]) -> str:
    """본문에서 인용했지만 REFERENCE에 빠진 ID를 카탈로그로 채운다 (LLM 누락을 결정론적으로 보완).

    추가 항목은 카탈로그에 실제로 있는 출처(source/url)만 쓰므로 근거를 지어내지 않는다.
    """
    from agents.quality_evaluation import extract_citations

    head, sep, tail = markdown.partition("## REFERENCE")
    if not sep:
        return markdown
    by_id = {c["ref_id"]: c for c in catalog}
    listed = set(extract_citations(tail))
    missing = [i for i in dict.fromkeys(extract_citations(head)) if i in by_id and i not in listed]
    if not missing:
        return markdown
    lines = []
    for ref_id in sorted(missing, key=lambda x: int(x[1:])):
        c = by_id[ref_id]
        title = c.get("source") or c.get("url") or "출처"
        url = f", {c['url']}" if c.get("url") and c.get("url") != title else ""
        lines.append(f"- [{ref_id}] {title}{url}")
    return head + sep + tail.rstrip() + "\n\n기타 (본문 인용 보완)\n" + "\n".join(lines) + "\n"


def run(state: GraphState) -> dict:
    catalog = build_reference_catalog(state.get("references", []))
    decisions = summarize_decisions(state.get("run_id", ""))
    payload = {
        "agent_definitions": [
            {
                "node": "supervisor",
                "name": "Supervisor",
                "rag": False,
                "output": "next_nodes / last_decision (라우팅·재작업·종료 결정)",
            },
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
                "output": "report_path (보고서 파일 URI)",
            },
            {
                "node": "quality_evaluation",
                "name": "보고서 품질 평가 노드 (Hybrid: 규칙 + LLM Judge)",
                "rag": False,
                "output": "quality_verdict",
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
        "reference_catalog": catalog,
        "orchestration": {
            "supervisor_steps_so_far": state.get("step_count", 0),
            "decisions": [
                {"step": d.get("step"), "action": d.get("action"), "targets": d.get("targets")}
                for d in decisions
            ],
            "rework_counts": state.get("rework_counts", {}),
            "sufficiency": state.get("sufficiency", {}),
            "excluded_agents": [n for n, st in state.get("node_status", {}).items() if st == "excluded"],
            "report_revision": state.get("report_revisions", 0),
        },
        "quality_feedback": state.get("quality_feedback", ""),
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
    report_markdown = complete_references(clean_report(response.content), catalog)

    settings.outputs_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    revision = state.get("report_revisions", 0)
    # 재작성·재작업마다 새 파일을 가리키도록 run_id 접두어를 붙인다(같은 초·같은 rev여도 run별로 구분).
    run_tag = (state.get("run_id") or "run")[:8]
    report_path = settings.outputs_path / f"report_{timestamp}_{run_tag}_rev{revision}.md"
    suffix = 1
    while report_path.exists():  # 같은 초에 다시 쓰는 경우에도 이전 파일을 덮어쓰지 않는다
        report_path = settings.outputs_path / f"report_{timestamp}_{run_tag}_rev{revision}_{suffix}.md"
        suffix += 1
    report_path.write_text(report_markdown, encoding="utf-8")
    print(f"[report_writer] 보고서 저장: {report_path}")

    # 본문은 파일로만 남기고 State에는 URI(경로)만 둔다.
    return {"report_path": str(report_path)}
