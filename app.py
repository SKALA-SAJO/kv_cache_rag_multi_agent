"""KV Cache 최적화 기술 다관점 평가 — Supervisor 패턴 Multi-Agent 실행 스크립트.

사전 준비:
    python -m scripts.download_papers   # RAG 코퍼스 다운로드
    python -m rag.ingest                # 색인 구축

실행:
    python app.py                         # 새 실행 (run_id 자동 발급)
    python app.py --question "다른 평가 질문"
    python app.py --resume <run_id>       # 중단된 실행을 마지막 체크포인트부터 재개
    python app.py --keep-checkpoints      # 정상 종료 후에도 중간 체크포인트를 지우지 않음 (기본은 정리)
"""

import argparse
import os
import sys
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402
from pypdf import PdfReader  # noqa: E402

from config import settings  # noqa: E402
from graph.observability import (  # noqa: E402
    count_reworks,
    print_timing_summary,
    record_run_feedback,
    render_decision_appendix,
    summarize_routing,
    trace_path,
)
from graph.checkpoint_maintenance import is_run_finished, prune_checkpoints  # noqa: E402
from graph.workflow import build_graph  # noqa: E402
from rag.external_search import register as register_external_search  # noqa: E402
from scripts.report_to_pdf import DEFAULT_CAMPUS, DEFAULT_CLASS, DEFAULT_TEAM_NAMES, OUTPUTS_DIR  # noqa: E402
from scripts.report_to_pdf import convert as convert_report_to_pdf  # noqa: E402

DEFAULT_QUESTION = (
    "장문맥 처리 애플리케이션 관점에서 DeepSeek-V2 MLA와 InfiniGen을 기술 성숙도, 시장성, "
    "이해관계자, 도메인 적합성 4가지 관점에서 비교 평가하라."
)
CHECKPOINT_DB = OUTPUTS_DIR / "checkpoints.sqlite"


def _print_orchestration_summary(result: dict) -> None:
    run_id = result.get("run_id", "")
    decisions = summarize_routing(run_id)
    print("\n=== Supervisor 결정 이력 ===")
    for d in decisions:
        print(f"  step {d['step']:>2} | {d['action']:<22} -> {d['targets'] or 'END'}")
    print(f"  라우팅 {len(decisions)}회, 재작업/재작성 {count_reworks(decisions)}회, "
          f"rework_counts={result.get('rework_counts', {})}")
    verdict = result.get("quality_verdict") or {}
    print(f"  품질 평가: {'PASS' if verdict.get('passed') else 'FAIL ' + str(verdict.get('failed_criteria'))}")
    print(f"  결정 로그: {trace_path(run_id)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="KV Cache 기술 다관점 평가 (Supervisor 패턴)")
    parser.add_argument("--question", default=DEFAULT_QUESTION, help="평가 질문")
    parser.add_argument("--resume", metavar="RUN_ID", help="중단된 실행을 체크포인트에서 재개")
    parser.add_argument(
        "--keep-checkpoints",
        action="store_true",
        help="정상 종료 후에도 중간 체크포인트를 지우지 않는다 (기본: 마지막 체크포인트만 남기고 정리)",
    )
    args = parser.parse_args()

    if not settings.vectorstore_path.exists():
        print(
            f"[app] {settings.vectorstore_path} 가 없습니다. 먼저 아래를 실행하세요:\n"
            "  python -m scripts.download_papers\n"
            "  python -m rag.ingest",
            file=sys.stderr,
        )
        sys.exit(1)
    if os.getenv("LANGSMITH_TRACING", "").lower() != "true" or not os.getenv("LANGSMITH_API_KEY"):
        print("[app] LangSmith 트레이싱 비활성 (.env의 LANGSMITH_TRACING/LANGSMITH_API_KEY 확인)", file=sys.stderr)

    register_external_search()
    run_id = args.resume or str(uuid.uuid4())
    # run_id = 체크포인트 thread_id = LangSmith metadata(+새 실행이면 루트 run id) = 결정 로그 파일명
    config = {
        "configurable": {"thread_id": run_id},
        "recursion_limit": settings.graph_recursion_limit,
        "run_name": "kv-cache-supervisor",
        "tags": ["supervisor-pattern", "resume" if args.resume else "fresh"],
        "metadata": {"run_id": run_id},
    }
    if not args.resume:
        config["run_id"] = uuid.UUID(run_id)
    print(f"[app] run_id={run_id}")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    total_start = time.perf_counter()
    with SqliteSaver.from_conn_string(str(CHECKPOINT_DB)) as checkpointer:
        workflow = build_graph(checkpointer=checkpointer)
        graph_input = None if args.resume else {"research_question": args.question, "run_id": run_id}
        result = workflow.invoke(graph_input, config=config)
        finished = is_run_finished(workflow, config)  # 체크포인터 연결이 열려 있을 때만 상태를 읽을 수 있다
    total_elapsed = time.perf_counter() - total_start

    if not result.get("final_report"):
        print("[app] 보고서가 생성되지 않았습니다. 결정 로그를 확인하세요.", file=sys.stderr)
        _print_orchestration_summary(result)
        sys.exit(2)

    print("\n=== 최종 평가 보고서 ===\n")
    print(result["final_report"])
    print_timing_summary()
    _print_orchestration_summary(result)
    print(f"\n[timing] 전체 실행 시간: {total_elapsed:.1f}초")

    # 최종본 = 품질 평가를 거친 보고서 + Supervisor 결정 이력 부록. 부록은 실행 메타데이터라 품질 평가
    # 대상이 아니고, 결정(품질 평가·종료 포함)이 모두 끝난 뒤에야 완성되므로 여기서 붙인다.
    report_path = Path(result["report_path"])
    final_path = report_path.with_name(f"{report_path.stem}_final.md")
    final_path.write_text(
        result["final_report"].rstrip() + "\n\n" + render_decision_appendix(result["run_id"]), encoding="utf-8"
    )
    print(f"[app] 최종 보고서(결정 이력 부록 포함): {final_path}")
    if not args.resume:  # 재개 실행은 루트 run id가 run_id와 달라 기록 대상이 없다
        record_run_feedback(result["run_id"], (result.get("quality_verdict") or {}).get("passed"))

    # 제출용 PDF(Agent-Output)를 이번 실행의 최종본으로 갱신한다. 실패해도 .md는 이미 저장됨.
    try:
        pdf_path = OUTPUTS_DIR / f"Agent-Output_{DEFAULT_CAMPUS}_{DEFAULT_CLASS}_{DEFAULT_TEAM_NAMES}.pdf"
        convert_report_to_pdf(final_path, pdf_path)
        pages = len(PdfReader(str(pdf_path)).pages)
        flag = "" if pages <= settings.max_report_pages else f" ⚠️ 최대 {settings.max_report_pages}쪽 초과"
        print(f"[app] 제출용 PDF: {pdf_path} ({pages}쪽{flag})")
    except Exception as exc:  # noqa: BLE001
        print(f"[app] PDF 변환 실패(보고서 .md는 저장됨): {exc}", file=sys.stderr)

    # 정상 종료한 run은 마지막 체크포인트만 남긴다(Ctrl+C·예외로 끊긴 run은 여기 오지 않아 재개 가능).
    # 산출물(.md/PDF)을 모두 만든 뒤에 하고, 실패해도 실행 결과에는 영향이 없다.
    if finished and not args.keep_checkpoints:
        try:
            before, after = prune_checkpoints(CHECKPOINT_DB, run_id)
            print(f"[app] 체크포인트 정리: {before / 1e6:.1f}MB → {after / 1e6:.1f}MB (--keep-checkpoints로 유지 가능)")
        except Exception as exc:  # noqa: BLE001
            print(f"[app] 체크포인트 정리 실패(실행 결과에는 영향 없음): {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
