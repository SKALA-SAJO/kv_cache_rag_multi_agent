"""KV Cache 최적화 기술 다관점 평가 Multi-Agent RAG 시스템 실행 스크립트.

사전 준비:
    python -m scripts.download_papers   # 기술 문서 RAG 코퍼스 다운로드
    python -m rag.ingest                # 색인 구축

실행:
    python app.py
    python app.py --question "다른 평가 질문"
"""

import argparse
import sys
import time

from dotenv import load_dotenv

load_dotenv(override=True)

from config import settings  # noqa: E402
from graph.workflow import build_graph, print_timing_summary  # noqa: E402
from rag.external_search import register as register_external_search  # noqa: E402

DEFAULT_QUESTION = (
    "장문맥 처리 애플리케이션 관점에서 DeepSeek-V2 MLA와 InfiniGen을 기술 성숙도, 시장성, "
    "이해관계자, 도메인 적합성 4가지 관점에서 비교 평가하라."
)


def main() -> None:
    parser = argparse.ArgumentParser(description="KV Cache 기술 다관점 평가 Multi-Agent RAG")
    parser.add_argument("--question", default=DEFAULT_QUESTION, help="평가 질문")
    args = parser.parse_args()

    if not settings.vectorstore_path.exists():
        print(
            f"[app] {settings.vectorstore_path} 가 없습니다. 먼저 아래를 실행하세요:\n"
            "  python -m scripts.download_papers\n"
            "  python -m rag.ingest",
            file=sys.stderr,
        )
        sys.exit(1)

    register_external_search()
    workflow = build_graph()

    total_start = time.perf_counter()
    result = workflow.invoke({"research_question": args.question})
    total_elapsed = time.perf_counter() - total_start

    print("\n=== 최종 평가 보고서 ===\n")
    print(result["final_report"])
    print_timing_summary()
    print(f"\n[timing] 전체 실행 시간: {total_elapsed:.1f}초")


if __name__ == "__main__":
    main()
