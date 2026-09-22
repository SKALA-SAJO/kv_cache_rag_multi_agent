"""Gold 질문셋으로 Hybrid Retriever의 Hit@K와 MRR을 계산한다.

실행 전 ``uv run python -m rag.ingest``로 현재 코퍼스와 일치하는 FAISS 색인을 만든다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rag.retriever import retrieve


ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = ROOT / "data" / "eval" / "golden_questions.json"


def _rank_of_expected_chunk(documents: list[Any], expected_chunk_id: str) -> int | None:
    for index, document in enumerate(documents, start=1):
        if document.metadata.get("chunk_id") == expected_chunk_id:
            return index
    return None


def evaluate(top_k: int) -> dict[str, float]:
    records = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    ranks: list[int | None] = []

    for record in records:
        documents = retrieve(
            record["question"],
            top_k=top_k,
            doc_types=record["doc_types"],
            technology=record["technology"],
        )
        rank = _rank_of_expected_chunk(documents, record["expected_chunk_id"])
        ranks.append(rank)
        status = f"rank {rank}" if rank is not None else f"miss (top {top_k})"
        print(f"[{record['id']}] {status}")

    total = len(ranks)
    metrics = {
        "Hit@1": sum(rank == 1 for rank in ranks) / total,
        "Hit@3": sum(rank is not None and rank <= 3 for rank in ranks) / total,
        f"Hit@{top_k}": sum(rank is not None for rank in ranks) / total,
        "MRR": sum(1 / rank if rank is not None else 0 for rank in ranks) / total,
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Gold 질문셋 기반 Retriever 평가")
    parser.add_argument("--top-k", type=int, default=5, choices=range(1, 21))
    args = parser.parse_args()

    metrics = evaluate(args.top_k)
    print("\n=== Retrieval evaluation ===")
    for name, score in metrics.items():
        print(f"{name}: {score:.3f}")


if __name__ == "__main__":
    main()
