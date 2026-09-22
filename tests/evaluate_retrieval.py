"""분리된 질문셋으로 Hybrid Retriever의 Hit@K와 MRR을 계산한다.

실행 전 ``uv run python -m rag.ingest``로 현재 코퍼스와 일치하는 FAISS 색인을 만든다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rag.retriever import retrieve


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATHS = {
    "smoke": ROOT / "data" / "eval" / "smoke_questions.json",
    "heldout": ROOT / "data" / "eval" / "heldout_questions.json",
}


def _rank_of_expected_chunk(documents: list[Any], expected_chunk_id: str) -> int | None:
    for index, document in enumerate(documents, start=1):
        if document.metadata.get("chunk_id") == expected_chunk_id:
            return index
    return None


def _load_records(dataset: str) -> list[dict[str, Any]]:
    path = DATASET_PATHS[dataset]
    if not path.exists():
        raise FileNotFoundError(
            f"{path}가 없습니다. data/eval/HELDOUT_GUIDE.md를 따라 질문셋을 먼저 작성하세요."
        )
    records = json.loads(path.read_text(encoding="utf-8"))
    if not records:
        raise ValueError(f"{path}에 평가 질문이 없습니다.")
    return records


def evaluate(top_k: int, dataset: str) -> dict[str, float]:
    records = _load_records(dataset)
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
    parser.add_argument(
        "--dataset",
        choices=DATASET_PATHS,
        required=True,
        help="smoke는 파이프라인 점검용, heldout은 성능 평가용입니다.",
    )
    args = parser.parse_args()

    metrics = evaluate(args.top_k, args.dataset)
    print(f"\n=== Retrieval evaluation ({args.dataset}) ===")
    for name, score in metrics.items():
        print(f"{name}: {score:.3f}")


if __name__ == "__main__":
    main()
