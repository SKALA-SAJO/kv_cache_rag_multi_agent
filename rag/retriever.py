"""Hybrid Retrieval + Rerank (RAG-Design PDF B.2 절).

1차: Dense(FAISS/bge-m3) + Sparse(BM25) 결합 Hybrid Retrieval (Top-20~30)
2차: Cross-encoder(BAAI/bge-reranker-base) 재정렬 후 Top-5~8만 최종 컨텍스트로 사용

기술 문서(DeepSeek-V2, InfiniGen 논문) RAG 검색 전용 모듈이다. 다른 관점
(시장성 등)의 검색에는 사용하지 않는다 (README '실제 구현 범위' 참고).
"""

from __future__ import annotations

import json
from functools import lru_cache

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from config import settings
from rag.embeddings import get_embeddings


def _load_chunks() -> list[Document]:
    if not settings.chunks_file.exists():
        raise RuntimeError(
            f"{settings.chunks_file} 가 없습니다. 먼저 `python -m rag.ingest` 를 실행하세요."
        )
    docs = []
    with open(settings.chunks_file, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            docs.append(Document(page_content=row["page_content"], metadata=row["metadata"]))
    return docs


@lru_cache(maxsize=1)
def _vectorstore() -> FAISS:
    if not (settings.vectorstore_path / "index.faiss").exists():
        raise RuntimeError(
            f"{settings.vectorstore_path} 에 FAISS 색인이 없습니다. "
            "먼저 `python -m rag.ingest` 를 실행하세요."
        )
    # 로컬에서 우리가 직접 만든 색인만 불러오므로 pickle 역직렬화 위험을 허용한다.
    return FAISS.load_local(
        str(settings.vectorstore_path),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )


@lru_cache(maxsize=1)
def _ensemble_retriever() -> EnsembleRetriever:
    dense = _vectorstore().as_retriever(
        search_kwargs={"k": settings.retrieval_top_k_candidates}
    )
    sparse = BM25Retriever.from_documents(_load_chunks())
    sparse.k = settings.retrieval_top_k_candidates
    return EnsembleRetriever(retrievers=[dense, sparse], weights=[0.5, 0.5])


@lru_cache(maxsize=1)
def _reranker() -> CrossEncoder:
    return CrossEncoder(settings.reranker_model, device=settings.embedding_device)


def retrieve(query: str, top_k: int | None = None) -> list[Document]:
    """주어진 질의로 기술 문서 RAG 검색을 수행하고 재정렬된 상위 문서를 반환한다."""
    top_k = top_k or settings.retrieval_top_k_final

    candidates = _ensemble_retriever().invoke(query)
    if not candidates:
        return []

    # dense/sparse 중복 문서 제거
    seen: set[str] = set()
    unique_candidates: list[Document] = []
    for doc in candidates:
        key = doc.metadata.get("chunk_id") or doc.page_content[:80]
        if key not in seen:
            seen.add(key)
            unique_candidates.append(doc)

    pairs = [(query, doc.page_content) for doc in unique_candidates]
    scores = _reranker().predict(pairs)
    ranked = sorted(zip(scores, unique_candidates), key=lambda pair: pair[0], reverse=True)
    return [doc for _, doc in ranked[:top_k]]
