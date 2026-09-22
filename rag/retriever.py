"""Hybrid Retrieval + Rerank (RAG-Design PDF B.2 절).

1차: Dense(FAISS/bge-m3) + Sparse(BM25) 결합 Hybrid Retrieval (Top-20~30, RRF 결합)
2차: Cross-encoder(BAAI/bge-reranker-base) 재정렬 후 Top-5~8만 최종 컨텍스트로 사용

data/raw/ 코퍼스는 기술 원문(technical_paper) / 구현 자료(implementation_document) /
도메인 벤치마크(domain_benchmark) / 시장 자료(market_document) 네 doc_type으로 태깅되어
있다(scripts/download_papers.py, rag/ingest.py). doc_types 인자로 특정 유형만 검색
대상으로 좁힐 수 있다 — 예: 시장 평가 Agent는 market_document만 본다. 전체 코퍼스가
아직 작아 정교한 인덱스 분리 대신 FAISS/BM25 양쪽 다 후보를 넉넉히 가져온 뒤
(fetch_k) 필터링하는 방식으로 구현했다.
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

# doc_type 필터가 걸렸을 때 FAISS가 필터 적용 전에 내부적으로 훑어볼 후보 수.
# 시장 자료(market_document)처럼 전체 코퍼스 중 비중이 작은 doc_type도 놓치지 않도록
# 기본 fetch_k(20)보다 훨씬 크게 잡는다 — 코퍼스가 아직 작아(수백 청크) 비용 문제는 없다.
FILTERED_FETCH_K = 500


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
def _all_chunks() -> tuple[Document, ...]:
    return tuple(_load_chunks())


def _filtered_chunks(doc_types: frozenset[str] | None) -> list[Document]:
    if doc_types is None:
        return list(_all_chunks())
    return [doc for doc in _all_chunks() if doc.metadata.get("doc_type") in doc_types]


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


@lru_cache(maxsize=8)
def _ensemble_retriever(doc_types: frozenset[str] | None) -> EnsembleRetriever:
    search_kwargs: dict = {"k": settings.retrieval_top_k_candidates}
    if doc_types is not None:
        search_kwargs["filter"] = lambda meta, dt=doc_types: meta.get("doc_type") in dt
        search_kwargs["fetch_k"] = FILTERED_FETCH_K
    dense = _vectorstore().as_retriever(search_kwargs=search_kwargs)

    sparse = BM25Retriever.from_documents(_filtered_chunks(doc_types))
    sparse.k = settings.retrieval_top_k_candidates

    return EnsembleRetriever(retrievers=[dense, sparse], weights=[0.5, 0.5])


@lru_cache(maxsize=1)
def _reranker() -> CrossEncoder:
    return CrossEncoder(settings.reranker_model, device=settings.embedding_device)


def retrieve(
    query: str,
    top_k: int | None = None,
    doc_types: frozenset[str] | None = None,
) -> list[Document]:
    """주어진 질의로 기술 문서 RAG 검색을 수행하고 재정렬된 상위 문서를 반환한다.

    doc_types를 지정하면 해당 doc_type의 청크만 검색 대상으로 삼는다(예:
    frozenset({"market_document"})). 지정하지 않으면 전체 코퍼스에서 검색한다.
    """
    top_k = top_k or settings.retrieval_top_k_final

    candidates = _ensemble_retriever(doc_types).invoke(query)
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
