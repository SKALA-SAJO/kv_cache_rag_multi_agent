"""Hybrid Retrieval + Rerank (RAG-Design PDF B.2 절).

1차: Dense(FAISS/bge-m3) + Sparse(BM25) 결합 Hybrid Retrieval (Top-20~30)
2차: Cross-encoder(BAAI/bge-reranker-v2-m3) 재정렬 후 Top-5~8만 최종 컨텍스트로 사용

코퍼스 전체(기술 원문, 구현 README, LongBench/RULER, Gemini 문서)를 대상으로 하는
RAG 검색 모듈이다. `doc_type`/`technology`로 검색 범위를 좁힐 수 있어, 호출하는
Agent가 자신의 관점에 맞는 문서군만 검색하도록 제한할 수 있다 (PDF B.3 문서군 표 참고).
doc_type 값은 이 모듈에서 새로 정의하지 않고 `scripts.download_papers`의 상수를
그대로 재사용한다 — 색인 메타데이터와 필터 값의 출처를 하나로 유지하기 위함이다.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Callable

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from config import settings
from rag.embeddings import MODEL_CALL_LOCK, get_embeddings

DocTypeFilter = str | tuple[str, ...] | None
_MetadataPredicate = Callable[[dict[str, Any]], bool]


def _normalize_doc_types(doc_types: DocTypeFilter) -> tuple[str, ...] | None:
    """단일 doc_type(str)과 복수 doc_type(tuple)을 캐시 키로 쓸 수 있는 tuple로 통일한다."""
    if doc_types is None:
        return None
    if isinstance(doc_types, str):
        return (doc_types,)
    return tuple(doc_types)


def _make_filter(
    doc_types: tuple[str, ...] | None, technology: str | None
) -> _MetadataPredicate | None:
    """doc_type/technology 조건의 메타데이터 predicate를 만든다.

    ingest.py가 technology 없는 문서(LongBench/RULER/Gemini)에는 "all"이라는
    문자열을 넣어두므로(파이썬 None이 아님) 단순 문자열 비교만으로 충분하다. 이
    때문에 domain_benchmark/market_document처럼 기술 비특정 문서군에 technology를
    함께 지정하면 의도적으로 빈 결과가 된다.
    """
    if doc_types is None and technology is None:
        return None

    def _match(metadata: dict[str, Any]) -> bool:
        if doc_types is not None and metadata.get("doc_type") not in doc_types:
            return False
        if technology is not None and metadata.get("technology") != technology:
            return False
        return True

    return _match


@lru_cache(maxsize=1)
def _all_chunks() -> tuple[Document, ...]:
    if not settings.chunks_file.exists():
        raise RuntimeError(
            f"{settings.chunks_file} 가 없습니다. 먼저 `python -m rag.ingest` 를 실행하세요."
        )
    docs = []
    with open(settings.chunks_file, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            docs.append(Document(page_content=row["page_content"], metadata=row["metadata"]))
    return tuple(docs)


@lru_cache(maxsize=32)
def _filtered_chunks(
    doc_types: tuple[str, ...] | None, technology: str | None
) -> tuple[Document, ...]:
    predicate = _make_filter(doc_types, technology)
    if predicate is None:
        return _all_chunks()
    return tuple(doc for doc in _all_chunks() if predicate(doc.metadata))


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


@lru_cache(maxsize=32)
def _ensemble_retriever(
    doc_types: tuple[str, ...] | None, technology: str | None
) -> EnsembleRetriever:
    predicate = _make_filter(doc_types, technology)

    search_kwargs: dict[str, Any] = {"k": settings.retrieval_top_k_candidates}
    if predicate is not None:
        # 필터가 있으면 FAISS가 fetch_k개를 먼저 뽑아 그중 필터를 통과하는 것만
        # 남기므로, 안 그러면 filter=None일 때보다 후보가 적게 나올 수 있다.
        search_kwargs["filter"] = predicate
        search_kwargs["fetch_k"] = settings.retrieval_fetch_k
    dense = _vectorstore().as_retriever(search_kwargs=search_kwargs)

    sparse = BM25Retriever.from_documents(list(_filtered_chunks(doc_types, technology)))
    sparse.k = settings.retrieval_top_k_candidates
    return EnsembleRetriever(retrievers=[dense, sparse], weights=[0.5, 0.5])


@lru_cache(maxsize=1)
def _reranker() -> CrossEncoder:
    return CrossEncoder(settings.reranker_model, device=settings.embedding_device)


def retrieve(
    query: str,
    top_k: int | None = None,
    doc_types: DocTypeFilter = None,
    technology: str | None = None,
) -> list[Document]:
    """주어진 질의로 RAG 검색을 수행하고 재정렬된 상위 문서를 반환한다.

    Args:
        query: 검색 질의.
        top_k: 최종 반환 개수. 기본값은 `settings.retrieval_top_k_final`.
        doc_types: 검색 범위를 제한할 doc_type. 단일 값(str) 또는 복수 값(tuple)을
            받으며, `scripts.download_papers`의 `DOC_TYPE_*` 상수를 사용해야 한다.
            생략(None)하면 코퍼스 전체를 검색한다.
        technology: 검색 범위를 제한할 technology (예: "DeepSeek-V2 MLA",
            "InfiniGen"). domain_benchmark/market_document처럼 technology가 없는
            문서군에는 함께 지정하지 않는다 — 지정하면 결과가 항상 비어 있다.
    """
    top_k = top_k or settings.retrieval_top_k_final
    normalized_doc_types = _normalize_doc_types(doc_types)

    if not _filtered_chunks(normalized_doc_types, technology):
        return []

    candidates = _ensemble_retriever(normalized_doc_types, technology).invoke(query)
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
    # rag/embeddings.py의 MODEL_CALL_LOCK 공유: 재정렬도 같은 GPU(MPS)를 쓰므로
    # 임베딩 호출과도 상호 배제되어야 한다.
    with MODEL_CALL_LOCK:
        scores = _reranker().predict(pairs)
    ranked = sorted(zip(scores, unique_candidates), key=lambda pair: pair[0], reverse=True)
    return [doc for _, doc in ranked[:top_k]]
