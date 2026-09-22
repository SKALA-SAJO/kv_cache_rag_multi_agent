"""Embedding 모델 wrapper (RAG-Design PDF B.4 절: BAAI/bge-m3 선정)."""

import threading
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from config import settings

# PyTorch의 MPS(Apple GPU) 백엔드는 동시 추론이 스레드 세이프하지 않다 — LangGraph가
# 여러 평가 Agent를 병렬로 실행하면서 reranker/embedding을 동시에 호출하면 세그폴트가
# 난다(실제로 재현된 문제). 이 락으로 임베딩·재정렬 호출을 전역적으로 직렬화해 MPS를
# EMBEDDING_DEVICE=mps로 안전하게 opt-in할 수 있게 한다. rag/retriever.py의 reranker
# 호출도 이 락을 공유한다. CPU에서는 원래도 문제없지만 락 오버헤드가 무시할 만큼
# 작아 device와 무관하게 항상 건다.
MODEL_CALL_LOCK = threading.Lock()


class _LockedHuggingFaceEmbeddings(HuggingFaceEmbeddings):
    """embed_query/embed_documents 호출을 MODEL_CALL_LOCK으로 직렬화하는 래퍼."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        with MODEL_CALL_LOCK:
            return super().embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        with MODEL_CALL_LOCK:
            return super().embed_query(text)


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return _LockedHuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": settings.embedding_device},
        encode_kwargs={"normalize_embeddings": True},
    )
