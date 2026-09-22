"""기술 문서 RAG 색인 구축 (RAG-Design PDF B.2, B.3 절).

data/raw/ 의 PDF(기술 원문)를 로드 -> 청크 분할 -> BAAI/bge-m3 임베딩 -> FAISS 색인.
BM25(Sparse) 검색을 위해 청크를 data/processed/chunks.jsonl 로도 저장한다.

Usage:
    python -m rag.ingest
"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings
from rag.embeddings import get_embeddings
from technologies import SELECTED_TECHNOLOGIES

# 파일명 -> 기술명 역매핑 (청크에 technology 메타데이터를 붙이기 위함)
DOC_TO_TECH = {info["source_document"]: name for name, info in SELECTED_TECHNOLOGIES.items()}


def load_and_split() -> list[Document]:
    """검색 단위: 섹션/문단 근사 - RecursiveCharacterTextSplitter로 문단 경계를 우선 보존한다."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks: list[Document] = []
    pdf_paths = sorted(settings.raw_data_path.glob("*.pdf"))
    if not pdf_paths:
        raise RuntimeError(
            f"{settings.raw_data_path} 에 PDF가 없습니다. "
            "먼저 `python -m scripts.download_papers` 를 실행하세요."
        )

    for pdf_path in pdf_paths:
        tech_name = DOC_TO_TECH.get(pdf_path.name, pdf_path.stem)
        pages = PyPDFLoader(str(pdf_path)).load()
        chunks = splitter.split_documents(pages)
        for i, chunk in enumerate(chunks):
            chunk.metadata["technology"] = tech_name
            chunk.metadata["source"] = pdf_path.name
            chunk.metadata["chunk_id"] = f"{pdf_path.stem}-{i}"
            # PyPDFLoader가 넣는 page는 0-base이므로 사람이 읽기 쉬운 1-base로 보정
            if "page" in chunk.metadata:
                chunk.metadata["page"] = chunk.metadata["page"] + 1
        all_chunks.extend(chunks)

    return all_chunks


def save_chunks_jsonl(chunks: list[Document], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            row = {"page_content": chunk.page_content, "metadata": chunk.metadata}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_index() -> None:
    chunks = load_and_split()
    save_chunks_jsonl(chunks, settings.chunks_file)

    settings.vectorstore_path.mkdir(parents=True, exist_ok=True)
    vectorstore = FAISS.from_documents(documents=chunks, embedding=get_embeddings())
    vectorstore.save_local(str(settings.vectorstore_path))
    print(f"[ingest] {len(chunks)}개 chunk 색인 완료 -> {settings.vectorstore_path}")


if __name__ == "__main__":
    build_index()
