"""설계서 B.2/B.3 기준의 다문서 RAG 색인을 구축한다.

``data/raw/``의 기술 원문, 구현 README, LongBench/RULER, Gemini 장문맥
문서를 읽어 토큰 기준으로 청킹한다. 모든 청크에는 ``doc_type``과
``technology``를 포함한 출처 메타데이터를 넣어 Retriever의 범위 필터링과
Faithfulness Check의 근거 추적에 사용한다.

Usage:
    python -m rag.ingest
"""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings
from rag.embeddings import get_embeddings
from scripts.download_papers import CORPUS_SOURCES, CorpusSource


# 설계서 B.2: 약 500 토큰, 50 토큰 overlap.
CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50
CAPTION_PATTERN = re.compile(
    r"^\s*(?:figure|fig\.?|table|그림|표)\s*\d+[\s.:：-]",
    re.IGNORECASE,
)
SECTION_HEADING_PATTERN = re.compile(
    r"^\s*(?:#{1,6}\s+|(?:\d+(?:\.\d+)*|[IVXLC]+)\.?\s+|"
    r"(?:abstract|introduction|background|method|methods|experiment(?:s)?|"
    r"results?|conclusion|references|appendix)\b)",
    re.IGNORECASE,
)


class _HTMLTextExtractor(HTMLParser):
    """외부 의존성 없이 공식 문서 HTML에서 색인 가능한 본문 텍스트를 추출한다."""

    _SKIP_TAGS = {"script", "style", "noscript", "svg", "nav", "footer", "header"}

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in self._SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self._SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        if not self._skip_depth and tag.lower() in {"p", "div", "li", "h1", "h2", "h3", "h4", "br"}:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self._parts.append(data)

    def text(self) -> str:
        lines = (re.sub(r"\s+", " ", line).strip() for line in "".join(self._parts).splitlines())
        return "\n".join(line for line in lines if line)


def _base_metadata(source: CorpusSource) -> dict[str, str]:
    return {
        "source": source.filename,
        "source_url": source.url,
        "source_title": source.title,
        "doc_type": source.doc_type,
        "technology": source.technology or "all",
    }


def _infer_section(text: str, fallback: str = "document") -> str:
    """문서 페이지·본문에서 식별 가능한 첫 섹션 제목을 출처 메타데이터로 남긴다."""
    for line in text.splitlines():
        candidate = re.sub(r"\s+", " ", line).strip()
        if SECTION_HEADING_PATTERN.match(candidate):
            return candidate[:160]
    return fallback


def _load_html(path: Path, source: CorpusSource) -> list[Document]:
    parser = _HTMLTextExtractor()
    parser.feed(path.read_text(encoding="utf-8", errors="ignore"))
    text = parser.text()
    if not text:
        raise ValueError(f"{path}에서 색인할 텍스트를 추출하지 못했습니다.")
    metadata = _base_metadata(source)
    metadata["section"] = _infer_section(text, fallback="web_document")
    return [Document(page_content=text, metadata=metadata)]


def _load_markdown(path: Path, source: CorpusSource) -> list[Document]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if not text.strip():
        raise ValueError(f"{path}가 비어 있습니다.")
    metadata = _base_metadata(source)
    metadata["section"] = _infer_section(text, fallback="document")
    return [Document(page_content=text, metadata=metadata)]


def _load_pdf(path: Path, source: CorpusSource) -> list[Document]:
    pages = PyPDFLoader(str(path)).load()
    base = _base_metadata(source)
    for page in pages:
        page.metadata.update(base)
        # PyPDFLoader page 값은 0-base이므로 보고서 출처에는 1-base로 기록한다.
        if "page" in page.metadata:
            page.metadata["page"] = page.metadata["page"] + 1
        page.metadata["section"] = _infer_section(
            page.page_content,
            fallback=f"page {page.metadata.get('page', '?')}",
        )
    return pages


def _load_source(path: Path, source: CorpusSource) -> list[Document]:
    if source.content_format == "pdf":
        return _load_pdf(path, source)
    if source.content_format == "markdown":
        return _load_markdown(path, source)
    if source.content_format == "html":
        return _load_html(path, source)
    raise ValueError(f"지원하지 않는 content_format: {source.content_format}")


def _caption_documents(document: Document) -> tuple[Document, list[Document]]:
    """표/그림 캡션을 본문에서 떼어 별도 검색 단위로 만든다.

    PDF 텍스트 추출만으로 표 셀 구조를 완벽히 재현할 수는 없으므로, 표 또는
    그림을 식별하는 caption을 우선 별도 청크로 분리한다. 본문 청크에는 캡션
    줄을 남기지 않아 캡션 검색 결과가 일반 본문에 묻히지 않게 한다.
    """
    lines = document.page_content.splitlines()
    body_lines: list[str] = []
    captions: list[Document] = []
    index = 0

    while index < len(lines):
        line = lines[index]
        if not CAPTION_PATTERN.match(line):
            body_lines.append(line)
            index += 1
            continue

        caption_lines = [line]
        index += 1
        # 빈 줄, 다음 캡션을 만나기 전까지의 짧은 캡션 이어쓰기만 포함한다.
        while index < len(lines) and len(" ".join(caption_lines)) < 1_200:
            candidate = lines[index]
            if not candidate.strip() or CAPTION_PATTERN.match(candidate):
                break
            caption_lines.append(candidate)
            index += 1

        metadata = dict(document.metadata)
        metadata["content_type"] = "table_caption" if re.match(
            r"^\s*(?:table|표)\s*\d+", caption_lines[0], re.IGNORECASE
        ) else "figure_caption"
        captions.append(Document(page_content="\n".join(caption_lines), metadata=metadata))

    body = Document(page_content="\n".join(body_lines).strip(), metadata=dict(document.metadata))
    body.metadata["content_type"] = "text"
    return body, captions


def _split_documents(documents: list[Document]) -> list[Document]:
    """문단 우선 분리와 tiktoken 기준 500/50 토큰 청킹을 적용한다."""
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=CHUNK_SIZE_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Document] = []
    per_source_count: dict[tuple[str, int | str, str], int] = {}
    for document in documents:
        body, captions = _caption_documents(document)
        for unit in [body, *captions]:
            if not unit.page_content.strip():
                continue
            for chunk in splitter.split_documents([unit]):
                page = chunk.metadata.get("page", "document")
                content_type = chunk.metadata["content_type"]
                key = (chunk.metadata["source"], page, content_type)
                sequence = per_source_count.get(key, 0)
                per_source_count[key] = sequence + 1
                source_stem = Path(chunk.metadata["source"]).stem
                chunk.metadata["chunk_id"] = f"{source_stem}-p{page}-{content_type}-{sequence}"
                chunks.append(chunk)
    return chunks


def load_and_split() -> list[Document]:
    """정의된 코퍼스를 불러오고 검색·출처 추적용 메타데이터를 붙여 청킹한다."""
    missing = [source.filename for source in CORPUS_SOURCES if not (settings.raw_data_path / source.filename).exists()]
    if missing:
        names = ", ".join(missing)
        raise RuntimeError(
            f"RAG 코퍼스 원문이 없습니다: {names}. "
            "먼저 `python -m scripts.download_papers` 를 실행하세요."
        )

    documents: list[Document] = []
    for source in CORPUS_SOURCES:
        path = settings.raw_data_path / source.filename
        loaded = _load_source(path, source)
        print(f"[load] {source.filename}: {len(loaded)}개 원문 단위 ({source.doc_type})")
        documents.extend(loaded)

    chunks = _split_documents(documents)
    if not chunks:
        raise RuntimeError("색인할 청크가 생성되지 않았습니다.")
    return chunks


def save_chunks_jsonl(chunks: list[Document], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for chunk in chunks:
            row = {"page_content": chunk.page_content, "metadata": chunk.metadata}
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_index() -> None:
    chunks = load_and_split()
    save_chunks_jsonl(chunks, settings.chunks_file)

    settings.vectorstore_path.mkdir(parents=True, exist_ok=True)
    vectorstore = FAISS.from_documents(documents=chunks, embedding=get_embeddings())
    vectorstore.save_local(str(settings.vectorstore_path))

    by_type: dict[str, int] = {}
    for chunk in chunks:
        doc_type = chunk.metadata["doc_type"]
        by_type[doc_type] = by_type.get(doc_type, 0) + 1
    detail = ", ".join(f"{doc_type}={count}" for doc_type, count in sorted(by_type.items()))
    print(f"[ingest] {len(chunks)}개 chunk 색인 완료 -> {settings.vectorstore_path} ({detail})")


if __name__ == "__main__":
    build_index()
