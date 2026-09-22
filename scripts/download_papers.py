"""RAG 코퍼스 원문을 ``data/raw/``에 수집한다.

수집 범위는 설계서 B.3의 기술 원문, 구현 자료, 장문맥 도메인 자료,
시장 자료다. 각 소스의 메타데이터는 ``CORPUS_SOURCES`` 한 곳에서 관리하며,
``rag.ingest``도 이 목록을 사용해 청크에 ``doc_type``과 ``technology``를
일관되게 부여한다.

Usage:
    python -m scripts.download_papers
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import urllib.request


RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
MANIFEST_PATH = RAW_DIR / "corpus_manifest.json"


@dataclass(frozen=True)
class CorpusSource:
    """색인할 원문 1개의 출처 및 검색 필터용 메타데이터."""

    filename: str
    url: str
    title: str
    doc_type: str
    technology: str | None
    content_format: str  # pdf | markdown | html


# Phase 1-B Retriever가 아래 doc_type 값을 필터 조건으로 사용한다.
DOC_TYPE_TECHNICAL_PAPER = "technical_paper"
DOC_TYPE_IMPLEMENTATION = "implementation_document"
DOC_TYPE_DOMAIN = "domain_benchmark"
DOC_TYPE_MARKET = "market_document"


CORPUS_SOURCES: tuple[CorpusSource, ...] = (
    CorpusSource(
        filename="deepseek_v2_mla.pdf",
        url="https://arxiv.org/pdf/2405.04434",
        title="DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model",
        doc_type=DOC_TYPE_TECHNICAL_PAPER,
        technology="DeepSeek-V2 MLA",
        content_format="pdf",
    ),
    CorpusSource(
        filename="infinigen.pdf",
        url="https://arxiv.org/pdf/2406.19707",
        title="InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management",
        doc_type=DOC_TYPE_TECHNICAL_PAPER,
        technology="InfiniGen",
        content_format="pdf",
    ),
    CorpusSource(
        filename="deepseek_v2_readme.md",
        url="https://raw.githubusercontent.com/deepseek-ai/DeepSeek-V2/main/README.md",
        title="DeepSeek-V2 GitHub Repository README",
        doc_type=DOC_TYPE_IMPLEMENTATION,
        technology="DeepSeek-V2 MLA",
        content_format="markdown",
    ),
    CorpusSource(
        filename="infinigen_readme.md",
        url="https://raw.githubusercontent.com/snu-comparch/infinigen/main/README.md",
        title="InfiniGen GitHub Repository README",
        doc_type=DOC_TYPE_IMPLEMENTATION,
        technology="InfiniGen",
        content_format="markdown",
    ),
    CorpusSource(
        filename="longbench.pdf",
        url="https://aclanthology.org/2024.acl-long.172.pdf",
        title="LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding",
        doc_type=DOC_TYPE_DOMAIN,
        technology=None,
        content_format="pdf",
    ),
    CorpusSource(
        filename="ruler.pdf",
        url="https://arxiv.org/pdf/2404.06654",
        title="RULER: What's the Real Context Size of Your Long-Context Language Models?",
        doc_type=DOC_TYPE_DOMAIN,
        technology=None,
        content_format="pdf",
    ),
    CorpusSource(
        filename="gemini_long_context.html",
        url="https://ai.google.dev/gemini-api/docs/long-context",
        title="Long context | Gemini API Docs",
        doc_type=DOC_TYPE_MARKET,
        technology=None,
        content_format="html",
    ),
)


def _download(source: CorpusSource, destination: Path) -> None:
    """공개 원문을 내려받는다. 실패한 파일을 남기지 않도록 임시 파일을 사용한다."""
    request = urllib.request.Request(source.url, headers={"User-Agent": "Mozilla/5.0"})
    temporary = destination.with_suffix(f"{destination.suffix}.part")
    try:
        with urllib.request.urlopen(request, timeout=60) as response, open(temporary, "wb") as file:
            file.write(response.read())
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_manifest() -> None:
    """수집 목록과 URL을 별도 파일에 남겨 ingest와 보고서 출처 추적에 사용한다."""
    payload = {"sources": [asdict(source) for source in CORPUS_SOURCES]}
    MANIFEST_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def download() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for source in CORPUS_SOURCES:
        destination = RAW_DIR / source.filename
        if destination.exists() and destination.stat().st_size > 0:
            print(f"[skip] {source.filename} 이미 존재함")
            continue
        print(f"[download] {source.url} -> {destination}")
        _download(source, destination)
        print(f"[done] {source.filename} ({destination.stat().st_size / 1024:.0f} KB)")

    _write_manifest()
    print(f"[manifest] {MANIFEST_PATH}")


if __name__ == "__main__":
    download()
