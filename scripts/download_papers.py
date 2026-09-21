"""기술 문서 RAG 코퍼스(DeepSeek-V2, InfiniGen 논문)를 data/raw/ 에 내려받는 스크립트.

RAG-Design 문서(초안3.pdf) REFERENCE 절에 명시된 arXiv 원문을 그대로 사용한다.
표준 라이브러리만 사용하므로 추가 의존성 설치 없이 실행할 수 있다.

Usage:
    python -m scripts.download_papers
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

PAPERS = {
    "deepseek_v2_mla.pdf": "https://arxiv.org/pdf/2405.04434",
    "infinigen.pdf": "https://arxiv.org/pdf/2406.19707",
}


def download() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in PAPERS.items():
        dest = RAW_DIR / filename
        if dest.exists():
            print(f"[skip] {filename} 이미 존재함")
            continue
        print(f"[download] {url} -> {dest}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response, open(dest, "wb") as f:
            f.write(response.read())
        print(f"[done] {filename} ({dest.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    download()
