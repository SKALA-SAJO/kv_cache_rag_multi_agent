"""outputs/report_*.md 최종 평가 보고서를 제출용 PDF(RAG-Output)로 변환한다.

파일명 규칙: RAG-Output_{캠퍼스}_{X반}_{이름들}.pdf (기본값 = RAG-Design PDF와 동일한 팀 정보).
pandoc 등 시스템 설치 없이 uv sync만으로 재현되도록 순수 Python(markdown + xhtml2pdf)을
쓰고, 한글 미지원인 기본 PDF 폰트 대신 assets/fonts/의 나눔고딕을 임베딩한다.

Usage:
    uv run python -m scripts.report_to_pdf                     # 최신 report_*.md 자동 선택
    uv run python -m scripts.report_to_pdf --input outputs/report_20260922_121006.md
    uv run python -m scripts.report_to_pdf --campus 판교 --class-name 10반 --names "이름1+이름2"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import markdown
from xhtml2pdf import pisa

BASE_DIR = Path(__file__).resolve().parent.parent
FONT_DIR = BASE_DIR / "assets" / "fonts"
OUTPUTS_DIR = BASE_DIR / "outputs"

DEFAULT_CAMPUS = "판교"
DEFAULT_CLASS = "10반"
DEFAULT_TEAM_NAMES = "박성우+박인애+이승준+서지원+전은배+최윤영"

_HTML_TEMPLATE = """<html>
<head>
<meta charset="utf-8" />
<style>
@font-face {{
    font-family: "NanumGothic";
    src: url("{regular_font}");
}}
@font-face {{
    font-family: "NanumGothic";
    font-weight: bold;
    src: url("{bold_font}");
}}
body {{ font-family: "NanumGothic"; font-size: 10pt; line-height: 1.55; }}
h1 {{ font-size: 17pt; border-bottom: 2px solid #333; padding-bottom: 4px; }}
h2 {{ font-size: 13.5pt; margin-top: 18px; border-bottom: 1px solid #999; padding-bottom: 2px; }}
h3 {{ font-size: 11.5pt; margin-top: 12px; }}
table {{ border-collapse: collapse; width: 100%; margin: 8px 0; }}
th, td {{ border: 1px solid #999; padding: 4px 6px; font-size: 9pt; text-align: left; }}
th {{ background: #eee; }}
code {{ background: #f0f0f0; padding: 1px 3px; font-size: 9pt; }}
ul, ol {{ margin: 4px 0; padding-left: 20px; }}
</style>
</head>
<body>
{body}
</body>
</html>"""


def _latest_report() -> Path:
    reports = sorted(OUTPUTS_DIR.glob("report_*.md"))
    if not reports:
        raise SystemExit(
            f"{OUTPUTS_DIR} 에 report_*.md 파일이 없습니다. 먼저 `uv run python app.py`를 실행하세요."
        )
    return reports[-1]


# 나눔고딕에 글리프가 없어 빈 박스로 나오는 특수문자를 LLM이 종종 섞어 쓴다 — 일반
# ASCII 문자로 정규화한다 (U+2011 논브레이킹 하이픈 등).
_GLYPH_FALLBACKS = {
    "‑": "-",  # non-breaking hyphen
}


def _normalize_text(text: str) -> str:
    for char, replacement in _GLYPH_FALLBACKS.items():
        text = text.replace(char, replacement)
    return text


def convert(input_path: Path, output_path: Path) -> None:
    if not FONT_DIR.exists():
        raise SystemExit(f"{FONT_DIR} 가 없습니다 — 나눔고딕 폰트를 assets/fonts/에 받아두세요.")

    markdown_text = _normalize_text(input_path.read_text(encoding="utf-8"))
    body_html = markdown.markdown(markdown_text, extensions=["tables", "fenced_code", "nl2br"])
    html = _HTML_TEMPLATE.format(
        regular_font=(FONT_DIR / "NanumGothic-Regular.ttf").as_uri(),
        bold_font=(FONT_DIR / "NanumGothic-Bold.ttf").as_uri(),
        body=body_html,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        result = pisa.CreatePDF(html, dest=f, encoding="utf-8")
    if result.err:
        raise SystemExit(f"PDF 변환 실패 (xhtml2pdf 오류 {result.err}건)")
    print(f"[report_to_pdf] {output_path} 생성 완료")


def main() -> None:
    parser = argparse.ArgumentParser(description="평가 보고서 Markdown -> 제출용 PDF(RAG-Output) 변환")
    parser.add_argument("--input", type=Path, default=None, help="변환할 .md 파일 (기본: 최신 report_*.md)")
    parser.add_argument("--output", type=Path, default=None, help="출력 PDF 경로 (기본: 과제 파일명 규칙)")
    parser.add_argument("--campus", default=DEFAULT_CAMPUS)
    parser.add_argument("--class-name", dest="class_name", default=DEFAULT_CLASS)
    parser.add_argument("--names", default=DEFAULT_TEAM_NAMES)
    args = parser.parse_args()

    input_path = args.input or _latest_report()
    output_path = (
        args.output
        or OUTPUTS_DIR / f"RAG-Output_{args.campus}_{args.class_name}_{args.names}.pdf"
    )
    convert(input_path, output_path)


if __name__ == "__main__":
    main()
