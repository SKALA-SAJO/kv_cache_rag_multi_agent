"""보고서 품질 평가 노드 (Hybrid = 규칙 기반 형식 검사 + LLM Judge 내용 판정).

평가 항목 4종 + 형식(분량·필수 목차):
  - groundedness         : 주장이 검색된 출처로 추적되는가
  - neutrality           : 특정 기술 추천/우열 판정이 없는가
  - bias_control         : 단일 출처·유리한 근거 편중이 없는가
  - perspective_coverage : 4개 관점(기술 성숙도·시장성·이해관계자·도메인 적용)을 포괄하는가
  - format               : SUMMARY/REFERENCE 필수 목차, PDF 기준 최대 N쪽

항목별 최종 판정 = 규칙 판정 AND LLM 판정. 규칙은 결정론적이라 "형식은 지켰는가"를 보장하고,
LLM은 "내용이 실제로 그러한가"(예: 인용은 있지만 인용된 출처가 그 주장을 담고 있는가)를 본다.
하나라도 미달이면 passed=False → Supervisor가 재작업/재작성 루프로 보낸다.
"""

from __future__ import annotations

import json
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from agents.base import build_reference_catalog, load_prompt, structured_call
from agents.schemas import QualityJudgement
from config import settings
from graph.state import GraphState

SYSTEM_PROMPT = load_prompt("quality_evaluation")

CRITERIA = ["groundedness", "neutrality", "bias_control", "perspective_coverage"]

# 4.x 관점 절을 찾기 위한 헤딩 키워드 → 관점 Agent
PERSPECTIVE_HEADINGS = {
    "trl_evaluation": ("기술 성숙도",),
    "market_evaluation": ("시장성",),
    "stakeholder_evaluation": ("이해관계자",),
    "domain_evaluation": ("장문맥 처리 애플리케이션", "도메인"),
}

# 우열 판정·추천 표현.
# - "조건 A에서는 X가 더 낫다" 같은 조건부 비교는 허용
# - "승자를 가릴 수 없다/어렵다" 같은 우열 판정 *부정* 문장은 위반으로 보지 않는다.
WINNER_RECOMMENDATION_PATTERNS = re.compile(
    r"최종 승자|승자로|채택을 권고|도입을 권고|추천한다|추천합니다|가장 좋은"
)
COMPARATIVE_PATTERNS = re.compile(
    r"우월|우세하다|더 우수|더 낫|더 유리|더 적합|압도적|superior|outperforms"
)
NEGATION_PATTERN = re.compile(r"하지\s*않|않는다|아니다|없다|수\s*없|어렵|불가|불가능|지양|배제")
CONDITIONAL_CUES = re.compile(r"조건|경우|환경|상황|에서는|에선|일\s*때|할\s*때|이라면|라면|가능\s*시|필요\s*시")

_CITATION_GROUP = re.compile(r"\[(R\d+(?:\s*[,;]\s*R\d+)*)\]")


def _headings(report: str) -> list[tuple[int, str, int]]:
    """(레벨, 제목, 시작 위치) 목록."""
    return [
        (len(m.group(1)), m.group(2).strip(), m.start())
        for m in re.finditer(r"^(#{1,4})\s+(.+)$", report, flags=re.MULTILINE)
    ]


def _section(report: str, keywords: tuple[str, ...], level: int | None = None) -> str | None:
    heads = _headings(report)
    for i, (lvl, title, start) in enumerate(heads):
        if (level is None or lvl == level) and any(k in title for k in keywords):
            end = next((s for l2, _, s in heads[i + 1:] if l2 <= lvl), len(report))
            return report[start:end]
    return None


def extract_citations(text: str) -> list[str]:
    ids: list[str] = []
    for group in _CITATION_GROUP.findall(text):
        ids.extend(re.findall(r"R\d+", group))
    return ids


def _body_without_reference(report: str) -> str:
    ref = _section(report, ("REFERENCE",))
    return report.replace(ref, "") if ref else report


# ── 규칙 기반 판정 (1안) ─────────────────────────────────────────────────────


def rule_groundedness(report: str, catalog: list[dict]) -> dict[str, Any]:
    valid = {c["ref_id"] for c in catalog}
    body = _body_without_reference(report)
    cited = extract_citations(body)
    issues = []
    unknown = sorted(set(cited) - valid)
    if unknown:
        issues.append(f"카탈로그에 없는 인용 ID 사용: {unknown}")
    if len(cited) < settings.min_report_citations:
        issues.append(f"본문 인용 {len(cited)}개 < 최소 {settings.min_report_citations}개")
    for node, keys in PERSPECTIVE_HEADINGS.items():
        section = _section(report, keys, level=3)
        if section is not None and not extract_citations(section):
            issues.append(f"'{keys[0]}' 관점 절에 인용 없음")
    ref_section = _section(report, ("REFERENCE",)) or ""
    missing_in_ref = sorted(set(cited) & valid - set(extract_citations(ref_section)))
    if missing_in_ref:
        issues.append(f"본문에서 인용했지만 REFERENCE에 없는 ID: {missing_in_ref}")
    return {"passed": not issues, "issues": issues, "citations": len(cited)}


def rule_neutrality(report: str) -> dict[str, Any]:
    issues = []
    for sentence in re.split(r"(?<=[.!?。])\s+|\n", _body_without_reference(report)):
        sentence = sentence.strip()
        if not sentence:
            continue

        if WINNER_RECOMMENDATION_PATTERNS.search(sentence) and not NEGATION_PATTERN.search(sentence):
            issues.append(f"우열/추천 표현: \"{sentence[:120]}\"")
            continue

        if COMPARATIVE_PATTERNS.search(sentence):
            if NEGATION_PATTERN.search(sentence):
                continue
            if CONDITIONAL_CUES.search(sentence):
                continue
            issues.append(f"우열/추천 표현: \"{sentence[:120]}\"")
    return {"passed": not issues, "issues": issues}


def rule_bias_control(report: str, catalog: list[dict], evidence_items: list[dict]) -> dict[str, Any]:
    issues = []
    sources = Counter(e.get("document_id") or e.get("source_url") for e in evidence_items)
    total = sum(sources.values())
    if total:
        top_source, top_count = sources.most_common(1)[0]
        ratio = top_count / total
        if ratio > settings.max_single_source_ratio:
            issues.append(f"단일 출처 편중: {top_source}가 근거의 {ratio:.0%}")
    cited = set(extract_citations(_body_without_reference(report)))
    if len(cited) < 4:
        issues.append(f"서로 다른 인용 출처 {len(cited)}종 < 4")
    by_id = {c["ref_id"]: c for c in catalog}
    external = {c["ref_id"] for c in catalog if c.get("source_type") == "external_search"}
    if external and not (cited & external):
        issues.append("외부 검색 출처가 수집됐지만 본문에서 하나도 인용되지 않음 (코퍼스 편중)")
    cited_types = {by_id[i].get("source_type") for i in cited if i in by_id}
    return {"passed": not issues, "issues": issues, "cited_source_types": sorted(t for t in cited_types if t)}


def rule_coverage(report: str) -> dict[str, Any]:
    issues = []
    for node, keys in PERSPECTIVE_HEADINGS.items():
        section = _section(report, keys, level=3)
        if section is None or len(section.strip()) < 150:
            issues.append(f"'{keys[0]}' 관점 절 누락 또는 내용 부족")
    return {"passed": not issues, "issues": issues}


def count_pdf_pages(report: str) -> int | None:
    """제출용 PDF와 같은 변환기로 렌더링해 실제 쪽수를 센다 (실패 시 None)."""
    from scripts.report_to_pdf import convert

    with tempfile.TemporaryDirectory() as tmp:
        md_path, pdf_path = Path(tmp) / "report.md", Path(tmp) / "report.pdf"
        md_path.write_text(report, encoding="utf-8")
        try:
            convert(md_path, pdf_path)
            return len(PdfReader(str(pdf_path)).pages)
        except (Exception, SystemExit):  # noqa: BLE001 - 형식 검사 실패가 그래프를 멈추면 안 됨
            return None


def rule_format(report: str) -> dict[str, Any]:
    issues = []
    if not _section(report, ("SUMMARY",)):
        issues.append("SUMMARY 절(헤딩) 없음")
    if not _section(report, ("REFERENCE",)):
        issues.append("REFERENCE 절(헤딩) 없음")
    pages = count_pdf_pages(report)
    if pages is None:
        if len(report) > 16000:
            issues.append(f"분량 초과 추정: {len(report)}자 (PDF 쪽수 측정 실패)")
    elif pages > settings.max_report_pages:
        issues.append(f"PDF {pages}쪽 > 최대 {settings.max_report_pages}쪽 — 분량 축소 필요")
    return {"passed": not issues, "issues": issues, "pdf_pages": pages}


# ── LLM Judge (2안) ──────────────────────────────────────────────────────────


def _judge_input(state: GraphState, report: str, catalog: list[dict]) -> str:
    evidence = state.get("evidence_items", [])
    per_agent = Counter(e.get("agent") for e in evidence)
    # 보고서가 실제로 인용한 출처의 근거를 우선 보여준다 — 앞에서부터 자르면 인용된 수치의
    # 원문이 샘플에서 빠져 Judge가 근거 있는 주장을 미달로 오판한다.
    cited_ids = set(extract_citations(_body_without_reference(report)))
    cited_sources = {c.get("url") or c.get("source") for c in catalog if c["ref_id"] in cited_ids}
    ranked = sorted(
        evidence,
        key=lambda e: (e.get("source_url") or e.get("document_id")) not in cited_sources,
    )
    samples = [
        {
            "agent": e.get("agent"),
            "source": e.get("document_id") or e.get("source_url"),
            "quote": (e.get("evidence_quote") or "")[:300],
        }
        for e in ranked[:80]
    ]
    payload = {
        "reference_catalog": [
            {k: c[k] for k in ("ref_id", "source", "url", "source_type", "technology")} for c in catalog
        ],
        "evidence_count_by_agent": dict(per_agent),
        "evidence_samples": samples,
        "excluded_agents": [n for n, s in state.get("node_status", {}).items() if s == "excluded"],
        "unverified_claims": (state.get("faithfulness_check") or {}).get("insufficient_evidence_claims", []),
    }
    return f"## 평가 대상 보고서\n{report}\n\n## 수집 근거 요약\n{json.dumps(payload, ensure_ascii=False, indent=1)}"


def combine(rules: dict[str, dict], judgement: QualityJudgement | None) -> dict[str, Any]:
    """규칙 판정과 LLM 판정을 항목별 AND로 결합해 quality_verdict를 만든다.

    Fail-closed: Judge 호출/파싱이 실패했거나(judgement=None) 특정 항목 판정이 빠지면 그 항목은
    통과로 치지 않는다. 품질 게이트가 "판정 불가 = 통과"가 되면 내용 검사가 조용히 사라진다.
    """
    llm = {c.criterion: c for c in judgement.criteria} if judgement else {}
    criteria: dict[str, dict] = {}
    rework_targets: set[str] = set()
    feedback_lines: list[str] = []
    for name in CRITERIA:
        rule = rules[name]
        judge = llm.get(name)
        llm_passed = judge.passed if judge else False
        passed = rule["passed"] and llm_passed
        criteria[name] = {
            "passed": passed,
            "rule": rule,
            "llm": judge.model_dump() if judge else None,
        }
        if not passed:
            feedback_lines += [f"[{name}/rule] {i}" for i in rule["issues"]]
            if judge is None:
                feedback_lines.append(f"[{name}/judge] LLM Judge 판정 없음 — 내용 검증 불가로 미달 처리")
            elif not judge.passed:
                feedback_lines += [f"[{name}/judge] {i}" for i in judge.issues]
                rework_targets.update(judge.rework_targets)
    criteria["format"] = {"passed": rules["format"]["passed"], "rule": rules["format"], "llm": None}
    if not rules["format"]["passed"]:
        feedback_lines += [f"[format/rule] {i}" for i in rules["format"]["issues"]]

    failed = [name for name, c in criteria.items() if not c["passed"]]
    if judgement and judgement.feedback and failed:
        feedback_lines.append(f"[judge 종합] {judgement.feedback}")
    return {
        "passed": not failed,
        "failed_criteria": failed,
        "criteria": criteria,
        "rework_targets": sorted(rework_targets),
        "feedback": "\n".join(feedback_lines),
        "method": "hybrid(rule AND llm_judge)",
        "judge_available": judgement is not None,
    }


def run(state: GraphState) -> dict:
    report = state.get("final_report", "")
    catalog = build_reference_catalog(state.get("references", []))
    rules = {
        "groundedness": rule_groundedness(report, catalog),
        "neutrality": rule_neutrality(report),
        "bias_control": rule_bias_control(report, catalog, state.get("evidence_items", [])),
        "perspective_coverage": rule_coverage(report),
        "format": rule_format(report),
    }
    try:
        judgement = structured_call(
            QualityJudgement, SYSTEM_PROMPT, _judge_input(state, report, catalog), role="judge"
        )
    except Exception as exc:  # noqa: BLE001 - Judge 장애는 노드 실패가 아니라 "미달" verdict로 남긴다
        print(f"[quality_evaluation] LLM Judge 실패 → 미달 처리: {type(exc).__name__}: {exc}", flush=True)
        judgement = None
    verdict = combine(rules, judgement)
    verdict["revision"] = state.get("report_revisions", 0)

    report_path = state.get("report_path")
    if report_path:
        Path(report_path).with_suffix(".quality.json").write_text(
            json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    status = "PASS" if verdict["passed"] else f"FAIL {verdict['failed_criteria']}"
    print(f"[quality_evaluation] {status} (pdf_pages={rules['format'].get('pdf_pages')})", flush=True)
    return {"quality_verdict": verdict}
