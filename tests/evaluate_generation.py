"""최종 보고서 Generation 품질을 정량 평가한다.

설계서의 Faithfulness Check 결과를 claim 통과율로 집계하고, 고정된 평가 Rubric을 기준으로
LLM Judge가 Answer Relevance를 1~5점으로 평가한다. 실제 전체 Graph와 외부 검색을 호출하므로
API 비용이 발생할 수 있다.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from agents.base import load_prompt, structured_call
from config import settings


ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "data" / "eval" / "generation_cases.json"
SYSTEM_PROMPT = load_prompt("generation_evaluator")


class GenerationCase(BaseModel):
    """Golden answer 대신 고정 Rubric으로 평가하는 Generation 평가 사례."""

    id: str
    question: str
    required_aspects: list[str] = Field(min_length=1)
    evaluation_scope: Literal["final_report"]


class AnswerRelevanceResult(BaseModel):
    score: int = Field(ge=1, le=5, description="질문 및 필수 관점 충족도")
    satisfied_aspects: list[str]
    missing_aspects: list[str]
    explanation: str


def load_cases(path: Path = CASES_PATH) -> list[GenerationCase]:
    if not path.exists():
        raise FileNotFoundError(f"Generation 평가 질문셋이 없습니다: {path}")
    raw_cases = json.loads(path.read_text(encoding="utf-8"))
    cases = [GenerationCase.model_validate(item) for item in raw_cases]
    if not cases:
        raise ValueError("Generation 평가 질문셋은 최소 1개 이상의 사례를 포함해야 합니다.")
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("Generation 평가 사례 id는 중복될 수 없습니다.")
    return cases


def faithfulness_metrics(faithfulness_check: dict[str, Any]) -> dict[str, int | float | None]:
    """기존 검증 Agent의 마지막 claim 판정을 재사용해 근거 충실도를 수치화한다."""
    checks = faithfulness_check.get("claim_checks", [])
    total = len(checks)
    passed = sum(check.get("status") == "pass" for check in checks)
    failed = total - passed
    return {
        "claim_count": total,
        "passed_claim_count": passed,
        "failed_claim_count": failed,
        "faithfulness_rate": passed / total if total else None,
    }


def judge_answer_relevance(case: GenerationCase, report: str) -> AnswerRelevanceResult:
    user_content = json.dumps(
        {
            "evaluation_question": case.question,
            "required_aspects": case.required_aspects,
            "evaluation_scope": case.evaluation_scope,
            "generated_report": report,
        },
        ensure_ascii=False,
        indent=2,
    )
    return structured_call(
        AnswerRelevanceResult,
        SYSTEM_PROMPT,
        user_content,
        role="judge",
    )


def evaluate_case(case: GenerationCase, workflow: Any) -> dict[str, Any]:
    """하나의 Golden Rubric 질문에 전체 Graph를 실행하고 두 Generation 지표를 산출한다."""
    state = workflow.invoke({"research_question": case.question})
    relevance = judge_answer_relevance(case, state.get("final_report", ""))
    return {
        "id": case.id,
        "question": case.question,
        "required_aspects": case.required_aspects,
        "faithfulness": faithfulness_metrics(state.get("faithfulness_check", {})),
        "answer_relevance": relevance.model_dump(),
        "retry_count": state.get("retry_count", 0),
        "evidence_item_count": len(state.get("evidence_items", [])),
        "reference_count": len(state.get("references", [])),
    }


def aggregate_metrics(case_results: list[dict[str, Any]]) -> dict[str, float | int | None]:
    faithfulness_rates = [
        result["faithfulness"]["faithfulness_rate"]
        for result in case_results
        if result["faithfulness"]["faithfulness_rate"] is not None
    ]
    relevance_scores = [result["answer_relevance"]["score"] for result in case_results]
    return {
        "case_count": len(case_results),
        "mean_faithfulness_rate": (
            sum(faithfulness_rates) / len(faithfulness_rates) if faithfulness_rates else None
        ),
        "mean_answer_relevance": sum(relevance_scores) / len(relevance_scores),
        "mean_retry_count": sum(result["retry_count"] for result in case_results) / len(case_results),
    }


def _validate_live_prerequisites() -> None:
    missing = []
    if not settings.openai_api_key:
        missing.append("OPENAI_API_KEY")
    if not settings.tavily_api_key:
        missing.append("TAVILY_API_KEY")
    if not settings.chunks_file.exists() or not (settings.vectorstore_path / "index.faiss").exists():
        missing.append("FAISS 색인 (uv run python -m rag.ingest)")
    if missing:
        raise RuntimeError("Generation 평가를 실행할 수 없습니다: " + ", ".join(missing))


def main() -> None:
    parser = argparse.ArgumentParser(description="Faithfulness + Answer Relevance Generation 평가")
    parser.add_argument("--case-id", help="특정 사례만 실행할 때 사용할 id")
    args = parser.parse_args()

    _validate_live_prerequisites()
    cases = load_cases()
    if args.case_id:
        cases = [case for case in cases if case.id == args.case_id]
        if not cases:
            raise ValueError(f"알 수 없는 case id입니다: {args.case_id}")

    from graph.workflow import build_graph
    from rag.external_search import register

    register()
    workflow = build_graph()
    results = []
    for case in cases:
        print(f"\n[{case.id}] 전체 Graph 실행 중...")
        result = evaluate_case(case, workflow)
        results.append(result)
        faithfulness = result["faithfulness"]["faithfulness_rate"]
        print(
            "Faithfulness="
            f"{faithfulness:.3f}" if faithfulness is not None else "Faithfulness=측정 불가"
        )
        print(f"Answer Relevance={result['answer_relevance']['score']}/5")

    payload = {
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "evaluator": {
            "faithfulness": "faithfulness_check.claim_checks의 pass 비율",
            "answer_relevance": "generation_evaluator Rubric 기반 LLM Judge 1~5점",
            "judge_model": settings.judge_model,
        },
        "aggregate": aggregate_metrics(results),
        "cases": results,
    }
    settings.outputs_path.mkdir(parents=True, exist_ok=True)
    path = settings.outputs_path / f"generation_eval_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Generation evaluation ===")
    print(json.dumps(payload["aggregate"], ensure_ascii=False, indent=2))
    print(f"결과 저장: {path}")


if __name__ == "__main__":
    main()
