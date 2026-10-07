"""Supervisor 패턴용 LangGraph State 정의.

설계 원칙 (README "State Schema" 절과 1:1 대응):
  - 제어 vs 페이로드 분리 : State를 "작업 페이로드"와 "제어 메타데이터" 두 블록으로 나눈다.
    Supervisor의 라우팅 정책(graph/supervisor.py)은 제어 블록 + 페이로드의 "존재·충분도"만 읽고,
    페이로드 본문 해석은 하위 Agent/평가 노드의 몫이다.
  - 레이어드 : 두 블록을 주석 구분이 아니라 별도 TypedDict로 선언한다. PayloadState(작업 결과)와
    ControlState(제어 메타)를 따로 정의하고 GraphState가 둘을 상속해 그래프의 State 스키마가 된다.
    "이 키는 어느 층인가"가 타입 수준에서 드러나고, 층별 키 집합이 겹치지 않음을 테스트로 고정한다
    (tests/test_state_layers.py). 하위 Agent별 Worker State 타입은 두지 않았다 — 각 Agent가 자기
    키만 쓰고(관점별 결과 키가 서로 다르다) 실행 상태는 ControlState의 node_status/attempts/errors가
    노드 키로 나눠 담기 때문이다. Agent 본체는 PayloadState 키만 반환하고, node_status/errors는
    graph/workflow.py의 _worker 래퍼가 노드 키로 기록한다(제어 키는 Supervisor와 래퍼만 쓴다).
  - 관측성 위치 : 결정 로그(사유 포함) 전문은 State가 아니라 외부 JSONL(outputs/traces/
    {run_id}.jsonl) + LangSmith 트레이스로 보낸다. State에는 "마지막 결정 1건"(last_decision)만
    덮어쓰기로 남겨, 트레이스 화면에서 노드 출력만 봐도 라우팅 사유가 보이게 한다.
  - 지속성 비용 : 체크포인트마다 직렬화되는 State가 무한 증식하지 않도록, 원문 청크
    (Document 전체 본문)는 State에 두지 않는다(이전 RAG 버전의 retrieved_documents 제거).
    근거는 300자 인용으로 잘린 evidence_items만 두고, 리듀서가 매 병합마다 중복을 제거한다.
    보고서 이력·결정 로그·PDF는 outputs/ 파일로만 남긴다.
    다만 SqliteSaver는 superstep마다 State 전체를 새로 저장하므로 용량은 State 한 건이 아니라 스냅샷 횟수에
    비례한다. 실행 중 상한은 재작업·스텝 예산이고, 정상 종료한 run은 마지막 체크포인트만 남기고 정리한다
    (graph/checkpoint_maintenance.py). 인용문 길이 축소는 근거 원본을 깎아 채택하지 않았다(README 참고).
  - 상관 : run_id 하나가 LangGraph thread_id(체크포인트), LangSmith 루트 run id/metadata,
    외부 결정 로그 파일명, 보고서 파일명을 모두 잇는 키다.
  - 재개/복구 : node_status/attempts/errors가 "어디까지 끝났고 무엇이 실패했는지"를, error_times가
    "언제 실패했는지"를 담는다(결과 키 타임스탬프 — last_decision에도 ts).
    SqliteSaver 체크포인트 + 이 필드만으로 `app.py --resume <run_id>` 재개가 가능하다.
  - 동시 처리 : Supervisor가 여러 관점 Agent를 한 superstep에 병렬 디스패치하므로 동시에
    쓰이는 필드(node_status, errors, error_times, evidence_items, references)에는 리듀서를 둔다.
    관점별 결과는 Agent마다 키가 달라(trl_evaluation 등) 충돌하지 않는다.
  - 종료 보장 : step_count/max_steps(Supervisor 스텝 상한), attempts(실패 재시도 상한),
    rework_counts(재작업 상한), faithfulness_rounds(검증 재작업 라운드 상한),
    report_revisions(보고서 재작성 상한) + LangGraph
    recursion_limit 이중 가드.
"""

from typing import Annotated, Any, Literal, TypedDict

NodeStatus = Literal["pending", "running", "done", "failed", "excluded"]


def merge_dict(existing: dict | None, new: dict | None) -> dict:
    """병렬 노드가 같은 dict 필드의 서로 다른 key를 동시에 갱신할 때 쓰는 리듀서.

    LangGraph 기본(마지막 쓰기 우선)은 같은 superstep에 두 노드가 쓰면 InvalidUpdateError를
    내므로, key 단위로 병합한다. 같은 key는 나중 값이 이긴다.
    """
    merged = dict(existing or {})
    merged.update(new or {})
    return merged


def dedupe_references(existing: list[dict], new: list[dict]) -> list[dict]:
    """references 리듀서: (source, page) 기준 중복 제거."""
    combined = existing + new
    seen: set = set()
    result: list[dict] = []
    for ref in combined:
        key = (ref.get("source"), ref.get("page"), ref.get("url"))
        if key in seen:
            continue
        seen.add(key)
        result.append(ref)
    return result


def dedupe_evidence_items(existing: list[dict], new: list[dict]) -> list[dict]:
    """evidence_items 리듀서: document_id/source_url + 페이지·섹션 + 인용문 앞부분 기준 중복 제거.

    재작업으로 같은 Agent가 같은 근거를 다시 가져와도 State가 커지지 않는다(지속성 비용).
    """
    combined = existing + new
    seen: set = set()
    result: list[dict] = []
    for item in combined:
        key = (
            item.get("agent"),
            item.get("technology"),  # 같은 청크라도 기술별 근거로는 따로 센다 (충분도 게이트)
            item.get("document_id") or item.get("source_url"),
            item.get("page_or_section"),
            (item.get("evidence_quote") or "")[:80],
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


class PayloadState(TypedDict, total=False):
    """작업 페이로드 층: 하위 Agent가 생산하고 Supervisor는 존재·충분도만 읽는다."""

    research_question: str
    selected_technologies: dict[str, dict[str, Any]]
    evaluation_rubric: dict[str, Any]

    technical_evidence: dict[str, Any]  # tech_research
    trl_evaluation: dict[str, Any]  # 기술 성숙도
    market_evaluation: dict[str, Any]  # 시장성
    stakeholder_evaluation: dict[str, Any]  # 이해관계자
    domain_evaluation: dict[str, Any]  # 도메인(장문맥) 적용

    # 전체 Agent가 누적 (동시 쓰기 → 중복 제거 리듀서)
    references: Annotated[list[dict[str, Any]], dedupe_references]
    evidence_items: Annotated[list[dict[str, Any]], dedupe_evidence_items]

    synthesis: dict[str, Any]
    faithfulness_check: dict[str, Any]  # claim-evidence 대조 verdict (구조화)
    final_report: str
    report_path: str
    quality_verdict: dict[str, Any]  # 품질 평가 verdict (구조화, Hybrid)


class ControlState(TypedDict, total=False):
    """제어 메타 층: 라우팅·종료·재개에 필요한 최소치. Supervisor와 _worker 래퍼만 쓰고 Agent 본체는 쓰지 않는다."""

    run_id: str  # ★ 상관 키: thread_id = LangSmith run id = 결정 로그 파일명
    step_count: int  # Supervisor 실행 횟수 (종료 가드)
    max_steps: int
    next_nodes: list[str]  # Supervisor 결정 → 조건부 엣지가 읽는 값 (END면 [])
    last_decision: dict[str, Any]  # {step, action, targets, reason, ts} 최신 1건만 (전문은 외부 로그)

    node_status: Annotated[dict[str, str], merge_dict]  # {node: NodeStatus}
    attempts: dict[str, int]  # 실패 재시도 판단용 디스패치 횟수 (Supervisor만 씀)
    errors: Annotated[dict[str, str], merge_dict]  # {node: 최근 에러 메시지}
    error_times: Annotated[dict[str, str], merge_dict]  # {node: 최근 실패 시각} — errors와 같은 키, 형식은 그대로
    rework_counts: dict[str, int]  # 근거 부족에 따른 재작업 요청 횟수 (Supervisor만 씀)
    retry_hints: dict[str, str]  # Supervisor → 하위 Agent 재작업 지시 (Agent 간 직접 통신 금지, 소비 후 Supervisor가 비움)
    sufficiency: dict[str, dict[str, Any]]  # 관점별 근거 충분도 판정 {agent: {sufficient, per_technology, reason}}
    faithfulness_rounds: int  # 검증 실패로 재작업을 요청한 라운드 수 (종료 가드)
    report_revisions: int  # 품질 평가 미달로 보고서를 다시 쓴 횟수
    quality_feedback: str  # Supervisor → report_writer 재작성 지시


class GraphState(PayloadState, ControlState, total=False):
    """그래프가 쓰는 State 스키마 = 페이로드 층 + 제어 층의 합성(키는 두 층이 겹치지 않는다)."""
