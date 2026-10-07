"""프로젝트 전역 설정.

.env 파일(.env.example 참고)에서 값을 읽는다. RAG-Design PDF의
B절(설계), Tech Stack 표에 정의된 모델·파라미터를 기본값으로 사용한다.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = BASE_DIR / "prompts"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = ""
    tavily_api_key: str = ""

    generator_model: str = "gpt-5-mini"
    judge_model: str = "gpt-5-mini"

    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    # 기본값은 팀 전체(비-Apple Silicon 포함)에서 항상 동작하는 "cpu". rag/embeddings.py의
    # MODEL_CALL_LOCK이 임베딩·재정렬 호출을 직렬화해 동시 호출 세그폴트를 막으므로,
    # Apple Silicon 사용자는 본인 .env에서만 EMBEDDING_DEVICE=mps로 바꿔 안전하게 GPU를
    # 쓸 수 있다(다만 전체 실행 시간은 로컬 연산보다 LLM API 왕복이 지배적이라 체감
    # 효과는 크지 않을 수 있음). CUDA 환경도 동일하게 .env에서 변경.
    embedding_device: str = "cpu"

    retrieval_top_k_candidates: int = 25
    retrieval_top_k_final: int = 6
    # doc_type/technology로 좁혀 검색할 때 FAISS가 필터 적용 전에 살펴볼 후보 수.
    # 기본(필터 없음) 검색에는 영향 없음 — 필터가 있을 때만 사용된다. 코퍼스가
    # 200페이지 이내로 작게 유지되므로(설계서 B.3) 전체를 커버할 만큼 넉넉히 잡는다.
    retrieval_fetch_k: int = 500

    # ── Supervisor 종료 보장·재작업 예산 (graph/supervisor.py) ──
    # Supervisor 스텝 상한. 정상 경로는 약 10스텝, 재작업이 모두 일어나도 25 이내라 여유를 둔다.
    max_supervisor_steps: int = 30
    # 하위 Agent가 예외로 실패했을 때 재시도 횟수. 초과하면 해당 Agent는 '제외(excluded)'되고
    # 보고서에 정보 부족으로 표기된다 (Fall-back).
    max_failure_retries: int = 1
    # 근거 부족(충분도 게이트·Faithfulness·품질 평가)으로 같은 Agent에 재작업을 요청하는 상한.
    max_rework_per_agent: int = 2
    # Faithfulness 검증 실패로 인한 재작업 "라운드" 상한. Agent별 예산과 별개로, 검증 루프가
    # 스텝 예산을 소진해 보고서·품질 평가 루프에 도달하지 못하는 일을 막는다.
    max_faithfulness_rounds: int = 2
    # 품질 평가 미달 시 보고서 재작성 상한.
    max_report_revisions: int = 2
    # 관점별 근거 충분도 게이트 기준 (Supervisor가 결정론적으로 판정).
    min_evidence_items: int = 3
    min_distinct_sources: int = 2
    # LangGraph 자체 recursion_limit (Supervisor 스텝 상한과 별개의 2차 가드).
    graph_recursion_limit: int = 80

    # ── 보고서 품질 평가 (agents/quality_evaluation.py) ──
    max_report_pages: int = 10
    # 단일 출처 편중 판정: 한 출처가 전체 근거에서 차지하는 비율 상한.
    max_single_source_ratio: float = 0.5
    min_report_citations: int = 8

    # 외부 정보 검색 도구 (시장/이해관계자 평가 Agent, agents/base.py의
    # register_external_search_tool 계약을 만족하는 rag/external_search.py에서 사용)
    external_search_max_results: int = 5

    vectorstore_dir: str = "vectorstore"
    chunks_path: str = "data/processed/chunks.jsonl"
    raw_data_dir: str = "data/raw"
    outputs_dir: str = "outputs"

    @property
    def vectorstore_path(self) -> Path:
        return BASE_DIR / self.vectorstore_dir

    @property
    def chunks_file(self) -> Path:
        return BASE_DIR / self.chunks_path

    @property
    def raw_data_path(self) -> Path:
        return BASE_DIR / self.raw_data_dir

    @property
    def outputs_path(self) -> Path:
        return BASE_DIR / self.outputs_dir


settings = Settings()
