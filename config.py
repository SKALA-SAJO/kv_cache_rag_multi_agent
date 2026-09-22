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

    generator_model: str = "gpt-4.1-mini"
    judge_model: str = "gpt-4.1-mini"

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

    max_verification_retries: int = 2

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
