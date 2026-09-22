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
    reranker_model: str = "BAAI/bge-reranker-base"
    # LangGraph가 4관점 평가 Agent를 병렬(멀티스레드)로 실행하는데, 그중 기술 성숙도·도메인
    # 평가 Agent가 동시에 reranker/embedding을 호출한다. PyTorch의 MPS(Apple GPU) 백엔드는
    # 동시 추론이 스레드 안전하지 않아 세그폴트가 발생하므로 기본값은 "cpu"로 둔다.
    # CUDA 환경 등에서는 .env에서 변경 가능.
    embedding_device: str = "cpu"

    retrieval_top_k_candidates: int = 25
    retrieval_top_k_final: int = 6

    max_verification_retries: int = 2

    # 외부 정보 검색 도구 (시장/이해관계자 평가 Agent 전용, rag/external_search.py)
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
