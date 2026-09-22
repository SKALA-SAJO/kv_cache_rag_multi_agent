"""외부 정보 검색 도구 등록: Tavily Search API.

팀 합의 기준([[justify_selections_structurally]]): LangChain 공식 통합, 구조화된 JSON
반환, 무료 티어로 팀 전원이 각자 키를 발급받아 재현 가능한지 — 이 세 가지 구조적
기준으로 후보를 비교해 Tavily를 선정함(벤치마크 점수 기준 아님).

agents/base.py의 register_external_search_tool(BaseTool) 계약에 맞춰 TavilySearch를
등록한다. TavilySearch는 이미 langchain_core.tools.BaseTool이고, 응답이
{"results": [{"title","url","content",...}]} 형태라 agents/base.py의
_normalize_search_results가 별도 어댑터 없이 그대로 처리한다.
"""

from __future__ import annotations

import os

from langchain_tavily import TavilySearch

from agents.base import register_external_search_tool
from config import settings


def register() -> None:
    """TAVILY_API_KEY가 설정된 경우에만 외부 검색 도구를 등록한다.

    키가 없으면 아무것도 등록하지 않는다 — get_external_search_tool()이 계속 None을
    반환하므로 시장/이해관계자 평가 Agent는 검색 없이 정직하게 '정보 부족'으로
    처리한다(근거를 지어내지 않는다는 기존 원칙 유지).

    TavilySearch에는 ``api_key`` 생성자 인자가 없다(내부 TavilySearchAPIWrapper가
    os.environ["TAVILY_API_KEY"]를 직접 읽는다) — 그래서 os.environ에 명시적으로
    반영한다. app.py가 load_dotenv()를 먼저 호출해두면 이미 채워져 있어 사실상
    no-op이지만, register()를 다른 진입점에서 단독 호출해도 안전하도록 명시했다.
    """
    if not settings.tavily_api_key:
        return
    os.environ.setdefault("TAVILY_API_KEY", settings.tavily_api_key)
    tool = TavilySearch(max_results=settings.external_search_max_results)
    register_external_search_tool(tool)
