"""외부 정보 검색 도구 (sample.pdf Agent 정의 표: 시장 평가·이해관계자 평가 Agent 전용).

Tavily Search API를 사용한다. 팀 합의 기준([[justify_selections_structurally]]):
LangChain 공식 통합, 구조화된 JSON 반환, 무료 티어로 팀 전원이 각자 키를 발급받아
재현 가능한지 — 이 세 가지 구조적 기준으로 후보를 비교해 선정함(벤치마크 점수 기준 아님).

rag/retriever.py(색인된 로컬 코퍼스 검색)와 달리, 여기는 실시간 웹 검색 결과를
반환한다. 결과는 agents/base.py의 search_results_to_evidence_items로 evidence_items
State에 등록한다.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_tavily import TavilySearch

from config import settings


@lru_cache(maxsize=1)
def _tool() -> TavilySearch:
    return TavilySearch(max_results=settings.external_search_max_results, api_key=settings.tavily_api_key)


def search(query: str) -> list[dict]:
    """Tavily로 검색하고 {"title", "url", "content"} 형태로 정규화된 결과를 반환한다.

    TAVILY_API_KEY가 없으면 조용히 빈 리스트를 반환하지 않고 예외를 올린다 — 그래야
    Agent가 "검색했지만 근거가 없었다"와 "검색 자체가 설정되지 않았다"를 구분해서
    limitations에 정확히 기록할 수 있다.
    """
    if not settings.tavily_api_key:
        raise RuntimeError(
            "TAVILY_API_KEY가 설정되지 않았습니다. .env에 TAVILY_API_KEY를 추가하세요."
        )
    response = _tool().invoke({"query": query})
    results = response.get("results", []) if isinstance(response, dict) else []
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
        }
        for r in results
    ]
