"""KV Cache RAG 프로젝트의 재현 가능한 자동 테스트 모음."""

import os

# 기본 테스트는 LangSmith로 트레이스를 보내지 않는다. 대역 Agent로 돌리는 그래프·검색 테스트가
# 실제 프로젝트(LANGSMITH_PROJECT)에 섞이면 제출용 실행 트레이스와 구분하기 어렵다.
# tests 패키지의 어떤 모듈보다 먼저 실행되며, .env를 다시 읽는 모듈(evaluate_generation)도 이 값을 유지한다.
os.environ["LANGSMITH_TRACING"] = "false"
