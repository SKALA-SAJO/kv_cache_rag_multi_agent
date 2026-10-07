"""KV Cache RAG 프로젝트의 재현 가능한 자동 테스트 모음.

테스트 격리 (tests 패키지가 처음 import될 때 1회 적용 — `test__isolation.py`가 discovery에서 가장 먼저
로드되도록 이름을 정해, `unittest discover -s tests`처럼 모듈이 최상위로 로드되는 경우에도 먼저 실행된다):
  - LangSmith 트레이싱 off: 대역 Agent로 돌리는 그래프·검색 테스트가 실제 프로젝트(LANGSMITH_PROJECT)에
    섞이면 제출용 실행 트레이스와 구분하기 어렵다. 환경변수가 아니라 langsmith 전역 설정으로 끄므로,
    이후 app.py 등이 .env(LANGSMITH_TRACING=true)를 override로 다시 읽어도 켜지지 않는다.
  - 산출물 경로를 임시 폴더로: outputs_dir를 따로 patch하지 않은 테스트(예: Supervisor 정책 테스트의
    fallback_exclude 로그)가 실제 outputs/traces/test-run.jsonl을 늘리던 문제를 막는다. config를 바로
    import해 설정을 고정하므로 이후 .env 재적용(OUTPUTS_DIR=outputs)의 영향을 받지 않는다.
    `python -m tests.evaluate_generation` 같은 평가 CLI는 결과를 outputs/에 저장해야 하므로
    테스트 러너(unittest/pytest)가 떠 있을 때만 적용한다.
"""

import os
import sys
import tempfile

import langsmith

os.environ["LANGSMITH_TRACING"] = "false"
langsmith.configure(enabled=False)

if "unittest" in sys.modules or "pytest" in sys.modules:
    os.environ["OUTPUTS_DIR"] = tempfile.mkdtemp(prefix="kv-cache-tests-")
    import config  # noqa: E402,F401  설정을 임시 경로로 고정
