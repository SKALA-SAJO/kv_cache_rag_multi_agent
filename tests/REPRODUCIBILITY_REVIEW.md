# 병합 후 재현성 점검 결과

- 점검일: 2026-10-07
- 담당: 박성우 — 기본 테스트 및 재현성 검증
- 검증 대상: `agent/supervisor`, 커밋 `8c1f9c690e3610d6102bb9496e8459c2bcadd223` (PR #50 병합)
- 환경: macOS arm64, Python 3.11.11, uv 0.12.17
- 방법: 원격 저장소를 새 임시 디렉터리에 클론. 개인 `.env`·API 키·데이터·검색 색인을 복사하지 않음.

## 재현 명령

```bash
git clone --branch agent/supervisor --single-branch https://github.com/SKALA-SAJO/kv_cache_rag_multi_agent.git
cd kv_cache_rag_multi_agent
uv sync
cp .env.example .env
uv run python -m unittest discover -s tests -v
uv run python app.py --help

# 아래 두 모듈은 실제 API 테스트 본문을 대역으로 교체하므로 비용 없이 실행한다.
RUN_LIVE_TESTS=1 uv run python -m unittest tests.test_live_test_gate tests.test_live_result_contract -v
```

## 확인 결과

| 항목 | 결과 |
|---|---|
| uv.lock 기준 의존성 설치 | 통과 |
| `.env.example` 복사 후 기본 테스트 | 109개 중 108개 통과, 실제 API 라이브 테스트 1개 생략 |
| `config.py`와 환경 예시의 설정 이름·기본값 | 계약 테스트 통과 |
| 원문·청크·검색 색인·출력 기본 경로 | 계약 테스트 통과 |
| `app.py --help` | 정상 종료, `--resume`·`--keep-checkpoints` 출력 |
| SQLite 중단 후 동일 thread 재개 | 통과 |
| 재개 시 완료된 병렬 작업의 중복 실행 방지 | 통과 |
| 대역 Agent app 실행의 체크포인트 정리 로그 | 정확히 1줄 |
| 정리 후 남은 체크포인트 | 최종 1개 |
| 최종 제출 Markdown 목차 | SUMMARY 첫 챕터, REFERENCE 마지막 챕터; 결정 이력 부록은 REFERENCE 앞 |
| 라이브 실행 조건·결과 판정 대역 테스트 | 6개 통과; 미설정·0은 생략, 1은 활성화 |

새 클론에는 `data/processed/chunks.jsonl`과 `vectorstore/index.faiss`가 없는 상태였으며,
위 기본 테스트 및 대역 app 실행은 이 파일 없이 통과했다. 대역 app 검증에서도 실제 그래프와
SQLite 정리 함수를 사용하고, LLM·외부 검색·트레이스 전송·PDF 생성은 대역으로 격리했다.

## 발견 사항과 검증 범위

이번 점검에서 담당 범위의 추가 결함은 발견하지 못했다. 앞서 최종 보고서에서 REFERENCE 뒤에
부록이 붙던 문제를 검출하는 PR #45의 테스트가 현재 출력 수정과 함께 통과하는 것도 확인했다.

이 기록은 위 커밋과 환경의 결과다. 이후 코드 변경 시 다시 검증해야 한다. 실제 OpenAI·Tavily
호출, 코퍼스 다운로드·색인 구축, 실제 PDF 생성·쪽수·내용 검토, 제출용 LangSmith 트레이스 대조는
이번 API 없는 점검에 포함하지 않았다. 따라서 실제 최종 보고서 생성 검증을 대신하지 않는다.
