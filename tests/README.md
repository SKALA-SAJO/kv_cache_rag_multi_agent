# 테스트 안내

## 기본 자동 테스트 (API 비용 없음)

```bash
uv run python -m unittest discover -s tests -p "test_*.py" -v
```

검사 대상은 State 리듀서, Supervisor 라우팅 정책·그래프 동적 실행(재작업·Fall-back·종료 보장), 품질 평가 규칙, 코퍼스 계획, 500/50 토큰 청킹, 표·그림 캡션
분리, Retriever 필터, 외부 검색 tool-calling 결과 변환, 보고서 SUMMARY/REFERENCE 계약,
프로젝트 구조다.

`test_state_workflow.py`의 체크포인트 테스트는 대역 관점 Agent에서 `KeyboardInterrupt`로
실행을 중단한 뒤 임시 SQLite DB를 닫고 다시 연다. 새 그래프에서 같은 `thread_id`로
`invoke(None, ...)`를 호출하여 보고서·품질 평가·종료까지 재개되는지, 이미 완료한 기술
조사를 반복하지 않는지, 다른 thread에 상태가 섞이지 않는지 확인한다. 실제 API를 호출하지 않는다.
병렬 관점의 완료 결과가 SQLite에 저장된 뒤 중단시키며, 재개 시 완료된 기술 성숙도·이해관계자
관점은 반복하지 않고 시장성 관점은 충분도 정책에 따른 재작업만 수행하는지 호출 횟수로 검증한다.

`test_checkpoint_prune.py`는 임시 SQLite DB에서 정상 종료한 run을 정리한 뒤 마지막 State가 그대로인지,
다른 thread를 건드리지 않는지, Ctrl+C로 끊긴 run은 정리 대상이 아니며 재개가 끝까지 가는지 확인한다.

보고서 저장 테스트는 LLM과 Retriever를 모두 대역으로 교체하여 로컬 데이터·색인·모델 없이 실행한다.

`test_live_test_gate.py`는 라이브 테스트 모듈을 격리해 로드하고, `RUN_LIVE_TESTS` 미설정·`0`이면
생략되고 `1`이면 실행되는지 unittest 실행 결과로 확인한다. 테스트 본문은 대역으로 교체하므로
활성화 조건 검증에서도 실제 API를 호출하지 않으며, 환경 변수와 기존 discovery 모듈은 유지한다.

`test_environment_contract.py`는 `.env.example`의 설정 이름·기본값이 `config.py`와 맞는지,
원문·청크·색인·출력 기본 경로가 다운로드 및 보고서 스크립트와 일치하는지 확인한다.
`test_app_cli.py`는 대역 Agent로 실제 `app.main()`과 SQLite 그래프를 실행한다. 정상 종료 시
`[app] 체크포인트 정리:` 로그가 정확히 한 줄 나오고 마지막 체크포인트만 남는지 확인하며,
`--keep-checkpoints`에서는 정리 로그 없이 이력이 유지되는지 확인한다. API·PDF 생성은 대역으로 격리한다.

## 실제 API 통합 테스트 (비용 발생 가능)

`.env`에 `OPENAI_API_KEY`, `TAVILY_API_KEY`가 설정되어 있고 RAG 색인이 만들어진 경우에만
실행한다.

```bash
RUN_LIVE_TESTS=1 uv run python -m unittest tests.test_integration_live -v
```

이 테스트는 실제 OpenAI·Tavily를 호출하고 `outputs/`에 Markdown 보고서를 만든다.
Supervisor 구조에 맞춰 `quality_verdict`와 종료 상태(`next_nodes == []`)도 확인한다.
품질 예산 소진으로 미달 상태에서 종료할 수도 있으므로 종료 여부와 품질 통과 여부는 구분한다.
최종 품질 노드의 `done` 상태도 확인하여, 평가 오류로 이전 보고서의 verdict만 남은 종료를 거부한다.
`test_live_result_contract.py`는 위 결과 판정을 대역 그래프와 임시 파일로 검증한다. 완료된 품질
통과·미달은 허용하고, `failed`·`excluded` 상태에 남은 이전 verdict는 거부하며 실제 API는 호출하지 않는다.

## Retrieval 평가 (Hit@K / MRR)

평가 질문셋은 역할에 따라 분리한다.

- `data/eval/smoke_questions.json`: 검색 순위를 확인하며 정리한 점검용 질문셋이다. 구현이
  깨지지 않았는지를 확인하는 회귀 테스트에만 사용한다.
- `data/eval/heldout_questions.json`: 검색 결과를 보기 전에 원문 근거부터 지정한 15문항의
  성능 평가용 질문셋이다. 추가·수정 시에는 `data/eval/HELDOUT_GUIDE.md`를 따른다.

색인을 만든 뒤 아래 명령으로 점검용 질문셋의 Hit@1, Hit@3, Hit@5, MRR을 계산한다.

```bash
uv run python -m tests.evaluate_retrieval --dataset smoke
```

Held-out 질문셋을 작성한 뒤에는 아래 명령으로 성능 지표를 측정한다.

```bash
uv run python -m tests.evaluate_retrieval --dataset heldout
```

모델·청킹·코퍼스가 바뀔 때마다 두 결과를 기록한다. 보고서의 검색 성능 지표에는 held-out 결과만
사용한다.

## Generation 평가 (Faithfulness / Answer Relevance)

`data/eval/generation_cases.json`의 고정 질문·필수 관점(Rubric)을 기준으로 전체 Graph를 실행한다.
기존 검증 Agent의 claim 통과율을 **Faithfulness**로 집계하고, 별도 Judge가 최종 보고서의
필수 관점 충족도를 **Answer Relevance 1~5점**으로 판정한다. 생성 문장을 정답지와 비교하지
않고, 설계서의 근거·한계·조건부 비교 원칙을 Golden Rubric으로 평가한다.

실제 OpenAI·Tavily 호출과 보고서 생성을 수행하므로 비용이 발생할 수 있다.

```bash
# 3개 Generation 평가 사례 전체 실행
uv run python -m tests.evaluate_generation

# 비용을 줄여 한 사례만 실행
uv run python -m tests.evaluate_generation --case-id four_perspective_comparison
```

결과는 `outputs/generation_eval_YYYYMMDD_HHMMSS.json`에 저장된다. 이 파일의 평균
Faithfulness와 평균 Answer Relevance를 Generation 평가 지표로 기록한다.
