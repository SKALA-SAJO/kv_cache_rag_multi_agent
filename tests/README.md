# 테스트 안내

## 기본 자동 테스트 (API 비용 없음)

```bash
uv run python -m unittest discover -s tests -p "test_*.py" -v
```

검사 대상은 State 리듀서, Graph 분기, 코퍼스 계획, 500/50 토큰 청킹, 표·그림 캡션
분리, Retriever 필터, 외부 검색 tool-calling 결과 변환, 보고서 SUMMARY/REFERENCE 계약,
프로젝트 구조다.

## 실제 API 통합 테스트 (비용 발생 가능)

`.env`에 `OPENAI_API_KEY`, `TAVILY_API_KEY`가 설정되어 있고 RAG 색인이 만들어진 경우에만
실행한다.

```bash
RUN_LIVE_TESTS=1 uv run python -m unittest tests.test_integration_live -v
```

이 테스트는 실제 OpenAI·Tavily를 호출하고 `outputs/`에 Markdown 보고서를 만든다.

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
