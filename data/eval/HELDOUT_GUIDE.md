# Held-out 검색 평가 질문 작성 안내

이 질문셋은 Retriever 결과를 **보기 전에** 원문을 읽고 작성한다. 따라서 `smoke_questions.json`의
점검 결과나 기존 검색 순위를 참고해 정답 청크를 정하면 안 된다.

## 작성 순서

1. 코퍼스 문서에서 질문으로 삼을 근거 문단을 먼저 고른다.
2. 해당 문단의 `chunk_id`, `source`, `doc_type`, `technology`를 기록한다.
3. 그 근거에 답할 수 있는 자연어 질문을 만든다.
4. 다른 팀원이 질문·원문 근거의 일치 여부를 한 번 검토한다.
5. 10~15문항이 모이면 `heldout_questions.json`으로 저장하고, 그 뒤에 처음으로 검색 평가를 실행한다.

## 파일 형식

`data/eval/heldout_questions.json`에 아래 형식의 객체를 배열로 저장한다.

```json
[
  {
    "id": "고유한_질문_ID",
    "question": "원문 근거를 찾기 위한 질문",
    "expected_chunk_id": "원문에서 먼저 확인한_chunk_id",
    "expected_source": "원문_파일명",
    "doc_types": ["technical_paper"],
    "technology": "DeepSeek-V2 MLA 또는 null"
  }
]
```

`expected_chunk_id`가 둘 이상 타당하면, 첫 번째 평가에서는 질문을 더 구체화해 대표 근거 하나를
정한다. 복수 정답을 지원하도록 평가기를 확장할 경우에만 `acceptable_chunk_ids`를 추가한다.
