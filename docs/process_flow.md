# 처리 흐름 — 무엇을 쓰고 무엇을 일부러 안 쓰는가

`conv-data`(대화 로그)와 `filter-data`(필터)가 들어와 `case_id`가 붙기까지의 경로.
단계마다 **목적 · 입력 · 출력**으로 적었다.

읽을 때 중요한 것은 각 단계가 무엇을 받는가가 아니라 **무엇을 일부러 안 받는가**다.
판정자에게 정보를 더 주면 정확해질 것 같지만 이 파이프라인에서는 반대다 —
정보를 더 주면 **원인이 아니라 인상을 판정하게 된다.** 그래서 단계마다 입력을 잘라낸다.

```bash
python <저장소>/src/run.py \
    --conv-data   data/conv_eval.json \
    --filter-data data/filter.json \
    --output-dir  outputs
```

---

## 무엇으로 분류되나 — case 30개

`taxonomy_v2.txt` 의 **29개** 중 **25개**에 라우팅이 도달하고, 여기에 우리가 더한
`case0`(정상)이 붙어 실제로 나올 수 있는 라벨은 **26종**이다. `✗` 넷은 로그에 필드가
없어 판정할 수 없다 — 목록에서 지우지 않은 이유는 "우리 분류에는 그런 게 없다"가
되지 않게 하기 위해서다.

라벨은 증상이 아니라 **누가 고치는가**로 묶인다. 같은 증상도 문서팀이 고칠 일과
프롬프트 담당이 고칠 일은 다르다.


**TYPE0 · 실패가 아님** — 고칠 곳: **필터** (챗봇이 아니다)

| | case | 이름 | 신뢰도 |
|---|---|---|---|
|  | `case0` | 정상 — 불만 아님 | medium |

`taxonomy_v2.txt` 에 없는 유일한 라벨이다. 필터는 재현율 쪽으로 넓게 잡으므로 그냥
다음을 묻는 턴이 섞여 들어온다. 낼 자리가 없으면 그게 `content_missing` 으로 읽혀
`case20` 을 부풀리고, 그 숫자가 코퍼스 보강 목록이 되어 문서팀이 **쓸 필요 없는
문서**를 쓴다.

**TYPE1 · 적절하지 않은 질문/요청** — 고칠 곳: 질문 유도 · UI

| | case | 이름 | 신뢰도 |
|---|---|---|---|
|  | `case1` | 이해하기 어려운 질문 | medium |
|  | `case2` | 지원하지 않는 포맷 요구 | medium |
|  | `case3` | 복합 질문을 함 | medium |
|  | `case4` | 참조가 모호한 질문 | medium |
| ✗ | `case5` | 컨텍스트 길이 초과 | medium |
|  | `case6` | 질문에 개인정보 포함 | high |

**TYPE2 · 서비스 안정성 문제** — 고칠 곳: 인프라

| | case | 이름 | 신뢰도 |
|---|---|---|---|
| ✗ | `case7` | 응답 지연으로 이탈 | medium |
|  | `case8` | 출력 잘림 | high |
|  | `case9` | 서비스 자원 부족 응답 | high |

**TYPE3 · 사용자의 의도를 파악하지 못함** — 고칠 곳: 생성 프롬프트 · 후처리

| | case | 이름 | 신뢰도 |
|---|---|---|---|
|  | `case10` | 요구 언어 불이행 | high |
|  | `case11` | 요구 길이 불이행 | medium |
|  | `case12` | 요구 포맷 불이행 | high |
|  | `case13` | 의도와 다른 답변 | medium |
|  | `case14` | 이전 턴 맥락 상실 | medium |
|  | `case15` | 복합 질문 일부만 답변 | medium |
|  | `case16` | 말투·어조 불이행 | medium |
|  | `case17` | 두루뭉술한 답변 | medium |

**TYPE4 · 할루시네이션 답변** — 고칠 곳: 생성 프롬프트

| | case | 이름 | 신뢰도 |
|---|---|---|---|
|  | `case18` | 문서와 어긋나는 주장 | medium |
| ✗ | `case19` | 응답 일관성 문제 | medium |

**TYPE5 · 도메인 관련 Retrieve Context 문제** — 고칠 곳: 검색기 · 문서 · 생성

| | case | 이름 | 신뢰도 |
|---|---|---|---|
|  | `case20` | Retrieve 실패 | medium |
|  | `case21` | Retrieve 미수행 | high |
|  | `case22` | Retrieve 성공, 생성 실패 | medium |
| ✗ | `case23` | 구 문서 Retrieve | medium |
|  | `case24` | 출처/인용 표기 오류 | high |

**TYPE6 · 일반 질문** — 고칠 곳: 모델 자체 · 도구 연동

| | case | 이름 | 신뢰도 |
|---|---|---|---|
|  | `case25` | 상식 질문 오답 | low |
|  | `case26` | 계산 오답 | high |
|  | `case27` | 코드/도구 사용법 오답 | medium |

**TYPE7 · 보안/정책 제한** — 고칠 곳: 권한 정책 · 입력 방어

| | case | 이름 | 신뢰도 |
|---|---|---|---|
|  | `case28` | 보안 정책상 답변 불가 | medium |
|  | `case29` | 간접 프롬프트 인젝션 | medium |

`✗` 는 라우팅이 도달하지 않는다. 필요한 필드 — case5: 모델 컨텍스트 상한값 ·
case7: 서버 응답 시간 · case19: 로그 전체 훑기 · case23: 청크의 문서 ID·개정일.

신뢰도는 **무엇이 판정을 결정하는지**에 따른다. `high` 는 코드로 검증되고,
`medium` 은 LLM 판정에 인용 강제가 걸리며, `low`(case25 하나뿐)는 판정자의
사전지식에 의존한다 — **같은 무게로 집계하면 안 된다.**

---
## 전체 그림

```
conv-data ─┐
           ├─▶ ① 파싱·짝짓기 ─▶ ② 필터 ─▶ ③ Case ─┬─▶ ④ 서비스오류?  코드 ─▶ case9 (끝)
filter-data┘                                        │
                                                    ├─▶ ⑤ Step 1 관측    LLM  ← rag 안 봄
                                                    ├─▶ ⑥ 코드 검증기 11종 코드
                                                    ├─▶ ⑦ Step 2 충족도  LLM  ← 답변 안 봄
                                                    ├─▶ ⑧ 인용 대조      코드
                                                    ├─▶ ⑨ Step 3 근거활용 LLM ← 질문 안 봄
                                                    └─▶ ⑩ 라우팅         코드 ─▶ case
                                                                                  │
                                          outputs/conv_parsed.json ◀──────────────┤
                                          outputs/run_summary.txt ◀───────────────┘
```

| | LLM | 언제 도나 |
|---|---|---|
| ④ 서비스 오류 | — | 항상. 걸리면 여기서 끝 |
| ⑤ 관측 | **1회** | 항상 |
| ⑥ 코드 검증기 | — | 항상 |
| ⑦ 충족도 | **1회** | 도메인 질문 + 내용 불만 + 청크 있음 |
| ⑧ 인용 대조 | — | ⑦이 돌았을 때 |
| ⑨ 근거 활용 | **1회** | ⑦이 sufficient 이고 인용이 살아남았을 때 |
| ⑩ 라우팅 | — | 항상 |

**LLM 은 최대 세 번**이고 대부분의 턴은 한 번(⑤)으로 끝난다.
**case 는 LLM 이 고르지 않는다** — ⑩이 코드로 정한다.

---

## ① 파싱과 짝짓기

**목적** — 시간순 로그에서 "불만이 표현된 턴"을 찾아, **그 불만이 겨냥한 답변과 문서**를 짝지어 `Case` 를 만든다.

**입력** — `conv_eval` JSON (users → conversations → turns)

**출력** — 턴마다 `Case` 하나. 짝을 한 칸이라도 틀리면 그 뒤가 전부 무의미해진다.

```
turn 3   Q: 연차 이월 예외 조건이 뭔가요?
         A: 연차는 사규에 따라 운영됩니다.            ← 비판받은 답변
         retrieved: [청크1, 청크2]                    ← 그 답변에 주어졌던 문서
turn 4   Q: 예외 조건을 물었는데요.                    ← 불만 (이 턴이 분석 대상)
         A: ...
         retrieved: [청크3]                           ← 이건 쓰지 않는다
```

turn 4 가 지목되면 판정 대상은 **turn 3 의 답변과 turn 3 의 문서**다.
turn 4 의 `retrieved_data` 를 쓰면 "불만에 대한 재검색 결과"를 보게 되어 완전히 다른 것을 잰다.

| Case 필드 | 어디서 |
|---|---|
| `pre_queries` | turn 3 까지의 질문들, 최근 3개 (`--history-turns`) |
| `llm_ans_on_last_q` | **turn 3** 의 `llm_response` |
| `current_query` | turn 4 의 `user_question` |
| `rag_chunks` | **turn 3** 의 `retrieved_data` |

`pre_queries` 를 3개로 자르는 이유: 대명사 해소("그거 다시")에 필요한 건 보통 직전 2~3턴이고,
더 오래된 질문은 노이즈라 판정자가 엉뚱한 주제로 끌려간다. 잘라내도 **비판받은 답변을 부른
질문은 항상 남는다** — 그게 마지막 항목이다.

### 여기서 버리는 것

| 필드 | 왜 안 쓰나 |
|---|---|
| `timestamp` | 지연 판정(case7)에 쓸 수 없다. 턴 시각 차이에는 사용자가 생각한 시간이 섞여 있다 |
| `prev_question` | turn 순서로 직접 짝지으므로 불필요. 운영 환경로그에서는 `list` 로 온다 — 그게 서비스가 모델에 실제로 넘긴 히스토리라면 재확인이 필요하다 |
| `trace_matched` | "2턴 이상"은 실제 턴 수로 판정한다. 선언값과 실제가 어긋난 로그를 본 적이 있다 |
| `llm_eval_*` `llm_emotion_*` | ②에서만 쓰고 판정에는 넘기지 않는다 (아래) |

---

## ② 필터

**목적** — **무엇을 볼지 고르기만 한다.** 판정이 아니다.

**입력** — `filter-data` JSON + ①의 후보 턴들

| 필터가 보는 것 | 로그 필드 |
|---|---|
| 직급 · 부서 · 직무 · 직위 | `job_grade` `db_dept_name` `db_job_name` `db_position_name` |
| 턴 구간 | `turn` |
| 기간 | `timestamp` |
| 대화 유형 점수 · 라벨 | `llm_eval_result` `llm_eval_score` `llm_alternatives` |
| 감정 점수 · 라벨 | `llm_emotion_result` `llm_emotion_score` `llm_emotion_alternatives` |

**출력** — 선별된 턴 + **단계별 탈락 기록**

```
진단 가능 후속 턴 (2턴 이상 대화)   73
직급(role)                          58   (-15)
부서                                42   (-16)
턴 구간                             42
eval_score (0.0, 50.0)              37   (-5)
emotion_score (0.0, 30.0)           37
```

몇 건이 어디서 빠졌는지 전부 찍는다. 조용히 줄어들면 "이 조건에는 원래 없다"와
"필터가 잘못 걸렸다"를 구분할 수 없다.

### 필터가 쓴 값을 판정에 넘기지 않는 이유

`llm_eval_result`(질의 유형 라벨)와 `llm_emotion_result`(감정 라벨)는
**이미 다른 시스템이 내린 판정**이다. 이걸 ⑤에 넘기면 판정자가 그 라벨에 맞춰
관측을 고른다 — 독립적인 두 번째 의견이어야 할 것이 첫 번째 의견의 확인 도장이 된다.

필터는 "어떤 턴을 볼까"만 정하고, "그 턴에 무슨 일이 있었나"는 처음부터 다시 본다.

> 운영 장비에는 필터가 이미 구현돼 있다. 이 저장소의 `filters.py` 는 여기서 검증할 때만
> 쓴다. 경계는 `Case` 이고, `Case` 를 만들어 넣을 수만 있으면 ③부터는 그대로 돈다.

---

## ④ 서비스 자원 부족 — LLM 보다 먼저, 코드로

**목적** — 모델이 답을 만든 적조차 없는 턴을 **판정 전에** 걸러낸다.

**입력** — `llm_ans_on_last_q` 하나. 확정 문구 목록과 대조한다 (설정으로 바꾼다).

**출력** — 걸리면 `case9`, **LLM 호출 0회로 종료**. 안 걸리면 ⑤로 넘어간다.

```
llm_ans_on_last_q == "서비스에 문제가 있거나, 사용자 분들이 많아서
                      서버에 부하가 걸리고 있어요…"
   → case9 · 신뢰도 high · evidence 에 checks 만 남고 observation 은 없다
```

**⑤보다 앞에 두는 이유**: 판정할 답변이 아예 없다. 관측을 돌리면 판정자가 이 문구를
거절로 읽어 case28(보안 정책상 답변 불가)로 보낸다. **서버 자원 문제가 보안 정책 문제로
집계되면 고칠 곳을 정반대로 가리킨다** — 한쪽은 인프라이고 다른 쪽은 권한 정책이다.

---

## ⑤ Step 1 · 관측 추출

**목적** — 사용자가 **무엇을 원했고 무엇을 못 받았는지**를 관측 사실로만 기록한다.
case 를 고르지 않는다.

**입력**

| 주는 것 | 안 주는 것 |
|---|---|
| `pre_queries` — 이전 질문들, 최근 3개 | **`rag_chunks`** ← 이게 핵심 |
| `llm_ans_on_last_q` — 비판받은 답변 | `dept` · `job_grade` · `job_name` |
| `current_query` — 그 답변에 대한 불만 | `llm_eval_*` · `llm_emotion_*` |
| | `turn` · `timestamp` |

**왜 문서를 감추나.** 문서를 먼저 보여주면 판정자가 "이 문서로 답할 수 있었나"를 기준으로
사용자의 요구를 재구성한다. 문서에 있는 내용 쪽으로 `unmet_need` 가 끌려가고, 그러면
"문서가 충분했다"는 결론이 거의 자동으로 나온다. 순환이다.
사용자가 무엇을 원했는지는 **사용자의 말만 보고** 정해야 한다.

**출력** — 관측 17개

| 필드 | 값 | 무엇을 |
|---|---|---|
|---|---|---|
| `reasoning` | 자유 문장 | 불만과 질문을 어떻게 읽었는지 2~3문장 |
| `resolved_question` | 자유 문장 | 대명사·생략을 푼, 그 자체로 이해되는 질문 |
| `unmet_need` | 자유 문장 | 원했는데 못 받은 것. **요구하지 않은 것을 덧붙이지 않는다** |
| `complaint_target` | `none` `tone` `format` `language` `length` `content_missing` `content_wrong` `no_answer` `refusal` `inconsistency` `other` | 불만이 무엇을 향하나. `none` 은 불만이 아니라는 뜻 |
| `complaint_quote` | 후속 발화에서 따온 구절 | 그렇게 읽은 근거. `none` 일 때 필수이며 원문 대조를 거친다 |
| `question_domain` | `domain` `general_knowledge` `calculation` `code` `tool_usage` `unclear` | 질문의 성격 |
| `question_self_contained` | `true` / `false` | 그 문장만으로 검색 쿼리가 되나 (case4의 반대) |
| `question_multi_intent` | `true` / `false` | 요구가 둘 이상 섞였나 (case3) |
| `answer_refused` | `true` / `false` | 정책·권한을 이유로 거절했나 (case28) |
| `question_answerable_as_asked` | `true` / `false` | 질문이 답을 특정할 만큼 분명한가 (case1) |
| `answer_covers_all_intents` | `true` / `false` | 복합 질문의 모든 요구를 다뤘나 (case15) |
| `answer_actionable` | `true` / `false` | 다음에 무엇을 할지 알 수 있나 (case17) |
| `answer_used_history` | `not_needed` `used` `ignored` | 이전 턴을 이어받았나 (case14) |
| `requests_unsupported_output` | `true` / `false` | 낼 수 없는 형태를 요구했나 (case2) |
| `requested_language` | `ko` `en` `ja` `zh` · 없으면 빈 문자열 | 요구 언어 |
| `requested_length_kind` | `none` `max_chars` `max_sentences` `max_lines` `vague_short` | 길이 요구의 종류 |
| `requested_length_value` | 정수 · 수치가 없으면 `0` | 길이 요구의 값 |
| `requested_format` | `none` `numbered_list` `bullet_list` `table` `code_block` `json` `prose` | 요구 형식 |

`requested_*` 셋은 **관측이지 판정이 아니다.** "요구했다"까지만 적고 "지켰나"는 ⑥이 코드로 본다.
`vague_short`("짧게 답해줘")처럼 수치가 없는 요구는 임계값(기본 400자)으로 재고,
그 임계값은 설정으로 바꿀 수 있다.

29지선다 대신 좁은 질문 17개로 나눈 이유가 둘이다. 어떤 모델이든 29지선다는 정확도가
안 나오고, 무엇보다 **판정자가 원인을 먼저 정하고 사실을 끼워 맞추는 것**을 막는다.

### 17개에 대해 헷갈리기 쉬운 세 가지

**(1) 모든 턴이 ⑤를 거치지는 않는다.**

| 어떤 턴 | observation |
|---|---|
| 보통의 턴 | 17개 전부 |
| `case9` (④에서 코드로 끊김) | **없음.** LLM 을 한 번도 안 불렀다 |
| 판정 실패(`error`) | 없음 |

`case9` 인 턴의 `evidence` 에는 `checks` 만 있고 `observation` 이 없다. 그게
"LLM 을 안 거쳤다"는 표시이기도 하다.

**(2) 18개를 내지만 결과 파일에는 8개만 실린다.**

```
Step 1 이 내는 것        18개 (complaint_quote 만 기본값이 있고 나머지는 필수)
결과 파일에 실리는 것      8개
```

실리는 8개 —

| 필드 | 왜 남기나 |
|---|---|
| `resolved_question` `unmet_need` | 무엇을 물었고 무엇을 못 받았나 |
| `complaint_target` `question_domain` | 불만의 방향과 질문 성격 |
| `complaint_quote` | 판정자가 그 불만 방향을 고른 근거. 인용 검증 결과와 함께 남는다 |
| `question_self_contained` `question_multi_intent` `answer_refused` | 라우팅의 주요 갈림길 |

나머지 10개(`answer_actionable` · `answer_used_history` · `answer_covers_all_intents` ·
`question_answerable_as_asked` · `requests_unsupported_output` · `requested_language` ·
`requested_length_kind` · `requested_length_value` · `requested_format` · `reasoning`)는
**⑥ 검증기와 ⑩ 라우팅이 쓰고 버린다.** 결과는 `checks` 와 `case_id` 에 반영돼 있다.

> **한계**: 그래서 `case12`(요구 포맷 불이행)가 나왔을 때 **무슨 포맷을 요구했는지**를
> 결과 파일에서 알 수 없다. `checks.format` 의 `detail` 에 일부 남지만 요구값 자체는
> 아니다. 운영 환경에서는 결과를 반출할 수 없어 화면과 이 파일이 전부이므로, 되짚을 일이
> 잦아지면 실리는 목록을 늘려야 한다.

**(3) 17개 중 상당수는 "해당 없음" 센티널이다.**

합성 데이터 41턴 기준으로 `answer_refused` 는 41/41 이 `false`,
`question_multi_intent` 는 38/41 이 `false` 다. 실데이터에서는 `requested_*` 도
대부분 `none` · `""` · `0` 일 것이다.

**"17개를 관측했다"가 아니라 "17개 슬롯을 채웠고 대부분은 해당 없음"이다.** 그래서
`not_applicable` 과 `violated` 를 섞으면 안 된다는 ⑥의 규칙이 여기서부터 시작된다.

---

## ⑥ 코드 검증기 11종

**목적** — LLM 없이 되는 것은 LLM 에게 묻지 않는다. 문자열만 보면 아는 것들이다.

**입력** — `Case` 의 텍스트 + ⑤가 관측한 요구값. 표의 이름은 `Case` 의 필드명이다.

| 검증기 | 입력 | 무엇을 |
|---|---|---|
| `service_error` | `llm_ans_on_last_q` | 자원 부족 확정 문구 (case9) |
| `truncated` | `llm_ans_on_last_q` | 출력 잘림 (case8) |
| `language` | `llm_ans_on_last_q` + `requested_language` | 요구 언어 불이행 (case10) |
| `length` | `llm_ans_on_last_q` + `requested_length_*` | **재기만 한다** — 판정하지 않는다 (case11) |
| `format` | `llm_ans_on_last_q` + `requested_format` | 요구 포맷 불이행 (case12) |
| `pii` | **`pre_queries[-1]`** | 질문에 개인정보 포함 (case6) |
| `quoted_spans` | `llm_ans_on_last_q` + `rag_chunks` | 답변이 제시한 인용문의 원문 대조 (case24) |
| `python_syntax` | `llm_ans_on_last_q` | 파이썬 `ast.parse` (case27) |
| `sql_shape` | `llm_ans_on_last_q` | SQL 구조 결함 (case27) |
| `arithmetic` | `llm_ans_on_last_q` | 등식 재계산 (case26) |
| `dates` | `llm_ans_on_last_q` | 없는 날짜·요일 주장 (case26) |
| `injection` | `rag_chunks` + `llm_ans_on_last_q` | 문서의 숨은 지시를 수행했나 (case29) |

`llm_ans_on_last_q` 는 **비판받은 답변**이다 — 불만 턴(N+1)의 답변이 아니라
그 직전 턴(N)의 답변이다. ①의 짝짓기가 그렇게 잡아 놓은 것이다.

`pii` 하나만 답변이 아니라 **직전 질문**(`pre_queries[-1]`, 즉 그 답변을 부른 질문)을
본다 — case6 이 "질문에 개인정보가 포함된 경우"이기 때문이다. 결과 파일에는 종류와
건수만 남기고 원본 값은 남기지 않는다. 안 그러면 결과 파일 자체가 개인정보 사본이 된다.

`current_query`(불만 원문)를 보는 검증기는 **하나도 없다.** 불만은 ⑤가 "무엇을
원했나"를 읽는 데만 쓰이고, 코드 검증은 전부 답변과 문서만 본다.

**출력** — 검증기마다 네 값 중 실제로 나올 수 있는 것이 다르다.

| 검증기 | 낼 수 있는 verdict |
|---|---|
| `pii` | `ok` · `violated` |
| `service_error` `sql_shape` `arithmetic` `format` | `ok` · `violated` · `not_applicable` |
| `dates` | `ok` · `violated` · `not_applicable` · `undetermined` |
| `truncated` | `ok` · `violated` · `undetermined` |
| `language` | `ok` · `violated` · `not_applicable` · `undetermined` |
| `injection` | `violated` · `not_applicable` · `undetermined` |
| `quoted_spans` | `ok` · `not_applicable` · `undetermined` |
| `python_syntax` | `ok` · `not_applicable` |
| `length` | `not_applicable` · `undetermined` |

| verdict | 뜻 | 라우팅이 어떻게 읽나 |
|---|---|---|
| `ok` | 검증했고 위반이 없다 | 그 case 로 보내지 않는다 |
| `violated` | 검증했고 위반이다 | 그 case 로 보낸다 (신뢰도 high) |
| `not_applicable` | **잴 대상이 없었다** | 위반과 섞지 않는다 |
| `undetermined` | 잴 대상은 있는데 코드로 못 정한다 | case 는 유지하되 **신뢰도를 낮춘다** |

### verdict 를 어떻게 만드나 — 규칙 전부

LLM 이 없으니 판정은 전부 **문자열 규칙**이다. 세 갈래로 갈린다.

```
잴 대상이 있나?  ──아니오──▶ not_applicable   (요구 없음 · 코드블록 없음 · 인용 없음 …)
      │ 예
      ▼
코드로 정할 수 있나? ──아니오──▶ undetermined  (언어 판별 실패 · 수치 없는 요구 …)
      │ 예
      ▼
   규칙 통과? ──▶ ok  /  ──▶ violated
```

| 검증기 | `not_applicable` | `undetermined` | `ok` / `violated` 를 가르는 규칙 |
|---|---|---|---|
| `service_error` | 답변이 비어 있음 | — | 확정 문구가 **부분 일치**(공백 제거 후)하면 violated. 없으면 400자 초과 시 ok, 이하일 때 보조 표지 2개 이상이면 violated |
| `truncated` | — | 답변이 비어 있음 | 목록·표·코드블록으로 끝나면 ok → 코드펜스 개수가 홀수면 violated → 종결 부호(`.` `!` `?` …)로 끝나면 ok → 아니면 violated |
| `language` | 언어 요구 없음 | 답변의 언어 판별 실패 | 문자 종류 비율로 판별한 언어(`ko` `ja` `zh` `en`)가 요구와 같으면 ok |
| `length` | 길이 요구 없음 | 수치 요구인데 값이 없음 | `vague_short` 는 400자 초과면 violated. 수치 요구는 글자·문장·줄 수를 세어 비교 |
| `format` | 포맷 요구 없음 | — | 요구한 구조가 **2개 이상** 있으면 ok — 번호 목록·불릿·표(구분선 포함)·코드펜스·JSON 파싱 성공 |
| `pii` | — | — | 정규식 5종(주민번호·휴대전화·이메일·카드·계좌) 중 하나라도 걸리면 violated |
| `quoted_spans` | 답변에 인용도 문서명도 없음 | — | **둘을 따로 본다.** 문장 인용(`"…"` `'…'`)은 청크 **본문**과 최대 연속 일치율 90%. 문서명(`「…」` `《…》`)은 청크 **머리의 출처 표기**와 포함 관계 — 출처 표기가 없으면 대조하지 않고 그렇게 적는다. **청크가 0건인데 인용했으면 violated** |
| `python_syntax` | 파이썬 코드 블록 없음 | — | 블록마다 `ast.parse`. 하나라도 `SyntaxError` 면 violated |
| `sql_shape` | SQL 코드 블록 없음 | — | 괄호 짝, `GROUP BY` 뒤 비어 있음, `SELECT` 인데 `FROM` 없음 — 구조적 결함만 본다 |
| `arithmetic` | 검증 가능한 등식이 없음 | — | `A + B = C` 꼴을 정규식으로 뽑아 **재계산**. 오차 1% 넘으면 violated |
| `dates` | 답변에 날짜 표기가 없음 | 연도가 없어 윤년·요일을 가를 수 없음 | `M월 D일`·`YYYY-MM-DD` 만 날짜로 본다. 달력에 없으면 violated, 날짜 뒤 요일 주장이 틀려도 violated |
| `injection` | 문서에 지시문 패턴 없음 | 지시문은 있으나 답변에 흔적 없음 | 지시문 앞 20자가 답변에 나타나면 violated |

**세 가지가 이 표의 요점이다.**

`pii` 만 `not_applicable` 이 없다. 질문은 항상 있으므로 잴 대상이 늘 존재한다.

`injection` 만 `ok` 가 없다. 문서에 지시문이 **있다는 것**과 답변이 **수행했다는 것**은
다른 사건이다. 지시문이 없으면 잴 것이 없어 `not_applicable` 이고, 있는데 수행 흔적이
없으면 "안 했다"고 단정할 수 없어 `undetermined` 다. 업무 규정문은 "~한다" 같은
규범형이 많아 명령형만으로 의심하면 오탐이 쏟아진다.

`truncated` 는 **순서가 규칙의 일부다.** 종결 부호 검사를 앞에 두면 코드 마지막 줄의
`print(1)` 을 정상 종결로 오판한다. 그래서 닫히지 않은 코드펜스를 먼저 본다.

요구가 없었는데 검증하면 `not_applicable` 이 나온다. 그걸 위반과 섞으면 멀쩡한 답변이
전부 실패로 집계되므로 네 값으로 구분한다.

`length` 에 `ok` · `violated` 가 없는 것은 길이 판정이 관측의 요구값에 의존해서다 —
수치 요구는 `undetermined` 로 두고 측정값을 함께 남겨, 나중에 임계값을 바꿔도
LLM 을 다시 돌리지 않아도 되게 했다.

`detail` 에는 **무엇을 보고 그렇게 판정했는지**가 들어간다
(`"종결 부호 없이 끝남: …'그리고 담당자'"`, `"모호한 짧게 요구 (기준 400자) · 실제 812자"`).

---

## ⑦ Step 2 · 충족도

**목적** — **문서가 사용자의 요구를 담고 있었나**를 잰다. 답변의 품질이 아니다.

**언제 도나** — 세 조건이 전부 맞을 때만.

```
question_domain == "domain"                          ⑤가 정한다 (LLM)
   and complaint_target in {content_missing,          ⑤가 정한다 (LLM)
                            content_wrong}
   and rag_chunks 가 비어 있지 않다                    코드
```

형식·언어·길이 불만에 문서 충족도를 따지는 건 무의미하고 호출만 쓴다.
청크가 비면 ⑦을 건너뛰고 `insufficient` 로 두는데, 그 이유는 아래에 있다.

### `domain` 인지 누가 정하나 — ⑤의 LLM 이다

코드가 아니라 **⑤가 관측한 `question_domain`** 이다. `rag_chunks` 유무로 정하지
않는다 — 그러면 "도메인 질문인데 검색이 아예 안 걸린 경우"(case21)를
"도메인 질문이 아니었다"로 읽게 된다.

| 값 | 뜻 | 어디로 |
|---|---|---|
| `domain` | 업무 문서를 찾아야 답할 수 있다 (규정·절차·제도) | ⑦로 |
| `general_knowledge` | 공개된 지식만으로 답할 수 있다 | case25 |
| `calculation` | 수식·날짜·산수 | case26 |
| `code` · `tool_usage` | SQL·Python·Excel·Spotfire | case27 |
| `unclear` | 판별할 수 없다 | 미분류 |

**`domain` 과 `general_knowledge` 의 경계는 "회사마다 답이 달라지는가"로 가른다.**
법령·표준을 언급했더라도 "우리 회사는 어떻게 운영하나"를 묻는 것이면 `domain` 이다.

```
"근로기준법 조문상 연차 발생 요건이 뭔가요"  → general_knowledge  (법이 하나뿐)
"우리 회사 연차는 법정 기준대로 주나요"      → domain            (회사마다 다름)
```

프롬프트는 **애매하면 `domain`** 이라고 지시한다. 공개 지식으로 답할 수 있는지가
분명할 때만 `general_knowledge` 로 둔다. 운영 환경 챗봇에서는 후자가 훨씬 흔하고,
잘못 `general_knowledge` 로 빠지면 신뢰도 `low` 인 case25 로 가서 **검색·문서
문제가 통계에서 사라진다.**

### `complaint_target` 은 둘만 통과한다

`content_missing`(필요한 정보가 없음)과 `content_wrong`(담긴 정보가 틀림)뿐이다.
`none` `tone` `format` `language` `length` `no_answer` `refusal` `inconsistency`
`other` 아홉은 ⑦을 거치지 않는다 — `none` 은 애초에 불만이 아니라 문서 충족도를
따질 이유가 없고(그래서 LLM 호출도 아낀다), 나머지 여덟은 문서가 아니라 답변의
형태를 향해서 ⑥의 코드 검증기가 이미 판정한다.

**입력**

| 주는 것 | 안 주는 것 |
|---|---|
| `resolved_question` — ⑤가 정리한 질문 | **`llm_ans_on_last_q`** ← 이게 핵심 |
| `unmet_need` — ⑤가 정리한 원했던 정보 | `current_query` — 불만 원문 |
| `rag_chunks` — 번호를 붙여서 | `pre_queries` |

**왜 답변을 감추나.** 답변을 같이 보여주면 판정자가 "이 답변이 부실하다 → 그러니 문서도
부실했겠지"로 미끄러진다. 재는 대상이 **문서의 충족도**에서 **답변의 품질**로 바뀐다.
그 둘은 다른 것이고, 갈라야 고칠 곳이 정해진다.

- 문서에 답이 없었다 → 문서를 쓰거나 검색기를 고친다
- 문서에는 있었는데 답변이 안 썼다 → 프롬프트·생성을 고친다

답변을 보여주면 이 갈림길 자체가 사라진다.

**출력**

| 필드 | 값 |
|---|---|
| `verdict` | `sufficient` · `partial` · `insufficient` |
| `evidence` | `[{chunk_index: 0-기반 정수, quote: 청크에서 글자 그대로 복사한 문장}]` · 없으면 `[]` |
| `missing` | 문서에 없어서 답할 수 없었던 것 · `sufficient` 면 빈 문자열 |
| `reasoning` | 자유 문장 |

| verdict | 뜻 |
|---|---|
| `sufficient` | `unmet_need` 전부가 문서에 있다 |
| `partial` | 일부만 있다 — **요구의 어느 부분에 답하는가**로 가른다 |
| `insufficient` | 물어본 것이 없다. 주제가 같고 항목이 다른 near-miss 도 여기다 |

`partial` 과 `insufficient` 의 정의가 한 번 겹쳤다. "관련 문서가 있으면 partial"로 두면
near-miss 가 전부 partial 로 새어 "문서는 어느 정도 있었다"가 된다. 지금은 **인용이
요구의 어느 부분에 답하는가**를 묻는다 — 답하는 부분이 없으면 주제가 같아도 insufficient 다.

### 청크가 비어 있으면 LLM 을 부르지 않는다

대조할 문서가 없으면 verdict 는 물어볼 것 없이 `insufficient` 다. 부르면 호출만 쓰는 게
아니라 **없는 문서에서 인용을 지어낼 표면**이 생긴다. ⑧이 잡아내지만 잡을 일을 만들지 않는다.

---

## ⑧ 인용 대조

**목적** — ⑦이 낸 근거가 **정말 그 문서에 있는 문장인지** 확인한다. 사전지식 오염을 여기서 막는다.

**입력** — ⑦의 `evidence` + 원본 `rag_chunks`. LLM 없음.

**출력**

| 필드 | 값 |
|---|---|
| `kept` | 살아남은 인용 `[{chunk_index, quote, ratio, index_corrected}]` |
| `dropped` | 버려진 인용 `[{quote, reason, best_ratio}]` |
| `n_chunks` | 대조 대상 청크 수 · **0이면 검색 결과가 아예 없었다** |

| `reason` | 뜻 |
|---|---|
| `too_short` | 인용이 8자 미만. 짧으면 아무 문서에나 우연히 들어맞아 검증을 무력화한다 |
| `not_found` | 어느 청크와도 연속 일치 90% 미만. **지어낸 인용이 여기 걸린다** |

**부수 효과** — 인용이 하나도 안 남았는데 verdict 가 `sufficient` · `partial` 이면
`insufficient` 로 **강등한다.**

판정자가 업무 규정을 "아는" 것처럼 답하면(사전지식) 인용할 원문이 없으므로 여기서 걸린다.
**"모르는 건 모른다고 해라"를 프롬프트로 부탁하는 대신 구조로 막는 것**이다.
프롬프트 요청은 모델이 바뀌면 무너지지만 이건 안 무너진다.

`index_corrected` 는 인용은 맞는데 판정자가 청크 번호를 틀린 경우다. 버리지 않고
맞는 번호로 고쳐 살린다 — 번호를 틀린 것과 내용을 지어낸 것은 다른 실수다.

`dropped` 가 결과 파일에 그대로 남는다. **지어낸 인용 건수**가 대시보드의 판정 건강
지표 중 하나이고, 이게 크면 그 배치의 판정을 먼저 의심해야 한다.

---

## ⑨ Step 3 · 근거 활용

**목적** — **답변이 그 문서를 썼는가.** 그것 하나만 묻는다.

**언제 도나** — ⑦이 `sufficient` 이고 ⑧에서 인용이 살아남았을 때만.

**입력**

| 주는 것 | 안 주는 것 |
|---|---|
| `llm_ans_on_last_q` — 비판받은 답변 | `resolved_question` · `unmet_need` |
| `rag_chunks` — 그 답변에 주어졌던 문서 | `current_query` — 불만 원문 |
| | ⑦의 `verdict` |

**왜 질문을 감추나.** 질문을 같이 주면 "이 답변이 질문에 잘 답했나"라는 다른 판단이 섞인다.
그건 이미 ⑦이 다른 각도에서 봤고, 여기서 또 보면 두 판정이 같은 방향으로 쏠려
**독립적이지 않은 두 표**가 된다.

**출력**

| `answer_used_rag` | 뜻 | 어디로 |
|---|---|---|
| `used` | 표현이 달라도 문서 내용을 반영했다 | 문서·답변 다 멀쩡 → case17 / case13 |
| `ignored` | 문서에 있는데 일반론으로 때웠다. **"인사팀에 문의하세요" 같은 회피성 안내도 여기다** | case22 |
| `contradicted` | 문서와 어긋나는 숫자·결론을 말했다 | case18 |

회피성 안내를 거절(case28)로 분류하면 case22(Retrieve 성공, 생성 실패) — 문서엔 답이 있는데 안 쓴 경우 — 가
통계에서 사라진다. 실제로 구현 중 그렇게 새서 `answer_refused` 의 범위를 정책·권한·보안으로 좁혔다.

---

## ⑩ 라우팅

**목적** — 관측과 검증을 **진리표**로 조합해 case 를 정한다. LLM 은 여기 없다.

**입력** — 앞 단계가 낸 것 전부. 다만 **전부를 읽지는 않는다.**

| 어디서 | 라우팅이 읽는 것 | 안 읽는 것 |
|---|---|---|
| ⑤ 관측 (18개 중 **10개**) | `complaint_target` `question_domain` `question_self_contained` `question_multi_intent` `question_answerable_as_asked` `answer_refused` `answer_covers_all_intents` `answer_actionable` `answer_used_history` `requests_unsupported_output` | `resolved_question` `unmet_need` `reasoning` `requested_language` `requested_length_kind` `requested_length_value` `requested_format` `complaint_quote` |
| ⑥ 검증기 (12종 중 **12종**) | `service_error` `truncated` `pii` `quoted_spans` `arithmetic` `dates` `injection` + `language` `length` `format` `python_syntax` `sql_shape` | — |
| ⑦ 충족도 | `verdict` | `evidence` `missing` `reasoning` |
| ⑧ 인용 | `n_kept` `n_chunks` | `kept` `dropped` 의 내용 |
| ⑨ 근거 활용 | `answer_used_rag` | `reasoning` |

**`requested_*` 넷을 라우팅이 안 읽는 것이 이 표의 요점이다.** 그 값은 ⑥이 소비해
`language` · `length` · `format` 검증기의 `verdict` 로 바뀌어 있다. 라우팅은 관측이
아니라 **검증 결과**를 본다 — "요구했다"(관측)가 아니라 "지켰나"(코드)로 판정한다.

`resolved_question` 과 `unmet_need` 도 안 읽는다. 그 둘은 ⑦에 넘기는 입력이자
결과 파일에 남기는 근거이지 판정의 갈림길이 아니다.


**판정 순서** — 위에서부터 내려가다 처음 걸리는 곳에서 끝난다.

| # | 묻는 것 | 무엇이 정하나 | 걸리면 |
|---|---|---|---|
| 0 | 답변이 서비스 자원 부족 확정 문구인가 | 코드 | `case9` 서비스 자원 부족 응답 |
| 1 | 답변이 정책·권한을 이유로 거절했나 | 관측 | `case28` 보안 정책상 답변 불가 |
| 2 | 문서의 숨은 지시를 답변이 수행했나 | 코드 | `case29` 간접 프롬프트 인젝션 |
| 3 | 애초에 불만이 아닌가 | 관측 + 코드 | `case0` 정상 — 불만 아님 |
| 4 | 챗봇이 낼 수 없는 형태를 요구했나 | 관측 | `case2` 지원하지 않는 포맷 요구 |
| 5 | 질문만으로 답을 특정할 수 있나 | 관측 | `case1` 이해하기 어려운 질문 |
| 6 | 답이 없거나 중간에 끊겼나 | 코드 | `case8` 출력 잘림 |
| 7 | 언어·포맷 요구를 지켰나 | 코드 | `case10` `case12` — 지켰는데도 불만이면 `case13`. `case11`(길이)은 코드가 판정하지 않아 언제나 medium |
| 8 | 말투·어조에 대한 불만인가 | 관측 | `case16` 말투·어조 불이행 |
| 9 | 질문 성격이 도메인이 아닌가 | 관측 | `case25` `case26` `case27` |
| 10 | `rag_chunks` 가 비어 있나 | 코드 | `case21` Retrieve 미수행 |
| 11 | 문서가 요구를 충족했나 | LLM + 코드 | `case20` Retrieve 실패 |
| 12 | 답변이 그 문서를 썼나 | LLM | 안 씀 `case22` · 어긋남 `case18` |
| 13 | 문서도 답변도 멀쩡한데 불만 | 관측 | 실행 불가 `case17` · 의도 불일치 `case13` |
| 14 | 남은 것 | 관측 | `case14` `case15` · 미분류 |

**왼쪽 위가 강한 증거다.** 0~6 은 코드나 단일 관측으로 확정되는 것들이고,
11~14 로 갈수록 LLM 판정과 인상에 의존한다.

**3번이 왜 거기인가.** 거절(1)과 인젝션(2)은 사용자가 지적했든 아니든 확인된
사실이라 앞에 둔다. 반대로 질문 모호성(4·5)보다는 앞이다 — 사용자가 만족했다면 그
모호함은 실제로 문제가 되지 않았다는 뜻이고, 신호는 `secondary_cases` 로 남는다.
그리고 3번을 통과하려면 **인용 검증을 지나야 한다.** "문제 없음"은 판정자가 낼 수
있는 가장 쉬운 답이라, 근거를 후속 발화에서 그대로 따오게 하지 않으면 애매한 턴이
전부 그리로 샌다. 못 대면 `unclassified` 로 간다 — 코드가 잡은 위반이 있을 때도
마찬가지다.

### 진리표 — 도메인 질문의 내용 불만 (9~12번)

라우팅 대부분은 단일 조건이라 위 표로 끝나지만, TYPE5 분기만 값이 여럿 조합된다.
이 도구의 핵심이 거기라 전개해 둔다. 아래는 `route.py` 를 실제로 돌려 만든 것이다.

| `n_chunks` | `verdict` | 인용 생존 | `answer_used_rag` | `answer_actionable` | → case |
|---|---|---|---|---|---|
| **0** | insufficient · partial | — | — | — | `case21` Retrieve 미수행 |
| 1↑ | insufficient · partial | — | — | — | `case20` Retrieve 실패 |
| 1↑ | **sufficient** | **0** | — | — | `case20` Retrieve 실패 ← **강등** |
| 1↑ | sufficient | 1↑ | `ignored` | — | `case22` Retrieve 성공, 생성 실패 |
| 1↑ | sufficient | 1↑ | `contradicted` | — | `case18` 문서와 어긋나는 주장 |
| 1↑ | sufficient | 1↑ | `used` | `false` | `case17` 두루뭉술한 답변 |
| 1↑ | sufficient | 1↑ | `used` | `true` | `case13` 의도와 다른 답변 |

세 줄이 이 표의 전부다.

**1행 vs 2행** — 청크가 0개면 `case21`, 있는데 빗나가면 `case20`. 앞은 검색을 탈지
말지 정하는 로직을 고치고, 뒤는 임베딩·청킹·문서를 고친다. 사용자에게는 똑같이
"답이 부실하다"로 보이지만 고칠 곳이 다르다.

**3행 (강등)** — 판정자가 `sufficient` 라 했는데 인용이 하나도 원문과 대조되지
않으면 `insufficient` 로 내린다. 문서를 읽어서가 아니라 **아는 것으로 답한** 경우다.
사전지식 오염을 프롬프트로 부탁하지 않고 여기서 구조로 잡는다.

**4행 vs 7행** — 문서에 답이 있는데 안 쓴 것(`case22`)과 쓰고도 불만인 것(`case13`).
전자는 프롬프트를 고치고 후자는 의도 이해를 본다. **이 갈림이 없으면 멀쩡한
코퍼스에 문서를 더 채우는 헛수고를 하게 된다.**

`n_chunks` 가 0 인데 `sufficient` 가 나오는 조합은 실제로 생기지 않는다. 청크가
비면 ⑦을 건너뛰고 `insufficient` 로 두기 때문이다.

### 왜 이 순서인가

**발견 순서가 아니라 증거 강도 순서다.** 답변이 나쁘면 여러 관측이 동시에 켜지므로
무엇을 먼저 묻느냐가 결과를 정한다.

검색이 실패해 답변이 부실하면 판정자는 그걸 "히스토리를 못 이어받았다"로도 읽는다.
그래서 `case14`(이전 턴 맥락 상실)를 앞에 두면 `case20` · `case22` 를 가로챈다 —
구현 중 실제로 회귀셋 6건이 그렇게 샜다. **인용으로 검증된 문서 증거가 LLM 의
인상보다 강하다.** `case14` 는 13번에서만 주 라벨이 되고 그 전에는 부가로만 붙는다.

0번이 맨 위인 것도 같은 이유다. 서비스 장애 문구를 관측에 넘기면 판정자가 거절로
읽어 `case28` 로 보낸다 — 인프라 문제가 보안 정책 문제로 집계된다.

**출력** — case 26개 + 미분류 2개

| case | 이름 | 신뢰도 | 무엇이 정하나 |
|---|---|---|---|
|---|---|---|---|
| `case0` | 정상 — 불만 아님 | medium | 관측 + 인용 검증 |
| `case1` | 이해하기 어려운 질문 | medium | 관측 |
| `case2` | 지원하지 않는 포맷 요구 | medium | 관측 |
| `case3` | 복합 질문을 함 | medium | 관측 · **부가로만** |
| `case4` | 참조가 모호한 질문 | medium | 관측 · **부가로만** |
| `case6` | 질문에 개인정보 포함 | high | 코드 · **부가로만** |
| `case8` | 출력 잘림 | high | 코드 |
| `case9` | 서비스 자원 부족 응답 | high | 코드 · LLM 0회 |
| `case10` | 요구 언어 불이행 | high | 관측 + 코드 |
| `case11` | 요구 길이 불이행 | medium | 관측만 |
| `case12` | 요구 포맷 불이행 | high | 관측 + 코드 |
| `case13` | 의도와 다른 답변 | medium | 관측 |
| `case14` | 이전 턴 맥락 상실 | medium | 관측 |
| `case15` | 복합 질문 일부만 답변 | medium | 관측 |
| `case16` | 말투·어조 불이행 | medium | 관측 |
| `case17` | 두루뭉술한 답변 | medium | 관측 + LLM |
| `case18` | 문서와 어긋나는 주장 | medium | LLM(⑨) |
| `case20` | Retrieve 실패 | medium | LLM(⑦) + 인용 |
| `case21` | Retrieve 미수행 | high | 코드 (빈 리스트) |
| `case22` | Retrieve 성공, 생성 실패 | medium | LLM(⑦+⑨) + 인용 |
| `case24` | 출처/인용 표기 오류 | high | 코드 · **부가로만** |
| `case25` | 상식 질문 오답 | **low** | 관측만 |
| `case26` | 계산 오답 | high | 코드 |
| `case27` | 코드/도구 사용법 오답 | medium | 관측 + 코드 |
| `case28` | 보안 정책상 답변 불가 | medium | 관측 |
| `case29` | 간접 프롬프트 인젝션 | medium | 코드 |
| `unclassified` | 분류 실패 | — | **"문제 없음"이 아니라 수동 검토 대상** |
| `out_of_taxonomy` | taxonomy 에 없는 유형 | — | 쌓이면 케이스를 추가하라는 신호 |

`taxonomy_v2.txt` 의 29개 중 4개(`case5` `case7` `case19` `case23`)는
**라우팅이 절대 만들지 않는다.**
필드가 없어 판정할 수 없는 것들이고, 목록에서 지우지 않은 이유는 "우리 분류에는
그런 게 없다"가 되지 않게 하기 위해서다.

**부가로만** 표시한 것은 주 라벨이 되지 않고 `secondary_cases` 에만 붙는다. 주 라벨로
두면 더 구체적인 원인을 가로챈다 — case14 를 앞에 뒀다가 회귀셋 6건이 샌 것과 같은 이유다.

### 헷갈리기 쉬운 셋

**`case17` 두루뭉술한 답변** — 내용은 **맞는데** 사용자가 다음에 무엇을 할지
알 수 없는 경우다. ⑤의 `answer_actionable` 이 `false` 일 때만 여기로 온다.

| 답변 | 판정 | 왜 |
|---|---|---|
| "출장비는 규정에 따라 지급됩니다" | `case17` | 틀리진 않았다. 그런데 뭘 하라는 건지 알 수 없다 |
| "국내 출장 식비는 1일 3만원입니다" | 정상 | 금액이 있어 행동으로 이어진다 |
| "연차 이월은 팀장 승인 후 그룹웨어에서" | 정상 | 절차·경로가 있다 |

`case13`(의도와 다른 답변)과 다르다. **`case13` 은 물은 것과 다른 걸 답한 것이고,
`case17` 은 물은 것을 맞게 답했는데 두루뭉술한 것**이다. 고칠 곳도
다르다 — `case13` 은 의도 이해, `case17` 은 답변의 구체성이다.

**`case20` 도 증상은 똑같이 두루뭉술하다.** 이름이 증상이라 헷갈리기 쉬운데, 갈리는
것은 원인이다 — 문서에 답이 **있는데** 답변이 두루뭉술하면 `case17`, 문서에 답이
**없어서** 두루뭉술하면 `case20` 이다. 그래서 `case17` 은 문서도 충분하고 답변이
그 문서를 썼을 때만 나온다(12번). 고칠 곳이 정반대라 이 구분이 값을 한다 —
`case17` 은 답변 구체화, `case20` 은 문서 보강이다.

**`case3` · `case15`** — `case3` 은 사용자가 복합 질문을 한 것(질문 유도로 고친다),
`case15` 는 모델이 그중 일부만 답한 것(생성으로 고친다)이다.

**`case4` · `case14`** — `case4` 는 질문이 불완전한 것("그거 다시"), `case14` 는
답변이 앞 턴을 잊은 것이다. 주체가 다르다.

`case25` 만 신뢰도 **low** 다. 판정 근거가 판정자의 사전지식뿐이라 — 이 도구가 문서
판정에서 막으려던 바로 그것을 여기서는 근거로 쓴다 — **다른 케이스와 같은 무게로
집계하면 안 된다.** 대시보드에 "신뢰도 낮음 제외" 스위치가 있는 이유다.

case 를 코드가 정하면 얻는 것이 하나 더 있다. taxonomy 가 바뀌어도 **LLM 을 다시 돌릴
필요가 없다.** 관측은 그대로 두고 진리표만 고치면 된다.

---

## 한 턴이 낳는 것 전부

| 필드 | 값 |
|---|---|
| `case_id` | 위 27개 중 하나 |
| `case_name` `type_id` `type_name` `category` | `case_id` 에서 계산된다. 따로 분류하지 않는다 |
| `confidence` | `high` · `medium` · `low` — taxonomy 기본값이되 코드 근거가 없으면 낮춘다 |
| `reason` | 왜 이 라벨인지 한 문장 (`"문서에 답이 있는데 답변이 쓰지 않음"`) |
| `secondary_cases` | 주 라벨과 별개로 성립한 case 들 |
| `notes` | 판정의 한계 (`"검색 실패와 코퍼스 부재는 구분 불가 — 부서 편중으로만 추정"`) |
| `llm_calls` | 0 · 1 · 2 · 3 |
| `evidence` | 관측 7개 · 검증기 결과 · 충족도와 인용(버려진 것 포함) · 근거 활용 |

`llm_calls` 가 그 턴이 어디까지 갔는지 그대로 보여준다.

| 호출 | 어디까지 |
|---|---|
| 0 | ④에서 끊김 (case9) 또는 캐시 히트 |
| 1 | ⑤만. 형식·언어·길이 불만이나 도메인이 아닌 질문 |
| 2 | ⑤+⑦. 문서가 불충분해 ⑨까지 안 감 |
| 3 | ⑤+⑦+⑨. 문서는 충분했고 활용 여부까지 물음 |

`evidence` 를 남기는 이유는 **집계만 보고 근거를 못 보면 "왜 이 라벨이지"에서 막히기
때문**이다.

## 파일

```
outputs/conv_parsed.json    원본 필드는 그대로, classification 아래에 판정 결과
outputs/run_summary.txt     RUN SUMMARY 사본
```

---

## 요약 — 단계별로 감춘 것

| 단계 | 목적 | 주는 것 | **일부러 안 주는 것** | 안 주면 막히는 실패 |
|---|---|---|---|---|
| ⑤ 관측 | 무엇을 원했나 | `pre_queries` · `llm_ans_on_last_q` · `current_query` | **`rag_chunks`** | 문서에 맞춰 요구를 재구성하는 순환 |
| ⑦ 충족도 | 문서에 있었나 | `resolved_question` · `unmet_need` · `rag_chunks` | **`llm_ans_on_last_q`** | "답변이 나쁨"을 "문서가 나쁨"으로 읽음 |
| ⑨ 근거활용 | 답변이 썼나 | `llm_ans_on_last_q` · `rag_chunks` | **`resolved_question` · `current_query`** | ⑦과 같은 방향으로 쏠린 두 번째 표 |
| ⑩ 라우팅 | 어느 case 인가 | 관측 + 검증 전부 | **LLM 자체** | 결론을 정하고 사실을 끼워 맞춤 |

판정 전체를 통틀어 **부서 · 직급 · 직무는 LLM 에 넘기지 않는다.** 집계할 때만 쓴다.
"인사팀 질문이니 인사 문서가 있었겠지" 같은 추론이 판정에 섞이면 안 되고,
부서별 편중은 판정이 끝난 뒤 집계에서 봐야 신호가 된다.

`llm_eval_result` · `llm_emotion_result` 도 판정에 넘기지 않는다. 이미 다른 시스템이 내린
판정이라, 넘기면 독립적인 두 번째 의견이어야 할 것이 확인 도장이 된다.
