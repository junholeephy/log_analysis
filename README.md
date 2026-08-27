# RAG 충족도 진단기

사내 지식 챗봇 대화 로그에서, **사용자가 직전 답변에 불만을 표한 턴**을 모아
"그때 검색된 문서(`rag_data`)가 그 질문에 만족스럽게 답하기에 충분했는가"를 판정한다.

목적은 개별 답변을 고치는 게 아니라, **집계했을 때 어디를 고쳐야 하는지**를 알아내는 것이다.

📊 **[파이프라인 흐름도](https://claude.ai/code/artifact/180f8cc5-d5fb-41e0-9084-8be60c271d5f)**
— 3단계 판정, 각 단계가 일부러 감추는 입력, 인용 검증, 관측 한계를 그림으로.
(비공개 링크. 공유 전에는 작성자만 열람 가능)

## 실행에 필요한 것 세 가지

```bash
pip install pydantic

export LLM_API_URL=http://<서버>:8000   # /v1 이 붙어 있어도, 스킴이 없어도 된다
export LLM_API_KEY=<키>

python run.py --input data/logs.json
```

끝이다. 나머지는 자동으로 정해진다.

| 자동으로 정해지는 것 | 어떻게 |
|---|---|
| 백엔드 | `LLM_API_URL`이 있으면 로컬 LLM |
| 모델 이름 | 서버의 `GET /v1/models` 에 물어본다 |
| 구조화 출력 강제 방식 | `json_schema` → `guided_json` → `json_object` → `none` 순으로 시도 |
| 추론 블록 처리 | `<think>` 블록과 `reasoning_content` 를 알아서 걷어낸다 |
| URL 형태 | `/v1`, 끝 슬래시, 전체 엔드포인트, 스킴 누락을 모두 흡수 |

문제가 생기면 먼저 이걸 돌린다:

```bash
python run.py --check-llm     # 서버·모델·강제방식·1회 소요시간·전체 예상시간
```

## 왜 이 구조인가

### 불만이 sufficiency를 판정 가능하게 만든다

"이 문서가 충분한가?"는 그 자체로는 판정할 수 없다. 무엇에 대해 충분한지 기준이 없기 때문이다.
그런데 `current_query`(불만)가 사용자의 진짜 정보 요구를 드러낸다. "예외 케이스가 안 나왔잖아"라는
불만은 원했던 게 예외 케이스였음을 알려주고, 판정은 "문서에 예외 케이스가 있었나?"라는
확인 가능한 질문으로 바뀐다.

### 단계마다 정보를 일부러 뺀다

이 표는 [흐름도](https://claude.ai/code/artifact/180f8cc5-d5fb-41e0-9084-8be60c271d5f)의 붉은 점선에 해당한다.

| 단계 | 주는 것 | **빼는 것** | 왜 |
|---|---|---|---|
| 1. 정보 요구 추출 | 질문 히스토리, 답변, 불만 | **`rag_data`** | 문서를 같이 주면 모델이 "사용자가 원한 것"을 문서에 있는 내용 쪽으로 끌어당긴다(anchoring). 요구와 문서가 저절로 일치해 보여서 sufficiency가 항상 후해진다 |
| 2. 충족도 판정 | 정리된 질문, 미충족 요구, 청크 | **챗봇 답변** | 답변을 보여주면 판정자가 답변을 문서의 대리물로 착각한다 — "답변이 이렇게 말했으니 문서에 있었겠지" |
| 3. 근거 활용 확인 | 답변, 청크 | — | 2가 sufficient일 때만 실행 |

### 인용 강제가 knowledge leakage를 막는다

판정자가 "문서에 답이 있다"고 말할 때, 문서를 읽어서인지 자기가 이미 알던 지식 때문인지
프롬프트로는 구분할 수 없다. 사내 코퍼스는 일반 상식과 상당히 겹치므로 이건 실제 위험이다.
leakage가 일어나면 **검색 실패가 '근거 미활용'으로 오분류되어 통계에서 사라진다.**

그래서 판정자에게 청크에서 글자 그대로 인용을 뽑게 하고 `verify.py`가 원문과 대조한다.
지어낸 인용은 일치하지 않으므로 걸러지고, 살아남은 인용이 하나도 없으면 verdict를
`insufficient`로 강등한다. "지어내지 마세요"라는 프롬프트와 달리 이건 검증 가능한 장치다.

### 라벨은 LLM이 아니라 코드가 결정한다

한 번의 호출로 원인까지 물으면 모델이 원인을 먼저 직감하고 판정값을 거기에 역으로
맞춘다(합리화). 판정값만 받아서 `decide.py`의 진리표가 조합하면 그 경로가 막히고,
라벨 체계를 바꿀 때 LLM을 다시 돌리지 않아도 되며, "왜 이 라벨이 붙었나"에 항상 답할 수 있다.

| complaint_type | verdict | used_rag | 라벨 | 고칠 곳 |
|---|---|---|---|---|
| format_or_style | — | — | `out_of_scope` | 없음 (rag 무관) |
| other | — | — | `unclassified` | 수동 검토 |
| content | insufficient | — | **`rag_insufficient`** | **코퍼스 / 검색** |
| content | partial | — | `rag_partial` | 코퍼스 보강 |
| content | sufficient | ignored / contradicted | `rag_sufficient_generation_failed` | 생성 프롬프트 |
| content | sufficient | used | `rag_sufficient_other` | 깊이 / 표현 |

## 알려진 한계

- **`rag_data`만으로는 '검색 실패'와 '코퍼스에 애초에 문서 없음'을 구분할 수 없다.**
  가져온 top-k에 답이 없다는 사실은 코퍼스 어딘가에 답이 있는지에 대해 아무것도 말해주지
  않는다. 그래서 두 라벨을 `rag_insufficient` 하나로 합쳤다. 구분 못 하는 걸 구분한 척하는
  라벨이 제일 나쁘다. **라벨 x 부서 교차표가 이 구분을 간접적으로 되살리는 유일한 신호다** —
  특정 부서에 몰려 있으면 그 도메인 문서가 비어 있다는 뜻이다.
  나중에 코퍼스에 접근할 수 있게 되면 재검색으로 두 라벨을 쪼갤 수 있다.
- `rag_sufficient_other`는 잔여 범주라 경계가 흐리다. 집계에서 이게 크면 라벨을 쪼갤 때다.
- `turns`에 불만 턴만 들어있다고 가정한다. 전체 턴이 들어오면 불만 탐지 단계가 앞에 필요하다.
- **`rag_data`는 청크를 `\n\n`(또는 `\n`)으로 이어붙인 통문자열이라 청크 경계를 복원해야 한다.**
  빈 줄을 먼저 시도하고 그걸로 안 쪼개질 때만 단일 개행으로 내려간다 — 청크 내부에도 개행이
  있을 수 있어서 순서가 반대면 한 청크가 여러 조각으로 찢어진다. 청크가 단일 개행으로
  이어붙여져 있고 내부에도 개행이 있으면 경계는 원리적으로 복원 불가능하다.
  다만 인용 검증이 전 청크를 훑기 때문에 잘못 쪼개진 경계는 `index_corrected`로 흡수되고,
  sufficiency 판정 자체는 영향을 받지 않는다. 리포트 `[4] 판정 건강 지표`의
  "청크 1개 이하인 케이스"가 이 실패를 드러낸다.

## 합성 데이터의 위치

`fixtures/synthetic.py`는 **정확도 측정용이 아니라 회귀 테스트용**이다.

1. 데이터와 판정 프롬프트를 같은 사람이 만들면 편향을 공유한다. 여기서 나온 일치율은 실전보다 후하다.
2. 합성 문서는 지어낸 사내 규정이라 판정자가 사전지식으로 알 리가 없다. `leakage_probe`
   케이스(상식으로 답 가능한 질문 + 그 답이 없는 문서)가 그 틈을 일부 메우지만 완전히는 못 메운다.

**이 셋은 이제 독립적인 측정 도구가 아니다.** 판정 실패를 보고 프롬프트를 네 차례 고치는 데
사용됐기 때문이다. 현재 23/23이 나오지만 이건 "알려진 회귀가 없다"는 뜻이지 정확도가 100%라는
뜻이 아니다. 프롬프트를 이 셋의 실패에 맞춰 조정한 이상, 같은 셋으로 잰 점수는 과대평가다.

실제 정확도는 실데이터 20~30건을 손으로 라벨링해서 재야 한다. 특히
**`false insufficient`(충분했는데 부족하다고 판정)** 를 따로 추적한다 — 이 방향 오류는
멀쩡한 코퍼스에 문서를 더 채워 넣게 만들어서 노력을 엉뚱한 데 쓰게 한다.

함정 유형: `near_miss`(주제 일치·답 부재), `partial`, `distractor`(그럴듯한 오답 문서),
`generation_failure`, `leakage_probe`, `context_dependent`, `format_complaint`.

### 이 셋이 실제로 잡아낸 것 (설계 검증 기록)

프롬프트를 돌려보기 전에는 보이지 않던 결함들이다.

| 결함 | 증상 | 수정 |
|---|---|---|
| 사용량 집계 동시성 경합 | 공유 카운터 차이로 케이스별 사용량을 재서 15콜·$1.08로 보고 (실제 6콜·$0.49) | 호출이 자기 사용량을 반환, 케이스가 지역 변수에 누적 |
| `partial`/`insufficient` 정의 중복 | 두 설명이 같은 상황을 서술해 distractor가 partial로 샘 | "이 인용이 요구의 **어느 부분에 답하는가**" 시험 추가 |
| `unmet_need` 부풀리기 | "정확한 금액이요"에 "직급별 금액표"까지 요구로 적어, 답이 있는 문서를 partial로 깎음 | "사용자가 요구한 범위를 넘지 마라" 제약 추가 |
| `context_dependent` 과탐 | "애매하면 true" 규칙이 23건 중 15건을 찍어 플래그가 무의미해짐 | 대상 명사 유무로 판별, "확신 없으면 false" |
| `false insufficient` 지표 사각지대 | `rag_insufficient`만 세어 `rag_partial` 과소평가를 놓침 | 두 라벨을 같은 방향으로 집계 + 회귀 테스트 |

프롬프트에 지시를 넣을 때는 **반대 방향 제약을 같이 넣어야 한다**는 게 반복된 교훈이다.
"구체적으로 써라"는 요구 부풀리기를, "애매하면 넓게 잡아라"는 과탐을 낳았다.

## 판정 백엔드

| | `--backend local` | `--backend cli` | `--backend api` |
|---|---|---|---|
| 대상 | **에어갭 장비의 로컬 LLM** | 개발 장비 | 개발 장비 |
| 연결 | OpenAI 호환 HTTP (표준 라이브러리) | `claude -p` | Anthropic SDK |
| 인증 | `LLM_API_URL` + `LLM_API_KEY` | 불필요 | `ANTHROPIC_API_KEY` |
| 스키마 강제 | 서버 능력에 따라 자동 협상 | 없음 (프롬프트 계약) | 서버가 강제 |

`LLM_API_URL`이 설정돼 있으면 기본 백엔드가 자동으로 `local`이 된다.

| | 인식하는 환경변수 (앞에 있는 것이 우선) |
|---|---|
| 주소 | `LLM_API_URL` · `API_URL` · `RAGDIAG_BASE_URL` · `OPENAI_BASE_URL` · `OPENAI_API_BASE` |
| 키 | `LLM_API_KEY` · `API_KEY` · `RAGDIAG_API_KEY` · `OPENAI_API_KEY` |

이름 하나가 목록에서 빠지면 그 장비에서 "주소가 없습니다"로 멈추므로, `tests/test_config.py`가
문서에 적힌 이름을 전부 검증한다.

### 에어갭 장비 이관

반입할 것은 소스와 `pydantic` 하나뿐이다. 로컬 LLM에는 표준 라이브러리로 붙으므로
HTTP 클라이언트가 필요 없고, `anthropic`은 반입하지 않아도 된다
(`ApiBackend`가 지연 import라 없어도 전체가 동작한다 — 테스트로 확인됨).

```bash
pip install pydantic                      # 사내 미러
export LLM_API_URL=http://<서버>:8000
export LLM_API_KEY=<키>

python run.py --check-llm                 # 1. 서버 규약 확정
python -m pytest tests/ -q                # 2. LLM 없이 도는 부분 (108개)
python run.py --synthetic                 # 3. 그 모델이 프롬프트를 따르는지
python run.py --input data/logs.json --limit 20   # 4. 실데이터
```

`--model`은 서버가 여러 모델을 서빙하고 첫 번째가 아닌 걸 쓰고 싶을 때만 필요하다.
`--check-llm`이 서버의 다른 모델 목록도 함께 보여준다.

**3번이 이 장비에서 가장 중요하다.** 개발 장비의 Claude Opus 5 기준선은 라벨 23/23이다.
로컬 모델이 크게 낮으면 프롬프트를 못 따르고 있다는 뜻이고, 실데이터 결과를 믿을 수 없다.
어느 함정 유형에서 깨지는지가 무엇을 고쳐야 하는지 알려준다. 397B급이면 기준선에 근접할
것으로 보지만, 확인 없이 넘어갈 일은 아니다.

`--check-llm`은 구조화 출력 강제 방식을 서버에 직접 물어 찾는다. 순서는
`json_schema`(OpenAI 규격) → `guided_json`(vLLM 고유) → `json_object` → `none`이고,
처음 통하는 것을 이후 계속 쓴다. 강제가 되면 재시도를 하지 않는다 — 로컬에서는 호출 한 번이 비싸다.

### 프록시(LiteLLM 등)를 거치는 경우

포트 4000 + `sk-...` 키는 보통 LiteLLM 프록시다. 프록시는 모르는 파라미터를 400으로
거절하지 않고 **조용히 버리는** 경우가 있다. 그러면 200 OK만 보고 "이 모드 된다"고
확정한 뒤 재시도를 1회로 줄여버려서, 첫 응답이 어긋나는 순간 케이스가 그냥 실패한다.

그래서 협상은 HTTP 200이 아니라 **강제가 실제로 걸렸는지**를 본다. 스키마와 무관한
탐침("Reply with a JSON object.")을 보내고, 돌아온 응답이 스키마에 맞는지 검사한다.
강제가 걸렸을 때만 맞을 수 있다. `--check-llm`이 협상 과정을 그대로 보여준다.

```
구조화 출력 강제 방식 : guided_json
  협상 과정:
    json_schema  200 OK지만 강제가 걸리지 않음 (프록시가 조용히 무시한 듯)
    guided_json  채택
```

모드별로 요구하는 것이 다르다. `json_schema`/`guided_json`은 스키마까지 맞아야 채택되고,
`json_object`는 유효한 JSON이기만 하면 된다(그 모드가 약속하는 게 거기까지다).
전부 실패하면 `none`으로 떨어지고, 그때는 `parse_with_repair`가 유일한 방어선이 된다.

### 하이브리드 추론 모델 (Qwen3 계열)

Qwen3 계열은 응답 앞에 `<think>...</think>` 추론 블록을 붙일 수 있다. **이게 JSON 추출을
망가뜨린다** — 추론 문장에 중괄호가 하나라도 있으면 `extract_json`이 그걸 집는다.
검증 실패 -> 재시도 -> 같은 실패로 케이스가 통째로 날아간다.

`strip_reasoning()`이 닫는 태그 뒤를 취해 이 문제를 없앤다. 여는 태그를 찾지 않는 이유는,
채팅 템플릿이 `<think>`를 미리 넣어주면 모델 출력에는 **닫는 태그만** 나오기 때문이다.
vLLM `--reasoning-parser`를 켜서 `reasoning_content`로 분리되는 경우도 함께 처리한다.

- `--thinking {auto,on,off}` — `auto`(기본)는 서버 기본값을 건드리지 않는다.
  `on`/`off`는 `chat_template_kwargs.enable_thinking`을 보낸다. 서버가 이 필드를 모르면
  모든 모드가 400이 되므로, 협상 실패 메시지가 그 가능성을 알려준다.
- `--max-tokens` 기본 16000. 추론 모드가 켜져 있으면 생각에만 수천 토큰을 쓰고,
  잘리면 JSON이 아예 안 나온다. 추론만 하고 답을 못 낸 경우는 별도 오류로 구분된다.

**thinking을 켤지 끌지는 측정해서 정해라.** 출력 스키마에 이미 `reasoning` 필드가 맨 앞에
있어서 모델은 어차피 근거를 먼저 쓴다. 추론 모드가 그 위에 더 얹을 값이 있는지는 모델과
과제에 따라 다르다. 23건짜리 합성 셋이 있으니 양쪽을 다 돌려 비교하는 게 추측보다 빠르다.

```bash
python run.py --synthetic --thinking off --out off.jsonl
python run.py --no-cache --synthetic --thinking on --out on.jsonl
```

특히 `partial`과 `insufficient`의 경계 판정에서 차이가 날 가능성이 크다. 그 4+3건이
갈리는지를 보면 된다.

### 서빙 쪽에서 확인할 것

- 397B MoE(활성 17B)는 다중 GPU 텐서 병렬이 필요하다. `--check-llm`이 1회 소요시간을
  알려주므로 거기서 전체 소요를 역산한다.
- `--workers`는 서버 처리량에 맞춘다. vLLM은 연속 배치를 잘 처리하므로 8~16도 무리가
  아니지만, `--check-llm`의 1회 시간이 수십 초라면 낮추는 게 낫다.
- 입력은 케이스당 2~3천 토큰 수준이라 컨텍스트 길이는 문제가 되지 않는다.

CLI가 3배쯤 무거운 이유는 Claude Code 기본 시스템 프롬프트(약 12k 토큰)가 매 호출에 실리기
때문이다. `--system-prompt`로 교체해도 줄지 않는다. 캐시가 더워지면 그 부분은 캐시 읽기가 된다.

**리포트의 달러 값은 청구액이 아니다.** CLI가 주는 `total_cost_usd`는 `costBasis="list"`,
즉 API 정가 환산치다. 구독(OAuth) 인증으로 붙으면 별도 청구가 발생하지 않고 구독 사용량만
소모하며, 한도를 넘으면 과금이 아니라 요청이 거절된다. 이 숫자는 어느 단계가 사용량을
많이 먹는지 비교하는 용도로만 읽어라. API 키로 붙는 `--backend api`에서는 실제 요금이 된다.

CLI 경로에는 서버측 스키마 강제가 없으므로 `prompts.output_contract()`가 Pydantic 모델에서
계약 문구를 생성해 시스템 프롬프트에 붙인다. 손으로 두 번 쓰면 `schema.py`와 어긋난다.

프롬프트 전문은 `python run.py --show-prompts`로 예시 입력과 함께 볼 수 있다.

## 사용법

```bash
python3 -m venv venv && ./venv/bin/pip install -r requirements.txt

./venv/bin/python -m pytest tests/ -q          # API 없이 도는 부분 검증
./venv/bin/python run.py --show-prompts        # 파이프라인 전체 프롬프트
./venv/bin/python run.py --trace C-4002:3     # 케이스 하나의 실제 통과 경로
./venv/bin/python run.py --synthetic           # 합성 데이터 회귀 검증 (23건)
./venv/bin/python run.py --input data/logs.json --limit 20   # 실데이터, 비용 확인용
./venv/bin/python run.py --input data/logs.json --workers 12
```

케이스당 LLM 호출은 평균 2회 내외다 (형식 불만이면 1회, sufficient일 때만 3회).
판정 결과는 `.cache/`에 저장되어 재실행 시 재사용된다 — 리포트 코드를 고칠 때마다
판정을 다시 살 필요가 없다.

주요 옵션: `--effort {low,medium,high,xhigh,max}`, `--no-cache`,
`--no-fallbacks`(refusal 서버측 폴백 beta가 조직에 미활성인 경우).

## 개인정보

`user_id` / `db_login_id`는 **로딩 단계에서** 해시로 치환된다. 리포트 단계가 아니라
로딩 단계에 둔 이유는 원본 식별자가 어떤 산출물에도 들어가지 않게 하기 위해서다.
해시는 salt 없는 결정적 값이라 그룹핑은 그대로 되고 필요하면 역조회도 가능하다.
집계는 부서·직급 단위로만 낸다 — "누가 못 쓰는가"가 아니라 "어디가 안 되는가"를 보는 도구다.

## 구조

```
ragdiag/
  schema.py    Case + Stage 1/2/3 출력 (Pydantic, 필드 순서에 의미 있음)
  load.py      중첩 JSON flatten, 청크 경계 복원, 식별자 마스킹
  prompts.py   판정 프롬프트 (단계별로 뺄 정보가 여기에 명시됨)
  judge.py     LLM 호출, 디스크 캐시, 케이스 단위 병렬
  backends.py  로컬 LLM(OpenAI 호환) / Claude Code CLI / Anthropic API 백엔드
  verify.py    인용 대조 — leakage 차단 장치
  decide.py    진리표 — 최종 라벨
  report.py    라벨 분포, 라벨 x 부서/직급 교차표, 코퍼스 보강 목록
fixtures/synthetic.py   함정 위주 합성 데이터 + 기대 라벨
```
