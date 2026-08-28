# 대화 로그 실패 분류기

사내 지식 챗봇의 대화 로그에서 **사용자가 직전 답변에 불만을 표한 턴**을 골라,
그 실패가 taxonomy 30개 케이스 중 어디에 해당하는지 분류한다.

목적은 개별 답변을 고치는 게 아니라 **집계했을 때 어디를 고쳐야 하는지**를
알아내는 것이다. 그래서 라벨은 증상이 아니라 **조치 주체**로 나뉜다 — 같은 증상도
문서팀이 고칠 일과 프롬프트 담당이 고칠 일은 다르다.

📋 **[실패 분류 체계](TAXONOMY.md)** — 30개 케이스 중 무엇이 이 로그로 판정 가능하고
무엇이 아닌지, 그 판단 근거.

📊 **[파이프라인 흐름도](https://claude.ai/code/artifact/180f8cc5-d5fb-41e0-9084-8be60c271d5f)**
— 3단계 판정과 각 단계가 일부러 감추는 입력을 그림으로. (비공개 링크)

## 두 장비로 나뉜다

**기능 개발은 이 저장소에서, 검증은 사내 머신에서.** 실데이터가 밖으로 나올 수
없고 사내에서는 코드를 고칠 수 없다. 그 분리가 구조에 박혀 있다 —
`IMPLEMENTATION_SPEC.md` 규격을 따른다.

```
[이 저장소] 구현 ──이식──▶ [사내] 실험 ──인사이트──▶ [이 저장소] 개선 ──▶ …
```

| | 무엇 |
|---|---|
| `src/ragdiag/contracts.py` | **입력 계약.** 사내에서 회수한 포맷 정보가 도착하는 유일한 지점 |
| `configs/example.yaml` | **모든 설정 키.** 사내 실값은 `AA/configs/local.yaml` |
| `src/ragdiag/fixtures/synth.py` | **가짜 데이터는 파일이 아니라 코드.** `generate(n, seed)` 가 런타임에 만든다 |
| `scripts/sync.sh` | 이식. `.git` 도 데이터도 넘기지 않는다 (규격 부록 A 전문) |
| `docs/insights/` | 사내에서 본 것을 적어 오는 자리 |

## 진입점

| | 무엇 | 언제 |
|---|---|---|
| **`python -m ragdiag`** | 본 파이프라인. conv_eval 로그를 29개 case 로 분류 | 실제 분석 |
| `scripts/legacy_run.py` | 구 파이프라인. case20/case22 판별만, 라벨 6개 | **회귀 기준선** |

`legacy_run.py` 는 지우지 않는다. 이 프로젝트에서 **실제 LLM 으로 검증된 최초의
파이프라인**이고, 그 23건 회귀셋이 새 파이프라인의 라우팅 결함을 잡아냈다.
검증된 기준선을 지우면 같은 종류의 회귀를 다음에 못 잡는다.

```bash
python src/run.py --conv-data <로그> --filter-data <필터> --output-dir <출력>
python src/run.py --config configs/local.yaml --dry-run   # 설정 파일로도 된다
python -m ragdiag --golden                                # 3단계 판정 품질
python -m ragdiag --legacy-regression                     # 회귀 기준선
python -m pytest tests/ -q                                # LLM 없이 도는 전부
```

## 사내 머신에서

```bash
# 최초 1회. clone 위치는 .staging/<저장소이름> 이어야 한다.
cd <사내작업폴더> && git clone <이 저장소> .staging/log_analysis

# 매번. 멱등하다 — 최초든 갱신이든 같은 명령이다.
bash .staging/log_analysis/scripts/sync.sh v0.2
```

`sync.sh` 는 이름을 스스로 유도한다. 저장소 이름은 자기 위치에서,
패키지 이름은 `src/` 아래에서 읽으므로 **손볼 것이 없다.** 위 명령이
다음에 실행할 것까지 찍어 준다:

```bash
source <기존 venv>/bin/activate
pip install --dry-run -r log_analysis/requirements.txt && pip check   # 충돌 먼저
pip install -r log_analysis/requirements.txt

python log_analysis/src/run.py --dry-run                     # ① 합성 스모크
python log_analysis/src/run.py \
    --conv-data data/conv_eval.json --limit 1000             # ② 계약 확인
python log_analysis/src/run.py \
    --conv-data   data/conv_eval.json \
    --filter-data data/filter.json \
    --output-dir  outputs                                    # ③ 전체
```

①에서 실패하면 **환경 문제**고, ②에서 나오는 계약 위반이 첫 사이클의 실제
수확이다. 계약이 깨끗해진 뒤에 ③으로 간다 — 틀린 계약 위에서 뽑은 숫자는
믿을 수 없다.

`PYTHONPATH` 도 설치도 필요 없다. 파이썬이 스크립트가 있는 `src/` 를
`sys.path[0]` 에 넣으므로 옆의 `ragdiag/` 가 그대로 import 된다. 공용 venv 에
우리 패키지를 남기지 않고, 사본 통째 교체가 무연산이 된다.

상대 경로는 전부 **실행 위치 기준**이다. 작업 폴더에서 실행하면 사본이
어디 있든 `outputs/` 가 맞아떨어진다.

`--output-dir` 에 두 파일이 생긴다.

| | |
|---|---|
| `conv_parsed.json` | 분류 결과 |
| `run_summary.txt` | RUN SUMMARY 사본. 손으로 옮겨 적을 때 스크롤을 뒤지지 않게 |

경로를 매번 치기 싫으면 설정에 넣고 `--config configs/local.yaml` 만 준다.
CLI 인자가 설정을 덮어쓰므로 한 번만 다르게 돌려볼 때도 섞어 쓸 수 있다.

규격이 정한 형태도 그대로 된다 — `src/run.py` 는 파이썬이 스크립트 디렉터리를 `sys.path[0]` 에
넣는 것을 쓸 뿐 로직이 없다.

```bash
PYTHONPATH=log_analysis/src python -m ragdiag --config configs/local.yaml
```

**패키지를 venv 에 설치하지 않는다** — `PYTHONPATH` 로만 붙인다. 공용 venv 를
오염시키지 않고 사본 통째 교체가 무연산이 된다. `--upgrade` 와
`--force-reinstall` 은 쓰지 않는다. 남의 환경을 조용히 깨뜨리고 되돌릴 수 없다.

### ⚠ 작업 폴더의 `.gitignore` 를 먼저 손볼 것

`sync.sh` 는 `.staging/` 만 무시 목록에 넣는다. 나머지는 그대로 두면 사내 git 에
커밋된다. 실행하면 작업 폴더에 이런 것들이 생긴다.

| | 무엇 | 커밋해도 되나 |
|---|---|---|
| `.cache/` | **LLM 판정 응답.** 실데이터에서 뽑은 관측·인용이 그대로 들어 있다 | 판단 필요 |
| `data/` | 실데이터 | 대개 아니다 |
| `outputs/` | 분류 결과 · RUN SUMMARY | 남기고 싶을 수 있다 |
| `configs/local.yaml` | 사내 실값 (경로·주소) | 판단 필요 |

`AA/log_analysis` 사본이 커밋되는 것은 목적이지만 — "어떤 코드로 돌렸는지"가
남는 유일한 형태다 — 나머지는 의도한 것만 남기는 편이 낫다. 특히 `.cache/` 는
분류를 다시 돌리면 재생성되는 파생물이고, 대화 내용이 그대로 들어 있다.

```bash
cd <사내작업폴더>
cat >> .gitignore <<'EOF'
.cache/
data/
EOF
```

**태그 없이 실행하지 않는다.** 결과 파일이 반출되지 않으므로 사내에 남은 사본이
"어떤 코드로 돌렸는지"를 알려주는 유일한 형태다.

`sync.sh` 가 지키는 것 — `configs/local.yaml` 은 있으면 **절대 건드리지 않고**
example 에만 있는 키를 경고한다(모르고 지나가면 조용히 기본값으로 돈다).
데이터 파일이 사본에 섞이면 **사본을 지우고** 실패로 끝낸다.

실행이 끝나면 화면 마지막에 이 블록이 찍힌다. 파일도 플롯도 못 가져오므로
**이게 유일한 출력**이다.

```
================ RUN SUMMARY =================================================
version   : v0.2 (a1b2c3d)
input     : 12,004 users / 48,221 conversations / 210,553 turns
contract  : 21 ok / 2 MISMATCH
  - turn.retrieved_data        : str|list 를 기대했으나 dict 가 왔다 (1,204건)
  - turn.권한코드               : 계약에 없는 키 (210,553건). 새 필드인지 확인할 것
metrics   : selected 3,912 turns
            classified 3,908 ok / 4 failed
runtime   : 412s, peak 6.2GB
status    : PARTIAL
==============================================================================
```

계약 위반 줄은 **그대로 옮겨 적어** `docs/insights/` 에 넣는다. 그게 포맷이
이쪽으로 돌아오는 유일한 경로다.

## 실행에 필요한 것

```bash
pip install pydantic PyYAML

export LLM_API_URL=http://<서버>:8000   # /v1 이 붙어 있어도, 스킴이 없어도 된다
export LLM_API_KEY=<키>
```

설정 파일 없이도 돈다. 나머지는 자동으로 정해진다.

| 자동으로 정해지는 것 | 어떻게 |
|---|---|
| 백엔드 | `LLM_API_URL`이 있으면 로컬 LLM |
| 모델 이름 | 서버의 `GET /v1/models` 에 물어본다 |
| 구조화 출력 강제 방식 | `json_schema` → `guided_json` → `json_object` → `none` 순으로 시도 |
| 추론 블록 처리 | `<think>` 블록과 `reasoning_content` 를 알아서 걷어낸다 |
| URL 형태 | `/v1`, 끝 슬래시, 전체 엔드포인트, 스킴 누락을 모두 흡수 |

문제가 생기면 먼저 이걸 돌린다:

```bash
python scripts/legacy_run.py --check-llm   # 서버·모델·강제방식·1회 소요시간
```

## 3단계 분류

```
Step 1  관측 추출     LLM 1회   ✂ rag_data 를 주지 않는다
Step 2  조건부 검증   코드 항상 · LLM 은 도메인 질문일 때만  ✂ 충족도 판정에 답변을 주지 않는다
Step 3  라우팅        코드      case 는 LLM 이 고르지 않는다
```

case 를 LLM 에게 직접 고르게 하지 않는 이유가 셋이다. 30지선다는 어떤 모델이든
정확도가 안 나오지만 좁은 질문 여러 개는 안정적이고, 한 호출로 case 까지 물으면
모델이 결론을 먼저 직감하고 관측값을 거기 맞추며, taxonomy 를 고쳐도 관측값은
그대로 재사용되어 라우팅만 다시 돌리면 된다.

category 는 따로 분류하지 않는다. 각 case 는 정확히 하나의 type 에, type 은 하나의
category 에 속하므로 case 만 정하면 나머지는 계산이다.

**Step 2 의 절반 이상이 코드다** — 언어·길이·포맷·잘림·개인정보·인용 대조·문법·계산.
LLM 에 맡기면 비용도 들지만 무엇보다 같은 입력에 다른 답이 나온다.

## 한 건을 끝까지 따라가기

실제로 돌린 결과다. 재무팀 대리의 2턴 대화 하나.

### 입력

```
turn 1  질문   "국내 출장 갈 때 식비는 얼마까지 쓸 수 있나요?"
        문서   ① 국내 출장 식비는 1일 3만원을 상한으로 한다.      <- 답이 여기 있다
               ② 국내 출장 숙박비는 1박 8만원을 상한으로 한다.
               ③ 출장비는 출장 종료 후 5영업일 이내에 정산한다.
        답변   "회사 규정에 따라 지급되며, 자세한 금액은 부서나 직급에 따라
                다를 수 있으니 총무팀에 확인해 보시기 바랍니다."

turn 2  질문   "부서별로 다르다는 게 아니라 규정상 정해진 금액이 있을 텐데요."
        라벨   명확화 요구(K, 25점) · 매우 부정(I, 3.1점)
```

문서에 "1일 3만원"이 명확히 있는데 답변은 "총무팀에 확인하세요"로 넘겼다.

### 필터

```
진단 가능 후속 턴 (2턴 이상)     1
eval_score (0~60)               1     명확화 요구 = 25점
emotion_score (0~20)            1     매우 부정 = 3.1점
emotion 라벨 (I. 매우부정)       1
```

점수는 기록값이 아니라 필터의 `query_scores` 로 재계산한 값이다.

### 짝짓기

```
불만     <- turn 2 의 질문
답변     <- turn 1 의 응답        비판받은 것
문서     <- turn 1 의 retrieved   ★ turn 2 것이 아니다
히스토리 <- turn 1 의 질문        (최대 3턴)
```

turn 2 의 검색 결과를 쓰면 "다음 질문으로 찾은 문서가 충분했나"라는 다른 질문이 된다.

### Step 1 · 관측 — LLM 1회, rag_data 를 주지 않는다

```
resolved_question        "국내 출장 시 규정상 정해진 식비 한도 금액은 얼마인가요?"
unmet_need               "국내 출장 식비의 규정상 정해진 정확한 한도 금액"
complaint_target         content_missing
question_domain          domain
question_self_contained  True
answer_refused           False
```

`answer_refused` 가 false 인 것이 중요하다. "총무팀에 확인하세요"는 회피성 안내지
정책상 거절이 아니다. 이걸 거절로 읽으면 case28 으로 빠지고 **진짜 원인이 통계에서
사라진다** — 실제로 났던 사고다.

문서를 주지 않은 덕에 `unmet_need` 가 "3만원"이 아니라 사용자가 원한 것으로 나왔다.
문서를 봤다면 거기 있는 내용 쪽으로 끌려갔을 것이다.

### Step 2 · 검증

코드 검증(LLM 0회)은 요구가 없던 항목이 `not_applicable` 이라 출력에 실리지 않는다.
실린 것은 항상 도는 둘뿐이다.

```
pii        ok    검출 없음
truncated  ok    종결 부호로 끝남
```

충족도 판정(LLM, 챗봇 답변을 주지 않는다):

```
verdict   sufficient
인용      청크0  ratio=1.0  "국내 출장 식비는 1일 3만원을 상한으로 한다."
폐기      없음
```

**`ratio=1.0` 이 이 판정의 무게를 결정한다.** 판정자가 "문서에 답이 있다"고 주장하며
뽑은 인용을 코드가 원문과 대조했고 완전히 일치했다. 지어낸 것이 아니다. 인용이 하나도
살아남지 못했다면 `insufficient` 로 강등됐을 것이다.

여기서 답변을 주지 않은 것이 결정적이다. 답변("부서별로 다를 수 있다")을 봤다면
판정자가 그걸 문서의 대리물로 착각해 "문서에도 명확한 금액이 없나 보다"로 흐를 수 있다.

### Step 3 · 근거 활용 — 여기서 처음 답변을 본다

```
answer_used_rag   ignored
```

이 시점엔 충족도가 이미 확정돼 있어 오염될 수 없다.

### Step 4 · 라우팅 — 코드, LLM 0회

```
complaint_target=content_missing   -> 내용 불만
question_domain=domain             -> TYPE5 분기
verdict=sufficient + 인용 1개 생존  -> 문서는 충분했다
answer_used_rag=ignored            -> 답변이 쓰지 않았다
                                   v
case22  Retrieve 성공, 생성 실패    (TYPE5 / category_2, 신뢰도 medium)
```

**case20(Retrieve 실패)가 아니라 case22 인 것이 이 도구의 존재 이유다.** 사용자에게는
똑같이 "답이 부실하다"로 보이지만, case20 면 문서를 써야 하고 case22 이면 프롬프트를
고쳐야 한다. 멀쩡한 코퍼스에 문서를 더 채우는 헛수고를 막는 것이 이 구분이다.

`category_2` 는 case22 에서 계산된다. 따로 분류하지 않는다.

### 최종 출력

원본 필드는 그대로 두고 분류는 `classification` 아래에 모인다.

```json
{
  "turn": 2,
  "pre_queries": ["국내 출장 갈 때 식비는 얼마까지 쓸 수 있나요?"],
  "llm_ans_on_last_q": "출장 식비는 회사 규정에 따라 지급되며 …",
  "current_query": "부서별로 다르다는 게 아니라 규정상 정해진 금액이 …",
  "chunk_data": ["국내 출장 식비는 1일 3만원을 상한으로 한다.", "…"],
  "classification": {
    "case_id": "case22", "case_name": "Retrieve 성공, 생성 실패",
    "type_id": "TYPE5", "category": "category_2",
    "confidence": "medium",
    "reason": "문서에 답이 있는데 답변이 쓰지 않음",
    "secondary_cases": [], "notes": [],
    "evidence": { "observation": {…}, "sufficiency": {…}, "grounding": {…}, "checks": [] },
    "llm_calls": 3, "answered_turn": 1
  }
}
```

`answered_turn` 으로 짝짓기를 사후 확인할 수 있고, `evidence` 전체가 남아
"왜 이 라벨이 붙었나"에 언제든 답할 수 있다.

**LLM 호출 3회.** 문서에 답이 없었다면 2회(근거 활용을 물을 이유가 없다),
형식 불만이었다면 1회로 끝난다.

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

### 검증 기록 — 두 셋이 서로 다른 층을 잡는다

| 셋 | 무엇을 재나 | 현재 |
|---|---|---|
| 관측 골든셋 44건 (`--golden`) | Step 1 관측 **하나하나**의 정확도 | 72/72 |
| 판정 골든셋 18건 (`--golden`) | Step 2·3 의 verdict·인용 위치 정확도 | 22/22 |
| 구 회귀셋 23건 (`--legacy-regression`) | 관측이 조합되어 case 로 가는 **경로** | 23/23 |
| 단위 테스트 (`pytest`) | LLM 없이 도는 전부 | 304개 |

**세 층은 서로를 대체하지 못한다.** 골든셋이 98% 일 때 회귀셋은 15/23 이었다. 관측 필드
자체는 멀쩡한데 그 값을 **라우팅 어디에 놓았느냐**가 틀렸던 것이고, 필드 단위
측정으로는 절대 보이지 않는다.

돌려보지 않았으면 못 찾았을 결함들:

| 결함 | 증상 | 고친 방식 |
|---|---|---|
| 사용량 집계 동시성 경합 | 15콜·$1.08 로 보고 (실제 6콜·$0.49) | 호출이 자기 사용량을 반환, 케이스가 지역 변수에 누적 |
| `partial`/`insufficient` 정의 중복 | 두 설명이 같은 상황을 서술 | "이 인용이 요구의 **어느 부분에 답하는가**" 시험 |
| `unmet_need` 부풀리기 | 답이 있는 문서를 partial 로 깎음 | "사용자가 요구한 범위를 넘지 마라" |
| `context_dependent` 과탐 | 23건 중 15건이 찍힘 | 대상 명사 유무로 판별, "확신 없으면 false" |
| 안전 지표 사각지대 | `rag_partial` 과소평가를 놓침 | 두 라벨을 같은 방향으로 집계 |
| **`case14` 가 도메인 분기를 가로챔** | **회귀셋 6건이 샘** | 부가 케이스로 강등 |
| **`answer_refused` 가 회피를 거절로 읽음** | **case22 탐지가 무너짐** | 정책·권한·보안으로 좁힘 |
| **빈 `rag_data` 를 '검색 실패'와 합쳐 셈** | **검색 트리거 문제가 임베딩 문제로 보임** | 빈 리스트를 case21 로 분리, 충족도 LLM 생략 |
| **`answer_refused` 가 서비스 장애 문구를 거절로 읽음** | **인프라 장애가 case28(보안 정책)로 집계** | 확정 문구를 코드로 대조해 LLM 이전에 단락 (case9) |

마지막 두 개는 같은 구조다 — **약한 증거가 강한 증거를 가로챘다.** 답변이 나쁘면
여러 관측이 동시에 켜지므로, 라우팅 순서는 발견 순서가 아니라 **증거의 강도 순서**여야 한다.

반복된 교훈: **프롬프트에 지시를 넣을 때는 반대 방향 제약을 같이 넣어야 한다.**
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

python scripts/legacy_run.py --check-llm                 # 1. 서버 규약 확정
python -m pytest tests/ -q                # 2. LLM 없이 도는 부분 (108개)
python scripts/legacy_run.py --synthetic                 # 3. 그 모델이 프롬프트를 따르는지
python scripts/legacy_run.py --input data/logs.json --limit 20   # 4. 실데이터
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
python scripts/legacy_run.py --synthetic --thinking off --out off.jsonl
python scripts/legacy_run.py --no-cache --synthetic --thinking on --out on.jsonl
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

프롬프트 전문은 `python scripts/legacy_run.py --show-prompts`로 예시 입력과 함께 볼 수 있다.

## 사용법

```bash
python3 -m venv venv && ./venv/bin/pip install -r requirements.txt

./venv/bin/python -m pytest tests/ -q          # API 없이 도는 부분 검증
./venv/bin/python scripts/legacy_run.py --show-prompts        # 파이프라인 전체 프롬프트
./venv/bin/python scripts/legacy_run.py --trace C-4002:3     # 케이스 하나의 실제 통과 경로
./venv/bin/python scripts/legacy_run.py --synthetic           # 합성 데이터 회귀 검증 (23건)
./venv/bin/python scripts/legacy_run.py --input data/logs.json --limit 20   # 실데이터, 비용 확인용
./venv/bin/python scripts/legacy_run.py --input data/logs.json --workers 12
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

## 사내 머신으로 무엇을 가져가나

경계는 **`Case`** 다.

```
  로그 → 필터 → Case → 판정 → 출력 JSON
         └ 그쪽 것 ┘   └──── 가져갈 것 ────┘
```

로그를 읽고 필터를 거는 부분은 사내 머신에 이미 있다. 이 저장소의 `conv.py` ·
`filters.py` 는 여기서 검증할 때만 쓴다. **`Case` 를 만들어 넣을 수만 있으면**
나머지는 그대로 돈다.

<!-- copy-list -->
```
src/ragdiag/settings.py    배포마다 바뀌는 값 — 여기부터 열 것
src/ragdiag/schema.py      Case + Step 1/2 출력 (Pydantic, 필드 순서에 의미 있음)
src/ragdiag/taxonomy.py    case 29개 메타데이터와 설명
src/ragdiag/prompts.py     판정 프롬프트 (단계별로 뺄 정보가 여기에 명시됨)
src/ragdiag/backends.py    로컬 LLM / Claude Code CLI / Anthropic API
src/ragdiag/judge.py       LLM 호출, 디스크 캐시, 케이스 단위 병렬
src/ragdiag/decide.py      구 진리표 (judge 가 참조)
src/ragdiag/verify.py      인용 대조 (사전지식 오염 차단)
src/ragdiag/checks.py      코드 검증기 — 언어·길이·포맷·잘림·PII·인용·문법·계산·인젝션·서비스오류
src/ragdiag/route.py       Step 3 진리표 — 관측+검증 → case
src/ragdiag/classify.py    3단계 오케스트레이션
src/ragdiag/output.py      pre_data_format 형태 출력
src/ragdiag/pipeline.py    단계별 함수
```
<!-- /copy-list -->

이 목록은 `tests/test_boundary.py` 가 **실제로 import 해서** 확인한다. 코어 모듈
하나가 입력 계층을 끌어오면 테스트가 깨진다. 문서로만 적어두면 누가 import 하나를
추가하는 순간 조용히 무너지고, 알아채는 건 사내 머신에서 `ImportError` 가 났을
때다.

### 붙이는 법

```python
from ragdiag.backends import backend_from_env
from ragdiag.pipeline import build_outcome, judge_cases, make_judge
from ragdiag.schema import Case

cases = [Case(...) for turn in 그쪽_필터_결과]   # 여기만 새로 쓰면 된다
judge = make_judge(backend_from_env())
results = judge_cases(cases, judge, workers=4)
build_outcome(owners, results).save("conv_parsed.json")
```

`owners` 는 결과를 되돌릴 대화 객체다. `conversation_id` 와 `user` 두 속성만
읽으므로 그쪽 파서가 만든 객체를 그대로 넣어도 된다.

## 대시보드

```bash
pip install streamlit pandas          # 분류 파이프라인에는 불필요하다

streamlit run log_analysis/src/dashboard.py -- \
    --result     outputs/conv_parsed.json \
    --dept-class outputs/class_dept.json \
    --job-class  outputs/class_job.json
```

`--` 가 있어야 한다. 앞은 streamlit 이 먹고 뒤가 스크립트로 넘어간다.
`--dept-class` · `--job-class` 는 없어도 돌아간다 — 부서·직급이 대분류로
접히지 않고 로그 원본 값으로 나올 뿐이다.

`src/run.py` 와 같은 이유로 **`src/` 직하**에 있다. streamlit 은 스크립트가 있는
디렉터리를 `sys.path[0]` 에 넣으므로, `src/ragdiag/` 안에 두면 그 디렉터리가
올라가고 `src/` 는 안 올라가서 `import ragdiag` 가 자기 자신을 못 찾는다.
실제로 한 번 그렇게 깨졌는데 **서버는 정상으로 뜨고 헬스체크도 통과했다** —
streamlit 이 스크립트를 브라우저 접속 시에 실행하고 예외를 화면에만 보이기
때문이다. `tests/test_dashboard.py` 가 스크립트를 끝까지 실행해 그걸 잡는다.

## 구조

```
src/ragdiag/__main__.py    본 진입점 — 분류 · --golden · --legacy-regression
run.py           구 파이프라인 (회귀 기준선) · --inspect · --check-llm

src/ragdiag/         (반입 목록은 위 참고)
  settings.py    배포마다 바뀌는 값을 한 곳에
  pipeline.py    단계별 함수 — 노트북·다른 스크립트에서 부를 수 있게

  ── 여기 전용 (사내 머신에는 그쪽 구현이 있다) ──
  conv.py        conv_eval 파싱, 턴 짝짓기 (N+1 불만 ↔ N 답변·문서)
  filters.py     필터 적용, 점수 재계산, 단계별 탈락 기록
  labels.py      llm_eval / llm_emotion 라벨 테이블과 점수
  load.py        구 포맷 로더 (회귀셋용)
  org.py         조직 분류 대분류/중분류/소분류 (대시보드용)
  survey.py      데이터 실태 조사 (--inspect)
  golden.py      골든셋 채점 (관측 · 판정)
  report.py      구 리포트 (회귀 기준선용)

fixtures/
  observations.py  Step 1 관측 골든셋 44건 (필드별 양성·음성)
  judgments.py     Step 2·3 판정 골든셋 18건 (충족도 10 · 근거 활용 8)
  synthetic.py     구 회귀셋 23건 + 구→신 case 매핑
  pseudo.py        대시보드 데모용 유사 데이터
```

`load.py` · `decide.py` · `report.py` 는 구 파이프라인 전용이다. 새 코드에서 쓰지 말 것 —
회귀 기준선을 그대로 두기 위해 남겨둔 것이지 현행 경로가 아니다.
