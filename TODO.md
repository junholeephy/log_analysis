# 실행 전에 작업 폴더에서 해야 할 것

`{AA}` 쪽에서 만들어야 돌아가는 것들. **이 저장소에는 없다** — 여기서 만들 수 없거나
(운영 실값), 그쪽 것을 쓰기로 한 것(필터)이다.


```
[ ] 1. configs/env.yaml 채우기      네 줄                    운영 실값
[ ] 2. 필터 구현                    todo/filter.md          이 프로젝트의 사정
[ ] 3. 실행 스크립트                todo/scripting.md       어느 프로젝트나
[ ] 4. 작업 폴더 .gitignore         ⚠ 실데이터가 운영 git 에  어느 프로젝트나
[ ] 5. 라벨 taxonomy 문서           2번을 --turns 로 하면 불필요
[ ] 6. 조직 분류 JSON               대시보드에서 부서·직급을 층으로 볼 때만
[ ] 7. 순서대로 점검                아래 4단계
```

**2번만 이 프로젝트의 사정이다.** 필터 로직이 운영 쪽에 이미 있고 그걸 쓰기로 했다.
나머지는 이 규격으로 도는 프로젝트라면 어디나 해야 하는 것들이다 — 실값을 채우고,
감싸는 스크립트를 두고, 작업 폴더에 무엇이 커밋될지 정하는 것.

만드는 방법은 `todo/` 의 두 문서에 있다. 이 파일은 **무엇이 남았나**만 센다.

---

## 1. `{AA}/configs/env.yaml` 채우기

`sync.sh` 가 만들어 둔다. `← 채우세요` 가 붙은 **네 줄**만 채우면 된다.

```yaml
paths:
  venv: /opt/shared/venv       # 이 파이썬으로 갈아타서 실행한다. activate 불필요
llm:
  url: http://<서버>:8000
  key: <키>                    # 서버가 검사하지 않으면 아무 값
  model: <모델 이름>            # 비우면 서버의 첫 모델. 여러 개면 실행마다 달라진다
```

나머지 25개 키는 전부 기본값으로 돈다. 경로(`conv_data` 등)는 실행할 때
`--conv-data` 로 줘도 되므로 여기 채우지 않아도 된다.

> **`{AA}/{BB}/configs/` 가 아니다.** 그쪽은 sync 때마다 통째로 지워진다.
> 거기 두면 프로그램이 거부하면서 옮길 명령을 알려준다.

---

## 2. 필터 구현 → 턴 목록

**규격: [`todo/filter.md`](todo/filter.md)**

필터 로직은 그쪽 것을 쓴다. 내놓을 것은 **어느 턴을 볼지**의 목록 하나다.

```json
[{"user_id": "EMP-0001", "conversation_id": "C-0001", "turn": 3}]
```

JSONL 도 그대로 받는다.

**로그는 자르지 말고 그대로 넘긴다.** 고른 턴만 남기면 전부 0건이 된다 — 직전 턴의
답변이 곧 판정 대상이다. 목록이 곧 필터이므로 로그를 또 자를 이유가 없다.

```bash
python log_analysis/src/run.py --conv-data <로그> --turns <목록>
```

---

## 3. 실행 스크립트

**규격: [`todo/scripting.md`](todo/scripting.md)**

실행은 이 스크립트로 한다. `{AA}` 직하에 둔다 — `{AA}/{BB}` 안에 두면 sync 때
지워진다.

손으로 매번 치지 않는 이유가 둘이다. 인자를 하나 빠뜨려도 프로그램은 기본값으로
돌아버리고(그래서 실행 조건 블록이 있다), 종료 코드를 사람이 보고 판단하면 `1` 과
`2` 의 차이가 묻힌다.

기댈 것은 **종료 코드**뿐이다 — 화면 문구나 표 모양은 계약이 아니다.

| 코드 | 뜻 |
|---|---|
| `0` | 정상 |
| `1` | 돌긴 했는데 결과가 온전치 않다. RUN SUMMARY 를 읽어야 한다 |
| `2` | 시작도 못 했다. 사람이 고쳐야 한다 |

결과는 `output/conv_parsed_<시각>.json` 으로 쌓인다. `ls -1 ... | tail -1` 로 최신을
고른다 — 이름 규칙이 계약이다.

---

## 4. ⚠ 작업 폴더의 `.gitignore`

**이게 제일 급하다.** `sync.sh` 는 `.staging/` 만 무시 목록에 넣는다. 나머지는 그대로
두면 운영 git 에 커밋된다.

| | 무엇이 들어 있나 | 커밋해도 되나 |
|---|---|---|
| `.cache/` | **LLM 판정 응답.** 실데이터에서 뽑은 관측·인용이 그대로 | 판단 필요 |
| `data/` | 실데이터 | 대개 아니다 |
| `configs/env.yaml` | 접속 정보 | 판단 필요 |
| `output/` | 분류 결과 · RUN SUMMARY | 남기고 싶을 수 있다 |

```bash
cd {AA}
cat >> .gitignore <<'EOF'
.cache/
data/
EOF
```

`{AA}/{BB}` 사본이 커밋되는 것은 **목적이다** — 결과가 반출되지 않는 상황에서
"어떤 코드로 돌렸는지"가 남는 유일한 형태다. 나머지는 의도한 것만 남긴다.

---

## 5. 라벨 taxonomy 문서 — 2번을 `--filter-data` 로 할 때만

라벨 이름·점수는 운영 코드값이라 저장소에 없다. 이 저장소의 필터로 **라벨·점수
조건**을 걸려면 실값이 필요하다.

```
{AA}/configs/query_taxonomy.md       형식: A. 이름 -> 점수
{AA}/configs/emotion_taxonomy.md
```
```yaml
labels:
  query:   configs/query_taxonomy.md
  emotion: configs/emotion_taxonomy.md
```

**2번을 `--turns` 로 하면 필요 없다.** 무엇을 볼지는 이미 정해져서 온다.

빠뜨린 채 라벨 조건을 건 필터를 주면 계산 전에 죽는다. 그대로 두면 필터가 에러 없이
0건을 돌려주기 때문이다.

---

## 6. 조직 분류 JSON — 대시보드에서 층으로 볼 때만

대시보드 사이드바가 부서·직급을 **대분류 → 중분류 → 소분류**로 좁혀 고르게 하려면
조직 체계가 필요하다. 없으면 원본 값(팀 이름·직급명)으로만 고른다 — 팀이 수십 개면
그 목록에서는 못 고른다.

```
{AA}/configs/dept_class.json
{AA}/configs/job_class.json
```
```yaml
paths:
  dept_class: configs/dept_class.json
  job_class:  configs/job_class.json
```

형식은 세 층이다. 루트 키 이름은 무엇이든 받는다.

```json
{ "dept_classes": [
  { "id": 1, "name": "경영지원본부",
    "subclasses": [{ "name": "인사", "items": ["인사팀", "인재개발팀"] }] } ] }
```

`items` 가 **로그의 실제 값**과 맞아야 한다 (`db_dept_name` · `job_grade` 등).

> **어느 필드에 붙는지는 값을 대조해 자동 판별한다.** 파일 이름으로 추측하지 않는다 —
> 잘못 붙이면 **에러 없이 전부 `(미분류)`** 가 되어 알아채기 어렵다. 매칭률은 사이드바에
> 뜬다: `· 부서 ← db_dept_name 87%`. 체계에 없는 값은 `(미분류)`로 모여 표에 남는다 —
> 버리면 "그 조직은 문제가 없다"로 잘못 읽힌다.

---

## 7. 순서대로 점검

위에서부터. **앞이 깨지면 뒤는 볼 필요 없다.**

```bash
cd {AA}

python log_analysis/src/run.py --check-llm
#   서버 규약·모델·1회 소요시간. 전체가 몇 분인지 여기서 나온다

python -m pytest log_analysis/tests -q
#   LLM 없이 도는 부분. 실패하면 반입 자체가 잘못됐다

python log_analysis/src/run.py --dry-run
#   합성 데이터로 끝까지. 깨지면 환경 문제이지 데이터 문제가 아니다

python log_analysis/src/run.py --conv-data <실데이터> --turns <목록> --limit 50
#   RUN SUMMARY 의 contract 줄이 첫 사이클의 실제 수확이다
```

**`--limit 50` 다음에 바로 전체로 가지 마라.** 계약이 깨끗해진 뒤에 간다 — 틀린 계약
위에서 뽑은 숫자는 믿을 수 없다.

---

## 돌린 뒤에 — 가지고 나올 것

결과 파일도 플롯도 로그도 반출되지 않는다. 돌아오는 것은 **화면을 보고 옮겨 적은 것**
뿐이다. `docs/insights/TEMPLATE.md` 를 채워서 가지고 나온다.

| 본 것 | 여기서 고칠 곳 |
|---|---|
| RUN SUMMARY 의 `contract` 줄 (전문) | `src/ragdiag/contracts.py` — `note` 에 확인된 사실도 |
| `truncated` · `filter FP` · `failed at` 줄 | 다음 실행의 `--thinking` · 필터 · 프롬프트 |
| "코드 한 줄만 고치면 되는데" 했던 순간 | 그 값을 설정이나 인자로 승격 |
| 의존성 충돌 메시지 | `requirements.txt` |
| 라벨이 틀린 표본 (case 와 이유) | `route.py` 의 순서, `prompts.py` 의 관측 정의 |
