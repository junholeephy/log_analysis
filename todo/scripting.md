# 실행 스크립트 규격 — `{AA}` 에서 이걸 감싸 쓸 때

> **이건 이렇게 도는 프로젝트라면 어디나 해당한다.** 반입본을 손으로 치지 않고
> 감싸는 스크립트로 돌리는 것은 이 규격을 쓰는 어느 프로젝트에서나 같다 — 인자를
> 하나 빠뜨려도 프로그램은 기본값으로 돌아버리고, 종료 코드를 사람이 눈으로 보면
> `1` 과 `2` 의 차이가 묻힌다.

실행은 작업 폴더 쪽 스크립트로 한다. 이 문서는 그 스크립트가 기댈 것을 정한다 —
**여기 적힌 것만 계약이다.** 화면에 찍히는 문구나 표의 모양은 언제든 바뀔 수 있으니
파싱하지 마라.

`filter.md` 가 "무엇을 골라 넘기나"를 정한다면, 이 문서는 "그걸 어떻게 돌리고 결과를
어떻게 집어 오나"를 정한다.

---

## 1. 스크립트를 어디에 두나

```
{AA}/
  run_daily.sh          ← 여기. 살아남는다
  configs/env.yaml      ← 설정
  log_analysis/         ← ✗ 여기 두면 sync 때 지워진다
  output/               ← 결과가 쌓이는 곳
```

**`{AA}/{BB}` 안에는 아무것도 두지 마라.** 갱신 때마다 `rm -rf` 로 통째 교체된다.
설정도 스크립트도 노트북도 전부 `{AA}` 직하다.

실행 위치도 `{AA}` 가 기준이다. 상대 경로가 전부 거기서 풀린다.

---

## 2. 무엇을 부르나

```bash
python log_analysis/src/run.py [옵션]
```

`PYTHONPATH` 도 설치도 필요 없다. 파이썬이 스크립트가 있는 디렉터리를 `sys.path[0]`
에 넣으므로 `src/run.py` 옆의 `ragdiag/` 가 그대로 잡힌다.

**`activate` 도 필요 없다.** `configs/env.yaml` 의 `paths.venv` 에 적어두면 진입점이
그 파이썬으로 갈아타서 다시 시작한다. 스크립트에서 venv 를 신경 쓰지 않아도 된다.

```yaml
paths:
  venv: /opt/shared/venv
```

---

## 3. 값을 어디에 두나

우선순위는 **CLI > 설정 > 환경변수 > 기본값**이다.

| 두는 곳 | 무엇 |
|---|---|
| `configs/env.yaml` | 매번 같은 것 — 접속 정보, venv, 임계값 |
| CLI 인자 | 실행마다 달라지는 것 — 입력 파일, 기간, `--limit` |

`--config` 를 안 줘도 **실행 위치의 `configs/env.yaml` 을 자동으로 쓴다.** 매번
가리키게 하면 한 번 빼먹는 순간 조용히 기본값으로 돈다.

환경변수(`LLM_API_URL` 등)는 설정보다 아래다. `.bashrc` 에 남은 옛 주소가 설정을
이기면 안 되기 때문이다.

---

## 4. 종료 코드

**이게 스크립트가 기댈 유일한 신호다.**

| 코드 | 뜻 | 스크립트가 할 일 |
|---|---|---|
| `0` | 정상 (`OK` · `DRY RUN`) | 다음으로 |
| `1` | **돌긴 했는데 결과가 온전치 않다** (`NO DATA` · `PARTIAL`) | RUN SUMMARY 를 읽어야 한다. 재시도해도 같다 |
| `2` | **시작도 못 했다** (설정·입력·백엔드) | 사람이 고쳐야 한다 |

`1` 과 `2` 를 가르는 기준은 "계산을 시작했나"다. `2` 는 인자 하나, 파일 하나 때문에
계산 전에 죽은 것이라 고치고 다시 돌리면 된다. `1` 은 이미 돈 것이고, 결과 파일도
RUN SUMMARY 도 남아 있다.

```bash
python log_analysis/src/run.py --conv-data "$LOG" --turns "$TURNS"
case $? in
  0) ;;
  1) echo "결과가 온전치 않다. RUN SUMMARY 확인" >&2 ;;
  2) echo "실행 조건이 잘못됐다. 위 메시지 확인" >&2; exit 2 ;;
esac
```

---

## 5. stdout 과 stderr

| | 무엇이 나오나 |
|---|---|
| **stdout** | RUN SUMMARY · 분류 결과 집계표 |
| **stderr** | 실행 조건 · 필터 리포트 · 진행 상황 · 오류 · 결과 파일 경로 |

로그로 남길 것은 stdout 이고, 사람이 보면서 판단할 것은 stderr 다.

```bash
python log_analysis/src/run.py ... > "logs/$(date +%F).txt"   # 요약만 남긴다
```

> **표를 파싱하지 마라.** 집계표의 모양은 바뀐다. 숫자가 필요하면 결과 JSON 에서
> 세라 — 그쪽이 계약이다.

---

## 6. 결과 파일 찾기

파일 이름에 **끝난 시각**이 붙는다. 같은 데이터를 여러 번 돌려도 덮어쓰지 않는다.

```
output/conv_parsed_20260901-153854.json    분류 결과
output/run_summary_20260901-153854.txt     RUN SUMMARY 사본 (같은 스탬프)
```

경로는 stderr 에 `결과: ...` 로 찍히지만 **거기서 긁어내지 마라.** 이름 규칙이
계약이므로 최신 것을 고르면 된다:

```bash
LATEST=$(ls -1 output/conv_parsed_*.json | tail -1)
```

정렬이 곧 시간순이다 (`YYYYMMDD-HHMMSS`). 대시보드도 같은 방식으로 고른다.

경로를 고정해야 하면 `--out` 을 주면 된다. 그러면 시각 스탬프를 붙이지 않는다.

---

## 7. 결과 JSON 의 모양

입력 로그와 같은 `users → conversations → turns` 구조를 유지한다. 원본 옆에 두고
조인할 수 있다. 판정한 턴에만 `classification` 이 붙는다.

```json
{"analysis_results": [{"user_id": "...", "conversations": [{"turns": [
  {"turn": 3, "classification": {
      "case_id": "case20", "case_name": "Retrieve 실패",
      "type_id": "TYPE5", "category": "category_2",
      "confidence": "medium", "reason": "...", "secondary_cases": [...],
      "evidence": {...}, "llm_calls": 2, "answered_turn": 2}}]}]}]}
```

판정에 실패한 턴은 `classification` 에 `error` 만 들어간다. 앞에 `[observe]` 같은
단계 이름이 붙으므로 어디서 깨졌는지 세어볼 수 있다.

```json
{"classification": {"error": "[sufficiency] Truncated: 토큰 한도(16,000)를 ..."}}
```

---

## 8. 하지 말 것

| | 왜 |
|---|---|
| `{AA}/{BB}` 안의 파일을 고치거나 두기 | sync 때마다 지워진다 |
| 화면 출력을 파싱해서 숫자 뽑기 | 표 모양은 계약이 아니다. 결과 JSON 을 써라 |
| `pip install --upgrade` · `--force-reinstall` | 공용 환경을 조용히 깨뜨린다 |
| 패키지를 venv 에 설치 | `src/run.py` 를 직접 부르면 필요 없다 |
| 종료 코드 `1` 에서 자동 재시도 | 이미 돈 것이다. 같은 결과가 나온다 |
| 결과 파일을 덮어쓰게 `--out` 고정 | 어느 실행 것인지 사후에 알 수 없다 |

---

## 9. 통째 예시

```bash
#!/usr/bin/env bash
# {AA}/run_daily.sh
set -euo pipefail
cd "$(dirname "$0")"          # {AA} 가 기준이다

LOG=${1:?분석할 로그 경로를 주세요}
TURNS=${2:-}                  # 필터가 고른 턴 목록 (filter.md)

args=(--conv-data "$LOG")
[[ -n "$TURNS" ]] && args+=(--turns "$TURNS")

# 설정(configs/env.yaml)은 자동으로 잡힌다. venv 도 거기 적혀 있다.
set +e
python log_analysis/src/run.py "${args[@]}"
code=$?
set -e

if [[ $code -eq 2 ]]; then
  echo "실행 조건이 잘못됐습니다. 위 메시지를 그대로 옮겨 적으세요." >&2
  exit 2
fi

LATEST=$(ls -1 output/conv_parsed_*.json | tail -1)
echo "결과: $LATEST"
[[ $code -eq 1 ]] && echo "일부가 온전치 않습니다. RUN SUMMARY 를 확인하세요." >&2
exit 0
```

---

## 10. 먼저 돌려볼 것

스크립트를 짜기 전에 순서대로 한 번씩:

```bash
python log_analysis/src/run.py --check-llm    # 서버 규약·모델·1회 소요시간
python log_analysis/src/run.py --dry-run      # 합성 데이터로 끝까지
```

`--check-llm` 이 전체 예상 소요를 알려주므로, 스크립트에 타임아웃을 걸 거라면
그 숫자를 기준으로 잡으면 된다.
