#!/usr/bin/env python
"""conv_eval 로그를 taxonomy case 로 분류한다.

  python <저장소>/src/run.py --conv-data <로그> --output-dir outputs --dry-run
  python <저장소>/src/run.py --conv-data <로그> --filter-data <필터> --output-dir outputs

필터가 고른 턴만 3단계로 분류한다:

  Step 1  관측 추출     LLM 1회 · rag_data 를 주지 않는다
  Step 2  조건부 검증   코드 검증은 항상 · LLM 검증은 도메인 질문일 때만
  Step 3  라우팅        코드. case 는 LLM 이 고르지 않는다

결과는 pre_data_format 형태로 나온다. 원본 필드는 그대로 두고 분류 결과는
`classification` 아래에 모은다.

설정은 --config 로 준다. 운영 환경에서는 코드를 못 고치므로 바뀔 값은 전부 거기 있다.
CLI 플래그는 설정을 덮어쓴다 - 한 번만 다르게 돌려볼 때 쓴다.

설정 없이 돌리려면 환경변수 두 개면 된다:
  export LLM_API_URL=http://<서버>:8000
  export LLM_API_KEY=<키>
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from ragdiag.backends import (
    KEY_VARS,
    URL_VARS,
    JudgeError,
    backend_from_env,
    env_first,
)
from ragdiag import contracts, settings
from ragdiag.config import ConfigError
from ragdiag.filters import LabelTableMissing
from ragdiag.config import apply as apply_config
from ragdiag.config import load as load_config
from ragdiag.judge import Judge
from ragdiag.pipeline import (
    build_outcome,
    judge_cases,
    load_and_select,
    select_turns,
    make_judge,
)
from ragdiag.summary import (
    Conditions,
    RunSummary,
    Timer,
    peak_memory_gb,
    version,
)

# --output-dir 도 설정도 없을 때. 실행 위치 기준이라 운영 환경에서는
# 작업 폴더 아래에 생긴다.
DEFAULT_OUTPUT_DIR = "output"


def make_backend(args, config=None, trace=None):
    """CLI 인자 > 설정 > 환경변수 순으로 고른다.

    trace(Conditions)를 주면 각 값이 **어디서 왔는지** 함께 적는다. 값만 찍으면
    "왜 저 값이지"를 못 푼다 - 운영 환경에는 .bashrc 의 환경변수, AA/configs/env.yaml,
    CLI 플래그가 겹쳐 있고 셋 다 화면에 안 보인다. 어느 쪽이 이겼는지가 안 보이면
    설정을 고쳐도 안 먹는 이유를 알 수 없다.
    """
    src: dict[str, str] = {}

    def pick(flag, key, default=None, flag_name=""):
        if flag is not None:
            src[key] = flag_name or "--" + key.rsplit(".", 1)[-1].replace("_", "-")
            return flag
        if config is not None and config.get(key) is not None:
            src[key] = f"설정 {key}"
            return config.get(key)
        src[key] = "기본값"
        return default

    # 이 진입점이 아는 백엔드는 하나다. 예전에는 LLM_API_URL 이 없으면 claude CLI 로
    # 떨어졌는데, 그 경로가 tools/ 로 나가면서 자동 선택도 같이 사라졌다.
    # 주소가 없으면 backend_from_env 가 무엇을 export 하라고 알려준다.
    backend = pick(args.backend, "llm.backend", "local")
    if src["llm.backend"] == "기본값":
        found = next((n for n in URL_VARS if os.environ.get(n)), None)
        src["llm.backend"] = f"기본값 · 주소는 {found}" if found else "기본값"
    model = pick(args.model, "llm.model")
    timeout = pick(args.timeout, "llm.timeout_sec", 600)
    json_mode = pick(args.json_mode, "llm.json_mode", "auto")
    thinking = pick(args.thinking, "llm.thinking", "auto")
    max_tokens = pick(args.max_tokens, "llm.max_tokens", 16000)

    def note(built):
        if trace is None:
            return built
        trace.add("백엔드", backend, src["llm.backend"])
        if backend != "cli":
            trace.add("주소", url or "(없음)", src["llm.url"])
            # 키는 값을 절대 찍지 않는다. 어디서 왔는지만 적는다.
            trace.add("키", "설정됨" if key else "(없음)", src["llm.key"])
        trace.add("모델", built.model,
                  src["llm.model"] if model else "자동 탐지",
                  "" if model else "서버 /v1/models 의 첫 항목")
        if backend != "cli":
            # 값마다 출처가 달라서 한 줄로 묶으면 어느 게 어디서 왔는지 사라진다.
            # 기본값이 아닌 것에만 출처를 붙인다.
            def tag(key, text):
                where = src[key]
                return text if where == "기본값" else f"{text}({where})"
            trace.add("판정", "  ".join([
                tag("llm.json_mode", f"json_mode={json_mode}"),
                tag("llm.thinking", f"thinking={thinking}"),
                tag("llm.max_tokens", f"max_tokens={max_tokens:,}")]))
        return built

    if backend != "local":
        # 이 진입점은 운영 환경에서 도는 경로만 안다. claude CLI 와 Anthropic API 는
        # 운영 환경에서 호출이 전부 실패하므로 tools/ 에 있고, src/ 는 tools/ 를
        # import 하지 않는다 (규격 §1.4).
        raise JudgeError(
            f"--backend {backend} 는 이 진입점에 없습니다.\n"
            "  운영 환경에서 실패할 호출은 src/ 에 두지 않습니다 (규격 §1.4 · C8).\n"
            "  개발 장비에서 그 백엔드로 돌리려면:\n"
            "    python tools/dev_run.py --backend " + backend + " ...\n"
            "  운영 환경·서버 경로는 --backend local 입니다."
        )

    url = pick(args.base_url, "llm.url", flag_name="--base-url")
    if url is None:
        for name in URL_VARS:
            if os.environ.get(name):
                url, src["llm.url"] = os.environ[name], f"환경변수 {name}"
                break
    key = pick(args.api_key, "llm.key", flag_name="--api-key")
    if key is None:
        for name in KEY_VARS:
            if os.environ.get(name):
                key, src["llm.key"] = os.environ[name], f"환경변수 {name}"
                break
    return note(backend_from_env(
        base_url=url, api_key=key, model=model,
        json_mode=json_mode, thinking=thinking, max_tokens=max_tokens,
        timeout=timeout,
    ))


def _probe_case():
    """점검용 최소 케이스. 문서에 답이 있는데 답변이 무시한 상황이라 정답은 ignored."""
    from ragdiag.schema import Case

    return Case(
        case_id="probe", user_id="probe", dept="-", job_grade="-", job_name="-",
        position_name="-", conversation_id="-", turn=0,
        pre_queries=["국내 출장 식비 상한이 얼마인가요?"],
        llm_ans_on_last_q="출장 식비는 회사 규정에 따라 지급되며 부서별로 다를 수 있습니다.",
        current_query="정확한 금액이요.",
        rag_chunks=["국내 출장 식비는 1일 3만원을 상한으로 한다.",
                    "국내 출장 숙박비는 1박 8만원을 상한으로 한다."],
    )


def check_llm(args, config=None) -> int:
    """로컬 LLM 서버가 붙는지, 구조화 출력을 어떻게 강제할 수 있는지 점검한다.

    **에어갭 장비에서 가장 먼저 돌릴 것이다.** 거기서는 물어볼 데가 없으므로 스스로
    진단이 되어야 하고, 전체를 돌리기 전에 서버 규약과 1회 소요시간을 확정해야
    한다 - 그래야 전체가 몇 분인지 알고 시작한다.

    이 함수는 로컬 서버만 본다. 개발 장비 전용 백엔드를 쓰지 않으므로 tools/ 가
    아니라 여기 있다 (규격 §1.4).
    """
    import time

    from ragdiag import prompts
    from ragdiag.schema import GroundingCheck

    try:
        backend = make_backend(args, config)
    except JudgeError as e:
        print(e, file=sys.stderr)
        return 2

    print(f"서버 : {backend.base_url}")
    print(f"모델 : {backend.model}" + ("   (자동 탐지)" if backend.discovered else ""))
    if backend.discovered:
        others = [m for m in backend.list_models() if m != backend.model]
        if others:
            print(f"       서버의 다른 모델: {', '.join(others)}")
            print(f"       다른 걸 쓰려면 --model <이름>")
    print(f"추론 모드 : {backend.thinking}   (auto = 서버 기본값 유지)")

    try:
        started = time.monotonic()
        mode = backend.negotiate(GroundingCheck)
        print(f"구조화 출력 강제 방식 : {mode}   ({time.monotonic() - started:.1f}s)")
        if len(backend.negotiation_log) > 1:
            print("  협상 과정:")
            for line in backend.negotiation_log:
                print(f"    {line}")
    except JudgeError as e:
        print(f"\n연결 실패:\n{e}", file=sys.stderr)
        return 1

    if mode == "none":
        print("\n  경고: 이 서버는 구조화 출력을 강제하지 못합니다.")
        print("  프롬프트만으로 JSON을 요구하게 되어 재시도가 잦아집니다.")
        print("  프록시(LiteLLM 등)를 거치고 있다면 파라미터를 버리고 있을 수 있습니다.")

    # 규약뿐 아니라 모델이 지시를 따르는지도 봐야 한다.
    print("\n실제 판정 1회 시도...")
    started = time.monotonic()
    try:
        result, usage = backend.complete(
            prompts.GROUNDING_SYSTEM,
            prompts.grounding_user_message(_probe_case()),
            GroundingCheck,
            prompts.output_contract(GroundingCheck),
        )
    except JudgeError as e:
        print(f"\n판정 실패:\n{e}", file=sys.stderr)
        return 1

    elapsed = time.monotonic() - started
    print(f"  answer_used_rag : {result.answer_used_rag}   (기대: ignored)")
    print(f"  소요            : {elapsed:.1f}s")
    print(f"  토큰            : 입력 {usage.input_tokens:,} / 출력 {usage.output_tokens:,}")
    if result.answer_used_rag != "ignored":
        print("\n  경고: 이 탐침은 'ignored' 가 정답입니다. 문서에 답이 있는데 답변이")
        print("  일반론으로 때운 경우입니다. 모델이 지시를 못 따르고 있을 수 있습니다.")

    if backend.fallbacks:
        print(f"\n  추론이 답에 도달 못 해 {len(backend.fallbacks)}회 폴백했습니다.")
        print("  전체 실행은 --thinking off 로 시작하는 편이 낫습니다.")

    workers = args.workers or settings.DEFAULT_WORKERS
    print(f"\n턴 1건당 LLM 호출은 최대 3회다. 1,000턴이면 대략 "
          f"{elapsed * 1000 * 2 / max(workers, 1) / 60:.0f}분 "
          f"(호출 2,000회 가정, 동시 {workers}).")
    return 0


def run_legacy_regression(args, backend=None) -> int:
    """구 회귀셋을 새 파이프라인으로 돌려 판별력이 유지되는지 본다.

    이 23건은 실제 LLM 으로 검증된 유일한 케이스 집합이다. 관측 골든셋이
    **관측 하나하나**를 재는 것과 달리, 이건 관측이 조합되어 case 로 가는
    **경로**를 잰다. 층이 달라서 한쪽만으로는 부족하다 — 실제로 골든셋이 98%
    였을 때 회귀셋은 15/23 이었고, 그 차이가 라우팅 순서의 결함을 드러냈다.
    """
    import collections

    sys.path.insert(0, str(Path(__file__).parent))
    from ragdiag.fixtures.synthetic import build, expected_cases

    from ragdiag.classify import classify_all
    from ragdiag.load import parse_cases

    data, expected = build()
    cases = parse_cases(data)
    if args.limit:
        cases = cases[: args.limit]

    try:
        backend = backend or make_backend(args)
    except JudgeError as e:
        print(e, file=sys.stderr)
        return 2

    judge = Judge(backend, cache_dir=None if args.no_cache else ".cache")
    print(f"구 회귀셋 {len(cases)}건 · {backend.model} · 동시 {args.workers}",
          file=sys.stderr)
    results = classify_all(cases, judge, max_workers=args.workers)

    hits, rows = 0, []
    for result in results:
        meta = expected[result.case.case_id]
        want = expected_cases(result.case.conversation_id, result.case.turn,
                              meta["expect"][0])
        got = (result.classification.primary_case if result.classification
               else f"ERROR:{result.error}")
        ok = got in want
        hits += ok
        rows.append((ok, meta["trap"], result, want, got))

    print("=" * 78)
    print(f"구 회귀셋 -> 신 taxonomy   일치 {hits}/{len(rows)}")
    print("=" * 78)
    for ok, trap, result, want, got in rows:
        secondary = (result.classification.secondary_cases
                     if result.classification else [])
        extra = f"  +{','.join(secondary)}" if secondary else ""
        print(f"{'OK ' if ok else 'X  '}[{trap:<18}] "
              f"{result.case.conversation_id}:{result.case.turn:<4} -> {got}{extra}")
        if not ok:
            print(f"      기대: {sorted(want)}")

    by_trap = collections.defaultdict(list)
    for ok, trap, *_ in rows:
        by_trap[trap].append(ok)
    print("\n함정 유형별:")
    for trap in sorted(by_trap):
        print(f"  {trap:<20} {sum(by_trap[trap])}/{len(by_trap[trap])}")
    print(f"\nLLM 호출 {sum(r.n_calls for r in results)}회")
    return 0 if hits == len(rows) else 1


def run_golden(args, backend=None) -> int:
    """Step 1 관측만 돌려 필드별 일치율을 잰다. Step 2·3 은 호출하지 않는다."""
    sys.path.insert(0, str(Path(__file__).parent))
    from ragdiag.fixtures.observations import build

    from ragdiag.conv import parse_conversations, to_case
    from ragdiag.golden import FieldScore, render, score_observation

    raw, expected = build()
    conversations = parse_conversations(raw)

    cases = []
    for conv in conversations:
        followup = max(t.turn for t in conv.turns)
        case = to_case(conv, followup, args.history_turns or 10**6)
        if case and case.case_id in expected:
            cases.append((case, expected[case.case_id]))
    if args.limit:
        cases = cases[: args.limit]

    try:
        backend = backend or make_backend(args)
    except JudgeError as e:
        print(e, file=sys.stderr)
        return 2

    judge = Judge(backend, cache_dir=None if args.no_cache else ".cache")
    print(f"관측 골든셋 {len(cases)}건 · {backend.model} · 동시 {args.workers}",
          file=sys.stderr)

    from concurrent.futures import ThreadPoolExecutor

    def observe(entry):
        case, meta = entry
        try:
            obs, _ = judge.observe(case)
            return meta, obs, None
        except Exception as e:
            return meta, None, f"{type(e).__name__}: {e}"

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        outcomes = list(pool.map(observe, cases))

    scores: dict[str, FieldScore] = {}
    per_case: dict[str, list[str]] = {}
    errors: list[tuple[str, str]] = []
    for meta, obs, error in outcomes:
        if error:
            errors.append((meta["id"], error))
            continue
        per_case[meta["id"]] = score_observation(meta["id"], meta["expect"], obs, scores)

    print(render(scores, per_case, errors))
    return run_judge_golden(args, judge)


def run_judge_golden(args, judge) -> int:
    """Step 2·3 판정을 채점한다. 관측 채점과 층이 다르다."""
    from concurrent.futures import ThreadPoolExecutor

    from ragdiag.fixtures.judgments import GROUNDING, SUFFICIENCY

    from ragdiag.golden import JudgeScore, render_judge, score_sufficiency
    from ragdiag.schema import Case
    from ragdiag.verify import verify_evidence

    def make_case(chunks, question="", answer=""):
        return Case(case_id="golden", user_id="-", dept="-", job_grade="-",
                    job_name="-", position_name="-", conversation_id="-", turn=0,
                    pre_queries=[question] if question else ["-"],
                    llm_ans_on_last_q=answer, current_query="-", rag_chunks=chunks)

    class _Need:
        def __init__(self, q, n):
            self.resolved_question, self.unmet_need = q, n

    suf, gnd = JudgeScore(), JudgeScore()

    def judge_suf(entry):
        case = make_case(entry["chunks"])
        judgment, _ = judge.judge_sufficiency_from(
            case, _Need(entry["question"], entry["unmet_need"]))
        return entry, judgment, verify_evidence(judgment.evidence, entry["chunks"])

    def judge_gnd(entry):
        case = make_case(entry["chunks"], answer=entry["answer"])
        check, _ = judge.check_grounding(case)
        return entry, check

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for entry, judgment, citation in pool.map(judge_suf, SUFFICIENCY):
            score_sufficiency(entry, judgment, citation, suf)
        for entry, check in pool.map(judge_gnd, GROUNDING):
            gnd.verdict_total += 1
            if check.answer_used_rag == entry["expect"]:
                gnd.verdict_hits += 1
            else:
                gnd.misses.append((entry["id"], entry["expect"],
                                   check.answer_used_rag, entry["note"]))

    print(render_judge(suf, gnd))
    return 0


def main(argv=None, backend=None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m ragdiag",
        description="conv_eval 로그를 taxonomy case 로 분류",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--config", help="설정 YAML. configs/env.example.yaml 참고")

    p.add_argument("--conv-data", help="conv_eval JSON 경로 (설정을 덮어쓴다)")
    p.add_argument("--golden", action="store_true",
                   help="Step 1 관측 골든셋을 돌려 필드별 일치율을 잰다")
    p.add_argument("--legacy-regression", action="store_true",
                   help="구 회귀셋 23건을 새 파이프라인으로 돌려 대조한다")
    p.add_argument("--turns", metavar="FILE",
                   help="운영 필터가 고른 턴 목록 (conversation_id + turn). "
                        "주면 여기 필터를 걸지 않는다 — 라벨 실값도 필요 없다")
    p.add_argument("--filter-data", "--filter", dest="filter_data",
                   help="필터 JSON. 없으면 진단 가능한 후속 턴 전부")
    p.add_argument("--output-dir", dest="output_dir",
                   help="산출물을 넣을 디렉터리 (기본 output). 없으면 만든다. 파일명에 끝난 시각이 붙는다")
    p.add_argument("--out", help="결과 파일 경로를 직접 지정. --output-dir 와 시각 스탬프를 무시한다")

    # 아래 기본값은 전부 None 이다. 설정과 CLI 를 구분하기 위해서다 -
    # argparse 기본값을 넣으면 "사용자가 준 것"과 "기본값"을 못 가린다.
    p.add_argument("--check-llm", action="store_true",
                   help="로컬 LLM 서버 점검 — 모델·강제방식·1회 소요시간. "
                        "운영 환경에서 전체를 돌리기 전에 먼저 돌린다")
    p.add_argument("--backend", choices=["local", "cli", "api"],
                   help="local: OpenAI 호환 서버 / cli: claude -p "
                        "(개발 장비 전용 — 저장소에 없다) / api: Anthropic SDK")
    p.add_argument("--base-url", help="LLM 주소 (또는 $LLM_API_URL)")
    p.add_argument("--api-key", help="(또는 $LLM_API_KEY)")
    p.add_argument("--model", help="생략하면 서버의 /v1/models 에서 자동 탐지")
    p.add_argument("--json-mode",
                   choices=["auto", "json_schema", "guided_json", "json_object", "none"])
    p.add_argument("--thinking", choices=["auto", "on", "off"])
    p.add_argument("--max-tokens", type=int)
    p.add_argument("--timeout", type=int)

    p.add_argument("--limit", type=int, help="앞에서 N건만 (비용 확인용)")
    p.add_argument("--workers", type=int, help="동시 실행 턴 수")
    p.add_argument("--history-turns", type=int,
                   help="Step 1 에 넘길 이전 질문 개수 상한. 0 이면 제한 없음")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="합성 데이터 스모크. 필터까지만 적용하고 LLM 을 부르지 않는다")
    args = p.parse_args(argv)

    # 설정은 계산 전에 읽고 검증한다. 30분 뒤에 키 하나로 죽으면 사이클 하나를 버린다.
    try:
        config = load_config(args.config)
    except ConfigError as e:
        print(e, file=sys.stderr)
        return 2
    changed = apply_config(config)
    if changed:
        print(f"설정 {config.source} 적용: {', '.join(changed)}", file=sys.stderr)

    if args.check_llm:
        return check_llm(args, config)
    if args.golden:
        return run_golden(args, backend)
    if args.legacy_regression:
        return run_legacy_regression(args, backend)

    conv_data = args.conv_data or config.get("paths.conv_data")

    # 데이터 없는 --dry-run 은 합성 데이터로 도는 스모크다. 여기서 실패하면
    # 환경 문제이지 데이터 문제가 아니다 - 그 둘을 갈라 보는 것이 목적이다.
    synthetic = False
    if not conv_data and args.dry_run:
        synthetic = True
    elif not conv_data:
        print("분석할 로그가 없습니다. --conv-data 로 경로를 주거나, "
              "--dry-run 으로 합성 데이터 스모크를 먼저 돌려보세요.", file=sys.stderr)
        return 2
    elif not Path(conv_data).exists():
        print(f"로그 파일이 없습니다: {conv_data}", file=sys.stderr)
        return 2

    filter_path = args.filter_data or config.get("paths.filter_data")
    turns_path = args.turns or config.get("paths.turns")
    if turns_path and filter_path:
        # 둘 다 "어느 턴을 볼지"를 정한다. 같이 주면 어느 쪽이 이겼는지 안 보인다.
        print("--turns 와 --filter-data 는 같이 쓸 수 없습니다. 둘 다 어느 턴을 "
              "볼지 정하는 것이라, 같이 주면 무엇이 적용됐는지 알 수 없습니다.",
              file=sys.stderr)
        return 2

    # --out(파일)이 --output-dir(디렉터리)보다 우선한다. 둘 다 없으면 현재 위치.
    out_dir = args.output_dir or config.get("paths.output_dir") or DEFAULT_OUTPUT_DIR
    out_path = args.out or config.get("paths.out")
    # 끝난 시각을 파일 이름에 박는다. 같은 데이터를 여러 번 돌리거나 설정을 바꿔
    # 다시 돌렸을 때 어느 것이 언제 것인지 파일 이름만 보고 알 수 있어야 한다 —
    # 운영 환경에서는 결과를 반출할 수 없어 이 파일들이 그 자리에 계속 쌓인다.
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if not out_path:
        out_path = str(Path(out_dir) / f"conv_parsed_{stamp}.json")
    if out_dir:
        # 없으면 만든다. 30분 돌린 뒤 디렉터리가 없어서 못 쓰면 그 사이클을 버린다.
        Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    workers = args.workers or config.get("run.workers")
    limit = args.limit or config.get("run.limit")
    history = args.history_turns if args.history_turns is not None else None

    summary = RunSummary(version=version(),
                         args=" ".join(argv if argv is not None else sys.argv[1:]))
    timer = Timer().__enter__()

    # 무엇을 읽어 무엇으로 돌리는지. **계산을 시작하기 전에** 찍는다 - 다 돌고
    # 나서 "모델이 그거였네"를 알면 이미 늦었다.
    def origin(flag, key: str, flag_name: str) -> str:
        if flag is not None:
            return flag_name
        if config.get(key) is not None:
            return f"설정 {key}"
        return "기본값"

    conditions = Conditions()
    # 지금 어느 파이썬으로 도는지. venv 경로는 설정 파일에 적을 수 없다 - 이미
    # 돌고 있는 파이썬 안에서 venv 를 바꿀 수는 없어서다. 대신 여기 찍어두면
    # 다음 사이클에 이 줄을 그대로 복사해서 activate 하면 된다.
    # VIRTUAL_ENV 는 activate 해야 생긴다. venv 의 python 을 경로로 직접 부르면
    # 비어 있어서 "venv 밖"으로 잘못 찍힌다 - prefix 로 본다.
    in_venv = sys.prefix != sys.base_prefix
    conditions.add("파이썬", sys.executable,
                   f"venv {Path(sys.prefix).name}" if in_venv else "시스템 파이썬",
                   "" if in_venv else "공용 환경을 건드리고 있을 수 있다")
    conditions.add("설정", config.source,
                   note=(f"{len(changed)}개 값을 덮어씀" if changed else "덮어쓴 값 없음"))
    conditions.add("로그", str(conv_data) if conv_data else "(합성 데이터)",
                   origin(args.conv_data, "paths.conv_data", "--conv-data"))
    if turns_path:
        conditions.add("턴 목록", str(turns_path),
                       origin(args.turns, "paths.turns", "--turns"),
                       "운영 필터가 고른 것 — 여기 필터는 안 돈다")
    else:
        conditions.add("필터", str(filter_path) if filter_path else "(없음)",
                       origin(args.filter_data, "paths.filter_data", "--filter-data"),
                       "" if filter_path else "진단 가능한 후속 턴 전부")
    conditions.add("출력", out_path,
                   origin(args.out or args.output_dir, "paths.output_dir", "--output-dir"))

    # --- 계약 대조. 분류 전에 한다 -------------------------------------------
    # 여기서 나온 줄들이 운영 환경에서 이쪽으로 돌아오는 포맷 정보의 전부다.
    if synthetic:
        from ragdiag.fixtures.synth import generate

        payload = generate(n=3, seed=0)
        summary.notes.append("합성 데이터로 돌았다. 실데이터로 확인한 것이 아니다.")
        conv_data = _write_temp(payload)
    else:
        payload = json.loads(Path(conv_data).read_text(encoding="utf-8"))
    summary.input_shape = contracts.shape(payload)
    report = contracts.check_log(payload)
    summary.contract_ok = report.n_ok
    summary.contract_mismatches = report.mismatches
    summary.contract_notes = report.notes

    def finish(status: str, code: int) -> int:
        summary.status = status
        summary.runtime_sec = timer.seconds
        summary.peak_gb = peak_memory_gb()
        text = summary.render()
        print()
        print(text)
        # 화면이 유일한 출력이지만, 손으로 옮겨 적을 때 스크롤을 거슬러 올라가는
        # 것보다 파일을 여는 편이 낫다. 운영 환경 밖으로 나가는 것은 아니다.
        if out_dir:
            try:
                (Path(out_dir) / f"run_summary_{stamp}.txt").write_text(
                    text + "\n", encoding="utf-8")
            except OSError:
                pass
        return code

    try:
        if turns_path:
            selection = select_turns(conv_data, turns_path,
                                     history_turns=history, limit=limit)
        else:
            selection = load_and_select(conv_data, filter_path,
                                        history_turns=history, limit=limit)
    except LabelTableMissing as e:
        # 트레이스백을 그대로 던지면 운영 환경에서 사이클 하나를 먹는다. 화면에
        # 적힌 것이 전부인 환경이라 무엇을 하라는지가 그대로 보여야 한다.
        print(f"\n{e}\n", file=sys.stderr)
        summary.notes.append("라벨 실값이 없어 필터를 걸 수 없다. "
                             "설정의 labels.query / labels.emotion 을 채울 것.")
        return finish("FAILED (라벨)", 2)
    except (OSError, ValueError, KeyError) as e:
        summary.notes.append(f"로그를 읽지 못했다: {type(e).__name__}: {e}")
        return finish("FAILED (입력)", 2)

    print(selection.report, file=sys.stderr)
    summary.metrics.append(("selected", f"{len(selection):,} turns"))

    if not selection.cases:
        # 첫 단계에서 다 빠졌으면 원인이 필터 조건이 아니라 로그 모양이다.
        # 앞 턴이 곧 "비판받은 답변"이라, 고른 턴만 남기면 판정할 대상이 없다.
        first = selection.steps[0] if selection.steps else None
        if first is not None and first.remaining == 0 and first.dropped:
            summary.notes.append(
                f"후속 턴 {first.dropped}건에 직전 턴이 없다. 고른 턴만 남기지 말고 "
                f"그 앞 턴도 함께 넘길 것 — 앞 턴이 판정 대상인 답변이다.")
        else:
            summary.notes.append("필터를 통과한 턴이 없다. 조건을 확인할 것.")
        return finish("NO DATA", 1)

    if args.dry_run:
        conditions.add("백엔드", "(안 씀)", "--dry-run", "LLM 호출 없음")
        conditions.hint = (f"{config.source} 를 고친다" if config.values else
                           "configs/env.example.yaml 을 복사해 --config 로 준다")
        print("\n" + conditions.render(), file=sys.stderr)
        summary.setup = conditions.compact()
        print(f"\n분류 대상 {len(selection)}턴 (LLM 호출 없음)", file=sys.stderr)
        for sel, case in zip(selection.selected[:10], selection.cases[:10]):
            print(f"  {case.case_id}  turn {case.turn}  "
                  f"{sel.turn.eval_result} / {sel.turn.emotion_result}",
                  file=sys.stderr)
        return finish("DRY RUN", 0)

    injected = backend is not None
    try:
        backend = backend or make_backend(args, config, trace=conditions)
    except JudgeError as e:
        print(conditions.render(), file=sys.stderr)
        print(e, file=sys.stderr)
        summary.notes.append("LLM 백엔드를 만들지 못했다. 위 메시지를 옮겨 적을 것.")
        return finish("FAILED (백엔드)", 2)

    if injected:
        # 백엔드를 밖에서 넣으면 make_backend 를 안 거쳐 이 줄들이 빈다.
        # "무엇으로 돌리는지"가 이 블록의 존재 이유라 비워 둘 수 없다.
        conditions.add("백엔드", type(backend).__name__, "주입됨 (tools/dev_run.py)")
        conditions.add("모델", backend.model, "주입됨")

    use_cache = not args.no_cache and config.get("paths.cache") is not False
    conditions.add("캐시", str(settings.CACHE_DIR) if use_cache else "(안 씀)",
                   "--no-cache" if args.no_cache else
                   origin(None, "paths.cache", ""))
    conditions.add("동시", str(workers or settings.DEFAULT_WORKERS),
                   origin(args.workers, "run.workers", "--workers"))
    conditions.hint = (f"{config.source} 를 고친다" if config.values else
                       "configs/env.example.yaml 을 복사해 --config 로 주거나 위 플래그로 덮어쓴다")
    print("\n" + conditions.render(), file=sys.stderr)
    print(f"분류 대상 {len(selection)}턴", file=sys.stderr)

    summary.metrics.append(("model", f"{backend.model}"))
    summary.setup = conditions.compact()

    judge = make_judge(backend, use_cache=use_cache)
    results = judge_cases(selection.cases, judge, workers=workers)
    outcome = build_outcome(selection.owners, results, selection.report)

    outcome.save(out_path)
    print(outcome.summary())
    print(f"\n결과: {out_path}", file=sys.stderr)

    summary.metrics.append(("classified", f"{len(results) - outcome.n_failed:,} ok / "
                                          f"{outcome.n_failed:,} failed"))
    summary.metrics.append(("llm calls", f"{outcome.n_llm_calls:,}"))
    for case_id, n in _top_cases(outcome):
        summary.metrics.append(("", f"{case_id:<14} {n:,}"))
    # 잘려서 조건을 바꿔 되살린 호출. 살아났어도 다음 실행에서는 처음부터
    # 그 조건으로 도는 게 낫다는 신호다.
    saved = getattr(backend, "fallbacks", [])
    if saved:
        for label, n in Counter(saved).most_common():
            summary.metrics.append(("truncated", f"{n:,} recovered by {label}"))
        summary.notes.append(
            f"추론이 답에 도달 못 해 {len(saved)}건을 조건을 바꿔 다시 물었다. "
            f"--thinking off 로 다시 돌릴 것.")
    # case0 은 챗봇 지표가 아니라 **필터 지표**다. 필터가 재현율 쪽으로 넓게
    # 잡아서 들어온 정상 턴이고, 필터는 운영 장비에 있어 여기서 못 고친다.
    # 어떤 eval 라벨에 몰리는지가 그쪽으로 돌아가는 유일한 피드백이다.
    #
    # 0 건이라고 좋은 게 아니다. 필터가 너무 좁아 놓치고 있다는 뜻일 수도 있어서,
    # 필터 리포트와 짝으로 읽어야 한다.
    normal = [sel for sel, r in zip(selection.selected, results)
              if r.classification and r.classification.primary_case == "case0"]
    if normal:
        share = 100 * len(normal) / max(1, len(results))
        summary.metrics.append(("filter FP", f"{len(normal):,} / {len(results):,} "
                                             f"({share:.0f}%) case0"))
        for label, n in Counter(s.turn.eval_result for s in normal).most_common(3):
            summary.metrics.append(("", f"{label[:20]:<20} {n:,}"))
        summary.notes.append(
            f"필터 오탐 후보 {len(normal)}건. 챗봇이 아니라 필터를 볼 것.")

    if outcome.n_failed:
        summary.notes.append(f"분류 실패 {outcome.n_failed}건. 결과 파일의 "
                             f"error 필드를 볼 것.")
        # 어느 단계에서 깨졌는지. 관측에서 몰려 깨지면 프롬프트·토큰 문제고,
        # 흩어져 깨지면 서버·입력 문제다. 조치가 갈린다.
        stages = Counter(r.error.split("]")[0].lstrip("[")
                         for r in results if r.error and r.error.startswith("["))
        for stage, n in stages.most_common():
            summary.metrics.append(("failed at", f"{stage:<14} {n:,}"))
    return finish("OK" if not outcome.n_failed else "PARTIAL",
                  1 if outcome.n_failed else 0)


def _write_temp(payload: dict) -> str:
    """합성 데이터를 임시 파일로. 파서가 경로를 받게 되어 있어서다.

    저장소에도 작업 폴더에도 남기지 않는다 - 가짜 데이터가 파일로 남으면
    운영 저장소로 흘러갈 길이 생긴다.
    """
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".json", prefix="ragdiag-smoke-")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    atexit.register(lambda: Path(path).unlink(missing_ok=True))
    return path


def _top_cases(outcome, limit: int = 5):
    """상위 case 몇 개. 지표 이름은 사이클 사이에 바뀌지 않아야 한다."""
    from collections import Counter

    counts = Counter(r.classification.primary_case
                     for r in outcome.results if r.classification)
    return counts.most_common(limit)


if __name__ == "__main__":
    raise SystemExit(main())
