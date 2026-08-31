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

설정은 --config 로 준다. 사내에서는 코드를 못 고치므로 바뀔 값은 전부 거기 있다.
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
from pathlib import Path

from ragdiag.backends import (
    URL_VARS,
    ClaudeCodeBackend,
    JudgeError,
    backend_from_env,
    env_first,
)
from ragdiag import contracts
from ragdiag.config import ConfigError
from ragdiag.config import apply as apply_config
from ragdiag.config import load as load_config
from ragdiag.judge import Judge
from ragdiag.pipeline import (
    build_outcome,
    judge_cases,
    load_and_select,
    make_judge,
)
from ragdiag.summary import RunSummary, Timer, peak_memory_gb, version


def make_backend(args, config=None):
    """CLI 인자 > 설정 > 환경변수 순으로 고른다."""
    pick = (lambda flag, key, default=None:
            flag if flag is not None else
            (config.get(key, default) if config else default))

    # 아무도 안 정했으면 환경으로 고른다. LLM_API_URL 이 있으면 로컬 서버,
    # 없으면 개발 장비의 claude CLI. 이 자동 선택이 없으면 --golden 같은
    # 검증 명령이 이 장비에서 안 돈다.
    backend = pick(args.backend, "llm.backend",
                   "local" if env_first(URL_VARS) else "cli")
    model = pick(args.model, "llm.model")
    timeout = pick(args.timeout, "llm.timeout_sec", 600)
    if backend == "cli":
        return ClaudeCodeBackend(model=model or "claude-opus-5", timeout=timeout)
    return backend_from_env(
        base_url=pick(args.base_url, "llm.url"),
        api_key=pick(args.api_key, "llm.key"),
        model=model,
        json_mode=pick(args.json_mode, "llm.json_mode", "auto"),
        thinking=pick(args.thinking, "llm.thinking", "auto"),
        max_tokens=pick(args.max_tokens, "llm.max_tokens", 16000),
        timeout=timeout,
    )


def run_legacy_regression(args) -> int:
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
        backend = make_backend(args)
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


def run_golden(args) -> int:
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
        backend = make_backend(args)
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


def main() -> int:
    p = argparse.ArgumentParser(
        prog="python -m ragdiag",
        description="conv_eval 로그를 taxonomy case 로 분류",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--config", help="설정 YAML. configs/example.yaml 참고")

    p.add_argument("--conv-data", help="conv_eval JSON 경로 (설정을 덮어쓴다)")
    p.add_argument("--golden", action="store_true",
                   help="Step 1 관측 골든셋을 돌려 필드별 일치율을 잰다")
    p.add_argument("--legacy-regression", action="store_true",
                   help="구 회귀셋 23건을 새 파이프라인으로 돌려 대조한다")
    p.add_argument("--filter-data", "--filter", dest="filter_data",
                   help="필터 JSON. 없으면 진단 가능한 후속 턴 전부")
    p.add_argument("--output-dir", dest="output_dir",
                   help="산출물을 넣을 디렉터리. 없으면 만든다")
    p.add_argument("--out", help="결과 파일 경로. --output-dir 보다 우선한다")

    # 아래 기본값은 전부 None 이다. 설정과 CLI 를 구분하기 위해서다 -
    # argparse 기본값을 넣으면 "사용자가 준 것"과 "기본값"을 못 가린다.
    p.add_argument("--backend", choices=["local", "cli", "api"],
                   help="local: OpenAI 호환 서버 / cli: claude -p (개발 장비 검증용)")
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
    args = p.parse_args()

    # 설정은 계산 전에 읽고 검증한다. 30분 뒤에 키 하나로 죽으면 사이클 하나를 버린다.
    try:
        config = load_config(args.config)
    except ConfigError as e:
        print(e, file=sys.stderr)
        return 2
    changed = apply_config(config)
    if changed:
        print(f"설정 {config.source} 적용: {', '.join(changed)}", file=sys.stderr)

    if args.golden:
        return run_golden(args)
    if args.legacy_regression:
        return run_legacy_regression(args)

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

    # --out(파일)이 --output-dir(디렉터리)보다 우선한다. 둘 다 없으면 현재 위치.
    out_dir = args.output_dir or config.get("paths.output_dir")
    out_path = args.out or config.get("paths.out")
    if not out_path:
        out_path = str(Path(out_dir or ".") / "conv_parsed.json")
    if out_dir:
        # 없으면 만든다. 30분 돌린 뒤 디렉터리가 없어서 못 쓰면 그 사이클을 버린다.
        Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    workers = args.workers or config.get("run.workers")
    limit = args.limit or config.get("run.limit")
    history = args.history_turns if args.history_turns is not None else None

    summary = RunSummary(version=version(), args=" ".join(sys.argv[1:]))
    timer = Timer().__enter__()

    # --- 계약 대조. 분류 전에 한다 -------------------------------------------
    # 여기서 나온 줄들이 사내에서 이쪽으로 돌아오는 포맷 정보의 전부다.
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

    def finish(status: str, code: int) -> int:
        summary.status = status
        summary.runtime_sec = timer.seconds
        summary.peak_gb = peak_memory_gb()
        text = summary.render()
        print()
        print(text)
        # 화면이 유일한 출력이지만, 손으로 옮겨 적을 때 스크롤을 거슬러 올라가는
        # 것보다 파일을 여는 편이 낫다. 사내 밖으로 나가는 것은 아니다.
        if out_dir:
            try:
                (Path(out_dir) / "run_summary.txt").write_text(text + "\n",
                                                               encoding="utf-8")
            except OSError:
                pass
        return code

    try:
        selection = load_and_select(conv_data, filter_path,
                                    history_turns=history, limit=limit)
    except (OSError, ValueError, KeyError) as e:
        summary.notes.append(f"로그를 읽지 못했다: {type(e).__name__}: {e}")
        return finish("FAILED (입력)", 2)

    print(selection.report, file=sys.stderr)
    summary.metrics.append(("selected", f"{len(selection):,} turns"))

    if not selection.cases:
        summary.notes.append("필터를 통과한 턴이 없다. 조건을 확인할 것.")
        return finish("NO DATA", 1)

    if args.dry_run:
        print(f"\n분류 대상 {len(selection)}턴 (LLM 호출 없음)", file=sys.stderr)
        for sel, case in zip(selection.selected[:10], selection.cases[:10]):
            print(f"  {case.case_id}  turn {case.turn}  "
                  f"{sel.turn.eval_result} / {sel.turn.emotion_result}",
                  file=sys.stderr)
        return finish("DRY RUN", 0)

    try:
        backend = make_backend(args, config)
    except JudgeError as e:
        print(e, file=sys.stderr)
        summary.notes.append("LLM 백엔드를 만들지 못했다. 위 메시지를 옮겨 적을 것.")
        return finish("FAILED (백엔드)", 2)

    detected = " (자동 탐지)" if getattr(backend, "discovered", False) else ""
    print(f"\n분류 대상 {len(selection)}턴 · {backend.model}{detected} "
          f"· 동시 {workers or 4}", file=sys.stderr)
    summary.metrics.append(("model", f"{backend.model}"))

    use_cache = not args.no_cache and config.get("paths.cache") is not False
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
    if outcome.n_failed:
        summary.notes.append(f"분류 실패 {outcome.n_failed}건. 결과 파일의 "
                             f"error 필드를 볼 것.")
    return finish("OK" if not outcome.n_failed else "PARTIAL",
                  1 if outcome.n_failed else 0)


def _write_temp(payload: dict) -> str:
    """합성 데이터를 임시 파일로. 파서가 경로를 받게 되어 있어서다.

    저장소에도 작업 폴더에도 남기지 않는다 - 가짜 데이터가 파일로 남으면
    사내 저장소로 흘러갈 길이 생긴다.
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
