#!/usr/bin/env python
"""conv_eval 로그를 taxonomy case 로 분류한다.

  python conv_parse.py --conv-data conv-eval.json --filter filter.json

필터가 고른 턴만 3단계로 분류한다:

  Step 1  관측 추출     LLM 1회 · rag_data 를 주지 않는다
  Step 2  조건부 검증   코드 검증은 항상 · LLM 검증은 도메인 질문일 때만
  Step 3  라우팅        코드. case 는 LLM 이 고르지 않는다

결과는 pre_data_format 형태로 나온다. 원본 필드는 그대로 두고 분류 결과는
`classification` 아래에 모은다.

필요한 환경변수는 두 개뿐이다:
  export LLM_API_URL=http://<서버>:8000
  export LLM_API_KEY=<키>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ragdiag.backends import (
    URL_VARS,
    ClaudeCodeBackend,
    JudgeError,
    backend_from_env,
    env_first,
)
from ragdiag.classify import classify_all
from ragdiag.conv import load_conversations
from ragdiag.filters import FilterSpec, apply_filter, load_filter, render_steps, to_cases
from ragdiag.judge import Judge
from ragdiag.output import build_output, summarize


def make_backend(args):
    if args.backend == "cli":
        return ClaudeCodeBackend(model=args.model or "claude-opus-5",
                                 timeout=args.timeout)
    return backend_from_env(
        base_url=args.base_url, api_key=args.api_key, model=args.model,
        json_mode=args.json_mode, thinking=args.thinking,
        max_tokens=args.max_tokens, timeout=args.timeout,
    )


def run_golden(args) -> int:
    """Step 1 관측만 돌려 필드별 일치율을 잰다. Step 2·3 은 호출하지 않는다."""
    sys.path.insert(0, str(Path(__file__).parent))
    from fixtures.observations import build

    from ragdiag.conv import parse_conversations, to_case
    from ragdiag.golden import FieldScore, render, score_observation

    raw, expected = build()
    conversations = parse_conversations(raw)

    cases = []
    for conv in conversations:
        followup = max(t.turn for t in conv.turns)
        case = to_case(conv, followup)
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
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description="conv_eval 로그를 taxonomy case 로 분류",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--conv-data", help="conv_eval JSON 경로")
    p.add_argument("--golden", action="store_true",
                   help="Step 1 관측 골든셋을 돌려 필드별 일치율을 잰다")
    p.add_argument("--filter", help="필터 JSON. 없으면 진단 가능한 후속 턴 전부")
    p.add_argument("--out", default="conv_parsed.json", help="결과 저장 경로")

    p.add_argument("--backend", choices=["local", "cli"],
                   default="local" if env_first(URL_VARS) else "cli",
                   help="local: OpenAI 호환 서버 / cli: claude -p (개발 장비 검증용)")
    p.add_argument("--base-url", help="LLM 주소 (또는 $LLM_API_URL)")
    p.add_argument("--api-key", help="(또는 $LLM_API_KEY)")
    p.add_argument("--model", help="생략하면 서버의 /v1/models 에서 자동 탐지")
    p.add_argument("--json-mode", default="auto",
                   choices=["auto", "json_schema", "guided_json", "json_object", "none"])
    p.add_argument("--thinking", default="auto", choices=["auto", "on", "off"])
    p.add_argument("--max-tokens", type=int, default=16000)
    p.add_argument("--timeout", type=int, default=600)

    p.add_argument("--limit", type=int, help="앞에서 N건만 (비용 확인용)")
    p.add_argument("--workers", type=int, default=4, help="동시 실행 턴 수")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="필터까지만 적용하고 LLM 호출 없이 대상 건수를 보여준다")
    args = p.parse_args()

    if args.golden:
        return run_golden(args)

    if not args.conv_data:
        print("--conv-data 또는 --golden 중 하나가 필요합니다.", file=sys.stderr)
        return 2

    conversations = load_conversations(args.conv_data)
    spec = load_filter(args.filter) if args.filter else FilterSpec()
    selected, steps = apply_filter(conversations, spec)
    print(render_steps(spec, steps), file=sys.stderr)

    if args.limit:
        selected = selected[: args.limit]
    if not selected:
        print("\n필터를 통과한 턴이 없습니다. 조건을 완화하세요.", file=sys.stderr)
        return 1

    cases = to_cases(selected)
    # to_cases 는 케이스를 만들 수 없는 턴을 걸러내므로 길이가 줄 수 있다.
    # 짝을 유지해야 결과를 원래 대화에 되돌릴 수 있다.
    pairs_in = [(s, c) for s, c in zip(selected, cases) if c is not None]

    if args.dry_run:
        print(f"\n분류 대상 {len(pairs_in)}턴 (LLM 호출 없음)", file=sys.stderr)
        for sel, case in pairs_in[:10]:
            print(f"  {case.case_id}  turn {case.turn}  "
                  f"{sel.turn.eval_result} / {sel.turn.emotion_result}", file=sys.stderr)
        return 0

    try:
        backend = make_backend(args)
    except JudgeError as e:
        print(e, file=sys.stderr)
        return 2

    detected = " (자동 탐지)" if getattr(backend, "discovered", False) else ""
    print(f"\n분류 대상 {len(pairs_in)}턴 · {backend.model}{detected} "
          f"· 동시 {args.workers}", file=sys.stderr)

    judge = Judge(backend, cache_dir=None if args.no_cache else ".cache")
    results = classify_all([c for _, c in pairs_in], judge, max_workers=args.workers)
    pairs = [(sel.conversation, result)
             for (sel, _), result in zip(pairs_in, results)]

    Path(args.out).write_text(
        json.dumps(build_output(pairs), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summarize(pairs))
    print(f"\n결과: {args.out}", file=sys.stderr)
    return 1 if any(r.error for _, r in pairs) else 0


if __name__ == "__main__":
    raise SystemExit(main())
