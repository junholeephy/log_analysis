#!/usr/bin/env python
"""[회귀 기준선] RAG 충족도 진단 CLI — case19/case20 판별, 라벨 6개.

현행 분석 경로는 conv_parse.py 다. 이 진입점은 검증된 기준선을 보존하고
데이터 실태 조사(--inspect)·서버 점검(--check-llm)을 제공한다.


  python run.py --show-prompts           # 파이프라인 전체 프롬프트 출력
  python run.py --synthetic              # 합성 데이터로 회귀 검증
  python run.py --input data/logs.json   # 실데이터 분석
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from ragdiag import prompts
from ragdiag.backends import (
    KEY_VARS,
    MODEL_VAR,
    URL_VARS,
    ApiBackend,
    ClaudeCodeBackend,
    JudgeError,
    OpenAICompatBackend,
    env_first,
)
from ragdiag.decide import decide
from ragdiag.judge import DEFAULT_MODEL, Judge, run_pipeline
from ragdiag.schema import GroundingCheck, NeedAnalysis, SufficiencyJudgment
from ragdiag.verify import verify_evidence
from ragdiag.conv import load_conversations
from ragdiag.filters import FilterSpec, apply_filter, load_filter, render_steps, to_cases
from ragdiag.load import load_cases, parse_cases
from ragdiag.survey import preview_filter, survey
from ragdiag.report import render, write_jsonl

# 사용자가 줘야 할 것은 URL과 키 둘뿐이다. 이름은 흔히 쓰는 것들을 모두 받는다.
URL_VARS = ("LLM_API_URL", "API_URL", "RAGDIAG_BASE_URL",
            "OPENAI_BASE_URL", "OPENAI_API_BASE")
KEY_VARS = ("LLM_API_KEY", "API_KEY", "RAGDIAG_API_KEY", "OPENAI_API_KEY")
ENV_MODEL = "RAGDIAG_MODEL"


def env_first(names, default=None):
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default

STAGES = [
    ("Stage 1 · 정보 요구 추출", prompts.NEED_SYSTEM, prompts.need_user_message,
     "rag_data를 주지 않는다 - 문서를 보면 '사용자가 원한 것'이 문서 쪽으로 끌려간다"),
    ("Stage 2 · 충족도 판정", prompts.SUFFICIENCY_SYSTEM, None,
     "챗봇 답변을 주지 않는다 - 답변을 문서의 대리물로 착각하게 된다"),
    ("Stage 3 · 근거 활용 확인", prompts.GROUNDING_SYSTEM, prompts.grounding_user_message,
     "Stage 2가 sufficient이고 인용이 검증됐을 때만 실행"),
]


def show_prompts() -> int:
    """파이프라인 전체 프롬프트를 예시 입력과 함께 출력한다."""
    sys.path.insert(0, str(Path(__file__).parent))
    from fixtures.synthetic import build

    case = parse_cases(build()[0])[0]
    need = prompts.NeedAnalysis(
        reasoning="(예시)",
        resolved_question="미주 지역 해외 출장 시 1일 숙박비 상한 금액은 얼마인가?",
        unmet_need="미주 지역 출장 숙박비의 1일 상한 금액",
        complaint_type="content_gap", context_dependent=False,
    )

    for title, system, user_fn, note in STAGES:
        print("\n" + "=" * 78)
        print(f"  {title}")
        print(f"  설계 포인트: {note}")
        print("=" * 78)
        print("\n--- SYSTEM ---\n")
        print(system)
        print("\n--- USER (예시 케이스로 채운 것) ---\n")
        if user_fn is None:
            print(prompts.sufficiency_user_message(case, need))
        else:
            print(user_fn(case))
    return 0



def _cut(text: str, n: int = 300) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n] + " …"


def trace_case(cases, case_id: str, judge: Judge) -> int:
    """한 케이스가 파이프라인을 어떻게 통과했는지 실제 값으로 보여준다.

    캐시에 저장된 판정만 읽는다. 새 LLM 호출은 하지 않는다.
    """
    matches = [c for c in cases if case_id in c.case_id]
    if not matches:
        print(f"'{case_id}'에 해당하는 케이스가 없습니다.", file=sys.stderr)
        return 1
    case = matches[0]

    def cached(stage, system, user, model):
        path = judge._cache_path(stage, system, user)
        if not (path and path.exists()):
            return None
        return model.model_validate_json(path.read_text(encoding="utf-8"))

    print("=" * 78)
    print(f"  {case.case_id}   [{case.dept} / {case.job_grade}]")
    print("=" * 78)
    print("\n■ 입력 (로그 원본)")
    for i, q in enumerate(case.pre_queries, 1):
        print(f"    질문 {i}: {q}")
    print(f"    챗봇 답변: {_cut(case.llm_ans_on_last_q, 150)}")
    print(f"    사용자 불만: {case.current_query}")
    for i, ch in enumerate(case.rag_chunks):
        print(f"    청크 {i}: {_cut(ch, 110)}")

    # ---- Stage 1 ----
    print("\n" + "-" * 78)
    print("■ Stage 1 · 정보 요구 추출")
    print("    준다:  질문 히스토리 · 챗봇 답변 · 불만")
    print("    안 준다: rag_data   <- 문서를 보면 '원한 것'이 문서 쪽으로 끌려간다")
    need = cached("need", prompts.NEED_SYSTEM, prompts.need_user_message(case), NeedAnalysis)
    if need is None:
        print("    (캐시 없음 - 먼저 실행하세요)")
        return 1
    print(f"    -> resolved_question : {_cut(need.resolved_question, 200)}")
    print(f"    -> unmet_need        : {_cut(need.unmet_need, 200)}")
    print(f"    -> complaint_type    : {need.complaint_type}")
    print(f"    -> context_dependent : {need.context_dependent}")

    if need.complaint_type not in ("content_gap", "wrong_content"):
        diag = decide(case.case_id, need, None, None, None)
        print("\n" + "-" * 78)
        print(f"■ Stage 2·3 건너뜀 - complaint_type이 '{need.complaint_type}'")
        print(f"    문서 충족도를 따질 필요가 없다. 호출 2회 절약.")
        print(f"\n■ 최종 라벨: {diag.label}   ({diag.reason})")
        return 0

    # ---- Stage 2 ----
    print("\n" + "-" * 78)
    print("■ Stage 2 · 충족도 판정")
    print("    준다:  resolved_question · unmet_need · 인덱스 붙은 청크")
    print("    안 준다: 챗봇 답변   <- 답변을 문서의 대리물로 착각하게 된다")
    judgment = cached("sufficiency", prompts.SUFFICIENCY_SYSTEM,
                      prompts.sufficiency_user_message(case, need), SufficiencyJudgment)
    if judgment is None:
        print("    (캐시 없음)")
        return 1
    print(f"    -> verdict (원본)    : {judgment.verdict}")
    print(f"    -> evidence          : {len(judgment.evidence)}개")
    for e in judgment.evidence:
        print(f"         [청크 {e.chunk_index}] {_cut(e.quote, 90)}")
    if judgment.missing:
        print(f"    -> missing           : {_cut(judgment.missing, 200)}")

    # ---- Stage 2b ----
    check = verify_evidence(judgment.evidence, case.rag_chunks)
    print("\n" + "-" * 78)
    print("■ Stage 2b · 인용 검증 (코드, LLM 없음)")
    print("    지어낸 인용을 원문과 대조해 걸러낸다 = knowledge leakage 차단")
    for e in check.kept:
        flag = " (인덱스 교정됨)" if e.index_corrected else ""
        print(f"    통과  ratio={e.ratio:.2f}  [청크 {e.chunk_index}]{flag}")
    for d in check.dropped:
        print(f"    폐기  {d['reason']}  {_cut(d['quote'], 60)}")
    if not judgment.evidence:
        print("    (제시된 인용 없음)")
    downgraded = judgment.verdict in ("sufficient", "partial") and check.n_kept == 0
    if downgraded:
        print("    => 살아남은 인용 0개. verdict를 insufficient로 강등한다.")

    # ---- Stage 3 ----
    grounding = None
    print("\n" + "-" * 78)
    if judgment.verdict == "sufficient" and check.n_kept > 0:
        print("■ Stage 3 · 근거 활용 확인")
        print("    준다: 챗봇 답변 · 청크   (여기서 처음으로 답변을 본다)")
        grounding = cached("grounding", prompts.GROUNDING_SYSTEM,
                           prompts.grounding_user_message(case), GroundingCheck)
        if grounding is None:
            print("    (캐시 없음)")
            return 1
        print(f"    -> answer_used_rag   : {grounding.answer_used_rag}")
    else:
        print("■ Stage 3 건너뜀 - verdict가 sufficient가 아니다")
        print("    문서가 모자랐다면 답변이 그걸 썼는지는 물을 필요가 없다.")

    # ---- Stage 4 ----
    diag = decide(case.case_id, need, judgment, check, grounding)
    print("\n" + "-" * 78)
    print("■ Stage 4 · 진리표 (코드가 결정, LLM 아님)")
    print(f"    complaint_type={diag.complaint_type}  verdict={diag.verdict_final}"
          f"  used_rag={diag.answer_used_rag}")
    print(f"\n■ 최종 라벨: {diag.label}")
    print(f"    근거: {diag.reason}")
    return 0


def check_llm(args) -> int:
    """로컬 LLM 서버가 붙는지, 구조화 출력을 어떻게 강제할 수 있는지 점검한다.

    에어갭 장비에서는 물어볼 데가 없으므로 스스로 진단이 되어야 한다.
    23건을 돌리기 전에 이걸 먼저 돌려서 서버 규약을 확정하는 순서다.
    """
    import time

    from ragdiag.schema import GroundingCheck

    backend = build_backend(args)
    if not isinstance(backend, OpenAICompatBackend):
        print("--check-llm은 --backend local 에서만 씁니다.", file=sys.stderr)
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
        print("  프록시 설정에서 response_format 통과를 허용하거나, 모델 서버에 직접 붙어보세요.")

    # 실제 판정 한 번. 규약뿐 아니라 모델이 지시를 따르는지도 봐야 한다.
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
        print("\n  경고: 이 탐침은 'ignored'가 정답입니다. 문서에 답이 있는데 답변이")
        print("  일반론으로 때운 경우입니다. 모델이 지시를 못 따르고 있을 수 있습니다.")

    n = 23
    print(f"\n전체 {n}건 예상 소요: 약 {elapsed * n * 2 / max(args.workers, 1) / 60:.0f}분"
          f" (호출 {n * 2}회, 동시 {args.workers})")
    return 0


def _probe_case():
    """점검용 최소 케이스. 문서에 답이 있는데 답변이 무시한 상황."""
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


def build_backend(args) -> object:
    if args.backend == "local":
        base_url = args.base_url or env_first(URL_VARS)
        if not base_url:
            raise JudgeError(
                "LLM 주소가 없습니다. 필요한 건 이 두 개뿐입니다:\n"
                "  export LLM_API_URL=http://<서버>:8000\n"
                "  export LLM_API_KEY=<키>\n"
                f"  (인식하는 이름: {', '.join(URL_VARS)} / {', '.join(KEY_VARS)})"
            )
        # 모델은 서버에 물어본다. 사용자가 알 필요 없는 정보다.
        return OpenAICompatBackend(
            base_url=base_url,
            model=args.model or os.environ.get(ENV_MODEL),
            api_key=args.api_key,
            timeout=args.timeout, json_mode=args.json_mode,
            thinking=args.thinking, max_tokens=args.max_tokens,
        )
    if args.backend == "cli":
        return ClaudeCodeBackend(model=args.model or DEFAULT_MODEL,
                                 cli_path=args.cli_path, timeout=args.timeout)
    import anthropic

    client = anthropic.Anthropic()
    if not (client.api_key or client.auth_token):
        raise JudgeError(
            "API 백엔드에 인증 정보가 없습니다.\n"
            "  export ANTHROPIC_API_KEY=sk-ant-...   또는 `ant auth login`\n"
            "  키 없이 쓰려면 --backend cli (기본값)"
        )
    return ApiBackend(model=args.model or DEFAULT_MODEL, effort=args.effort,
                      use_fallbacks=not args.no_fallbacks, client=client)


def main() -> int:
    p = argparse.ArgumentParser(description="RAG 충족도 진단")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="분석할 로그 JSON 경로")
    src.add_argument("--synthetic", action="store_true",
                     help="합성 데이터로 회귀 검증 (기대 라벨과 대조)")
    src.add_argument("--show-prompts", action="store_true",
                     help="파이프라인 전체 프롬프트를 출력하고 종료")
    src.add_argument("--conv", metavar="FILE",
                     help="conv_eval JSON. --filter 와 함께 쓴다")
    src.add_argument("--inspect", metavar="FILE",
                     help="conv_eval JSON의 라벨·점수 분포를 조사한다 (LLM 호출 없음). "
                          "필터 조건을 정하기 전에 먼저 돌린다")
    src.add_argument("--check-llm", action="store_true",
                     help="로컬 LLM 서버 연결과 구조화 출력 강제 방식을 점검 (에어갭 장비용)")
    src.add_argument("--trace", metavar="CASE_ID",
                     help="한 케이스가 파이프라인을 통과한 실제 경로를 단계별로 출력 "
                          "(캐시된 판정만 읽고 새 호출은 하지 않음)")

    p.add_argument("--backend", choices=["local", "cli", "api"],
                   default="local" if env_first(URL_VARS) else "cli",
                   help="LLM_API_URL이 설정돼 있으면 자동으로 local")
    p.add_argument("--base-url", help="LLM 주소 (또는 $LLM_API_URL)")
    p.add_argument("--api-key", default=env_first(KEY_VARS, "EMPTY"),
                   help="(또는 $LLM_API_KEY)")
    p.add_argument("--thinking", default="auto", choices=["auto", "on", "off"],
                   help="Qwen3 계열 추론 모드 토글. auto면 서버 기본값을 건드리지 않는다")
    p.add_argument("--max-tokens", type=int, default=16000,
                   help="추론 모드가 켜져 있으면 생각에만 수천 토큰을 쓴다")
    p.add_argument("--json-mode", default="auto",
                   choices=["auto", "json_schema", "guided_json", "json_object", "none"],
                   help="구조화 출력 강제 방식. auto면 통하는 것을 실행 시점에 찾는다")
    p.add_argument("--cli-path", help="claude 실행 파일 경로 (기본: PATH 또는 ~/.local/bin)")
    p.add_argument("--timeout", type=int, default=300, help="CLI 호출 타임아웃(초)")
    p.add_argument("--limit", type=int, help="앞에서 N건만 (비용 확인용)")
    p.add_argument("--workers", type=int, default=4, help="동시 실행 케이스 수")
    p.add_argument("--model", default=None,
                   help="생략하면 서버의 /v1/models 에서 자동으로 알아낸다")
    p.add_argument("--effort", default="high",
                   choices=["low", "medium", "high", "xhigh", "max"],
                   help="api 백엔드에서만 적용")
    p.add_argument("--no-cache", action="store_true", help="디스크 캐시 사용 안 함")
    p.add_argument("--no-fallbacks", action="store_true",
                   help="api 백엔드의 refusal 서버측 폴백 beta를 쓰지 않음")
    p.add_argument("--filter", metavar="FILE",
                   help="필터 JSON. 주지 않으면 진단 가능한 후속 턴을 전부 대상으로 한다")
    p.add_argument("--out", default="results.jsonl", help="케이스별 결과 저장 경로")

    # --inspect 와 함께 쓰는 필터 미리보기 조건
    p.add_argument("--eval-label", action="append",
                   help="llm_eval_result 가 이 값인 턴만 (여러 번 지정 가능)")
    p.add_argument("--emotion-label", action="append",
                   help="llm_emotion_result 가 이 값인 턴만 (여러 번 지정 가능)")
    p.add_argument("--max-eval-score", type=float, help="llm_eval_score 상한")
    p.add_argument("--max-emotion-score", type=float, help="llm_emotion_score 상한")
    args = p.parse_args()

    if args.show_prompts:
        return show_prompts()

    if args.inspect:
        import json as _json

        raw = _json.loads(Path(args.inspect).read_text(encoding="utf-8"))
        conversations = load_conversations(args.inspect)
        print(survey(conversations, raw.get("metadata")))
        if args.filter:
            spec = load_filter(args.filter)
            selected, steps = apply_filter(conversations, spec)
            print()
            print(render_steps(spec, steps))
            if selected:
                print("\n  남은 케이스 미리보기 (최대 5건)")
                for s_ in selected[:5]:
                    print(f"    {s_.conversation.conversation_id[:18]:<20} "
                          f"turn {s_.turn.turn:<3} "
                          f"eval {s_.eval_score or 0:>5.1f} · "
                          f"emo {s_.emotion_score or 0:>5.1f}  "
                          f"{s_.turn.eval_result} / {s_.turn.emotion_result}")
            return 0
        if args.eval_label or args.emotion_label or args.max_eval_score is not None \
                or args.max_emotion_score is not None:
            print()
            print(preview_filter(
                conversations,
                eval_labels=set(args.eval_label) if args.eval_label else None,
                emotion_labels=set(args.emotion_label) if args.emotion_label else None,
                max_eval_score=args.max_eval_score,
                max_emotion_score=args.max_emotion_score,
            ))
        return 0

    if args.check_llm:
        try:
            return check_llm(args)
        except JudgeError as e:
            print(e, file=sys.stderr)
            return 2

    expected = None
    if args.synthetic or args.trace:
        sys.path.insert(0, str(Path(__file__).parent))
        from fixtures.synthetic import build

        data, expected = build()
        Path("fixtures/synthetic.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        cases = parse_cases(data)
    elif args.conv:
        conversations = load_conversations(args.conv)
        spec = load_filter(args.filter) if args.filter else FilterSpec()
        selected, steps = apply_filter(conversations, spec)
        print(render_steps(spec, steps), file=sys.stderr)
        cases = to_cases(selected)
    elif args.input:
        cases = load_cases(args.input)
    else:
        cases = []

    if args.trace:
        return trace_case(cases, args.trace, Judge(None, cache_dir=".cache"))

    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        print("분석할 케이스가 없습니다.", file=sys.stderr)
        return 1

    try:
        backend = build_backend(args)
    except JudgeError as e:
        print(e, file=sys.stderr)
        return 2

    est = f" · 예상 사용량 list 환산 ~${len(cases) * 2 * 0.06:.1f}" if args.backend == "cli" else ""
    detected = " (자동 탐지)" if getattr(backend, "discovered", False) else ""
    print(f"케이스 {len(cases)}건 · {args.backend} · {backend.model}{detected} "
          f"· 동시 {args.workers}{est}", file=sys.stderr)

    judge = Judge(backend, cache_dir=None if args.no_cache else ".cache")
    results = run_pipeline(cases, judge, max_workers=args.workers)

    write_jsonl(results, args.out)
    print(render(results, expected))
    print(f"\n케이스별 결과: {args.out}", file=sys.stderr)
    return 1 if any(r.error for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
