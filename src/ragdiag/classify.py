"""3단계 분류 파이프라인.

  Step 1  관측 추출     LLM · rag_data 를 주지 않는다
  Step 2  충족도 판정   LLM · 챗봇 답변을 주지 않는다 (도메인 질문일 때만)
  Step 3  근거 활용     LLM · 질문과 불만을 주지 않는다 (문서가 충분할 때만)
  라우팅                코드. case 는 LLM 이 고르지 않는다

코드 검증기는 Step 과 나란히 항상 돈다. 언어·길이·포맷·잘림·개인정보·인용 대조·문법·
계산은 문자열만 보면 안다. LLM 에 맡기면 비용도 들지만 무엇보다 같은 입력에 다른 답이
나온다.

**Step 이라는 이름은 LLM 호출 세 개에만 쓴다.** 라우팅을 Step 3 이라 부르던 때가 있었는데,
골든셋과 대시보드는 근거 활용을 Step 3 이라 부르고 있어서 같은 이름이 두 가지를 가리켰다.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

from ragdiag import prompts
from ragdiag.backends import Usage
from ragdiag.checks import (
    Check,
    LengthRequest,
    check_arithmetic,
    check_injection,
    check_sql_shape,
    check_format,
    check_language,
    check_length,
    check_pii,
    check_python_syntax,
    check_quoted_spans,
    check_service_error,
    check_truncated,
)
from ragdiag.judge import Judge
from ragdiag.route import Classification, route, service_unavailable
from ragdiag.schema import Case, GroundingCheck, Observation, SufficiencyJudgment
from ragdiag.verify import CitationCheck, verify_evidence

# 내용에 대한 불만일 때만 문서 충족도를 따진다. 형식 불만에 그걸 묻는 건 무의미하다.
CONTENT_COMPLAINTS = {"content_missing", "content_wrong"}


@dataclass
class TurnResult:
    case: Case
    observation: Optional[Observation] = None
    checks: dict[str, Check] = field(default_factory=dict)
    judgment: Optional[SufficiencyJudgment] = None
    citation: Optional[CitationCheck] = None
    grounding: Optional[GroundingCheck] = None
    classification: Optional[Classification] = None
    error: Optional[str] = None
    usage: Usage = field(default_factory=Usage)
    n_calls: int = 0


def run_checks(case: Case, obs: Observation) -> dict[str, Check]:
    """코드로 되는 검증. LLM 호출 없음.

    항상 도는 것과 관측이 요구를 찾았을 때만 도는 것을 나눈다. 요구가 없었는데
    검증하면 not_applicable 이 나오는데, 그걸 위반과 섞으면 멀쩡한 답변이
    전부 실패로 집계된다.
    """
    checks: dict[str, Check] = {
        "pii": check_pii(case.last_query),
        "service_error": check_service_error(case.llm_ans_on_last_q),
        "truncated": check_truncated(case.llm_ans_on_last_q),
        "quoted_spans": check_quoted_spans(case.llm_ans_on_last_q, case.rag_chunks),
        "python_syntax": check_python_syntax(case.llm_ans_on_last_q),
        "sql_shape": check_sql_shape(case.llm_ans_on_last_q),
        "arithmetic": check_arithmetic(case.llm_ans_on_last_q),
        "injection": check_injection(case.rag_chunks, case.llm_ans_on_last_q),
        "language": check_language(case.llm_ans_on_last_q, obs.requested_language or None),
        "format": check_format(
            case.llm_ans_on_last_q,
            None if obs.requested_format == "none" else obs.requested_format,
        ),
    }
    if obs.requested_length_kind == "none":
        checks["length"] = check_length(case.llm_ans_on_last_q, None)
    else:
        checks["length"] = check_length(
            case.llm_ans_on_last_q,
            LengthRequest(obs.requested_length_kind, obs.requested_length_value or None),
        )
    return checks


def classify_turn(case: Case, judge: Judge) -> TurnResult:
    """한 턴을 case 로 분류한다."""
    usage, calls = Usage(), 0

    def track(pair):
        nonlocal calls
        value, used = pair
        usage.add(used)
        calls += bool(used.input_tokens or used.output_tokens or used.cost_usd)
        return value

    try:
        # 서비스가 자원을 확보하지 못했을 때 내보내는 확정 문구는 모델이 만든 답이
        # 아니다. 판정할 답변이 없으므로 LLM 을 한 번도 부르지 않고 여기서 끝낸다.
        #
        # 관측을 돌리면 판정자가 이 문구를 거절로 읽어 case28(보안 정책)로 간다.
        # 서버 자원 문제를 보안 정책 문제로 세면 고칠 곳을 정반대로 가리킨다.
        service = check_service_error(case.llm_ans_on_last_q)
        if service.violated:
            return TurnResult(
                case=case, checks={"service_error": service},
                classification=service_unavailable(service),
                usage=usage, n_calls=0,
            )

        obs = track(judge.observe(case))
        checks = run_checks(case, obs)

        judgment = citation = grounding = None
        # 도메인 질문 + 내용 불만일 때만 LLM 검증을 더 쓴다.
        if obs.question_domain == "domain" and obs.complaint_target in CONTENT_COMPLAINTS:
            if not case.rag_chunks:
                # 판정할 문서가 하나도 없다. verdict 는 물어볼 것 없이 insufficient 이고
                # 인용할 대상도 없다. LLM 을 부르면 호출만 쓰는 게 아니라 없는 문서에서
                # 인용을 지어낼 표면이 생긴다 - verify 가 잡아내지만 잡을 일을 안 만든다.
                judgment = SufficiencyJudgment(
                    reasoning="rag_data 가 비어 있어 대조할 문서가 없다.",
                    evidence=[], verdict="insufficient", missing=obs.unmet_need)
            else:
                judgment = track(judge.judge_sufficiency_from(case, obs))
            citation = verify_evidence(judgment.evidence, case.rag_chunks)
            if judgment.verdict == "sufficient" and citation.n_kept > 0:
                grounding = track(judge.check_grounding(case))

        return TurnResult(
            case=case, observation=obs, checks=checks, judgment=judgment,
            citation=citation, grounding=grounding,
            classification=route(obs, checks, judgment, citation, grounding),
            usage=usage, n_calls=calls,
        )
    except Exception as e:
        # 배치 중 한 턴의 실패가 나머지를 날리면 안 된다. 타입명을 남겨서
        # 예상 못 한 예외가 조용히 묻히지도 않게 한다.
        return TurnResult(case=case, error=f"{type(e).__name__}: {e}",
                          usage=usage, n_calls=calls)


def classify_all(cases: list[Case], judge: Judge, max_workers: int = 4) -> list[TurnResult]:
    results: list[TurnResult] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(classify_turn, c, judge) for c in cases]
        for future in as_completed(futures):
            results.append(future.result())
    order = {c.case_id: i for i, c in enumerate(cases)}
    results.sort(key=lambda r: order.get(r.case.case_id, 1 << 30))
    return results
