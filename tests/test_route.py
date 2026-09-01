"""Step 3 라우팅 테스트.

case 를 코드가 정하므로 여기서 전 경로를 커버할 수 있다. LLM 호출 없음.
"""

import pytest

from ragdiag import taxonomy
from ragdiag.checks import Check
from ragdiag.route import route, secondary_from, service_unavailable
from ragdiag.schema import Evidence, GroundingCheck, Observation, SufficiencyJudgment
from ragdiag.verify import CitationCheck, QuoteCheck, VerifiedEvidence


def obs(**kw) -> Observation:
    base = dict(
        reasoning="r", resolved_question="q", unmet_need="n",
        complaint_target="content_missing", question_domain="domain",
        question_self_contained=True, question_multi_intent=False,
        answer_refused=False, requested_language="",
        requested_length_kind="none", requested_length_value=0,
        requested_format="none",
        question_answerable_as_asked=True, requests_unsupported_output=False,
        answer_covers_all_intents=True, answer_actionable=True,
        answer_used_history="not_needed",
    )
    base.update(kw)
    return Observation(**base)


def checks(**kw) -> dict[str, Check]:
    table = {
        name: Check(name, "not_applicable")
        for name in ("pii", "truncated", "quoted_spans", "python_syntax",
                     "sql_shape", "arithmetic", "injection",
                     "language", "format", "length")
    }
    table.update({k: v for k, v in kw.items()})
    return table


def judgment(verdict, n_evidence=1) -> SufficiencyJudgment:
    return SufficiencyJudgment(
        reasoning="r",
        evidence=[Evidence(chunk_index=0, quote="q" * 12)] * n_evidence,
        verdict=verdict, missing="" if verdict == "sufficient" else "금액",
    )


def citation(n_kept=1, n_chunks=3) -> CitationCheck:
    # n_chunks 기본값을 0 이 아닌 값으로 둔다. 0 은 "검색 결과가 아예 없었다"는
    # 별개의 사실이고 case21 로 갈리므로, 기본값으로 두면 모든 테스트가 그리로 샌다.
    return CitationCheck(kept=[VerifiedEvidence(0, "q" * 12, 1.0)] * n_kept,
                         n_chunks=n_chunks)


def ground(used) -> GroundingCheck:
    return GroundingCheck(reasoning="r", answer_used_rag=used)


# ---------------------------------------------------------------------------
# 갈림길 순서
# ---------------------------------------------------------------------------

def test_refusal_wins_over_everything():
    result = route(obs(answer_refused=True, complaint_target="content_missing"), checks())
    assert result.primary_case == "case28"
    # 권한 부족으로 인한 거절과 구분할 수 없다. 구분한 척하지 않는다.
    assert any("권한 조회 결과" in n for n in result.notes)


def test_truncated_answer_is_case9():
    result = route(
        obs(complaint_target="no_answer"),
        checks(truncated=Check("truncated", "violated", "종결 부호 없이 끝남")),
    )
    assert result.primary_case == "case8"
    assert result.confidence == "high"


def test_no_answer_complaint_but_answer_is_intact():
    """답이 없다는 불만인데 답변은 온전하다. 서비스 끊김은 로그로 판정 불가."""
    result = route(obs(complaint_target="no_answer"),
                   checks(truncated=Check("truncated", "ok", "정상")))
    assert result.primary_case == taxonomy.UNCLASSIFIED
    assert any("서비스 끊김" in n for n in result.notes)


# ---------------------------------------------------------------------------
# 요청 불이행 — 코드 검증이 판정을 확정한다
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("target,check_name,case_id", [
    ("format", "format", "case12"),
    ("language", "language", "case10"),
    ("length", "length", "case11"),
])
def test_verified_violation_gives_high_confidence(target, check_name, case_id):
    result = route(
        obs(complaint_target=target),
        checks(**{check_name: Check(check_name, "violated", "불일치")}),
    )
    assert result.primary_case == case_id
    assert result.confidence == "high"


@pytest.mark.parametrize("target,check_name", [
    ("format", "format"), ("language", "language"), ("length", "length"),
])
def test_requirement_met_but_still_complaining_is_intent_miss(target, check_name):
    """요구를 지켰는데도 불만이면 형식 문제가 아니라 기대와 다른 것이다."""
    result = route(
        obs(complaint_target=target),
        checks(**{check_name: Check(check_name, "ok", "충족")}),
    )
    assert result.primary_case == "case13"


def test_complaint_without_a_found_requirement_lowers_confidence():
    """불만은 포맷을 가리키는데 명시적 요구를 못 찾았다.

    case 는 유지하되 코드 근거가 없으므로 신뢰도를 낮추고 이유를 남긴다.
    """
    result = route(obs(complaint_target="format"), checks())
    assert result.primary_case == "case12"
    assert result.confidence == "medium"
    assert any("요구를 찾지 못함" in n for n in result.notes)


def test_inconsistency_is_parked_not_guessed():
    result = route(obs(complaint_target="inconsistency"), checks())
    assert result.primary_case == taxonomy.UNCLASSIFIED
    assert any("case19" in n for n in result.notes)


# ---------------------------------------------------------------------------
# 질문 성격
# ---------------------------------------------------------------------------

def test_general_knowledge_is_low_confidence():
    """판정자의 사전지식으로 판정한다 — 다른 라벨과 같은 무게로 집계하면 안 된다."""
    result = route(obs(question_domain="general_knowledge"), checks())
    assert result.primary_case == "case25"
    assert result.confidence == "low"


def test_calculation_question():
    assert route(obs(question_domain="calculation"), checks()).primary_case == "case26"


def test_broken_code_is_verified():
    result = route(
        obs(question_domain="code"),
        checks(python_syntax=Check("python_syntax", "violated", "문법 오류")),
    )
    assert result.primary_case == "case27"
    assert "문법 오류" in result.reason


def test_code_passing_syntax_still_case23_with_a_caveat():
    result = route(obs(question_domain="code"),
                   checks(python_syntax=Check("python_syntax", "ok", "통과")))
    assert result.primary_case == "case27"
    assert any("실행 검증" in n for n in result.notes)


def test_unclear_domain_is_unclassified():
    assert route(obs(question_domain="unclear"), checks()).primary_case \
        == taxonomy.UNCLASSIFIED


def test_other_complaint_is_out_of_taxonomy():
    """taxonomy 에 없는 유형. 쌓이면 케이스를 추가하라는 신호다."""
    result = route(obs(complaint_target="other", question_domain="unclear"), checks())
    assert result.primary_case == taxonomy.OUT_OF_TAXONOMY


# ---------------------------------------------------------------------------
# TYPE5 분기 — 검증된 case20/case22 판별
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("verdict", ["insufficient", "partial"])
def test_documents_did_not_cover_the_need(verdict):
    result = route(obs(), checks(), judgment(verdict), citation())
    assert result.primary_case == "case20"
    assert any("구분 불가" in n for n in result.notes)


def test_sufficient_without_verified_citation_is_downgraded():
    """인용이 하나도 검증되지 않은 sufficient 주장은 사전지식에서 나온 것으로 본다."""
    result = route(obs(), checks(), judgment("sufficient"), citation(n_kept=0))
    assert result.primary_case == "case20"
    assert any("강등" in n for n in result.notes)


def test_documents_sufficient_but_answer_ignored_them():
    result = route(obs(), checks(), judgment("sufficient"), citation(), ground("ignored"))
    assert result.primary_case == "case22"


def test_answer_contradicting_documents_is_hallucination():
    result = route(obs(), checks(), judgment("sufficient"), citation(),
                   ground("contradicted"))
    assert result.primary_case == "case18"
    assert any("문서 밖 허구" in n for n in result.notes)


def test_documents_used_but_user_still_unhappy_is_intent_miss():
    result = route(obs(), checks(), judgment("sufficient"), citation(), ground("used"))
    assert result.primary_case == "case13"


def test_domain_without_sufficiency_judgment_is_unclassified():
    assert route(obs(), checks()).primary_case == taxonomy.UNCLASSIFIED


# ---------------------------------------------------------------------------
# 부가 케이스 — 주 라벨과 독립적으로 성립한다
# ---------------------------------------------------------------------------

def test_context_dependent_question_adds_case4():
    result = route(obs(question_self_contained=False), checks(),
                   judgment("insufficient"), citation(0))
    assert result.primary_case == "case20"
    assert "case4" in result.secondary_cases


def test_multi_intent_adds_case3():
    result = route(obs(question_multi_intent=True), checks(),
                   judgment("insufficient"), citation(0))
    assert "case3" in result.secondary_cases


def test_pii_and_citation_problems_are_secondary():
    result = route(
        obs(), checks(pii=Check("pii", "violated", "휴대전화 1건"),
                      quoted_spans=Check("quoted_spans", "violated", "원문에 없음")),
        judgment("insufficient"), citation(0),
    )
    assert {"case6", "case24"} <= set(result.secondary_cases)


def test_secondary_never_duplicates_the_primary():
    result = route(obs(complaint_target="no_answer", question_self_contained=False),
                   checks(truncated=Check("truncated", "violated", "끊김")))
    assert result.primary_case not in result.secondary_cases


def test_not_applicable_checks_do_not_become_secondary_cases():
    """요구가 없던 항목을 위반으로 세면 멀쩡한 답변이 전부 실패로 집계된다."""
    assert secondary_from(obs(), checks()) == []


# ---------------------------------------------------------------------------
# 전역 규약
# ---------------------------------------------------------------------------

def test_routing_never_produces_an_undiagnosable_case():
    """로그로 판정할 수 없는 케이스로 보내면 안 된다.

    taxonomy 에는 남겨두되(목록에서 지우면 '그런 게 없다'가 된다) 라우팅은
    절대 그쪽으로 가지 않는다.
    """
    produced = set()
    targets = ["tone", "format", "language", "length", "content_missing",
               "content_wrong", "no_answer", "refusal", "inconsistency", "other"]
    domains = ["domain", "general_knowledge", "calculation", "code", "tool_usage", "unclear"]
    for target in targets:
        for domain in domains:
            for refused in (True, False):
                for j, c, g in [(None, None, None),
                                (judgment("insufficient"), citation(0), None),
                                (judgment("sufficient"), citation(), ground("used")),
                                (judgment("sufficient"), citation(), ground("ignored"))]:
                    result = route(
                        obs(complaint_target=target, question_domain=domain,
                            answer_refused=refused),
                        checks(), j, c, g,
                    )
                    produced.add(result.primary_case)
                    produced.update(result.secondary_cases)
    assert not (produced & taxonomy.UNDIAGNOSABLE), \
        f"판정 불가 케이스로 라우팅됨: {produced & taxonomy.UNDIAGNOSABLE}"


def test_every_produced_case_exists_in_the_taxonomy():
    result = route(obs(), checks(), judgment("insufficient"), citation(0))
    payload = result.as_dict()
    assert payload["case_id"] == "case20"
    assert payload["type_id"] == "TYPE5"
    assert payload["category"] == "category_2"


def test_category_is_derived_not_classified():
    """case 가 정해지면 category 는 계산이다. 따로 분류하지 않는 이유."""
    for case_id, case in taxonomy.CASES.items():
        assert taxonomy.describe(case_id)["category"] == case.category


# ---------------------------------------------------------------------------
# 문서 드리프트 — TAXONOMY.md 의 주장과 코드의 실제가 어긋나지 않는지
# ---------------------------------------------------------------------------

def reachable_cases() -> set[str]:
    """route.py 가 실제로 만들어낼 수 있는 case 를 전수 열거한다."""
    import itertools

    produced = set()
    targets = ["tone", "format", "language", "length", "content_missing",
               "content_wrong", "no_answer", "refusal", "inconsistency", "other"]
    domains = ["domain", "general_knowledge", "calculation", "code", "tool_usage", "unclear"]
    outcomes = [
        (None, None, None),
        (judgment("insufficient"), citation(0), None),
        # 검색 결과가 0건인 경우. n_chunks=0 이 아니면 case21 이 영영 안 나온다.
        (judgment("insufficient"), citation(0, n_chunks=0), None),
        (judgment("partial"), citation(1), None),
        (judgment("sufficient"), citation(0), None),
        (judgment("sufficient"), citation(1), ground("used")),
        (judgment("sufficient"), citation(1), ground("ignored")),
        (judgment("sufficient"), citation(1), ground("contradicted")),
    ]
    variants = [
        checks(),
        checks(truncated=Check("truncated", "violated", "x")),
        checks(format=Check("format", "violated", "x"),
               language=Check("language", "violated", "x"),
               length=Check("length", "violated", "x")),
        checks(format=Check("format", "ok", "x"),
               language=Check("language", "ok", "x"),
               length=Check("length", "ok", "x")),
        checks(python_syntax=Check("python_syntax", "violated", "x")),
        checks(pii=Check("pii", "violated", "x"),
               quoted_spans=Check("quoted_spans", "violated", "x")),
    ]
    # 관측 필드를 늘릴 때마다 여기도 늘려야 한다. 안 그러면 도달 범위를 실제보다
    # 적게 세고, 드리프트 테스트가 통과하면서 문서가 뒤처진다.
    flags = list(itertools.product((True, False), (True, False),
                                   ("not_needed", "used", "ignored"),
                                   (True, False), (True, False)))
    variants = variants + [checks(injection=Check("injection", "violated", "x")),
                           checks(arithmetic=Check("arithmetic", "violated", "x"))]

    for target, domain, refused, (j, c, g), ck, \
            (answerable, unsupported, history, covers, actionable) in \
            itertools.product(targets, domains, (True, False), outcomes, variants, flags):
        result = route(
            obs(complaint_target=target, question_domain=domain, answer_refused=refused,
                question_self_contained=False, question_multi_intent=True,
                question_answerable_as_asked=answerable,
                requests_unsupported_output=unsupported,
                answer_covers_all_intents=covers, answer_actionable=actionable,
                answer_used_history=history),
            ck, j, c, g,
        )
        produced.add(result.primary_case)
        produced.update(result.secondary_cases)
    # complaint_target="none" 은 라우팅 첫머리에서 갈리므로 위의 큰 조합을 다 돌
    # 필요가 없다. 대신 인용 검증의 세 상태를 여기서 다 밟는다 - 검증을 통과해야만
    # case0 이고, 못 대면 통과시키지 않는다.
    for ck in variants:
        for comp in (None, QuoteCheck("q" * 12, 1.0, True), QuoteCheck("", 0.0, False)):
            result = route(obs(complaint_target="none"), ck, complaint=comp)
            produced.add(result.primary_case)
            produced.update(result.secondary_cases)

    # case9 는 route() 가 아니라 service_unavailable() 이 만든다. 관측을 거치지
    # 않는 유일한 경로라 여기서 빠지면 "도달 불가"로 잘못 집계된다.
    produced.add(service_unavailable(
        Check("service_error", "violated", "확정 문구 일치")).primary_case)
    return {c for c in produced if c.startswith("case")}


# ---------------------------------------------------------------------------
# case0 — 실패가 아닌 턴
#
# 필터는 재현율 쪽으로 넓게 잡는다. 낼 자리가 없으면 정상 후속 질문이
# content_missing 으로 읽혀 case20 을 부풀리고, 그 숫자가 코퍼스 보강 목록이 되어
# 문서팀이 쓸 필요 없는 문서를 쓴다. 다만 "문제 없음"은 판정자가 낼 수 있는 가장
# 쉬운 답이라, 열어두기만 하면 애매한 턴이 전부 그리로 샌다.
# ---------------------------------------------------------------------------

def test_no_complaint_with_a_verified_quote_is_case0():
    result = route(obs(complaint_target="none"), checks(),
                   complaint=QuoteCheck("그럼 반차는 어떻게 되나요", 1.0, True))
    assert result.primary_case == "case0"
    assert any("분모" in n for n in result.notes), result.notes


def test_no_complaint_without_a_verified_quote_does_not_pass():
    """근거를 못 대면 통과시키지 않는다. 인용 강제가 여기서 값을 한다."""
    result = route(obs(complaint_target="none"), checks(),
                   complaint=QuoteCheck("지어낸 구절", 0.2, False))
    assert result.primary_case == taxonomy.UNCLASSIFIED
    assert result.confidence == "low"


def test_no_complaint_but_a_code_violation_does_not_pass():
    """불만이 없다고 결함이 없는 건 아니다. 사용자가 지적하지 않았을 뿐이다."""
    result = route(obs(complaint_target="none"),
                   checks(truncated=Check("truncated", "violated", "문장 중간에서 끊김")),
                   complaint=QuoteCheck("그럼 반차는", 1.0, True))
    assert result.primary_case == taxonomy.UNCLASSIFIED
    assert "truncated" in result.reason


def test_refusal_and_injection_beat_case0():
    """거절·인젝션은 사용자가 지적했든 아니든 확인된 사실이다."""
    verified = QuoteCheck("그럼 반차는", 1.0, True)
    assert route(obs(complaint_target="none", answer_refused=True),
                 checks(), complaint=verified).primary_case == "case28"
    assert route(obs(complaint_target="none"),
                 checks(injection=Check("injection", "violated", "지시 수행")),
                 complaint=verified).primary_case == "case29"


def test_case0_beats_question_side_problems():
    """사용자가 만족했다면 그 모호함은 실제로 문제가 되지 않았다.

    신호는 secondary 로 남는다 - 지워버리면 되묻기 유도를 고칠 근거가 사라진다.
    """
    result = route(obs(complaint_target="none", question_answerable_as_asked=False,
                       question_self_contained=False, question_multi_intent=True),
                   checks(), complaint=QuoteCheck("그럼 반차는", 1.0, True))
    assert result.primary_case == "case0"
    assert "case4" in result.secondary_cases and "case3" in result.secondary_cases


def test_taxonomy_doc_matches_the_reachable_cases():
    """TAXONOMY.md 가 코드보다 앞서 나가지 않도록.

    한 번 어긋난 적이 있다. 문서에 '판정 가능'이라 써두고 라우팅은 그 case 를
    만들지 못하는 상태였는데, 문서만 읽으면 알 수 없었다.
    """
    import re
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "TAXONOMY.md"
    if not path.exists():
        pytest.skip("TAXONOMY.md 없음")
    text = path.read_text(encoding="utf-8")
    line = next(l for l in text.splitlines() if "라우팅 도달 가능" in l)
    claimed = {f"case{n}" for n in re.findall(r"\b(\d{1,2})\b", line.split("|")[2])}

    actual = reachable_cases()
    assert claimed == actual, (
        f"문서만 있음: {sorted(claimed - actual)} / "
        f"코드만 있음: {sorted(actual - claimed)}"
    )


def test_undiagnosable_cases_are_never_reachable():
    assert not (reachable_cases() & taxonomy.UNDIAGNOSABLE)
