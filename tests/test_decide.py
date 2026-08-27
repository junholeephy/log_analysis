"""진리표 테스트. 라벨이 코드로 결정되므로 여기서 전부 커버할 수 있다."""

import pytest

from ragdiag.decide import decide
from ragdiag.schema import Evidence, GroundingCheck, NeedAnalysis, SufficiencyJudgment
from ragdiag.verify import CitationCheck, VerifiedEvidence


def need(complaint_type="content_gap", context_dependent=False):
    return NeedAnalysis(
        reasoning="r", resolved_question="q", unmet_need="n",
        complaint_type=complaint_type, context_dependent=context_dependent,
    )


def judgment(verdict, n_evidence=1):
    return SufficiencyJudgment(
        reasoning="r",
        evidence=[Evidence(chunk_index=0, quote="q" * 10)] * n_evidence,
        verdict=verdict, missing="" if verdict == "sufficient" else "금액",
    )


def check(n_kept=1):
    return CitationCheck(kept=[VerifiedEvidence(0, "q" * 10, 1.0)] * n_kept)


def ground(used):
    return GroundingCheck(reasoning="r", answer_used_rag=used)


def test_format_complaint_is_out_of_scope():
    # 형식 불만에 문서 충족도를 따지는 건 무의미하다.
    d = decide("c1", need("format_or_style"), None, None, None)
    assert d.label == "out_of_scope"


def test_unknown_complaint_is_unclassified_not_out_of_scope():
    # 판별 실패를 out_of_scope로 뭉개면 "rag는 문제없었다"로 잘못 읽힌다.
    d = decide("c1", need("other"), None, None, None)
    assert d.label == "unclassified"


def test_insufficient_becomes_rag_insufficient():
    d = decide("c1", need(), judgment("insufficient", 0), CitationCheck(), None)
    assert d.label == "rag_insufficient"
    assert not d.citation_failed


def test_partial_becomes_rag_partial():
    d = decide("c1", need(), judgment("partial"), check(), None)
    assert d.label == "rag_partial"


def test_sufficient_and_used_is_other():
    d = decide("c1", need(), judgment("sufficient"), check(), ground("used"))
    assert d.label == "rag_sufficient_other"


@pytest.mark.parametrize("used", ["ignored", "contradicted"])
def test_sufficient_but_answer_did_not_use_docs(used):
    # 문서는 충분했으므로 코퍼스 보강 대상이 아니다. 이걸 구분하지 못하면
    # 멀쩡한 코퍼스에 문서를 채워 넣는 헛수고를 하게 된다.
    d = decide("c1", need(), judgment("sufficient"), check(), ground(used))
    assert d.label == "rag_sufficient_generation_failed"


def test_sufficient_with_no_verified_citation_is_downgraded():
    # 인용이 하나도 검증되지 않은 sufficient 주장은 사전지식에서 나온 것으로 본다.
    d = decide("c1", need(), judgment("sufficient"), check(n_kept=0), None)
    assert d.label == "rag_insufficient"
    assert d.citation_failed
    assert "인용 검증 실패" in d.reason


def test_partial_with_no_verified_citation_is_downgraded():
    d = decide("c1", need(), judgment("partial"), check(n_kept=0), None)
    assert d.label == "rag_insufficient"
    assert d.citation_failed


def test_missing_judgment_for_content_complaint_raises():
    with pytest.raises(ValueError):
        decide("c1", need(), None, None, None)


def test_sufficient_without_grounding_raises():
    with pytest.raises(ValueError):
        decide("c1", need(), judgment("sufficient"), check(), None)


def test_context_dependent_flag_survives():
    d = decide("c1", need(context_dependent=True), judgment("insufficient", 0),
               CitationCheck(), None)
    assert d.context_dependent
