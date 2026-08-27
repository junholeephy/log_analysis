"""인용 검증 테스트. leakage 차단 장치가 실제로 막는지가 핵심."""

import pytest

from ragdiag.schema import Evidence
from ragdiag.verify import MATCH_THRESHOLD, match_ratio, normalize, verify_evidence

CHUNKS = [
    "해외 출장 시 미주 지역의 1일 숙박비 상한은 250달러이다.",
    "국내 출장 식비는 1일 3만원을 상한으로 한다.",
]


def ev(idx, quote):
    return Evidence(chunk_index=idx, quote=quote)


def test_exact_quote_is_kept():
    r = verify_evidence([ev(0, "1일 숙박비 상한은 250달러")], CHUNKS)
    assert r.n_kept == 1
    assert r.kept[0].ratio == 1.0
    assert not r.kept[0].index_corrected


def test_whitespace_differences_are_tolerated():
    # 청크 분할 과정에서 공백은 쉽게 달라진다. 그걸로 근거를 버리면 안 된다.
    r = verify_evidence([ev(0, "1일  숙박비\n상한은 250달러")], CHUNKS)
    assert r.n_kept == 1


def test_fabricated_quote_is_dropped():
    # 문서에 없는 내용을 지어낸 경우. 이게 막히지 않으면 검색 실패가
    # '근거 미활용'으로 오분류되어 통계에서 사라진다.
    r = verify_evidence([ev(0, "유럽 지역의 1일 숙박비 상한은 300달러이다")], CHUNKS)
    assert r.n_kept == 0
    assert r.dropped[0]["reason"] == "not_found"


def test_short_quote_is_dropped():
    # 짧은 인용은 아무 문서에나 우연히 맞아 검증을 무력화한다.
    r = verify_evidence([ev(0, "출장")], CHUNKS)
    assert r.n_kept == 0
    assert r.dropped[0]["reason"] == "too_short"


def test_wrong_index_but_real_quote_is_kept_and_flagged():
    # leakage를 막는 건 인용의 실재성이지 인덱스의 정확성이 아니다.
    r = verify_evidence([ev(1, "미주 지역의 1일 숙박비 상한은 250달러")], CHUNKS)
    assert r.n_kept == 1
    assert r.kept[0].chunk_index == 0
    assert r.kept[0].index_corrected
    assert r.any_corrected


def test_out_of_range_index_falls_back_to_scan():
    r = verify_evidence([ev(99, "국내 출장 식비는 1일 3만원")], CHUNKS)
    assert r.n_kept == 1
    assert r.kept[0].chunk_index == 1


def test_no_chunks_means_nothing_can_be_verified():
    r = verify_evidence([ev(0, "무엇이든 상관없는 충분히 긴 인용문")], [])
    assert r.n_kept == 0


def test_normalize_removes_whitespace_and_normalizes_unicode():
    assert normalize("가 나\t다\n") == "가나다"
    assert normalize("２５０달러") == "250달러"


@pytest.mark.parametrize("quote,expected_min", [
    ("미주 지역의 1일 숙박비 상한은 250달러이다", 1.0),
    ("전혀 관계없는 문장입니다 여기에는", 0.0),
])
def test_match_ratio_bounds(quote, expected_min):
    ratio = match_ratio(quote, CHUNKS[0])
    assert ratio >= expected_min if expected_min == 1.0 else ratio < MATCH_THRESHOLD
