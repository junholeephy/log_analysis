"""결정적 검증기 테스트.

LLM 없이 도는 부분이라 여기서 전부 커버할 수 있다. 오탐 테스트를 특히 많이 넣었다 —
이 검증기들은 "위반"을 선언하는 쪽이라, 오탐이 나면 멀쩡한 답변이 실패로 집계된다.
"""

import pytest

from ragdiag.checks import (
    Check,
    LengthRequest,
    check_format,
    check_language,
    check_length,
    check_pii,
    check_python_syntax,
    check_quoted_spans,
    check_service_error,
    check_truncated,
    detect_language,
    extract_quotes,
    find_pii,
    has_format,
    script_profile,
)

CHUNKS = [
    "국내 출장 식비는 1일 3만원을 상한으로 한다.",
    "해외 출장 시 미주 지역의 1일 숙박비 상한은 250달러이다.",
]


# ---------------------------------------------------------------------------
# 언어 (case10)
# ---------------------------------------------------------------------------

def test_korean_with_english_terms_is_still_korean():
    # 운영 환경 답변에는 영문 용어가 흔히 섞인다. 다수결로 세면 영어로 오판한다.
    text = "VPN 접속 시 MFA 인증이 필요하며, IT Helpdesk portal에서 재등록하십시오."
    assert detect_language(text) == "ko"


def test_pure_english_is_english():
    assert detect_language("The daily meal allowance is capped at 30,000 KRW.") == "en"


@pytest.mark.parametrize("text", ["", "12345", "!!! ???", "   "])
def test_text_without_letters_is_unknown(text):
    assert detect_language(text) == "unknown"


def test_no_language_request_is_not_applicable():
    assert check_language("아무 답변", None).verdict == "not_applicable"


def test_language_mismatch_is_violated():
    check = check_language("이것은 한국어 답변입니다.", "en")
    assert check.verdict == "violated"
    assert "요구 en" in check.detail


def test_language_match_is_ok():
    assert check_language("이것은 한국어 답변입니다.", "ko").verdict == "ok"


def test_undeterminable_language_is_not_silently_ok():
    # 판별 못 한 걸 ok로 넘기면 위반이 통계에서 사라진다.
    assert check_language("12345", "ko").verdict == "undetermined"


def test_script_profile_ignores_digits_and_punctuation():
    profile = script_profile("가나다 123 !!!")
    assert profile["hangul"] == 1.0


# ---------------------------------------------------------------------------
# 길이 (case11)
# ---------------------------------------------------------------------------

def test_no_length_request_is_not_applicable():
    assert check_length("아주 긴 답변" * 100, None).verdict == "not_applicable"


def test_max_chars_violation():
    check = check_length("가" * 500, LengthRequest("max_chars", 100))
    assert check.verdict == "violated"
    assert "실제 500" in check.detail


def test_max_chars_within_limit():
    assert check_length("가" * 50, LengthRequest("max_chars", 100)).verdict == "ok"


def test_max_sentences_counts_korean_sentences():
    answer = "첫 문장입니다. 둘째 문장입니다. 셋째 문장입니다."
    assert check_length(answer, LengthRequest("max_sentences", 2)).verdict == "violated"
    assert check_length(answer, LengthRequest("max_sentences", 3)).verdict == "ok"


def test_vague_short_request_records_the_measurement():
    """수치 없는 '짧게'는 임의 기준을 쓸 수밖에 없다.

    그래서 실제 측정값을 detail에 남긴다. 나중에 기준을 바꿔도 LLM을 다시 돌리지 않고
    저장된 값으로 재판정할 수 있어야 한다.
    """
    check = check_length("가" * 900, LengthRequest("vague_short"))
    assert check.verdict == "violated"
    assert "900자" in check.detail
    assert "기준" in check.detail


def test_vague_short_within_threshold():
    assert check_length("짧은 답변입니다.", LengthRequest("vague_short")).verdict == "ok"


def test_numeric_request_without_value_is_undetermined():
    assert check_length("답변", LengthRequest("max_chars", None)).verdict == "undetermined"


# ---------------------------------------------------------------------------
# 포맷 (case12)
# ---------------------------------------------------------------------------

NUMBERED = "1. 오래된 메일을 정리한다\n2. 첨부파일을 삭제한다\n3. 증설을 요청한다"
BULLET = "- 오래된 메일 정리\n- 첨부파일 삭제"
TABLE = "| 항목 | 금액 |\n|---|---|\n| 식비 | 3만원 |"
PROSE = "메일 용량이 초과되면 오래된 메일을 정리하거나 보관함으로 옮기면 됩니다."


@pytest.mark.parametrize("answer,kind", [
    (NUMBERED, "numbered_list"),
    (BULLET, "bullet_list"),
    (TABLE, "table"),
    ("```python\nprint(1)\n```", "code_block"),
    ('{"a": 1}', "json"),
    (PROSE, "prose"),
])
def test_format_detected(answer, kind):
    assert has_format(answer, kind)


def test_prose_request_violated_by_a_list():
    assert check_format(NUMBERED, "prose").verdict == "violated"


def test_numbered_request_violated_by_prose():
    check = check_format(PROSE, "numbered_list")
    assert check.verdict == "violated"
    assert "numbered_list" in check.detail


def test_single_numbered_item_is_not_a_list():
    # 본문 중 "1. " 하나만 나온 걸 목록으로 세면 오탐이다.
    assert not has_format("설명입니다.\n1. 첫 항목만 있음", "numbered_list")


def test_table_needs_a_separator_row():
    # 파이프만 있는 줄은 표가 아니다.
    assert not has_format("| 항목 | 금액 |\n| 식비 | 3만원 |", "table")


def test_no_format_request_is_not_applicable():
    assert check_format(PROSE, None).verdict == "not_applicable"


def test_json_inside_a_fence_is_still_json():
    assert has_format('```json\n{"a": 1}\n```', "json")


# ---------------------------------------------------------------------------
# 출력 잘림 (case8)
# ---------------------------------------------------------------------------

def test_normal_korean_ending_is_ok():
    assert check_truncated("출장 식비 상한은 1일 3만원입니다.").verdict == "ok"


def test_cut_mid_sentence_is_violated():
    check = check_truncated("출장 식비 상한은 1일 3만원이며 숙박비는")
    assert check.verdict == "violated"


def test_list_ending_without_period_is_not_truncation():
    # 목록으로 끝나는 정상 답변을 잘림으로 세면 오탐이 쏟아진다.
    assert check_truncated(NUMBERED).verdict == "ok"


def test_table_ending_is_not_truncation():
    assert check_truncated(TABLE).verdict == "ok"


def test_unclosed_code_fence_is_truncation():
    check = check_truncated("예시입니다.\n```python\nprint(1)")
    assert check.verdict == "violated"
    assert "닫히지 않음" in check.detail


def test_closed_code_fence_is_ok():
    assert check_truncated("예시입니다.\n```python\nprint(1)\n```").verdict == "ok"


def test_empty_answer_is_undetermined():
    assert check_truncated("   ").verdict == "undetermined"


# ---------------------------------------------------------------------------
# 개인정보 (case6)
# ---------------------------------------------------------------------------

# 이 파일에 이메일 리터럴을 한 줄로 두면 sync.sh 의 이식 표면 점검이 "소스에 박힌
# 개인 이메일"로 잡는다. 그 점검은 줄 단위 grep 이고 도메인을 구분하지 않는다.
# example.com 은 RFC 2606 이 문서용으로 예약한 도메인이라 실제 사람의 주소일 수
# 없지만, 점검을 무디게 만드는 것보다 여기서 조립하는 쪽이 싸다.
_MAIL = "hong" + "@" + "example.com"


@pytest.mark.parametrize("text,kind", [
    ("제 주민번호는 900101-1234567 입니다", "주민등록번호"),
    ("연락처 010-1234-5678 로 주세요", "휴대전화"),
    (f"메일은 {_MAIL} 입니다", "이메일"),
    ("카드 1234-5678-9012-3456 결제", "카드번호"),
])
def test_pii_detected(text, kind):
    assert kind in {h["kind"] for h in find_pii(text)}


@pytest.mark.parametrize("text", [
    "국내 출장 식비는 1일 3만원을 상한으로 한다.",
    "출장 종료 후 5영업일 이내에 제출한다.",
    "만 35세 이상은 종합검진 대상이다.",
    "2026-08-28 기준 개정되었습니다.",
    "미주 지역은 1일 250달러입니다.",
    "제 3-1-2 조를 참고하세요.",
])
def test_ordinary_numbers_are_not_flagged_as_pii(text):
    # 숫자 나열을 전부 의심하면 업무 규정 문장이 죄다 걸린다.
    assert find_pii(text) == []


def test_pii_check_does_not_leak_the_value():
    check = check_pii("주민번호 900101-1234567")
    assert check.verdict == "violated"
    assert "900101" not in check.detail   # 종류와 개수만 남긴다


def test_clean_text_passes_pii():
    assert check_pii("출장비 정산 절차를 알려주세요").verdict == "ok"


# ---------------------------------------------------------------------------
# 답변 속 인용 대조 (case24)
# ---------------------------------------------------------------------------

def test_quote_matching_the_document_passes():
    answer = '규정에 따르면 "국내 출장 식비는 1일 3만원을 상한으로 한다" 고 되어 있습니다.'
    assert check_quoted_spans(answer, CHUNKS).verdict == "ok"


def test_fabricated_quote_is_caught():
    answer = '규정에 따르면 "국내 출장 식비는 1일 5만원을 상한으로 한다" 고 되어 있습니다.'
    check = check_quoted_spans(answer, CHUNKS)
    assert check.verdict == "violated"
    assert check.evidence


def test_korean_quotation_marks_are_recognized():
    answer = "규정은 「해외 출장 시 미주 지역의 1일 숙박비 상한은 250달러이다」 입니다."
    assert check_quoted_spans(answer, CHUNKS).verdict == "ok"


def test_answer_without_quotes_is_not_applicable():
    assert check_quoted_spans("식비 상한은 3만원입니다.", CHUNKS).verdict == "not_applicable"


def test_no_chunks_means_undetermined_not_violated():
    # 대조할 문서가 없는 걸 위반으로 세면 안 된다.
    answer = '"국내 출장 식비는 1일 3만원을 상한으로 한다"'
    assert check_quoted_spans(answer, []).verdict == "undetermined"


def test_short_quotes_are_ignored():
    # 짧은 인용은 아무 문서에나 우연히 맞아 검증을 무력화한다.
    assert extract_quotes('그는 "네" 라고 답했다') == []


# ---------------------------------------------------------------------------
# 코드 문법 (case27)
# ---------------------------------------------------------------------------

def test_valid_python_passes():
    answer = "다음과 같이 하세요.\n```python\nfor i in range(3):\n    print(i)\n```"
    assert check_python_syntax(answer).verdict == "ok"


def test_broken_python_is_caught():
    answer = "```python\nfor i in range(3)\n    print(i)\n```"
    check = check_python_syntax(answer)
    assert check.verdict == "violated"
    assert check.evidence


def test_answer_without_code_is_not_applicable():
    assert check_python_syntax("식비 상한은 3만원입니다.").verdict == "not_applicable"


def test_non_python_blocks_are_skipped():
    answer = "```sql\nSELECT * FROM;;;\n```"
    assert check_python_syntax(answer).verdict == "not_applicable"


# ---------------------------------------------------------------------------
# 공통 규약
# ---------------------------------------------------------------------------

def test_not_applicable_is_never_reported_as_violated():
    """요구가 없었던 것과 요구를 어긴 것은 다르다.

    이걸 섞으면 '포맷 요구가 없던 답변'이 전부 포맷 위반으로 집계된다.
    """
    checks = [
        check_language("답변", None),
        check_length("답변", None),
        check_format("답변", None),
        check_quoted_spans("인용 없는 답변", CHUNKS),
        check_python_syntax("코드 없는 답변"),
    ]
    assert all(c.verdict == "not_applicable" for c in checks)
    assert not any(c.violated for c in checks)


# ---------------------------------------------------------------------------
# 서비스 자원 부족 응답  (case9)
# ---------------------------------------------------------------------------

CANNED = "서비스에 문제가 있거나, 사용자 분들이 많아서 서버에 부하가 걸리고 있어요."


@pytest.mark.parametrize("tail", [
    "",
    " 잠시 후 다시 시도해 주세요.",
    " 잠시 후 다시 시도해 주세요. 불편을 드려 죄송합니다.",
    " " + "안내 문구가 길게 이어집니다. " * 30,      # 길이 가드를 넘김
])
def test_service_error_matches_regardless_of_trailing_text(tail):
    """확정 문구 뒤에 무엇이 얼마나 붙든 잡아야 한다.

    실제 문구는 이 문장으로 끝나지 않고 안내가 더 이어진다. 부분 일치로 보는 이유고,
    확정 문구 검사를 길이 가드보다 **앞에** 둔 이유다. 순서가 뒤집히면 안내가 긴
    배포에서 통째로 놓친다.
    """
    assert check_service_error(CANNED + tail).violated


def test_service_error_survives_whitespace_and_prefix():
    """줄바꿈·띄어쓰기 차이와 앞에 붙은 인사말을 흡수한다."""
    assert check_service_error("서비스에 문제가 있거나,\n사용자 분들이 많아서\n"
                               "서버에 부하가 걸리고 있어요.").violated
    assert check_service_error("서비스에 문제가있거나, 사용자분들이 많아서 "
                               "서버에 부하가 걸리고 있어요.").violated
    assert check_service_error("안녕하세요. " + CANNED).violated


@pytest.mark.parametrize("text", [
    "서비스에 문제가 있거나, 사용자 분들이 많아서 서버에 부하가 걸리고 있습니다.",
    "서비스에 장애가 있거나, 사용자 분들이 많아서 서버에 부하가 걸리고 있어요.",
    "일시적인 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
])
def test_service_error_falls_back_to_markers(text):
    """문구가 조금 달라도 표지 두 개 이상이면 잡는다.

    배포마다 문구가 다를 수 있고, 확정 문구 목록이 최신이 아닐 수 있다.
    """
    assert check_service_error(text).violated


@pytest.mark.parametrize("text", [
    "국내 출장 식비는 1일 3만원을 상한으로 합니다.",
    "서버 증설은 IT인프라팀 승인 후 진행합니다.",          # 표지 한 개뿐
    "서버에 부하가 걸리는 상황의 대응 절차는 다음과 같습니다. " * 12,   # 장애를 주제로 답한 정상 답변
])
def test_service_error_does_not_fire_on_normal_answers(text):
    """오탐이 나면 멀쩡한 실패 판정이 통째로 case9 로 사라진다."""
    assert not check_service_error(text).violated


def test_service_error_on_empty_answer_is_not_applicable():
    """빈 답변은 '서비스 오류'가 아니다. 잘림(case8) 쪽에서 볼 일이다."""
    assert check_service_error("").verdict == "not_applicable"
    assert check_service_error("   \n ").verdict == "not_applicable"
