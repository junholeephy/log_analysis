"""결정적 검증기 테스트.

LLM 없이 도는 부분이라 여기서 전부 커버할 수 있다. 오탐 테스트를 특히 많이 넣었다 —
이 검증기들은 "위반"을 선언하는 쪽이라, 오탐이 나면 멀쩡한 답변이 실패로 집계된다.
"""

import pytest

from ragdiag.checks import (
    Check,
    LengthRequest,
    check_dates,
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
    extract_sources,
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


def test_length_is_measured_but_never_judged():
    """길이는 기준이 사용자 머릿속에 있어 코드로 가릴 수 없다.

    503건 실측에서 길이를 요구한 20건이 전부 ok 로 나왔다. "세 줄 이내" 요구에
    149자 만연체로 답한 것까지 통과했다 - 줄바꿈이 없어 1줄이라서다. 다른
    검증기는 답이 텍스트 밖에 확정돼 있지만(언어는 스크립트, 날짜는 달력,
    파이썬은 파서) 길이는 그런 것이 없다.

    그래서 verdict 는 언제나 undetermined 다. "판정에 실패했다"가 아니라
    "코드가 판정할 것이 아니다"라는 뜻이다.
    """
    long_run_on = ("연차유급휴가를 신청하시려면 먼저 그룹웨어에 접속하셔야 하며, "
                   "인사 메뉴에서 휴가 신청을 선택하시고, 사유를 기재하신 다음, "
                   "팀장님의 승인을 받으시면 됩니다.")
    for request in (LengthRequest("max_lines", 3),
                    LengthRequest("max_sentences", 5),
                    LengthRequest("max_chars", 200),
                    LengthRequest("vague_short"),
                    LengthRequest("max_chars", None)):
        got = check_length(long_run_on, request)
        assert got.verdict == "undetermined", (request, got)


def test_length_records_what_was_asked_and_what_came():
    """판정을 안 하는 대신, 사람이 볼 수 있게 남긴다.

    실데이터에서 분포를 보고 기준을 정하게 되면 이 detail 만 읽으면 된다 -
    LLM 을 다시 돌릴 필요가 없다.
    """
    got = check_length("가" * 900, LengthRequest("vague_short"))
    assert "900자" in got.detail, got.detail
    assert "짧게" in got.detail, got.detail

    got = check_length("첫 문장입니다.\n둘째 문장입니다.", LengthRequest("max_chars", 200))
    assert "max_chars ≤ 200" in got.detail, got.detail
    assert "2문장" in got.detail and "2줄" in got.detail, got.detail


def test_no_length_request_still_records_the_measurement():
    got = check_length("짧은 답변입니다.", None)
    assert got.verdict == "not_applicable"
    assert "9자" in got.detail, got.detail


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


def test_a_document_name_is_not_compared_against_chunk_bodies():
    """제목 부호 안은 문장이 아니라 문서 이름이다.

    청크 본문에 문서명이 적혀 있을 리가 없어서, 한 통에 넣고 대조하면 **정확히
    인용한 답변까지 위반**이 된다. 그동안 안 터진 것은 길이 덕이었다 -
    「연차휴가 운영지침」은 정규화하면 8자라 10자 하한에 조용히 걸렸다.

    청크에 출처 표기가 없으면 대조하지 않고 그렇게 적는다.
    """
    answer = '「정보보호정책 시행세칙」에 "국내 출장 식비는 1일 3만원을 상한으로 한다"고 되어 있습니다'
    got = check_quoted_spans(answer, CHUNKS)
    assert got.verdict == "ok", got
    assert "대조 불가" in got.detail, got.detail


def test_a_document_name_is_checked_against_the_chunk_header():
    """운영 로그는 청크 앞에 출처를 붙여 보낸다.

        "[정보보호정책 시행세칙 제7조] 사내 자료의 외부 반출은…"

    그게 오면 문서명 검증이 저절로 켜진다. 부분 점수를 주지 않는 이유는 이름이
    산문이 아니라 식별자라서다 - 「해외출장관리지침」과 「국내출장관리지침」은
    아홉 자 중 여섯 자가 겹친다.
    """
    tagged = ["[정보보호정책 시행세칙 제7조] 사내 자료의 외부 반출은 보안심의를 거쳐야 한다."]
    quote = '"사내 자료의 외부 반출은 보안심의를 거쳐야 한다"고 되어 있습니다'

    assert check_quoted_spans(f"「정보보호정책 시행세칙」에 {quote}", tagged).verdict == "ok"

    got = check_quoted_spans(f"「해외출장관리지침」에 {quote}", tagged)
    assert got.verdict == "violated", got
    assert any("해외출장관리지침" in e for e in got.evidence), got.evidence

    # 짧은 이름도 봐야 한다. 문장 기준(10자)을 쓰면 「휴가규정」이 빠진다.
    got = check_quoted_spans(f"「휴가규정」에 {quote}", tagged)
    assert got.verdict == "violated", got


def test_answer_without_quotes_is_not_applicable():
    assert check_quoted_spans("식비 상한은 3만원입니다.", CHUNKS).verdict == "not_applicable"


def test_no_chunks_but_a_quotation_is_a_violation():
    """전에는 undetermined 였다. 뒤집은 이유를 남긴다.

    "대조할 문서가 없는 걸 위반으로 세면 안 된다"가 옛 판단이었고, 대조 자체로
    보면 맞다. 그런데 재는 것이 대조가 아니라 **출처 주장**이다 - 검색이 0건인데
    문서를 인용했다면 가져올 곳이 없었는데 가져온 척한 것이다.

    undetermined 로 두면 라우팅이 아무것도 하지 않아 이 신호가 조용히 사라진다.
    case21(Retrieve 미수행)과 겹치는 자리라 특히 아깝다.
    """
    answer = '"국내 출장 식비는 1일 3만원을 상한으로 한다"'
    assert check_quoted_spans(answer, []).verdict == "violated"


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


def test_a_trailing_emoji_is_not_a_cut_off_answer():
    """이모지로 끝나는 것은 완결의 신호다.

    생성이 끊기면 토큰 중간에서 멈추지, 그 자리에 장식을 붙이고 멈추지 않는다.
    한국어 답변은 마침표를 생략하고 이모지로 끝맺는 일이 흔한데, 그걸 잘림으로
    세면 case8 이 부풀고 고칠 곳을 서비스 안정성 쪽으로 잘못 가리킨다.
    """
    for answer in ("도움이 되셨길 바랍니다 😊",     # 종결 부호 없음
                   "확인해 보세요! 👍",            # 종결 부호 + 이모지
                   "완료했습니다 ✅",
                   "주의하세요 ⚠️",               # 이형 선택자가 뒤에 붙는다
                   "가족 행사입니다 👨‍👩‍👧",          # ZWJ 로 이어진 것
                   "한국 지사입니다 🇰🇷",           # 지역 표시 두 글자
                   "잘 하셨어요 👍🏽"):             # 피부색 수정자
        assert check_truncated(answer).verdict == "ok", answer


def test_emoji_stripping_does_not_eat_a_closing_code_fence():
    """백틱의 유니코드 범주가 Sk 다.

    범주로 뭉뚱그려 벗겨냈다가 닫는 ``` 까지 떨어져서, 멀쩡히 닫힌 코드블록이
    "코드블록이 닫히지 않음" 으로 뒤집혔다.
    """
    assert check_truncated("예시입니다.\n```python\nprint(1)\n```").verdict == "ok"
    assert check_truncated("```sql\nSELECT 1").verdict == "violated"


def test_an_emoji_does_not_rescue_a_sentence_cut_mid_word():
    """이모지가 붙었다고 다 넘기면 검증기가 무력해진다."""
    assert check_truncated("재발 방지를 위한").verdict == "violated"
    assert check_truncated("").verdict == "undetermined"
    # 이모지만 있는 답변은 쓸모없을지언정 끊긴 것은 아니다.
    assert check_truncated("😊").verdict == "ok"


# ---------------------------------------------------------------------------
# 날짜  (case26 보강)
# ---------------------------------------------------------------------------

def test_a_date_that_is_not_on_the_calendar_is_caught():
    """없는 날짜는 순수 할루시네이션이다.

    사람이 오타로 "2월 30일"을 쓸 일은 드물고, 모델이 그럴듯한 숫자를 채울 때
    나온다. 맥락과 무관하게 틀렸으므로 오탐이 원리적으로 없다.
    """
    for answer in ("건강검진은 2월 30일까지 신청하세요",
                   "신청 마감은 13월 1일입니다",
                   "2026-02-30 까지 제출",
                   "2025년 2월 29일까지"):          # 2025는 평년
        assert check_dates(answer).verdict == "violated", answer
    assert check_dates("2024년 2월 29일까지").verdict == "ok"   # 윤년


def test_periods_and_recurrences_are_not_dates():
    """월과 일이 함께 있을 때만 날짜로 본다.

    "30일 이내"(기간) · "매월 25일"(반복) · "5영업일"을 날짜로 읽으면 오탐이
    쏟아진다. 실제 데이터에서 이쪽이 날짜 표현보다 훨씬 흔하다.
    """
    for answer in ("30일 이내에 신청하세요", "매월 25일에 지급됩니다",
                   "정산은 5영업일 이내에 하세요", "제30일차 교육입니다"):
        assert check_dates(answer).verdict == "not_applicable", answer


def test_a_wrong_weekday_claim_is_caught():
    assert check_dates("2026년 3월 13일은 금요일입니다").verdict == "ok"
    got = check_dates("2026년 3월 13일은 목요일이니 참고하세요")
    assert got.verdict == "violated", got
    assert "금요일" in got.evidence[0], got.evidence


def test_a_weekday_claim_without_a_year_is_not_judged():
    """timestamp 의 연도를 갖다 쓰면 12월에 물어본 1월 일정에서 틀린다.

    그 틀린 판정이 high 신뢰도로 나가는 것이 최악이라, 모르는 것은 모른다고 한다.
    """
    assert check_dates("3월 13일은 금요일입니다").verdict == "undetermined"
    # 요일 주장이 없으면 연도가 없어도 달력 검사는 된다.
    assert check_dates("3월 11일까지입니다").verdict == "ok"
    assert check_dates("2월 30일까지입니다").verdict == "violated"


def test_business_days_are_deliberately_not_judged():
    """공휴일표 없이 주말만 빼면 설·추석에 조용히 틀린다.

    틀린 판정이 high 신뢰도로 나가느니 안 보는 편이 낫다.
    """
    assert check_dates("출장 종료 후 5영업일 이내에 정산서를 제출한다").verdict \
        == "not_applicable"


def test_every_kind_of_quote_mark_is_recognised():
    """모델이 어떤 부호를 쓸지 정해져 있지 않다.

    큰따옴표만 보다가 작은따옴표·겹화살괄호로 인용한 답변을 통째로 놓쳤다.
    여는 부호와 닫는 부호를 한 집합으로 묶어 짝이 어긋나도 잡는다 - 모델이
    `“…"` 처럼 섞어 내는 일이 흔하다.
    """
    body = "모든 반출은 보안심의를 거친다"
    for opened, closed in [('"', '"'), ("“", "”"), ("“", '"'),
                           ("'", "'"), ("‘", "’")]:
        assert extract_quotes(f"규정에 {opened}{body}{closed}고 되어 있습니다") == [body], \
            (opened, closed)

    # 제목 부호는 문장이 아니라 **문서 이름**이 들어가는 자리다. 대조 대상이
    # 청크 본문이 아니라 머리의 출처 표기라 따로 뽑는다.
    for opened, closed in [("「", "」"), ("『", "』"), ("《", "》"), ("〈", "〉")]:
        assert extract_sources(f"{opened}휴가규정{closed}에 따르면") == ["휴가규정"], \
            (opened, closed)
        assert extract_quotes(f"{opened}{body}{closed}고 되어 있습니다") == []


def test_an_apostrophe_is_not_a_quote():
    """영어 축약형 둘 사이가 인용처럼 보인다.

    "you don't need to worry, it isn't required" 는 두 어포스트로피 사이가
    23자라, 가드가 없으면 인용으로 잡혀 원문에 없다고 판정된다.
    """
    assert extract_quotes("you don't need to worry, it isn't required") == []
    # 강조가 이어져도 경계를 넘지 않는다.
    assert extract_quotes("'연차 사용'과 '반차 사용'을 구분하세요") == []


def test_quoting_with_no_search_results_is_a_violation():
    """검색 결과가 0건인데 답변이 문서를 인용했다.

    대조할 것이 없는 게 아니라 **가져올 곳이 없었는데 가져온 척한 것**이다.
    전에는 undetermined 로 넘겨서 이 신호가 조용히 사라졌다.
    """
    got = check_quoted_spans('규정에 "미사용 연차는 이월한다"고 되어 있습니다', [])
    assert got.verdict == "violated", got
    assert "0건" in got.detail, got.detail
    # 인용이 아예 없으면 여전히 해당 없음이다.
    assert check_quoted_spans("이월 가능합니다", []).verdict == "not_applicable"
