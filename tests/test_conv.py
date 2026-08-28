"""conv_eval 포맷 로더 테스트.

이 포맷의 함정은 두 가지다. retrieved_data 가 JSON을 담은 **문자열**이라는 것과,
불만이 그 턴이 아니라 **다음 턴**에 있다는 것. 둘 다 조용히 틀리는 종류라 테스트로 못박는다.
"""

import json

import pytest

from ragdiag.conv import (
    parse_conversations,
    parse_retrieved,
    to_case,
)
from ragdiag.load import mask

RAW = {
    "metadata": {"generated_at": "2026-07-14T18:08:10", "total_users": 1, "total_turns": 3},
    "users": [{
        "user_id": "66b452",
        "db_login_id": "",
        "job_grade": "Staff Engineer",
        "db_dept_name": "해외영업팀",
        "db_job_name": "해외영업",
        "db_position_name": "파트장",
        "conversations": [
            {
                "conversation_id": "asbdsa",
                "turns": [{
                    "turn": 1, "timestamp": "2026-03-06 10:49:08.298",
                    "prev_question": None, "retrieved_data": "[]",
                    "llm_response": "answer on aaaa", "user_question": "aaaa",
                    "trace_matched": "False",
                    "llm_eval_result": None, "llm_eval_score": None,
                    "llm_eval_score_top1": None, "llm_alternatives": [],
                    "llm_emotion_result": None, "llm_emotion_score": None,
                    "llm_emotion_score_top1": None, "llm_emotion_alternatives": [],
                }],
            },
            {
                # conversation_id 가 없는 대화가 실제로 있다
                "turns": [
                    {
                        "turn": 1, "timestamp": "2026-03-07 11:29:04.218",
                        "prev_question": None,
                        "retrieved_data": json.dumps(["미주 숙박비 250달러", "정산 절차"]),
                        "llm_response": "숙박비는 실비 정산입니다.", "user_question": "숙박비 정산은?",
                        "trace_matched": "True",
                        "llm_eval_result": None, "llm_eval_score": None,
                        "llm_eval_score_top1": None, "llm_alternatives": [],
                        "llm_emotion_result": None, "llm_emotion_score": None,
                        "llm_emotion_score_top1": None, "llm_emotion_alternatives": [],
                    },
                    {
                        "turn": 2, "timestamp": "2026-03-07 11:31:04.218",
                        "prev_question": "숙박비 정산은?",
                        "retrieved_data": json.dumps(["다른 문서 A", "다른 문서 B"]),
                        "llm_response": "지역별로 다릅니다.", "user_question": "상한 금액을 물었는데요",
                        "trace_matched": "True",
                        "llm_eval_result": "조건 변경", "llm_eval_score": 45.57,
                        "llm_eval_score_top1": 45,
                        "llm_alternatives": [
                            {"label": "F", "name": "조건 변경", " probability": 0.9},
                            {"label": "B", "name": "맥락 추가", "probability": 0.095},
                        ],
                        "llm_emotion_result": "긍정적 중립", "llm_emotion_score": 61.69,
                        "llm_emotion_score_top1": 62.5,
                        "llm_emotion_alternatives": [
                            {"lable": "D", "name": "긍정적 중립", "probability": 0.931},
                            {"label": "E", "name": "중립", "probability": 0.067},
                        ],
                    },
                ],
            },
        ],
    }],
}


def _convs():
    return parse_conversations(RAW)


# ---------------------------------------------------------------------------
# retrieved_data — JSON을 담은 문자열
# ---------------------------------------------------------------------------

def test_json_string_becomes_a_list():
    assert parse_retrieved('["aaaaa", "bbbbb"]') == ["aaaaa", "bbbbb"]


def test_empty_array_string_is_empty():
    assert parse_retrieved("[]") == []
    assert parse_retrieved("") == []
    assert parse_retrieved(None) == []


def test_actual_list_also_works():
    assert parse_retrieved(["a", "b"]) == ["a", "b"]


def test_dict_items_are_unwrapped():
    assert parse_retrieved('[{"text": "a"}, {"content": "b"}]') == ["a", "b"]


def test_broken_json_falls_back_to_chunk_splitting():
    # 이스케이프가 깨진 채 오는 경우. 버리지 말고 경계 복원을 시도한다.
    assert parse_retrieved("청크 A 내용\n\n청크 B 내용") == ["청크 A 내용", "청크 B 내용"]


def test_blank_items_are_dropped():
    assert parse_retrieved('["a", "   ", "b"]') == ["a", "b"]


# ---------------------------------------------------------------------------
# 구조 파싱
# ---------------------------------------------------------------------------

def test_users_and_conversations_are_flattened():
    convs = _convs()
    assert len(convs) == 2
    assert all(c.user.dept == "해외영업팀" for c in convs)


def test_missing_conversation_id_is_synthesized_and_unique():
    convs = _convs()
    ids = [c.conversation_id for c in convs]
    assert ids[0] == "asbdsa"
    assert ids[1] and ids[1] != "asbdsa"
    assert len(set(ids)) == len(ids)


def test_user_id_is_masked():
    conv = _convs()[0]
    assert conv.user.user_id == mask("66b452")
    assert "66b452" not in conv.user.user_id


def test_trace_matched_string_becomes_bool():
    convs = _convs()
    assert convs[0].turns[0].trace_matched is False
    assert convs[1].turns[0].trace_matched is True


def test_turns_are_sorted_by_number():
    turns = _convs()[1].turns
    assert [t.turn for t in turns] == [1, 2]


def test_null_eval_fields_become_empty_not_none():
    first = _convs()[1].turns[0]
    assert first.eval_result == ""
    assert first.eval_score is None
    assert first.is_followup is False


def test_followup_turn_carries_eval_labels():
    second = _convs()[1].turns[1]
    assert second.is_followup is True
    assert second.eval_result == "조건 변경"
    assert second.eval_score == pytest.approx(45.57)
    assert second.emotion_result == "긍정적 중립"


# ---------------------------------------------------------------------------
# 대안 목록 키 오타
# ---------------------------------------------------------------------------

def test_leading_space_in_probability_key_is_fixed():
    """실데이터에 " probability"(앞 공백)가 있다.

    그대로 두면 확률 기반 필터가 조용히 0을 읽는다 — 에러가 안 나서 더 위험하다.
    """
    alts = _convs()[1].turns[1].eval_alternatives
    assert alts[0]["probability"] == pytest.approx(0.9)


def test_lable_typo_is_fixed():
    alts = _convs()[1].turns[1].emotion_alternatives
    assert alts[0]["label"] == "D"
    assert alts[0]["name"] == "긍정적 중립"


def test_alternatives_always_have_the_three_keys():
    for alt in _convs()[1].turns[1].eval_alternatives:
        assert set(alt) == {"label", "name", "probability"}


# ---------------------------------------------------------------------------
# 케이스 변환 — 짝짓기가 핵심
# ---------------------------------------------------------------------------

def test_case_pairs_the_followup_with_the_previous_turn():
    conv = _convs()[1]
    case = to_case(conv, followup_turn=2)
    # 불만은 turn 2의 질문
    assert case.current_query == "상한 금액을 물었는데요"
    # 비판받은 답변은 turn 1의 응답
    assert case.llm_ans_on_last_q == "숙박비는 실비 정산입니다."


def test_case_uses_the_documents_behind_the_criticized_answer():
    """rag_data 는 turn N의 것이어야 한다.

    turn N+1의 검색 결과를 쓰면 "다음 질문으로 찾은 문서가 충분했나"를 묻게 되어
    질문 자체가 달라진다. 이걸 틀리면 판정이 통째로 무의미해진다.
    """
    case = to_case(_convs()[1], followup_turn=2)
    assert case.rag_chunks == ["미주 숙박비 250달러", "정산 절차"]
    assert "다른 문서 A" not in case.rag_chunks


def test_history_includes_every_prior_question():
    case = to_case(_convs()[1], followup_turn=2)
    assert case.pre_queries == ["숙박비 정산은?"]
    assert case.last_query == "숙박비 정산은?"


def test_first_turn_cannot_be_a_complaint():
    # 직전 턴이 없으면 비판할 답변도 없다.
    assert to_case(_convs()[1], followup_turn=1) is None


def test_unknown_turn_number_returns_none():
    assert to_case(_convs()[1], followup_turn=99) is None


def test_case_id_is_stable_and_masked():
    case = to_case(_convs()[1], followup_turn=2)
    assert case.case_id.endswith(":2")
    assert "66b452" not in case.case_id


def test_case_carries_user_metadata_for_crosstabs():
    case = to_case(_convs()[1], followup_turn=2)
    assert case.dept == "해외영업팀"
    assert case.job_grade == "Staff Engineer"


def test_missing_optional_fields_do_not_crash():
    raw = {"users": [{"user_id": "x", "conversations": [
        {"turns": [{"turn": 1}, {"turn": 2}]}]}]}
    conv = parse_conversations(raw)[0]
    assert conv.turns[0].user_question == ""
    assert conv.turns[0].retrieved == []
    assert to_case(conv, 2) is not None


# ---------------------------------------------------------------------------
# 히스토리 상한 — 대명사 해소에 필요한 것은 직전 2~3턴이다
# ---------------------------------------------------------------------------

def _long_conv(n=8):
    turns = [{"turn": i, "user_question": f"질문{i}", "llm_response": f"답변{i}",
              "retrieved_data": json.dumps([f"청크{i}"]),
              "llm_eval_result": None if i == 1 else "조건 변경"}
             for i in range(1, n + 1)]
    return parse_conversations({"users": [{"user_id": "u", "conversations": [
        {"conversation_id": "c", "turns": turns}]}]})[0]


def test_history_is_capped_to_the_most_recent_turns():
    case = to_case(_long_conv(), followup_turn=8, history_turns=3)
    assert case.pre_queries == ["질문5", "질문6", "질문7"]


def test_the_question_that_produced_the_answer_always_survives():
    """잘라내더라도 비판받은 답변을 부른 질문은 남아야 한다.

    이게 없으면 Step 1 이 무엇에 대한 답변인지 모른 채 불만을 읽는다.
    """
    for cap in (1, 2, 3, 5):
        case = to_case(_long_conv(), followup_turn=8, history_turns=cap)
        assert len(case.pre_queries) == cap
        assert case.last_query == "질문7"
        assert case.llm_ans_on_last_q == "답변7"


def test_zero_means_no_limit():
    case = to_case(_long_conv(), followup_turn=8, history_turns=0)
    assert len(case.pre_queries) == 7


def test_short_conversation_is_unaffected():
    case = to_case(_long_conv(n=3), followup_turn=3, history_turns=3)
    assert case.pre_queries == ["질문1", "질문2"]


def test_default_cap_is_three():
    from ragdiag.conv import MAX_HISTORY_TURNS

    assert MAX_HISTORY_TURNS == 3
    assert len(to_case(_long_conv(), followup_turn=8).pre_queries) == 3
