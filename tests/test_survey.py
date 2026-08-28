"""실태 조사 테스트.

필터 조건은 이 조사 결과를 보고 정한다. 여기서 숫자가 틀리면 필터가 통째로 틀린다.
"""

import json

from ragdiag.conv import parse_conversations
from ragdiag.survey import preview_filter, survey


def _raw(convs):
    return {"users": [{"user_id": "u1", "db_dept_name": "인사팀",
                       "job_grade": "대리", "conversations": convs}]}


def _turn(n, eval_result=None, emotion=None, eval_score=None, emotion_score=None):
    return {
        "turn": n, "user_question": f"질문 {n}", "llm_response": f"답변 {n}",
        "retrieved_data": json.dumps([f"청크 {n}"]),
        "llm_eval_result": eval_result, "llm_eval_score": eval_score,
        "llm_emotion_result": emotion, "llm_emotion_score": emotion_score,
    }


def test_first_turns_are_excluded_from_candidates():
    """첫 턴은 비판할 직전 답변이 없어 진단 대상이 아니다."""
    convs = parse_conversations(_raw([
        {"conversation_id": "c1", "turns": [_turn(1), _turn(2, "명확화 요구")]},
    ]))
    text = survey(convs)
    assert "전체 턴                     2" in text
    assert "후속 턴 (eval 라벨 있음)      1" in text


def test_metadata_mismatch_is_flagged():
    """선언과 실제가 다르면 파일이 잘렸거나 중복이 있다는 뜻이다."""
    convs = parse_conversations(_raw([{"conversation_id": "c", "turns": [_turn(1)]}]))
    assert "불일치" in survey(convs, {"total_turns": 999})
    assert "불일치" not in survey(convs, {"total_turns": 1})


def test_label_distribution_is_counted():
    convs = parse_conversations(_raw([{"conversation_id": "c", "turns": [
        _turn(1), _turn(2, "명확화 요구"), _turn(3, "명확화 요구"), _turn(4, "조건 변경")]}]))
    text = survey(convs)
    assert "명확화 요구" in text and "조건 변경" in text


def test_filter_preview_shows_each_step_separately():
    """한 번에 다 걸고 최종 숫자만 보면 어느 조건이 과했는지 알 수 없다."""
    convs = parse_conversations(_raw([{"conversation_id": "c", "turns": [
        _turn(1),
        _turn(2, "명확화 요구", "부정", 30.0),
        _turn(3, "조건 변경", "부정", 80.0),
        _turn(4, "명확화 요구", "긍정", 20.0),
    ]}]))
    text = preview_filter(convs, eval_labels={"명확화 요구"},
                          emotion_labels={"부정"}, max_eval_score=50.0)
    assert "eval_result 포함" in text
    assert "emotion_result 포함" in text
    assert "최종 진단 대상" in text
    # 명확화 요구 2건 -> 부정 1건 -> 점수 50 이하 1건
    assert text.rstrip().count("turn 2") == 1


def test_filter_preview_drops_followups_without_a_previous_turn():
    """eval 라벨이 있어도 직전 턴이 실제로 없으면 케이스를 만들 수 없다.

    턴 번호가 띄엄띄엄한 데이터에서 실제로 생긴다.
    """
    convs = parse_conversations(_raw([{"conversation_id": "c", "turns": [
        _turn(5, "명확화 요구", "부정", 10.0)]}]))
    assert "후속 턴 (직전 턴 존재)                       0" in preview_filter(convs)


def test_no_conditions_means_everything_survives():
    convs = parse_conversations(_raw([{"conversation_id": "c", "turns": [
        _turn(1), _turn(2, "조건 변경", "중립", 50.0)]}]))
    text = preview_filter(convs)
    assert "최종 진단 대상                              1" in text


def test_survey_runs_on_empty_input():
    assert "턴 0개" in survey([])
