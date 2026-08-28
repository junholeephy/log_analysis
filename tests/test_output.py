"""출력 빌더 테스트. pre_data_format 형태를 유지하는지 본다."""

import json

from ragdiag.checks import Check
from ragdiag.classify import TurnResult
from ragdiag.conv import parse_conversations, to_case
from ragdiag.output import build_output, summarize
from ragdiag.route import route
from tests.test_route import checks, citation, judgment, obs

RAW = {"users": [{
    "user_id": "EMP-원본", "db_login_id": "login1", "job_grade": "대리",
    "db_dept_name": "인사팀", "db_job_name": "인사운영", "db_position_name": "팀원",
    "conversations": [{"conversation_id": "c1", "turns": [
        {"turn": 1, "user_question": "q1", "llm_response": "a1",
         "retrieved_data": json.dumps(["청크 A"])},
        {"turn": 2, "user_question": "q2", "llm_response": "a2",
         "retrieved_data": json.dumps(["청크 B"]),
         "llm_eval_result": "명확화 요구"},
    ]}],
}]}


def _pairs():
    conv = parse_conversations(RAW)[0]
    case = to_case(conv, 2)
    result = TurnResult(
        case=case, observation=obs(), checks=checks(),
        judgment=judgment("insufficient"), citation=citation(0),
        classification=route(obs(), checks(), judgment("insufficient"), citation(0)),
        n_calls=2,
    )
    return [(conv, result)]


def test_output_keeps_the_pre_data_format_shape():
    out = build_output(_pairs())
    assert list(out) == ["analysis_results"]
    user = out["analysis_results"][0]
    for key in ("user_id", "db_login_id", "job_grade", "db_dept_name",
                "db_job_name", "db_position_name", "conversations"):
        assert key in user
    turn = user["conversations"][0]["turns"][0]
    for key in ("turn", "pre_queries", "llm_ans_on_last_q", "current_query", "chunk_data"):
        assert key in turn


def test_original_identifier_is_preserved_for_joining():
    """출력은 원본 로그 옆에 놓여 조인에 쓰인다. 마스킹본만 남기면 되돌릴 수 없다."""
    user = build_output(_pairs())["analysis_results"][0]
    assert user["user_id"] == "EMP-원본"
    assert user["user_id_hashed"].startswith("u_")


def test_classification_is_kept_in_its_own_block():
    """원본 필드와 섞으면 '이게 원본인가 우리가 붙인 건가'를 매번 확인해야 한다."""
    turn = build_output(_pairs())["analysis_results"][0]["conversations"][0]["turns"][0]
    assert set(turn) == {"turn", "pre_queries", "llm_ans_on_last_q",
                         "current_query", "chunk_data", "classification"}
    assert turn["classification"]["case_id"] == "case17"


def test_chunk_data_comes_from_the_answered_turn():
    turn = build_output(_pairs())["analysis_results"][0]["conversations"][0]["turns"][0]
    assert turn["chunk_data"] == ["청크 A"]      # turn 1 의 것
    assert turn["classification"]["answered_turn"] == 1


def test_not_applicable_checks_are_not_written_out():
    turn = build_output(_pairs())["analysis_results"][0]["conversations"][0]["turns"][0]
    written = turn["classification"]["evidence"].get("checks", [])
    assert all(c["verdict"] != "not_applicable" for c in written)


def test_violated_checks_are_written_out():
    conv, result = _pairs()[0]
    result.checks["truncated"] = Check("truncated", "violated", "끊김")
    turn = build_output([(conv, result)])["analysis_results"][0]["conversations"][0]["turns"][0]
    names = {c["name"] for c in turn["classification"]["evidence"]["checks"]}
    assert "truncated" in names


def test_failed_turn_records_the_error():
    conv = parse_conversations(RAW)[0]
    result = TurnResult(case=to_case(conv, 2), error="JudgeError: 연결 실패")
    turn = build_output([(conv, result)])["analysis_results"][0]["conversations"][0]["turns"][0]
    assert turn["classification"]["error"].startswith("JudgeError")


def test_output_is_json_serializable():
    json.dumps(build_output(_pairs()), ensure_ascii=False)


def test_summary_reports_case_and_confidence():
    text = summarize(_pairs())
    assert "case17" in text and "medium" in text
