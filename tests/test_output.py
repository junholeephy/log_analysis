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
         "llm_eval_result": "질의 킬로"},
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
    assert turn["classification"]["case_id"] == "case20"


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


def test_failed_turn_names_the_step_it_died_in():
    """관측에서 몰려 깨지는 것과 흩어져 깨지는 것은 조치가 다르다.

    관측이면 프롬프트·토큰 쪽이고, 충족도에만 몰리면 청크가 길어서 잘리는 것이다.
    단계 이름이 없으면 실데이터 한 번 돌린 뒤 어디를 고칠지 정할 수 없다.
    """
    from ragdiag.backends import Usage
    from ragdiag.classify import classify_turn
    from tests.test_route import obs as make_obs

    class _Dies:
        def __init__(self, at):
            self.at = at

        def observe(self, case):
            if self.at == "observe":
                raise RuntimeError("잘림")
            return make_obs(), Usage(input_tokens=1, output_tokens=1)

        def judge_sufficiency_from(self, case, obs):
            raise RuntimeError("잘림")

    case = to_case(parse_conversations(RAW)[0], 2)
    assert case.rag_chunks, "충족도 단계까지 가려면 청크가 있어야 한다"

    assert classify_turn(case, _Dies("observe")).error.startswith("[observe]")
    assert classify_turn(case, _Dies("sufficiency")).error.startswith("[sufficiency]")


def test_no_complaint_skips_the_sufficiency_step():
    """불만이 없는데 문서 충족도를 묻는 건 무의미하고 호출만 쓴다.

    없는 문서에서 인용을 지어낼 표면도 생긴다 - verify 가 잡지만 잡을 일을 안 만든다.
    """
    from ragdiag.backends import Usage
    from ragdiag.classify import classify_turn
    from tests.test_route import obs as make_obs

    class _Judge:
        def __init__(self):
            self.asked = []

        def observe(self, case):
            self.asked.append("observe")
            return (make_obs(complaint_target="none",
                             complaint_quote=case.current_query[:14]),
                    Usage(input_tokens=1, output_tokens=1))

        def judge_sufficiency_from(self, case, obs):
            self.asked.append("sufficiency")
            raise AssertionError("불만이 없는데 충족도를 물었다")

    import dataclasses

    # 인용은 최소 길이를 넘겨야 검증을 통과한다. RAW 의 "q2" 로는 짧아서
    # 검증이 실패하고, 그러면 case0 이 아니라 unclassified 로 간다.
    # 답변도 온전해야 한다. RAW 의 "a2" 는 truncated 검증기가 잡는다 - 사용자가
    # 지적하지 않았어도 잘린 답변은 결함이므로 case0 으로 안 보낸다.
    case = dataclasses.replace(
        to_case(parse_conversations(RAW)[0], 2),
        current_query="그럼 반차는 어떻게 되나요?",
        llm_ans_on_last_q="연차는 입사일 기준으로 매년 15일이 부여됩니다. "
                          "자세한 내용은 인사규정 제12조를 확인해 주세요.")
    judge = _Judge()
    result = classify_turn(case, judge)

    assert judge.asked == ["observe"], judge.asked
    assert result.classification.primary_case == "case0", result.classification
    assert result.judgment is None and result.citation is None


def test_output_is_json_serializable():
    json.dumps(build_output(_pairs()), ensure_ascii=False)


def test_summary_reports_case_and_confidence():
    text = summarize(_pairs())
    assert "case20" in text and "medium" in text
