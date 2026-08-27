"""로더 테스트. 중첩 구조 flatten과 메타 상속, 마스킹."""

from ragdiag.load import mask, parse_cases

RAW = {
    "analysis_results": [{
        "user_id": "EMP1", "db_login_id": "a.b", "job_grade": "과장",
        "db_dept_name": "해외영업팀", "db_job_name": "해외영업", "db_position_name": "파트장",
        "conversations": [{
            "conversation_id": "C1",
            "turns": [
                {"turn": 3, "pre_queries": ["q1", "q2"], "llm_ans_on_last_q": "a",
                 "current_query": "c", "rag_data": ["ch1", "ch2"]},
                {"turn": 10, "pre_queries": ["q3"], "llm_ans_on_last_q": "a2",
                 "current_query": "c2", "rag_data": []},
            ],
        }],
    }]
}


def test_flatten_and_meta_inheritance():
    cases = parse_cases(RAW)
    assert len(cases) == 2
    assert all(c.dept == "해외영업팀" and c.job_grade == "과장" for c in cases)
    assert cases[0].turn == 3 and cases[1].turn == 10


def test_case_id_is_stable_and_masked():
    c = parse_cases(RAW)[0]
    assert c.case_id == f"{mask('EMP1')}:C1:3"
    assert "EMP1" not in c.case_id  # 원본 식별자가 산출물에 새어나가면 안 된다


def test_last_query_is_the_one_that_produced_the_answer():
    assert parse_cases(RAW)[0].last_query == "q2"


def test_rag_data_as_plain_string_becomes_one_chunk():
    raw = {"analysis_results": [{"user_id": "x", "conversations": [
        {"conversation_id": "C", "turns": [{"turn": 1, "rag_data": "통문자열"}]}]}]}
    assert parse_cases(raw)[0].rag_chunks == ["통문자열"]


def test_rag_data_as_dicts_extracts_text():
    raw = {"analysis_results": [{"user_id": "x", "conversations": [
        {"conversation_id": "C", "turns": [{"turn": 1, "rag_data": [
            {"text": "t1", "score": 0.9}, {"content": "t2"}, {"text": "  "}]}]}]}]}
    assert parse_cases(raw)[0].rag_chunks == ["t1", "t2"]


def test_missing_fields_do_not_crash():
    raw = {"analysis_results": [{"user_id": "x", "conversations": [
        {"conversation_id": "C", "turns": [{"turn": 1}]}]}]}
    c = parse_cases(raw)[0]
    assert c.pre_queries == [] and c.rag_chunks == [] and c.dept == "unknown"


def test_synthetic_fixture_loads():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from fixtures.synthetic import CASES, build

    data, expected = build()
    cases = parse_cases(data)
    assert len(cases) == len(CASES) == len(expected)
    assert {c.case_id for c in cases} == set(expected)
    assert all(c.rag_chunks and c.current_query for c in cases)


def test_concatenated_string_is_split_on_blank_lines():
    # 실데이터는 청크를 \n\n으로 이어붙인 통문자열로 온다.
    from ragdiag.load import split_concatenated

    assert split_concatenated("첫 청크입니다.\n\n둘째 청크입니다.\n\n셋째입니다.") == [
        "첫 청크입니다.", "둘째 청크입니다.", "셋째입니다."]


def test_blank_line_split_wins_over_single_newline():
    # 청크 내부에도 개행이 있을 수 있다. 단일 개행부터 쪼개면 한 청크가 찢어진다.
    from ragdiag.load import split_concatenated

    text = "제1조 목적\n이 규정은 출장비를 정한다.\n\n제2조 범위\n전 임직원에 적용한다."
    assert split_concatenated(text) == [
        "제1조 목적\n이 규정은 출장비를 정한다.", "제2조 범위\n전 임직원에 적용한다."]


def test_falls_back_to_single_newline_when_no_blank_lines():
    from ragdiag.load import split_concatenated

    assert split_concatenated("청크 하나\n청크 둘\n청크 셋") == ["청크 하나", "청크 둘", "청크 셋"]


def test_single_chunk_string_stays_one_chunk():
    from ragdiag.load import split_concatenated

    assert split_concatenated("경계가 없는 한 덩어리 문장.") == ["경계가 없는 한 덩어리 문장."]
    assert split_concatenated("   ") == []


def test_concatenated_rag_data_flows_through_loader():
    raw = {"analysis_results": [{"user_id": "x", "conversations": [
        {"conversation_id": "C", "turns": [
            {"turn": 1, "rag_data": "청크 A 내용\n\n청크 B 내용"}]}]}]}
    assert parse_cases(raw)[0].rag_chunks == ["청크 A 내용", "청크 B 내용"]


def test_synthetic_fixture_defaults_to_production_string_format():
    # 실데이터가 통문자열이므로 픽스처도 같은 형태여야 청크 경계 복원 경로가 검증된다.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from fixtures.synthetic import CASES, build

    data, _ = build()
    turn = data["analysis_results"][0]["conversations"][0]["turns"][0]
    assert isinstance(turn["rag_data"], str)

    by_id = {c["conv"] + str(c["turn"]): c for c in CASES}
    for case in parse_cases(data):
        original = by_id[case.conversation_id + str(case.turn)]
        assert case.rag_chunks == original["chunks"]  # 경계가 정확히 복원되어야 한다
