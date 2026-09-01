"""라벨 테이블과 필터 적용 테스트.

가장 위험한 실패는 에러가 아니라 **조용한 0건**이다. 라벨 표기가 세 군데에서 다르고
(필터 "I. 감정인디아" · 문서 "감정 인디아" · 로그 "감정 인디아"), 문자열을 그대로 비교하면
아무 경고 없이 아무것도 안 걸린다. 그래서 표기 변형을 전부 테스트한다.
"""

import json
from pathlib import Path

import pytest

from ragdiag.conv import parse_conversations
from ragdiag.filters import (
    FilterSpec,
    apply_filter,
    parse_filter,
    render_steps,
    score_turn,
    to_cases,
)
from ragdiag.labels import (
    DEFAULT_EMOTION_SCORES,
    DEFAULT_QUERY_SCORES,
    EMOTION_LABELS,
    QUERY_LABELS,
    expected_score,
    load_markdown_table,
    normalize_name,
    resolve,
    resolve_all,
)

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# 라벨 테이블
# ---------------------------------------------------------------------------

def test_shipped_table_carries_no_real_label_values():
    """실값은 사내 코드값이라 저장소에 없다 (규격 §1.1 · C3).

    이 저장소는 public 이고, 라벨 집합은 그 자체로 사내 분류 체계를 드러낸다.
    전에는 labels.py 와 query_taxonomy.md 양쪽에 실값이 박혀 있었다.
    """
    import re as _re

    from ragdiag import labels as mod

    source = (ROOT / "src/ragdiag/labels.py").read_text(encoding="utf-8")
    for banned in ("질의 킬로", "질의 에코", "감정 인디아", "질의 알파",
                   "질의 리마"):
        assert banned not in source, f"실제 라벨 이름이 소스에 있다: {banned}"

    assert not (ROOT / "query_taxonomy.md").exists() or \
        _re.search(r"^query_taxonomy\.md$", (ROOT / ".gitignore").read_text(encoding="utf-8"),
                   _re.M), "실값 파일이 저장소에 추적되고 있다"

    # 구조는 남는다 - 파서가 alternatives 의 글자를 읽어야 하고 그건 형식이다.
    assert set(mod.QUERY_LETTERS) == set("ABCDEFGHIJKLMNOPQR")
    assert set(mod.EMOTION_LETTERS) == set("ABCDEFGHI")


def test_placeholder_is_detectable(placeholder_labels):
    """실값 없이 도는 상태를 코드가 알아야 막을 수 있다."""
    from ragdiag import labels as mod

    assert mod.is_placeholder()
    assert len(mod.QUERY_LABELS) == 18 and len(mod.EMOTION_LABELS) == 9


def test_installing_a_table_replaces_names_and_scores():
    """설정으로 실값을 끼우면 이름도 점수도 바뀐다. 별칭 조회까지 따라와야 한다."""
    from ragdiag import labels as mod

    assert not mod.is_placeholder(), "conftest 가 테스트 테이블을 끼웠어야 한다"
    assert DEFAULT_QUERY_SCORES["L"] == 0
    assert DEFAULT_EMOTION_SCORES["A"] == 100.0
    assert resolve("질의 뎔타", QUERY_LABELS).letter == "D", "별칭이 따라오지 않았다"


# ---------------------------------------------------------------------------
# 라벨 표기 변형 — 조용한 0건의 원인
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spec", ["I", "i", "I. 감정인디아", "I. 감정 인디아", "감정 인디아", "감정인디아"])
def test_every_written_form_resolves_to_the_same_label(spec):
    label = resolve(spec, EMOTION_LABELS)
    assert label is not None and label.letter == "I"


def test_filter_spelling_matches_log_spelling():
    """필터는 '감정인디아', 로그는 '감정 인디아'. 띄어쓰기 하나로 갈리면 안 된다."""
    assert resolve("I. 감정인디아", EMOTION_LABELS) is resolve("감정 인디아", EMOTION_LABELS)


def test_unknown_label_is_reported_not_swallowed():
    letters, unknown = resolve_all(["I. 감정인디아", "Z. 없는라벨"], EMOTION_LABELS)
    assert letters == {"I"}
    assert unknown == ["Z. 없는라벨"]


def test_resolve_returns_none_for_garbage():
    assert resolve("존재하지 않는 이름", QUERY_LABELS) is None
    assert resolve("", QUERY_LABELS) is None


# ---------------------------------------------------------------------------
# 점수 계산 — 확률가중 기대점수
# ---------------------------------------------------------------------------

EVAL_ALTS = [
    {"label": "F", "name": "질의 폭스", "probability": 0.9},
    {"label": "B", "name": "맥락 추가", "probability": 0.095},
    {"label": "I", "name": "질의 인디아", "probability": 0.003},
    {"label": "A", "name": "질의 알파", "probability": 0.002},
    {"label": "K", "name": "질의 킬로", "probability": 0.001},
]
EMOTION_ALTS = [
    {"label": "D", "name": "긍정적 감정 에코", "probability": 0.931},
    {"label": "E", "name": "감정 에코", "probability": 0.067},
    {"label": "B", "name": "긍정", "probability": 0.001},
]


def test_expected_score_reproduces_the_recorded_value():
    """예시 데이터의 기록값을 재현하는지.

    llm_eval_score=45.57, llm_emotion_score=61.69 이 계산으로 나온다는 것이
    query_scores 를 필터에서 덮어쓸 수 있는 근거다.
    """
    assert expected_score(EVAL_ALTS, DEFAULT_QUERY_SCORES) == pytest.approx(45.57, abs=0.1)
    assert expected_score(EMOTION_ALTS, DEFAULT_EMOTION_SCORES) == pytest.approx(61.69, abs=0.1)


def test_probabilities_are_normalized():
    # 상위 5개만 실려 확률합이 1이 아니어도 기대점수가 왜곡되면 안 된다.
    half = [{"label": "M", "probability": 0.5}]      # M = 100
    assert expected_score(half, DEFAULT_QUERY_SCORES) == pytest.approx(100.0)


def test_falls_back_to_the_recorded_score_without_alternatives():
    assert expected_score([], DEFAULT_QUERY_SCORES, fallback=42.0) == 42.0
    assert expected_score([], DEFAULT_QUERY_SCORES) is None


def test_overriding_scores_changes_the_result():
    changed = dict(DEFAULT_QUERY_SCORES, F=0)   # 질의 폭스을 0점으로
    before = expected_score(EVAL_ALTS, DEFAULT_QUERY_SCORES)
    after = expected_score(EVAL_ALTS, changed)
    assert after < before - 30


# ---------------------------------------------------------------------------
# 필터 파일 파싱
# ---------------------------------------------------------------------------

EXAMPLE = {
    "name": "trial_filter",
    "state": {
        "role": "전체",
        "org": {"db_position_name": "전체"},
        "org_tree": {"db_dept_name": {}, "db_job_name": {}},
        "turn": [],
        "use_date": False,
        "start_date": "2026-03-01",
        "end_date": "2026-08-01",
        "eval_range": [0, 60],
        "emotion_range": [0, 20],
        "eval_labels": [],
        "emotion_labels": ["I. 감정인디아"],
        "query_scores": {"A": 80, "K": 25, "L": 0, "M": 100},
    },
}


def test_example_filter_parses():
    spec = parse_filter(EXAMPLE)
    assert spec.name == "trial_filter"
    assert spec.eval_range == (0.0, 60.0)
    assert spec.emotion_range == (0.0, 20.0)
    assert spec.emotion_letters == {"I"}
    assert spec.eval_letters == set()        # 빈 목록 = 제한 없음
    assert spec.unknown_labels == []


def test_everything_means_no_filter():
    spec = parse_filter(EXAMPLE)
    assert spec.positions == set()   # "전체"
    assert spec.depts == set()       # {}
    assert spec.job_names == set()
    assert spec.turn_buckets == []   # []


def test_org_tree_keys_become_the_department_filter():
    raw = {"state": {"org_tree": {"db_dept_name": {"인사팀": {}, "재무팀": {}}}}}
    assert parse_filter(raw).depts == {"인사팀", "재무팀"}


def test_query_scores_override_only_the_given_letters():
    spec = parse_filter(EXAMPLE)
    assert spec.query_scores["A"] == 80
    assert spec.query_scores["C"] == DEFAULT_QUERY_SCORES["C"]   # 안 준 건 기본값 유지


def test_real_example_file_parses_if_present():
    path = ROOT / "filter_data_ex.json"
    if not path.exists():
        pytest.skip("filter_data_ex.json 없음")
    spec = parse_filter(json.loads(path.read_text(encoding="utf-8")))
    assert spec.emotion_letters == {"I"}
    assert spec.unknown_labels == []


# ---------------------------------------------------------------------------
# 필터 적용
# ---------------------------------------------------------------------------

def _turn(n, eval_letter=None, emotion_letter=None, ts="2026-03-06 10:00:00.000"):
    turn = {
        "turn": n, "timestamp": ts,
        "user_question": f"질문 {n}", "llm_response": f"답변 {n}",
        "retrieved_data": json.dumps([f"청크 {n}"]),
        "llm_eval_result": None, "llm_emotion_result": None,
        "llm_alternatives": [], "llm_emotion_alternatives": [],
    }
    if eval_letter:
        turn["llm_eval_result"] = QUERY_LABELS[eval_letter].name
        turn["llm_alternatives"] = [{"label": eval_letter, "probability": 1.0}]
    if emotion_letter:
        turn["llm_emotion_result"] = EMOTION_LABELS[emotion_letter].name
        turn["llm_emotion_alternatives"] = [{"label": emotion_letter, "probability": 1.0}]
    return turn


def _convs(turns, dept="인사팀"):
    return parse_conversations({"users": [{
        "user_id": "u1", "db_dept_name": dept, "db_job_name": "인사운영",
        "db_position_name": "팀원", "job_grade": "대리",
        "conversations": [{"conversation_id": "c1", "turns": turns}],
    }]})


def test_only_followups_with_a_previous_turn_are_candidates():
    convs = _convs([_turn(1), _turn(2, "K", "I")])
    selected, steps = apply_filter(convs, FilterSpec())
    assert len(selected) == 1 and selected[0].turn.turn == 2
    assert steps[0].remaining == 1


def test_example_filter_selects_very_negative_low_score_turns():
    convs = _convs([
        _turn(1),
        _turn(2, "K", "I"),   # 질의 킬로(25) + 감정 인디아(0)  -> 통과
        _turn(3, "M", "I"),   # 명시적 긍정(100)               -> eval_range 탈락
        _turn(4, "K", "A"),   # 매우 긍정(100)                 -> emotion 탈락
    ])
    selected, _ = apply_filter(convs, parse_filter(EXAMPLE))
    assert [s.turn.turn for s in selected] == [2]


def test_each_step_records_what_it_dropped():
    """어느 조건이 과했는지 보이지 않으면 필터를 조정할 수 없다."""
    convs = _convs([_turn(1), _turn(2, "M", "A"), _turn(3, "M", "A")])
    _, steps = apply_filter(convs, parse_filter(EXAMPLE))
    names = [s.name for s in steps]
    assert any("eval_score" in n for n in names)
    dropped = {s.name: s.dropped for s in steps}
    assert sum(dropped.values()) == 2


def test_missing_score_is_dropped_not_passed():
    """점수를 계산할 수 없는 턴을 통과시키면 범위 조건이 무의미해진다."""
    turn = _turn(2, "K")
    turn["llm_emotion_alternatives"] = []
    turn["llm_emotion_score"] = None
    convs = _convs([_turn(1), turn])
    spec = FilterSpec(emotion_range=(0.0, 20.0))
    selected, _ = apply_filter(convs, spec)
    assert selected == []


def test_date_filter_is_ignored_when_use_date_is_false():
    convs = _convs([_turn(1), _turn(2, "K", "I", ts="2020-01-01 00:00:00.000")])
    spec = parse_filter(EXAMPLE)          # use_date=False, 기간 2026-03~08
    assert len(apply_filter(convs, spec)[0]) == 1


def test_date_filter_applies_when_enabled():
    convs = _convs([_turn(1), _turn(2, "K", "I", ts="2020-01-01 00:00:00.000")])
    raw = json.loads(json.dumps(EXAMPLE))
    raw["state"]["use_date"] = True
    assert apply_filter(convs, parse_filter(raw))[0] == []


def test_department_filter():
    convs = _convs([_turn(1), _turn(2, "K", "I")], dept="재무팀")
    assert apply_filter(convs, FilterSpec(depts={"인사팀"}))[0] == []
    assert len(apply_filter(convs, FilterSpec(depts={"재무팀"}))[0]) == 1


def test_selected_turns_become_cases_with_the_right_pairing():
    convs = _convs([_turn(1), _turn(2, "K", "I")])
    selected, _ = apply_filter(convs, parse_filter(EXAMPLE))
    cases = to_cases(selected)
    assert len(cases) == 1
    assert cases[0].current_query == "질문 2"          # 불만은 turn 2
    assert cases[0].llm_ans_on_last_q == "답변 1"      # 비판받은 답변은 turn 1
    assert cases[0].rag_chunks == ["청크 1"]           # 문서도 turn 1의 것


def test_unknown_labels_are_surfaced_in_the_report():
    raw = json.loads(json.dumps(EXAMPLE))
    raw["state"]["emotion_labels"] = ["Z. 오타라벨"]
    spec = parse_filter(raw)
    text = render_steps(spec, apply_filter(_convs([_turn(1)]), spec)[1])
    assert "해석하지 못한 라벨" in text and "Z. 오타라벨" in text


def test_score_is_recomputed_with_the_filter_table():
    turn_raw = _turn(2, "F", "D")
    turn_raw["llm_alternatives"] = EVAL_ALTS
    turn_raw["llm_eval_score"] = 45.57
    convs = _convs([_turn(1), turn_raw])
    spec = FilterSpec(query_scores=dict(DEFAULT_QUERY_SCORES, F=0))
    selected, _ = apply_filter(convs, spec)
    # 기록값 45.57 이 아니라 새 점수표로 계산된 값이어야 한다
    assert selected[0].eval_score < 15


def test_document_typo_still_resolves():
    """문서에 '질의 뎔타' 오타가 있었다.

    eval 시스템이 문서대로 설정됐다면 그 오타를 그대로 뱉을 수 있다. 어느 쪽이
    실제인지 확인할 수 없으므로 둘 다 D로 해석한다 — 안 그러면 조용히 0건이 된다.
    """
    assert resolve("질의 뎔타", QUERY_LABELS).letter == "D"
    assert resolve("질의 델타", QUERY_LABELS).letter == "D"
    assert resolve("D. 질의 뎔타", QUERY_LABELS).letter == "D"


# ---------------------------------------------------------------------------
# 턴 구간 — 정수가 아니라 구간 문자열로 온다
# ---------------------------------------------------------------------------

from ragdiag.filters import in_buckets, parse_turn_buckets


@pytest.mark.parametrize("text,expected", [
    ("1-5 턴", [(1, 5)]),
    ("6-10 턴", [(6, 10)]),
    ("11-50 턴", [(11, 50)]),
    ("1~5턴", [(1, 5)]),
    ("3", [(3, 3)]),
    ("10 턴 이하", [(1, 10)]),
])
def test_turn_bucket_forms(text, expected):
    assert parse_turn_buckets([text]) == expected


def test_open_ended_bucket_has_no_upper_bound():
    (low, high), = parse_turn_buckets(["51 턴 이상"])
    assert low == 51 and high == float("inf")
    assert in_buckets(9999, [(low, high)])


def test_multiple_buckets_are_a_union():
    buckets = parse_turn_buckets(["1-5 턴", "51 턴 이상"])
    assert [in_buckets(n, buckets) for n in (1, 5, 6, 50, 51, 99)] == [
        True, True, False, False, True, True]


def test_empty_turn_filter_means_all_turns():
    assert parse_turn_buckets([]) == []
    assert parse_turn_buckets(None) == []


def test_unparseable_bucket_is_skipped_not_crashing():
    assert parse_turn_buckets(["최근 턴", ""]) == []


def test_turn_bucket_filters_selected_turns():
    convs = _convs([_turn(1), _turn(2, "K", "I"), _turn(3, "K", "I"),
                    _turn(4, "K", "I")])
    spec = FilterSpec(turn_buckets=parse_turn_buckets(["3-4 턴"]))
    assert [s.turn.turn for s in apply_filter(convs, spec)[0]] == [3, 4]


# ---------------------------------------------------------------------------
# role -> job_grade
# ---------------------------------------------------------------------------

def test_role_filters_on_job_grade():
    convs = parse_conversations({"users": [{
        "user_id": "u", "job_grade": "Staff Engineer", "db_dept_name": "인사팀",
        "conversations": [{"conversation_id": "c", "turns": [_turn(1), _turn(2, "K", "I")]}],
    }]})
    assert len(apply_filter(convs, FilterSpec(job_grades={"Staff Engineer"}))[0]) == 1
    assert apply_filter(convs, FilterSpec(job_grades={"Manager"}))[0] == []


def test_role_everything_means_no_filter():
    assert parse_filter({"state": {"role": "전체"}}).job_grades == set()


def test_role_is_read_from_the_filter_file():
    assert parse_filter({"state": {"role": "Principal Engineer"}}).job_grades == \
        {"Principal Engineer"}


# ---------------------------------------------------------------------------
# 분석 대상은 2턴 이상 대화
# ---------------------------------------------------------------------------

def test_single_turn_conversations_are_excluded():
    """1턴짜리 대화에는 비판받을 직전 답변이 없다."""
    convs = _convs([_turn(1)])
    selected, steps = apply_filter(convs, FilterSpec())
    assert selected == []
    assert "2턴 이상" in steps[0].name
