"""라벨 테이블과 필터 적용 테스트.

가장 위험한 실패는 에러가 아니라 **조용한 0건**이다. 라벨 표기가 세 군데에서 다르고
(필터 "I. 매우부정" · 문서 "매우 부정" · 로그 "매우 부정"), 문자열을 그대로 비교하면
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

@pytest.mark.parametrize("filename,table", [
    ("query_taxonomy.md", QUERY_LABELS),
    ("emotion_taxonomy.md", EMOTION_LABELS),
])
def test_code_table_matches_the_document(filename, table):
    """코드 테이블이 taxonomy 문서와 어긋나지 않았는지.

    테이블을 코드에 둔 건 에어갭 배포 때문이다. 그 대가로 문서와 벌어질 수 있으므로
    여기서 잡는다.
    """
    path = ROOT / filename
    if not path.exists():
        pytest.skip(f"{filename} 없음 — 드리프트 검사 생략")
    from_doc = load_markdown_table(path)
    assert set(from_doc) == set(table), "라벨 글자 집합이 다름"
    for letter, label in from_doc.items():
        assert label.score == table[letter].score, f"{letter} 점수 불일치"
        known = {normalize_name(n) for n in (table[letter].name,) + table[letter].aliases}
        assert normalize_name(label.name) in known, (
            f"{letter}: 문서의 '{label.name}' 가 코드 테이블에 없음. "
            "표기가 바뀌었으면 name 을 고치고, 오타면 aliases 에 추가하세요."
        )


def test_query_table_has_all_eighteen_labels():
    assert len(QUERY_LABELS) == 18
    assert set(QUERY_LABELS) == set("ABCDEFGHIJKLMNOPQR")


def test_emotion_scores_span_the_full_scale():
    assert DEFAULT_EMOTION_SCORES["A"] == 100.0   # 매우 긍정
    assert DEFAULT_EMOTION_SCORES["I"] == 0.0     # 매우 부정


def test_explicit_feedback_anchors_the_query_scale():
    # 점수가 만족도 대리 지표라는 근거. 필터가 낮은 점수를 고르는 이유이기도 하다.
    assert DEFAULT_QUERY_SCORES["L"] == 0     # 명시적 부정 피드백
    assert DEFAULT_QUERY_SCORES["M"] == 100   # 명시적 긍정 피드백


# ---------------------------------------------------------------------------
# 라벨 표기 변형 — 조용한 0건의 원인
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spec", ["I", "i", "I. 매우부정", "I. 매우 부정", "매우 부정", "매우부정"])
def test_every_written_form_resolves_to_the_same_label(spec):
    label = resolve(spec, EMOTION_LABELS)
    assert label is not None and label.letter == "I"


def test_filter_spelling_matches_log_spelling():
    """필터는 '매우부정', 로그는 '매우 부정'. 띄어쓰기 하나로 갈리면 안 된다."""
    assert resolve("I. 매우부정", EMOTION_LABELS) is resolve("매우 부정", EMOTION_LABELS)


def test_unknown_label_is_reported_not_swallowed():
    letters, unknown = resolve_all(["I. 매우부정", "Z. 없는라벨"], EMOTION_LABELS)
    assert letters == {"I"}
    assert unknown == ["Z. 없는라벨"]


def test_resolve_returns_none_for_garbage():
    assert resolve("존재하지 않는 이름", QUERY_LABELS) is None
    assert resolve("", QUERY_LABELS) is None


# ---------------------------------------------------------------------------
# 점수 계산 — 확률가중 기대점수
# ---------------------------------------------------------------------------

EVAL_ALTS = [
    {"label": "F", "name": "조건 변경", "probability": 0.9},
    {"label": "B", "name": "맥락 추가", "probability": 0.095},
    {"label": "I", "name": "범위 좁히기", "probability": 0.003},
    {"label": "A", "name": "심화 확장", "probability": 0.002},
    {"label": "K", "name": "명확화 요구", "probability": 0.001},
]
EMOTION_ALTS = [
    {"label": "D", "name": "긍정적 중립", "probability": 0.931},
    {"label": "E", "name": "중립", "probability": 0.067},
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
    changed = dict(DEFAULT_QUERY_SCORES, F=0)   # 조건 변경을 0점으로
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
        "emotion_labels": ["I. 매우부정"],
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
    assert spec.turns == set()       # []


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
        _turn(2, "K", "I"),   # 명확화 요구(25) + 매우 부정(0)  -> 통과
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
    """문서에 '예시 요첟' 오타가 있었다.

    eval 시스템이 문서대로 설정됐다면 그 오타를 그대로 뱉을 수 있다. 어느 쪽이
    실제인지 확인할 수 없으므로 둘 다 D로 해석한다 — 안 그러면 조용히 0건이 된다.
    """
    assert resolve("예시 요첟", QUERY_LABELS).letter == "D"
    assert resolve("예시 요청", QUERY_LABELS).letter == "D"
    assert resolve("D. 예시 요첟", QUERY_LABELS).letter == "D"
