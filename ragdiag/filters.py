"""필터 파일 적용 — 어느 턴을 진단할지 고른다.

필터는 llm_eval / llm_emotion 의 점수와 라벨, 그리고 조직·기간으로 턴을 좁힌다.

두 가지를 특히 조심한다.

**1. 라벨 이름 표기가 세 군데에서 다르다.**
필터는 "I. 매우부정"(붙여쓰기), taxonomy 문서는 "매우 부정"(띄어쓰기),
로그는 "매우 부정". 문자열을 그대로 비교하면 에러 없이 0건이 나온다.
labels.resolve() 가 글자·붙여쓰기·띄어쓰기를 모두 받아준다.

**2. 점수는 다시 계산한다.**
필터가 query_scores 를 들고 있다는 건 점수표를 바꿔 쓰겠다는 뜻이다. 기록된
llm_eval_score 는 옛 점수표로 계산된 값이라 그대로 쓰면 필터가 의도와 다르게 걸린다.
llm_alternatives 가 있으면 새 점수표로 재계산하고, 없을 때만 기록값을 쓴다.

떨어진 건수를 단계별로 남기는 것도 의도적이다. 한 번에 다 걸고 "3건 남았다"만 보면
어느 조건이 과했는지 알 수 없다.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from ragdiag.conv import Conversation, Turn
from ragdiag.labels import (
    DEFAULT_QUERY_SCORES,
    EMOTION_LABELS,
    QUERY_LABELS,
    expected_score,
    resolve,
    resolve_all,
)
from ragdiag.schema import Case

# 필터에서 "제한 없음"을 뜻하는 값들
ANY_VALUES = {"전체", "all", "ALL", "", "*"}


def _as_set(value: Any) -> set[str]:
    """'전체' · 빈 값 · 문자열 · 리스트 · 트리(dict) 를 모두 집합으로."""
    if value is None:
        return set()
    if isinstance(value, str):
        return set() if value.strip() in ANY_VALUES else {value.strip()}
    if isinstance(value, dict):
        # org_tree 는 {부서: {직무: ...}} 형태. 비어 있으면 제한 없음.
        return {k for k in value if k not in ANY_VALUES}
    if isinstance(value, (list, tuple, set)):
        return {str(v).strip() for v in value if str(v).strip() not in ANY_VALUES}
    return {str(value)}


def _as_range(value: Any) -> Optional[tuple[float, float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        low, high = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    return (min(low, high), max(low, high))



# 턴 필터는 정수가 아니라 구간 문자열로 온다: "1-5 턴", "6-10 턴", "51 턴 이상".
# 정수로 파싱하면 조용히 0건이 된다.
_RANGE = re.compile(r"(\d+)\s*[-~]\s*(\d+)")
_AT_LEAST = re.compile(r"(\d+)[^\d]*이상")
_AT_MOST = re.compile(r"(\d+)[^\d]*이하")
_BARE = re.compile(r"^\D*(\d+)\D*$")


def parse_turn_buckets(values: Any) -> list[tuple[int, float]]:
    """구간 문자열들을 (하한, 상한) 목록으로. 여러 개면 합집합(OR)이다."""
    buckets: list[tuple[int, float]] = []
    for raw in (values or []):
        text = str(raw).strip()
        if not text:
            continue
        match = _RANGE.search(text)
        if match:
            low, high = int(match.group(1)), int(match.group(2))
            buckets.append((min(low, high), max(low, high)))
            continue
        match = _AT_LEAST.search(text)
        if match:
            buckets.append((int(match.group(1)), math.inf))
            continue
        match = _AT_MOST.search(text)
        if match:
            buckets.append((1, int(match.group(1))))
            continue
        match = _BARE.match(text)
        if match:
            n = int(match.group(1))
            buckets.append((n, n))
    return buckets


def in_buckets(turn: int, buckets: list[tuple[int, float]]) -> bool:
    return any(low <= turn <= high for low, high in buckets)


@dataclass
class FilterSpec:
    name: str = "filter"
    job_grades: set[str] = field(default_factory=set)   # 필터의 role
    positions: set[str] = field(default_factory=set)
    depts: set[str] = field(default_factory=set)
    job_names: set[str] = field(default_factory=set)
    turn_buckets: list[tuple[int, float]] = field(default_factory=list)
    use_date: bool = False
    start_date: str = ""
    end_date: str = ""
    eval_range: Optional[tuple[float, float]] = None
    emotion_range: Optional[tuple[float, float]] = None
    eval_letters: set[str] = field(default_factory=set)
    emotion_letters: set[str] = field(default_factory=set)
    query_scores: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_QUERY_SCORES))
    # 해석하지 못한 라벨 지정자. 조용히 버리면 오타 하나로 필터가 통째로 빗나간다.
    unknown_labels: list[str] = field(default_factory=list)


def parse_filter(raw: dict) -> FilterSpec:
    state = raw.get("state", raw)
    org = state.get("org") or {}
    tree = state.get("org_tree") or {}

    eval_letters, unknown_eval = resolve_all(state.get("eval_labels"), QUERY_LABELS)
    emotion_letters, unknown_emotion = resolve_all(
        state.get("emotion_labels"), EMOTION_LABELS
    )

    scores = dict(DEFAULT_QUERY_SCORES)
    for letter, value in (state.get("query_scores") or {}).items():
        try:
            scores[letter] = float(value)
        except (TypeError, ValueError):
            continue

    return FilterSpec(
        name=raw.get("name") or "filter",
        job_grades=_as_set(state.get("role")),
        positions=_as_set(org.get("db_position_name")),
        depts=_as_set(tree.get("db_dept_name")),
        job_names=_as_set(tree.get("db_job_name")),
        turn_buckets=parse_turn_buckets(state.get("turn")),
        use_date=bool(state.get("use_date")),
        start_date=state.get("start_date") or "",
        end_date=state.get("end_date") or "",
        eval_range=_as_range(state.get("eval_range")),
        emotion_range=_as_range(state.get("emotion_range")),
        eval_letters=eval_letters,
        emotion_letters=emotion_letters,
        query_scores=scores,
        unknown_labels=unknown_eval + unknown_emotion,
    )


def load_filter(path: str | Path) -> FilterSpec:
    return parse_filter(json.loads(Path(path).read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# 적용
# ---------------------------------------------------------------------------

@dataclass
class Selected:
    conversation: Conversation
    turn: Turn
    eval_score: Optional[float]
    emotion_score: Optional[float]


@dataclass
class Step:
    name: str
    remaining: int
    dropped: int


def score_turn(turn: Turn, spec: FilterSpec) -> tuple[Optional[float], Optional[float]]:
    """필터의 점수표로 다시 계산한 (eval, emotion) 점수.

    emotion 점수표는 taxonomy 에 고정이라 필터가 덮어쓰지 않는다.
    """
    from ragdiag.labels import DEFAULT_EMOTION_SCORES

    return (
        expected_score(turn.eval_alternatives, spec.query_scores, turn.eval_score),
        expected_score(
            turn.emotion_alternatives, DEFAULT_EMOTION_SCORES, turn.emotion_score
        ),
    )


def _in_range(value: Optional[float], bounds: Optional[tuple[float, float]]) -> bool:
    if bounds is None:
        return True
    if value is None:
        return False        # 판정 못 한 걸 통과시키면 조건이 무의미해진다
    return bounds[0] <= value <= bounds[1]


def apply_filter(
    conversations: list[Conversation], spec: FilterSpec
) -> tuple[list[Selected], list[Step]]:
    """필터를 걸고, 각 단계에서 몇 건이 떨어졌는지 함께 돌려준다."""
    candidates = [
        Selected(conv, turn, *score_turn(turn, spec))
        for conv in conversations
        for turn in conv.turns
        # 분석 대상은 2턴 이상인 대화뿐이다. 후속 턴이면서 직전 턴이 실제로
        # 있어야 "비판받은 답변"이 존재한다. 1턴짜리 대화는 여기서 전부 빠진다.
        if turn.is_followup and conv.turn_at(turn.turn - 1) is not None
    ]
    steps = [Step("진단 가능 후속 턴 (2턴 이상 대화)", len(candidates), 0)]

    def narrow(name: str, active: bool, keep: Callable[[Selected], bool]) -> None:
        nonlocal candidates
        if not active:
            return
        before = len(candidates)
        candidates = [s for s in candidates if keep(s)]
        steps.append(Step(name, len(candidates), before - len(candidates)))

    narrow("직급(role)", bool(spec.job_grades),
           lambda s: s.conversation.user.job_grade in spec.job_grades)
    narrow("부서", bool(spec.depts), lambda s: s.conversation.user.dept in spec.depts)
    narrow("직무", bool(spec.job_names),
           lambda s: s.conversation.user.job_name in spec.job_names)
    narrow("직위", bool(spec.positions),
           lambda s: s.conversation.user.position_name in spec.positions)
    narrow("턴 구간", bool(spec.turn_buckets),
           lambda s: in_buckets(s.turn.turn, spec.turn_buckets))
    narrow(
        f"기간 {spec.start_date}~{spec.end_date}",
        spec.use_date and bool(spec.start_date or spec.end_date),
        lambda s: (not spec.start_date or s.turn.timestamp[:10] >= spec.start_date)
        and (not spec.end_date or s.turn.timestamp[:10] <= spec.end_date),
    )
    narrow(
        f"eval_score {spec.eval_range}", spec.eval_range is not None,
        lambda s: _in_range(s.eval_score, spec.eval_range),
    )
    narrow(
        f"emotion_score {spec.emotion_range}", spec.emotion_range is not None,
        lambda s: _in_range(s.emotion_score, spec.emotion_range),
    )
    narrow(
        "eval 라벨", bool(spec.eval_letters),
        lambda s: _letter_of(s.turn.eval_result, QUERY_LABELS) in spec.eval_letters,
    )
    narrow(
        "emotion 라벨", bool(spec.emotion_letters),
        lambda s: _letter_of(s.turn.emotion_result, EMOTION_LABELS)
        in spec.emotion_letters,
    )
    return candidates, steps


def _letter_of(result: str, table) -> str:
    label = resolve(result, table)
    return label.letter if label else ""


def to_cases(selected: list[Selected], history_turns: int = 0) -> list[Case]:
    from ragdiag.conv import MAX_HISTORY_TURNS, to_case

    turns = history_turns or MAX_HISTORY_TURNS
    cases = [to_case(s.conversation, s.turn.turn, turns) for s in selected]
    return [c for c in cases if c is not None]


def render_steps(spec: FilterSpec, steps: list[Step]) -> str:
    from ragdiag.report import _pad

    lines = ["=" * 78, f"필터 적용: {spec.name}", "=" * 78, ""]
    for step in steps:
        suffix = f"   (-{step.dropped})" if step.dropped else ""
        lines.append(f"  {_pad(step.name, 36)}{step.remaining:>6}{suffix}")
    if spec.unknown_labels:
        lines += [
            "",
            f"  [!] 해석하지 못한 라벨 {len(spec.unknown_labels)}개: "
            + ", ".join(spec.unknown_labels),
            "      이 조건은 적용되지 않았습니다. 표기를 확인하세요.",
        ]
    return "\n".join(lines)
