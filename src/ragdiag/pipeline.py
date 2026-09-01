"""파이프라인을 단계별 함수로 노출한다.

전에는 이 흐름이 conv_parse.py 의 main() 안에 인자 파싱·출력·종료코드와 섞여
있었다. 노트북이나 다른 스크립트에서 부르려면 셸을 거치는 수밖에 없었다.

    로그 → 필터 → Case → 판정 → 출력 JSON
           └ 여기 전용 ┘ └──── 코어 ────┘

**경계는 Case 다.** 왼쪽(로그를 읽고 필터를 거는 부분)은 운영 장비에 이미
구현돼 있으므로 이 저장소의 것은 여기서 검증할 때만 쓴다. 오른쪽(판정과 출력)이
반입 대상이고, Case 를 만들어 넣을 수만 있으면 그대로 돈다.

그래서 오른쪽 함수들은 conv·filters 를 import 하지 않는다. 왼쪽을 쓰는
run_from_conv_eval() 만 함수 안에서 늦게 불러온다 - 이 모듈을 import 하는 것만으로
입력 계층이 딸려오면 "필요한 것만 복사"가 성립하지 않는다.
tests/test_boundary.py 가 그 경계를 실제 import 로 확인한다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ragdiag import settings
from ragdiag.classify import TurnResult, classify_all
from ragdiag.judge import Judge
from ragdiag.output import build_output, summarize
from ragdiag.schema import Case

# ---------------------------------------------------------------------------
# 코어 — Case 를 받아 판정하고 출력 모양으로 만든다
# ---------------------------------------------------------------------------


@dataclass
class Outcome:
    """한 번의 실행 결과. 저장은 호출자가 정한다."""

    results: list[TurnResult] = field(default_factory=list)
    payload: dict = field(default_factory=dict)
    # 필터 단계의 드롭 리포트. 필터를 안 거쳤으면 빈 문자열.
    filter_report: str = ""

    @property
    def n_failed(self) -> int:
        return sum(1 for r in self.results if r.error)

    @property
    def n_llm_calls(self) -> int:
        return sum(r.n_calls for r in self.results)

    def summary(self) -> str:
        return summarize(self._pairs)

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.write_text(json.dumps(self.payload, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        return out

    _pairs: list = field(default_factory=list, repr=False)


def judge_cases(cases: list[Case], judge: Judge,
                workers: Optional[int] = None) -> list[TurnResult]:
    """Case 목록을 판정한다. 3스텝 판정이 도는 곳이다.

    한 턴이 실패해도 나머지는 계속 간다 - 결과의 error 필드로 확인할 것.
    """
    # 기본 인자는 def 시점에 굳어 --config 적용이 안 먹는다. 여기서 푼다.
    return classify_all(cases, judge,
                        max_workers=workers or settings.DEFAULT_WORKERS)


def build_outcome(owners: list, results: list[TurnResult],
                  filter_report: str = "") -> Outcome:
    """판정 결과를 출력 JSON 모양으로 묶는다.

    owners 는 결과를 되돌릴 대화 객체들이다. conversation_id 와 user 두 속성만
    쓰므로 운영 장비의 파서가 만든 객체를 그대로 넣어도 된다.
    """
    pairs = list(zip(owners, results))
    return Outcome(results=results, payload=build_output(pairs),
                   filter_report=filter_report, _pairs=pairs)


# ---------------------------------------------------------------------------
# 여기 전용 — conv_eval 로그를 읽고 필터를 건다
#
# 운영 장비에는 이 단계가 이미 있다. 여기 것은 검증용이다.
# ---------------------------------------------------------------------------


@dataclass
class Selection:
    """필터를 통과한 턴과 거기서 만든 Case 를 짝지어 둔 것.

    짝을 유지해야 판정 결과를 원래 대화로 되돌릴 수 있다. to_cases() 가
    Case 를 만들 수 없는 턴을 걸러내므로 zip 만으로는 어긋난다.
    """

    owners: list = field(default_factory=list)   # Conversation
    cases: list[Case] = field(default_factory=list)
    report: str = ""
    # 필터가 만든 원본 선별 결과. 코어는 쓰지 않고 여기서 점검할 때만 본다
    # (dry-run 이 eval_result·emotion_result 를 보여주는 데 쓴다).
    selected: list = field(default_factory=list)
    # 필터 각 단계의 통과·탈락 건수. 0건이 나왔을 때 어디서 빠졌는지 보려면 필요하다.
    steps: list = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.cases)

    def head(self, n: int) -> "Selection":
        return Selection(self.owners[:n], self.cases[:n], self.report,
                         self.selected[:n], self.steps)


def load_and_select(conv_path: str | Path, filter_path: Optional[str | Path] = None,
                    history_turns: Optional[int] = None,
                    limit: Optional[int] = None) -> Selection:
    """로그를 읽고 필터를 걸어 Case 까지 만든다."""
    from ragdiag.conv import load_conversations
    from ragdiag.filters import FilterSpec, apply_filter, load_filter, render_steps, to_cases

    conversations = load_conversations(conv_path)
    spec = load_filter(filter_path) if filter_path else FilterSpec()
    selected, steps = apply_filter(conversations, spec)
    report = render_steps(spec, steps)

    if limit:
        selected = selected[:limit]

    cases = to_cases(selected,
                     history_turns=history_turns or settings.MAX_HISTORY_TURNS)
    pairs = [(s, c) for s, c in zip(selected, cases) if c is not None]
    return Selection(owners=[s.conversation for s, _ in pairs],
                     cases=[c for _, c in pairs], report=report,
                     selected=[s for s, _ in pairs], steps=steps)


@dataclass
class TurnRef:
    """운영 필터가 고른 턴 하나. user_id 는 있으면 쓰고 없으면 대화 id 로만 찾는다."""

    conversation_id: str
    turn: int
    user_id: str = ""


def read_turn_list(path: str | Path) -> list[TurnRef]:
    """고른 턴 목록을 읽는다. JSON 배열과 JSONL 을 모두 받는다.

    운영 필터의 출력 형식을 여기서 강제하지 않으려고 둘 다 받는다 - 파이프라인이
    JSONL 을 뱉는 경우가 흔하고, 그것 때문에 운영 환경에서 변환 스크립트를 하나 더
    짜게 만들 이유가 없다 (C2 — 거기서는 코드를 못 고친다).

        [{"conversation_id": "C-0001", "turn": 3}, ...]
        {"conversation_id": "C-0001", "turn": 3}      ← 한 줄에 하나
    """
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"{path} 가 비어 있습니다.")

    rows: list = []
    if text.lstrip().startswith("["):
        rows = json.loads(text)
    else:
        for n, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"{path}:{n} 을 읽을 수 없습니다: {e}\n"
                    f"  JSON 배열이거나 한 줄에 객체 하나여야 합니다.") from e

    refs = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{path}[{i}] 가 객체가 아닙니다: {type(row).__name__}")
        cid = row.get("conversation_id") or row.get("conv_id")
        turn = row.get("turn")
        if not cid or turn is None:
            raise ValueError(
                f"{path}[{i}] 에 conversation_id 나 turn 이 없습니다. "
                f"있는 키: {sorted(row)}")
        refs.append(TurnRef(str(cid), int(turn), str(row.get("user_id") or "")))
    return refs


def select_turns(conv_path: str | Path, turns_path: str | Path,
                 history_turns: Optional[int] = None,
                 limit: Optional[int] = None) -> Selection:
    """**필터를 운영 환경에 두고** 고른 턴만 받아 Case 로 만든다.

    필터 로직은 운영 환경 것을 쓰고 여기 것은 검증용이다. 그런데 턴을 Case 로 엮는 일
    (후속 턴 ↔ 직전 턴의 답변 ↔ 그 답변을 만든 청크)은 짝을 틀리면 조용히 엉뚱한
    답변을 판정하게 된다. 그래서 그 부분만은 검증된 이쪽 코드가 한다 - 운영 환경에서
    다시 구현하면 어긋나도 드러나지 않는다.

    라벨 실값도 필요 없다. 무엇을 볼지는 이미 정해져서 왔다.
    """
    from ragdiag.conv import load_conversations
    from ragdiag.filters import Selected, Step, to_cases

    conversations = load_conversations(conv_path)
    refs = read_turn_list(turns_path)

    # (user_id, conversation_id) 와 conversation_id 두 갈래로 찾는다. 대화 id 가
    # 사용자마다 겹칠 수 있어서다 - Case.case_id 가 user_id 를 앞에 두는 이유다.
    #
    # 원본 id 와 마스킹된 id 를 둘 다 등록한다. 파서가 user_id 를 해시로 바꾸므로
    # 목록에 원본을 적어 오면 마스킹된 쪽과 절대 안 맞는다 - 그러면 전부 "로그에
    # 없는 턴"으로 세어지고, 목록은 멀쩡한데 0건이 나온다.
    by_pair, by_conv = {}, {}
    for conv in conversations:
        for uid in {conv.user.user_id, conv.user.raw_user_id}:
            if uid:
                by_pair[(uid, conv.conversation_id)] = conv
        by_conv.setdefault(conv.conversation_id, []).append(conv)

    selected, missing, ambiguous, no_prior = [], [], [], []
    for ref in refs:
        conv = by_pair.get((ref.user_id, ref.conversation_id))
        if conv is None:
            found = by_conv.get(ref.conversation_id) or []
            if len(found) > 1:
                ambiguous.append(ref)
                continue
            conv = found[0] if found else None
        if conv is None:
            missing.append(ref)
            continue
        turn = conv.turn_at(ref.turn)
        if turn is None:
            missing.append(ref)
            continue
        if conv.turn_at(ref.turn - 1) is None:
            # 직전 턴이 판정 대상인 답변이다. 없으면 판정할 것이 없다.
            no_prior.append(ref)
            continue
        selected.append(Selected(conv, turn, turn.eval_score, turn.emotion_score))

    if ambiguous:
        raise ValueError(
            f"대화 id 가 여러 사용자에게 있어 어느 것인지 정할 수 없습니다 "
            f"({len(ambiguous)}건, 예: {ambiguous[0].conversation_id}).\n"
            f"  목록에 user_id 를 함께 넣으세요.")

    steps = [Step(f"운영 필터가 고른 턴 ({Path(turns_path).name})", len(refs), 0)]
    if missing:
        steps.append(Step("로그에 없는 턴", len(refs) - len(missing), len(missing)))
    if no_prior:
        steps.append(Step("직전 턴이 없어 판정 불가", len(selected), len(no_prior)))

    if limit:
        selected = selected[:limit]
    cases = to_cases(selected,
                     history_turns=history_turns or settings.MAX_HISTORY_TURNS)
    pairs = [(sel, case) for sel, case in zip(selected, cases) if case is not None]

    from ragdiag.filters import render_steps

    return Selection(owners=[s.conversation for s, _ in pairs],
                     cases=[c for _, c in pairs],
                     report=render_steps(None, steps),
                     selected=[s for s, _ in pairs], steps=steps)


def run_from_conv_eval(conv_path: str | Path, judge: Judge,
                       filter_path: Optional[str | Path] = None,
                       history_turns: Optional[int] = None,
                       limit: Optional[int] = None,
                       workers: Optional[int] = None) -> Outcome:
    """로그 경로 하나로 끝까지 돌린다. 편의 함수일 뿐 특별한 로직은 없다."""
    selection = load_and_select(conv_path, filter_path, history_turns, limit)
    if not selection.cases:
        return Outcome(filter_report=selection.report)
    results = judge_cases(selection.cases, judge, workers)
    outcome = build_outcome(selection.owners, results, selection.report)
    return outcome


def make_judge(backend: Any, use_cache: bool = True) -> Judge:
    return Judge(backend, cache_dir=settings.CACHE_DIR if use_cache else None)
