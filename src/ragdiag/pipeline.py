"""파이프라인을 단계별 함수로 노출한다.

전에는 이 흐름이 conv_parse.py 의 main() 안에 인자 파싱·출력·종료코드와 섞여
있었다. 노트북이나 다른 스크립트에서 부르려면 셸을 거치는 수밖에 없었다.

    로그 → 필터 → Case → 판정 → 출력 JSON
           └ 여기 전용 ┘ └──── 코어 ────┘

**경계는 Case 다.** 왼쪽(로그를 읽고 필터를 거는 부분)은 사내 머신에 이미
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
    쓰므로 사내 머신의 파서가 만든 객체를 그대로 넣어도 된다.
    """
    pairs = list(zip(owners, results))
    return Outcome(results=results, payload=build_output(pairs),
                   filter_report=filter_report, _pairs=pairs)


# ---------------------------------------------------------------------------
# 여기 전용 — conv_eval 로그를 읽고 필터를 건다
#
# 사내 머신에는 이 단계가 이미 있다. 여기 것은 검증용이다.
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

    def __len__(self) -> int:
        return len(self.cases)

    def head(self, n: int) -> "Selection":
        return Selection(self.owners[:n], self.cases[:n], self.report,
                         self.selected[:n])


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
                     selected=[s for s, _ in pairs])


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
