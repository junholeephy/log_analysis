"""데이터 실태 조사 — 진단 전에 무엇이 들어있는지부터 본다.

필터 조건은 분포를 보고 정해야 한다. "eval_score 40점 미만"이 10건을 거를지 3000건을
거를지 모르는 채로 임계값을 쓰면, 돌려보고 나서야 알게 된다. 이 도구는 필터를 걸기 전에
각 축이 어떻게 생겼는지, 어떤 조건이 몇 건을 남기는지 먼저 보여준다.

LLM을 쓰지 않는다. 순수 집계다.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional, Sequence

from ragdiag.conv import Conversation, Turn
from ragdiag.report import _pad, _w


def _bar(count: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return ""
    filled = round(width * count / total)
    return "█" * filled + "·" * (width - filled)


def _dist(title: str, counter: Counter, total: int, limit: int = 12) -> list[str]:
    if not counter:
        return [f"  {title}: (값 없음)"]
    lines = [f"  {title}"]
    name_width = min(28, max(_w(str(k)) for k in counter))
    for name, count in counter.most_common(limit):
        label = str(name) if str(name) else "(비어 있음)"
        lines.append(
            f"    {_pad(label, name_width + 2)}{count:>6}  "
            f"{count / total:>5.1%}  {_bar(count, total)}"
        )
    remaining = len(counter) - limit
    if remaining > 0:
        lines.append(f"    … 그 외 {remaining}종")
    return lines


def _numeric(title: str, values: Sequence[float]) -> list[str]:
    present = [v for v in values if v is not None]
    if not present:
        return [f"  {title}: (값 없음)"]
    ordered = sorted(present)
    n = len(ordered)

    def q(p: float) -> float:
        return ordered[min(n - 1, int(n * p))]

    return [
        f"  {title}  n={n}  결측 {len(values) - n}",
        f"    최소 {ordered[0]:.1f} · p25 {q(.25):.1f} · 중앙 {q(.5):.1f} · "
        f"p75 {q(.75):.1f} · p90 {q(.9):.1f} · 최대 {ordered[-1]:.1f}",
    ]


def _histogram(values: Sequence[float], bins: int = 10) -> list[str]:
    present = [v for v in values if v is not None]
    if not present:
        return []
    low, high = min(present), max(present)
    if high == low:
        return [f"    전부 {low:.1f}"]
    width = (high - low) / bins
    counts = Counter(min(bins - 1, int((v - low) / width)) for v in present)
    lines = []
    for i in range(bins):
        lo, hi = low + i * width, low + (i + 1) * width
        c = counts.get(i, 0)
        lines.append(
            f"    {lo:>6.1f}~{hi:<6.1f} {c:>6}  {_bar(c, len(present))}"
        )
    return lines


def survey(conversations: list[Conversation], metadata: Optional[dict] = None) -> str:
    turns: list[Turn] = [t for c in conversations for t in c.turns]
    followups = [t for t in turns if t.is_followup]

    out = [
        "=" * 78,
        "데이터 실태 조사 (LLM 호출 없음)",
        "=" * 78,
        "",
        "[1] 규모",
        f"  사용자 {len({c.user.user_id for c in conversations})}명 · "
        f"대화 {len(conversations)}건 · 턴 {len(turns)}개",
    ]
    if metadata:
        declared_users = metadata.get("total_users")
        declared_turns = metadata.get("total_turns")
        actual_users = len({c.user.user_id for c in conversations})
        # metadata 와 실제가 어긋나면 파일이 잘렸거나 중복이 있다는 뜻이다.
        for name, declared, actual in [
            ("total_users", declared_users, actual_users),
            ("total_turns", declared_turns, len(turns)),
        ]:
            if declared is not None:
                mark = "" if declared == actual else "   <-- 불일치"
                out.append(f"  metadata.{name}: 선언 {declared} / 실제 {actual}{mark}")

    lengths = Counter(len(c.turns) for c in conversations)
    out += ["", "[2] 대화당 턴 수"]
    for n in sorted(lengths):
        out.append(f"    {n:>3}턴  {lengths[n]:>6}건  {_bar(lengths[n], len(conversations))}")

    # 진단 케이스가 될 수 있는 것은 후속 턴뿐이다. 첫 턴은 비판할 답변이 없다.
    out += [
        "",
        "[3] 진단 가능 후보",
        f"  전체 턴                {len(turns):>6}",
        f"  후속 턴 (eval 라벨 있음) {len(followups):>6}  "
        f"{len(followups) / max(len(turns), 1):.1%}",
        "  └ 이 후속 턴들이 필터의 대상이다. 첫 턴은 직전 답변이 없어 제외된다.",
    ]

    out += ["", "[4] llm_eval_result — 후속 질문의 성격"]
    out += _dist("라벨 분포", Counter(t.eval_result for t in followups), len(followups))
    out += [""]
    out += _numeric("llm_eval_score", [t.eval_score for t in followups])
    out += _histogram([t.eval_score for t in followups])

    out += ["", "[5] llm_emotion_result — 사용자 감정"]
    with_emotion = [t for t in followups if t.emotion_result]
    out += _dist("라벨 분포", Counter(t.emotion_result for t in with_emotion),
                 max(len(with_emotion), 1))
    out += [""]
    out += _numeric("llm_emotion_score", [t.emotion_score for t in followups])
    out += _histogram([t.emotion_score for t in followups])

    out += ["", "[6] retrieved_data"]
    empty = sum(1 for t in turns if not t.retrieved)
    out += [
        f"  검색 결과 없음          {empty:>6}  {empty / max(len(turns), 1):.1%}",
        f"  검색 결과 있음          {len(turns) - empty:>6}",
    ]
    chunk_counts = [len(t.retrieved) for t in turns if t.retrieved]
    if chunk_counts:
        out += _numeric("  청크 수", chunk_counts)

    out += ["", "[7] trace_matched"]
    out += _dist("값 분포", Counter(str(t.trace_matched) for t in turns), len(turns))

    out += ["", "[8] 부서 · 직급"]
    out += _dist("부서", Counter(c.user.dept for c in conversations), len(conversations), 8)
    out += _dist("직급", Counter(c.user.job_grade for c in conversations),
                 len(conversations), 8)

    out += [
        "",
        "=" * 78,
        "필터를 걸기 전에 [4]와 [5]를 보고 조건을 정하세요.",
        "조건을 걸었을 때 몇 건이 남는지는 --eval-label / --emotion-label /",
        "--max-eval-score / --max-emotion-score 를 함께 주면 이어서 보여줍니다.",
    ]
    return "\n".join(out)


def preview_filter(
    conversations: list[Conversation],
    eval_labels: Optional[set[str]] = None,
    emotion_labels: Optional[set[str]] = None,
    max_eval_score: Optional[float] = None,
    max_emotion_score: Optional[float] = None,
) -> str:
    """조건을 걸었을 때 몇 건이 남는지 미리 본다.

    조건을 하나씩 누적하면서 각 단계에서 몇 건이 떨어지는지 보여준다.
    한 번에 다 걸고 "3건 남았다"만 보면 어느 조건이 과했는지 알 수 없다.
    """
    followups = [
        (c, t) for c in conversations for t in c.turns
        if t.is_followup and c.turn_at(t.turn - 1) is not None
    ]
    lines = ["=" * 78, "필터 미리보기", "=" * 78, "",
             f"  후속 턴 (직전 턴 존재)                  {len(followups):>6}"]

    current = followups
    steps = [
        ("eval_result 포함", eval_labels,
         lambda ct: ct[1].eval_result in eval_labels),
        ("emotion_result 포함", emotion_labels,
         lambda ct: ct[1].emotion_result in emotion_labels),
        (f"eval_score <= {max_eval_score}", max_eval_score,
         lambda ct: ct[1].eval_score is not None and ct[1].eval_score <= max_eval_score),
        (f"emotion_score <= {max_emotion_score}", max_emotion_score,
         lambda ct: ct[1].emotion_score is not None
         and ct[1].emotion_score <= max_emotion_score),
    ]
    for name, condition, predicate in steps:
        if condition is None:
            continue
        before = len(current)
        current = [ct for ct in current if predicate(ct)]
        lines.append(
            f"  {_pad(name, 36)}{len(current):>6}   (-{before - len(current)})"
        )

    lines += ["", f"  최종 진단 대상                         {len(current):>6}"]
    if current:
        lines += ["", "  남은 케이스 미리보기 (최대 5건)"]
        for conv, turn in current[:5]:
            question = turn.user_question[:52]
            lines.append(
                f"    {conv.conversation_id[:16]:<18} turn {turn.turn:<3} "
                f"{turn.eval_result or '-':<10} {question}"
            )
    return "\n".join(lines)
