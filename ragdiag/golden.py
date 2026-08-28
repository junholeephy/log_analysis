"""Step 1 관측 품질 채점.

관측 8개가 라우팅 전체를 좌우하는데 지금까지 실제 모델로 검증된 적이 없다.
케이스마다 **확실한 필드만** 채점한다 — 모든 필드에 정답을 억지로 붙이면
설계자의 추측을 정답으로 만드는 셈이 된다.

필드별 일치율을 따로 내는 게 핵심이다. 전체 평균만 보면 어느 관측이 약한지
보이지 않고, 약한 관측이 무엇이냐에 따라 고칠 프롬프트가 달라진다.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

from ragdiag.report import _pad, _w
from ragdiag.schema import Observation


@dataclass
class FieldScore:
    field: str
    hits: int = 0
    total: int = 0
    misses: list[tuple[str, Any, Any]] = field(default_factory=list)  # (case, 기대, 실제)

    @property
    def rate(self) -> float:
        return self.hits / self.total if self.total else 0.0


def score_observation(
    case_id: str, expect: dict, obs: Optional[Observation], scores: dict[str, FieldScore]
) -> list[str]:
    """기대 필드만 채점하고, 어긋난 필드 이름을 돌려준다."""
    wrong = []
    for name, want in expect.items():
        entry = scores.setdefault(name, FieldScore(name))
        entry.total += 1
        got = getattr(obs, name, None) if obs else None
        # 정답이 하나로 확정되지 않는 케이스는 허용 집합으로 둔다. 억지로 하나를
        # 고르게 만들면 설계자의 추측이 정답이 된다.
        ok = got in want if isinstance(want, (set, frozenset)) else got == want
        if ok:
            entry.hits += 1
        else:
            entry.misses.append((case_id, want, got))
            wrong.append(name)
    return wrong


def render(scores: dict[str, FieldScore], per_case: dict[str, list[str]],
           errors: list[tuple[str, str]]) -> str:
    graded = [s for s in scores.values() if s.total]
    total_hits = sum(s.hits for s in graded)
    total_all = sum(s.total for s in graded)
    clean = [c for c, wrong in per_case.items() if not wrong]

    lines = [
        "=" * 78,
        "Step 1 관측 채점 (합성 골든셋 — 실데이터가 아님)",
        "=" * 78,
        "",
        f"  전 필드 일치        {total_hits}/{total_all}  "
        f"({total_hits / total_all:.0%})" if total_all else "  채점할 항목 없음",
        f"  모든 필드가 맞은 케이스  {len(clean)}/{len(per_case)}",
        "",
        "[1] 관측 필드별 일치율",
    ]
    width = max((_w(s.field) for s in graded), default=10) + 2
    for entry in sorted(graded, key=lambda s: (s.rate, -s.total)):
        bar = "█" * round(20 * entry.rate) + "·" * (20 - round(20 * entry.rate))
        lines.append(
            f"  {_pad(entry.field, width)}{entry.hits:>3}/{entry.total:<3} "
            f"{entry.rate:>5.0%}  {bar}"
        )

    misses = [(e.field, m) for e in graded for m in e.misses]
    if misses:
        lines += ["", "[2] 어긋난 판정"]
        for name, (case_id, want, got) in misses:
            lines.append(f"  {_pad(name, width)}{case_id:<10} 기대 {want!r} · 실제 {got!r}")

    if errors:
        lines += ["", f"[!] 관측 실패 {len(errors)}건"]
        for case_id, message in errors[:5]:
            lines.append(f"  {case_id}: {message}")

    lines += [
        "",
        "=" * 78,
        "이 점수는 합성 데이터 기준이다. 내가 만든 케이스이므로 실데이터의 표현 방식과",
        "다를 수 있고, 프롬프트를 이 셋에 맞춰 고치면 과대평가된다.",
    ]
    return "\n".join(lines)
