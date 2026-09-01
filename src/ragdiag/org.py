"""조직 분류 체계 — 대분류 / 중분류 / 소분류.

class_dept.json · class_job.json 의 구조:

    { "dept_classes": [                     루트 키는 파일마다 다르다
        { "id": 1,
          "name": "A1",                     대분류
          "subclasses": [
            { "name": "a1-aa",              중분류
              "items": ["xxx", "yyy"] } ]   소분류 - 로그의 실제 값과 매칭되는 지점
        } ] }

집계 축을 소분류(팀 이름)로만 두면 팀이 수십 개일 때 표가 읽히지 않는다. 대분류로
접어야 "어느 본부가 문제인가"가 보이고, 그다음 중분류로 좁혀 들어간다.

**매칭되지 않은 값을 조용히 버리지 않는다.** 로그에 있는데 분류 체계에 없는 부서가
있으면 그 건들이 집계에서 사라지거나 빈 칸으로 뭉친다. 어느 쪽이든 "그 조직은 문제가
없다"로 잘못 읽힌다. coverage() 가 미매칭 값을 그대로 돌려주는 이유다.

class_job 이 로그의 어느 필드에 붙는지는 파일만으로 알 수 없다 - conv_eval 에는
db_job_name(직무)과 job_grade(직급)가 둘 다 있다. detect_field() 가 값을 대조해
판별한다. 추측해서 붙이면 매칭률이 0에 가까워도 에러가 안 나서 알아채기 어렵다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ragdiag import settings
from ragdiag.labels import normalize_name

UNMAPPED = "(미분류)"


@dataclass(frozen=True)
class Node:
    """소분류 하나가 속한 자리."""

    item: str
    middle: str      # 중분류 (subclasses[].name)
    major: str       # 대분류 (name)
    major_en: str = ""
    major_id: Optional[int] = None


@dataclass
class Classification:
    name: str                                   # dept / job
    nodes: dict[str, Node] = field(default_factory=dict)   # 정규화된 소분류 -> Node

    def lookup(self, value: str) -> Optional[Node]:
        return self.nodes.get(normalize_name(value or ""))

    def rollup(self, value: str, level: str) -> str:
        """소분류 값을 원하는 층으로 접는다. 매칭 안 되면 (미분류)."""
        node = self.lookup(value)
        if node is None:
            return UNMAPPED
        return {"major": node.major, "middle": node.middle, "item": node.item}[level]

    @property
    def items(self) -> set[str]:
        return {n.item for n in self.nodes.values()}


def parse_classification(raw: dict, name: str = "") -> Classification:
    """루트 키가 무엇이든 (dept_classes / job_classes / classes) 받아준다."""
    entries: list[Any] = []
    for key, value in raw.items():
        if isinstance(value, list):
            entries = value
            name = name or key.replace("_classes", "")
            break

    table = Classification(name=name or "org")
    for major in entries:
        if not isinstance(major, dict):
            continue
        major_name = str(major.get("name") or "")
        for sub in major.get("subclasses") or []:
            if not isinstance(sub, dict):
                continue
            middle = str(sub.get("name") or "")
            for item in sub.get("items") or []:
                if not isinstance(item, str) or not item.strip():
                    continue
                table.nodes[normalize_name(item)] = Node(
                    item=item.strip(), middle=middle, major=major_name,
                    major_en=str(major.get("name_en") or ""),
                    major_id=major.get("id") if isinstance(major.get("id"), int) else None,
                )
    return table


def load_classification(path: str | Path) -> Classification:
    file = Path(path)
    return parse_classification(json.loads(file.read_text(encoding="utf-8")), file.stem)


# ---------------------------------------------------------------------------
# 층으로 좁혀 고르기 — 대분류 → 중분류 → 소분류
#
# 팀이 수십 개면 소분류를 평평하게 늘어놓은 목록에서 고를 수 없다. 위층을 고르면
# 아래층 후보가 거기 매이게 한다. 각 층은 여러 개를 동시에 고를 수 있다.
#
# 이 로직을 화면 밖에 두는 이유: streamlit 없이 시험할 수 있어야 한다. 고르는
# 규칙이 틀리면 표가 조용히 다른 것을 세는데, 그건 화면만 봐서는 안 보인다.
# ---------------------------------------------------------------------------


def observed_tree(table: Optional["Classification"],
                  values: list[str]) -> dict[str, dict[str, list[str]]]:
    """대분류 → 중분류 → 소분류. **로그에 실제로 나온 값만** 담는다.

    체계 전부를 늘어놓으면 이 데이터에 없는 팀까지 고르게 되어, 고른 뒤 0건이
    나오고 사람은 필터를 의심한다. 체계에 없는 값은 UNMAPPED 로 모은다 -
    버리면 "그 조직은 문제가 없다"로 잘못 읽힌다.
    """
    tree: dict[str, dict[str, list[str]]] = {}
    for value in sorted({v for v in values if v}):
        node = table.lookup(value) if table else None
        major, middle = (node.major, node.middle) if node else (UNMAPPED, UNMAPPED)
        tree.setdefault(major, {}).setdefault(middle, [])
        if value not in tree[major][middle]:
            tree[major][middle].append(value)
    return tree


def level_options(tree: dict, majors: list[str],
                  middles: list[str]) -> tuple[list[str], list[str], list[str]]:
    """각 층에 보여줄 후보. 위층에서 고른 것이 있으면 거기 매인다."""
    major_options = sorted(tree)
    live_majors = [m for m in (majors or major_options) if m in tree]

    middle_options = sorted({mid for m in live_majors for mid in tree[m]})
    live_middles = [m for m in (middles or middle_options) if m in middle_options]

    item_options = sorted({item for m in live_majors
                           for mid, items in tree[m].items() if mid in live_middles
                           for item in items})
    return major_options, middle_options, item_options


def allowed_values(tree: dict, majors: list[str], middles: list[str],
                   items: list[str]) -> Optional[set[str]]:
    """세 층의 선택을 통과하는 **원본 값**들. 아무것도 안 골랐으면 None(전체).

    None 과 빈 집합은 다르다 - 전자는 "안 좁혔다"이고 후자는 "골랐는데 해당 없음"
    이다. 둘을 같게 다루면 조건을 걸었는데 전체가 나오는 일이 생긴다.
    """
    if not (majors or middles or items):
        return None
    if items:
        return set(items)
    _, _, candidates = level_options(tree, majors, middles)
    return set(candidates)


# ---------------------------------------------------------------------------
# 진단 — 붙이기 전에 붙을지부터 본다
# ---------------------------------------------------------------------------

@dataclass
class Coverage:
    field_name: str
    matched: int
    total: int
    unmapped: list[str] = field(default_factory=list)   # 로그에는 있는데 체계에 없는 값
    unused: list[str] = field(default_factory=list)     # 체계에는 있는데 로그에 없는 값

    @property
    def rate(self) -> float:
        return self.matched / self.total if self.total else 0.0


def coverage(values: list[str], table: Classification, field_name: str = "") -> Coverage:
    """로그의 값들이 분류 체계에 얼마나 붙는지."""
    seen = [v for v in values if v]
    matched = sum(1 for v in seen if table.lookup(v))
    unmapped = sorted({v for v in seen if not table.lookup(v)})
    used = {normalize_name(v) for v in seen}
    unused = sorted({n.item for key, n in table.nodes.items() if key not in used})
    return Coverage(field_name or table.name, matched, len(seen), unmapped, unused)


# conv_eval 에서 조직 분류가 붙을 수 있는 필드들
CANDIDATE_FIELDS = settings.ORG_CANDIDATE_FIELDS


def detect_field(
    records: list[dict], table: Classification,
    candidates: Optional[tuple[str, ...]] = None,
) -> tuple[Optional[str], dict[str, Coverage]]:
    """어느 필드에 붙는 분류인지 값 대조로 판별한다.

    파일 이름만으로는 알 수 없다 - class_job 이 직무(db_job_name)일 수도 직급
    (job_grade)일 수도 있다. 매칭률이 가장 높은 필드를 고르되, 전부 낮으면
    None 을 돌려준다. 억지로 붙이면 집계가 통째로 (미분류)가 된다.
    """
    # 기본 인자는 def 시점에 굳어 설정 적용이 안 먹는다. 여기서 푼다.
    if candidates is None:
        candidates = settings.ORG_CANDIDATE_FIELDS
    scores = {
        name: coverage([str(r.get(name, "")) for r in records], table, name)
        for name in candidates
    }
    best = max(scores.values(), key=lambda c: c.rate)
    return (best.field_name if best.rate >= 0.5 else None), scores


def render_coverage(scores: dict[str, Coverage], chosen: Optional[str]) -> str:
    lines = []
    for name, cov in sorted(scores.items(), key=lambda kv: -kv[1].rate):
        mark = "  <- 선택" if name == chosen else ""
        lines.append(f"  {name:<20} {cov.matched:>4}/{cov.total:<4} {cov.rate:>5.0%}{mark}")
    if chosen is None:
        lines.append("  어느 필드에도 절반 이상 붙지 않는다. 분류 체계가 이 로그의 것이 맞는지 확인할 것.")
    else:
        cov = scores[chosen]
        if cov.unmapped:
            lines.append(f"  체계에 없는 값 {len(cov.unmapped)}종: "
                         f"{', '.join(cov.unmapped[:8])}"
                         + (" …" if len(cov.unmapped) > 8 else ""))
            lines.append("    이 값들은 (미분류)로 묶인다. 집계에서 빠지는 것이 아니다.")
    return "\n".join(lines)
