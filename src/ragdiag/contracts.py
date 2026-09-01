"""입력 데이터 계약 — 운영 환경에서 회수한 포맷 정보가 도착하는 유일한 지점.

실데이터는 밖으로 나오지 않는다. 운영 실험에서 이쪽으로 돌아오는 것은 **포맷**과
**사람의 인사이트** 둘뿐이고, 포맷이 여러 파일에 흩어져 있으면 반영할 때마다
어디를 고쳐야 하는지부터 찾아야 한다. 그래서 여기 한 곳에 둔다.

**구조만 적는다.** 실제 값·분포·식별 가능한 코드값 목록은 적지 않는다. 부서명이나
직급 코드의 실제 목록은 그 자체가 운영 환경 정보다. allowed 에 적는 것은 파서가 의미를
갖고 분기하는 값(true/false 표기 같은 것)뿐이다.

`note` 는 운영 환경에서 확인된 사실을 적는 자리다. "실제로는 null 이 온다", "이 필드는
turn 1 에만 비어 있다" 같은 것. 다음 사이클의 코드가 그걸 근거로 바뀐다.

validate() 는 계약 위반을 **사람이 그대로 옮겨 적을 수 있는 문장**으로 돌려준다.
결과 파일을 반출할 수 없으므로 화면에 찍히는 그 문장이 포맷 회수의 주 채널이다.
"validation failed" 같은 메시지는 여기서 결함이다 — 옮겨 적을 것이 없다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class Field:
    name: str
    dtype: str                      # int | float | str | bool | list | dict | datetime
    nullable: bool
    allowed: Optional[tuple] = None  # 파서가 분기하는 값만. 운영 코드값 목록은 적지 않는다
    rng: Optional[tuple] = None      # (min, max)
    note: str = ""                   # 운영 환경에서 확인된 사실을 적는 자리
    # 파이프라인이 읽지 않는 필드. 어긋나도 결과가 달라지지 않으므로 MISMATCH 로
    # 세지 않는다 - 계약 위반 줄은 "판정이 틀렸을 수 있다"는 뜻이어야 하고,
    # 거기 잡음이 섞이면 운영 환경에서 그 줄을 안 보게 된다.
    unused: bool = False

    def describe(self) -> str:
        bits = [self.dtype]
        if self.unused:
            bits.append("파이프라인 미사용")
        if self.nullable:
            bits.append("nullable")
        if self.allowed:
            bits.append(f"allowed={self.allowed}")
        if self.rng:
            bits.append(f"range={self.rng}")
        return " · ".join(bits)


# ---------------------------------------------------------------------------
# conv_eval 로그
#
# 중첩 구조라 층마다 나눠 적는다.
#   users[] → conversations[] → turns[]
# ---------------------------------------------------------------------------

USER_SCHEMA = (
    Field("user_id", "str", False,
          note="사번 등 원본 식별자. 출력에 실어 원본 로그와 조인한다"),
    Field("db_login_id", "str", True),
    Field("db_dept_name", "str", True, note="조직 분류의 소분류와 매칭되는 값"),
    Field("db_job_name", "str", True, note="직무. job_grade(직급)와 다른 축이다"),
    Field("job_grade", "str", True, note="직급"),
    Field("db_position_name", "str", True, note="직위"),
)

CONVERSATION_SCHEMA = (
    Field("conversation_id", "str", False),
    Field("turns", "list", False, note="시간순. 2턴 이상이어야 분석 대상이다"),
)

TURN_SCHEMA = (
    Field("turn", "int", False, rng=(1, None), note="1부터. 순서가 어긋난 로그를 본 적 있다"),
    Field("timestamp", "str", True,
          note="지연 판정에는 쓸 수 없다 — 턴 시각 차이에 사용자가 생각한 시간이 섞인다"),
    Field("user_question", "str", False),
    Field("llm_response", "str", True,
          note="자원 부족 시 서비스가 정해진 안내 문구를 여기 넣는다 (case9)"),
    Field("retrieved_data", "str|list", True,
          note="청크가 \\n\\n 또는 \\n 으로 연결된 문자열로 오는 배포가 있다. "
               "파서가 쪼갠다. 비어 있으면 검색 결과 0건이고, 서비스가 "
               "'검색 없이 답할 수 있다'고 판단한 경우도 여기 해당한다 (case21)"),
    Field("prev_question", "str|list", True, unused=True,
          note="운영 환경로그에서 list 로 관측됨 (2026-09-01, 16,141건). 파이프라인은 "
               "읽지 않는다 - pre_queries 를 turn 순서로 직접 만든다. "
               "다만 이게 서비스가 모델에 실제로 넘긴 히스토리라면 우리가 재구성한 "
               "것과 다를 수 있다. case14 판정의 전제가 걸려 있으니 내용 확인 필요"),
    Field("trace_matched", "bool|str", True,
          note="문자열 'true'/'yes'/'y'/'1'/'n' 로도 온다. 파서가 흡수한다. "
               "2턴 이상 대화라는 뜻이지만 선언값과 실제 턴 수가 어긋난 로그를 본 적 있다"),
    Field("llm_eval_result", "str", True,
          note="직전 턴과의 관계 분류. turn 1 에서는 비어 있다. "
               "실제 라벨 목록은 운영 taxonomy 에 있다"),
    Field("llm_eval_score", "float", True, rng=(0, 100)),
    Field("llm_eval_score_top1", "float", True, rng=(0, 100),
          note="1순위 라벨의 점수. llm_eval_score 는 확률가중 기대점수일 수 있다"),
    Field("llm_alternatives", "list", True,
          note="[{label, probability}]. 확률가중 기대점수 계산에 쓴다"),
    Field("llm_emotion_result", "str", True),
    Field("llm_emotion_score", "float", True, rng=(0, 100)),
    Field("llm_emotion_score_top1", "float", True, rng=(0, 100)),
    Field("llm_emotion_alternatives", "list", True),
)

# 로그 최상위에서 사용자 배열이 들어 있을 수 있는 키. 배포마다 다르다.
USER_ROOT_KEYS = ("users", "analysis_results", "data")

SCHEMAS = {
    "user": USER_SCHEMA,
    "conversation": CONVERSATION_SCHEMA,
    "turn": TURN_SCHEMA,
}


# ---------------------------------------------------------------------------
# 검증
# ---------------------------------------------------------------------------

_DTYPE = {
    "int": (int,), "float": (int, float), "str": (str,),
    "bool": (bool,), "list": (list,), "dict": (dict,),
}


def _types(dtype: str) -> tuple:
    """`str|list` 처럼 둘 다 오는 필드가 있다. 로그가 실제로 그렇다."""
    out: tuple = ()
    for part in dtype.split("|"):
        out += _DTYPE.get(part.strip(), ())
    return out


@dataclass
class Mismatch:
    """계약 위반 하나. 한 줄로 찍혀 사람이 그대로 옮겨 적는다."""

    layer: str
    field: str
    kind: str
    detail: str
    count: int = 1

    def line(self) -> str:
        where = f"{self.layer}.{self.field}"
        return f"{where:<28} : {self.detail}"

    @property
    def key(self) -> tuple:
        return (self.layer, self.field, self.kind)


def _sample_values(rows: list[dict], name: str, limit: int = 4) -> str:
    """예상 밖 값의 **모양**만 보여준다. 실데이터 값을 그대로 찍지 않는다."""
    kinds = {type(r.get(name)).__name__ for r in rows if name in r}
    return ", ".join(sorted(kinds)[:limit]) or "없음"


def validate(rows: list[dict], schema: tuple[Field, ...], layer: str) -> list[Mismatch]:
    """한 층의 레코드들을 계약과 대조한다.

    개별 값을 찍지 않는다 - 타입 이름과 건수만 남긴다. 실데이터가 화면을
    거쳐 밖으로 나가는 경로를 만들지 않기 위해서다.
    """
    found: dict[tuple, Mismatch] = {}

    def add(field: str, kind: str, detail: str):
        m = Mismatch(layer, field, kind, detail)
        if m.key in found:
            found[m.key].count += 1
        else:
            found[m.key] = m

    for field in schema:
        present = [r for r in rows if field.name in r]
        if rows and not present:
            add(field.name, "missing", "로그에 이 키가 하나도 없다")
            continue

        nulls = sum(1 for r in present if r.get(field.name) in (None, ""))
        if nulls and not field.nullable:
            add(field.name, "null",
                f"{nulls:,}건이 비어 있는데 nullable=False")

        want = _types(field.dtype)
        if want:
            bad = [r for r in present
                   if r.get(field.name) is not None
                   and not isinstance(r[field.name], want)]
            if bad:
                add(field.name, "dtype",
                    f"{field.dtype} 를 기대했으나 {_sample_values(bad, field.name)} "
                    f"가 왔다 ({len(bad):,}건)")

        if field.rng:
            low, high = field.rng
            out = 0
            for r in present:
                v = r.get(field.name)
                if not isinstance(v, (int, float)):
                    continue
                if (low is not None and v < low) or (high is not None and v > high):
                    out += 1
            if out:
                add(field.name, "range", f"범위 {field.rng} 밖 {out:,}건")

        if field.allowed:
            out = sum(1 for r in present
                      if r.get(field.name) is not None
                      and r[field.name] not in field.allowed)
            if out:
                add(field.name, "allowed",
                    f"허용값 {field.allowed} 밖 {out:,}건")

    # 계약에 없는 키. 새 필드가 생겼다는 신호이고, 그게 인사이트가 된다.
    known = {f.name for f in schema}
    extra: dict[str, int] = {}
    for r in rows:
        for key in r:
            if key not in known and key not in ("conversations", "turns"):
                extra[key] = extra.get(key, 0) + 1
    for key, n in sorted(extra.items()):
        add(key, "unknown", f"계약에 없는 키 ({n:,}건). 새 필드인지 확인할 것")

    return list(found.values())


@dataclass
class ContractReport:
    checked: int = 0
    mismatches: list[Mismatch] = None
    # 안 쓰는 필드에서 나온 어긋남. 결과에 영향이 없으므로 따로 담는다.
    notes: list[Mismatch] = None

    def __post_init__(self):
        if self.mismatches is None:
            self.mismatches = []
        if self.notes is None:
            self.notes = []

    @property
    def n_ok(self) -> int:
        return self.checked - len({m.field for m in self.mismatches})

    @property
    def ok(self) -> bool:
        return not self.mismatches


def check_log(payload: dict) -> ContractReport:
    """conv_eval 페이로드 전체를 계약과 대조한다.

    분류를 돌리기 전에 부른다. 여기서 나온 줄들이 운영 환경에서 이쪽으로 돌아오는
    포맷 정보의 전부다.
    """
    users: list[dict] = []
    for key in USER_ROOT_KEYS:
        if isinstance(payload.get(key), list):
            users = payload[key]
            break

    convs = [c for u in users for c in (u.get("conversations") or [])
             if isinstance(c, dict)]
    turns = [t for c in convs for t in (c.get("turns") or []) if isinstance(t, dict)]

    report = ContractReport(
        checked=len(USER_SCHEMA) + len(CONVERSATION_SCHEMA) + len(TURN_SCHEMA))
    found = (validate(users, USER_SCHEMA, "user")
             + validate(convs, CONVERSATION_SCHEMA, "conversation")
             + validate(turns, TURN_SCHEMA, "turn"))
    skip = {f.name for schema in SCHEMAS.values() for f in schema if f.unused}
    report.mismatches = [m for m in found if m.field not in skip]
    report.notes = [m for m in found if m.field in skip]
    return report


def shape(payload: dict) -> str:
    """`1,204,331 rows x 27 cols` 자리에 들어갈 한 줄."""
    users: list[dict] = []
    for key in USER_ROOT_KEYS:
        if isinstance(payload.get(key), list):
            users = payload[key]
            break
    convs = [c for u in users for c in (u.get("conversations") or [])]
    turns = [t for c in convs if isinstance(c, dict) for t in (c.get("turns") or [])]
    return (f"{len(users):,} users / {len(convs):,} conversations / "
            f"{len(turns):,} turns")
