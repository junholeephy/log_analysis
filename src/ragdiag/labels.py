"""llm_eval / llm_emotion 라벨 테이블.

**여기 있는 이름·점수는 자리표시자다.** 실제 라벨 이름과 점수는 운영 코드값이라
저장소에 두지 않는다 (규격 §1.1 · C3 — "식별 가능한 코드값 목록은 적지 않는다").
이 저장소는 public 이고, 라벨 집합은 그 자체로 운영 환경 분류 체계를 드러낸다.

실값은 설정으로 온다:

    labels:
      query:   configs/query_taxonomy.md
      emotion: configs/emotion_taxonomy.md

두 파일은 `.gitignore` 에 있고 운영 환경에서는 `{AA}/configs/` 에 둔다. 형식은 그대로다
(`A. 이름 -> 점수`) — 운영 환경에 이미 있는 문서를 그대로 쓰라는 뜻이다.

구조(글자 A~R / A~I, 개수)는 남긴다. 파서가 `llm_alternatives` 의 글자를 읽어야
하고, 그건 값이 아니라 형식이다.

**실값 없이 라벨 조건을 건 필터를 돌리면 조용히 0건이 나온다.** 가장 찾기 어려운
실패라, `is_placeholder()` 로 그 조합을 계산 전에 막는다.

각 라벨에는 점수가 붙어 있고, 이 점수가 **만족도 대리 지표**다. 명시적 부정
피드백이 0점, 명시적 긍정 피드백이 100점인 척도라 방향이 분명하다. 낮은 점수 =
직전 답변이 만족스럽지 않았다는 신호이므로, 필터가 여기에 걸린다.

기록된 점수의 계산식(예시 데이터로 검증함):

    *_score       = Σ(확률 × 라벨점수) / Σ확률      확률가중 기대점수
    *_score_top1  = argmax 라벨의 점수

필터 파일이 `query_scores`를 들고 있는 이유가 이것이다. 점수표를 바꾸면 기록된
`llm_eval_score`는 낡은 값이 되므로 `llm_alternatives`에서 다시 계산해야 한다.

테이블을 코드에 두고 .md 파일과 일치하는지는 테스트로 확인한다. 파일을 런타임에
읽으면 에어갭 배포에 파일을 같이 넣어야 하고, 코드에만 두면 문서와 어긋난다.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Label:
    letter: str
    name: str
    score: float
    group: str = ""
    # 문서·로그에서 다르게 쓰이는 표기. 오타도 포함한다.
    # eval 시스템이 문서의 오타를 그대로 뱉을 수 있는데, 어느 쪽이 실제인지
    # 확인할 방법이 없으므로 둘 다 받는 편이 안전하다.
    aliases: tuple[str, ...] = ()


def normalize_name(name: str) -> str:
    """라벨 이름 대조용 정규화.

    같은 라벨이 세 군데에서 다르게 적힌다. 필터 파일은 "I. 어떤라벨"(붙여쓰기),
    taxonomy 문서는 "어떤 라벨"(띄어쓰기), 로그의 result 값은 또 다를 수 있다.
    공백을 지우고 맞춘다 — 안 하면 필터가 에러 없이 0건을 돌려준다.
    가장 찾기 어려운 실패다.
    """
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", name)).lower()


# ---------------------------------------------------------------------------
# 자리표시자 테이블
#
# 개수와 글자만 실제와 같다. 이름·점수·그룹은 운영 환경 값이라 여기 두지 않는다.
# 점수를 균등하게 두는 것도 의도다 - 실값 없이 점수 조건을 걸면 결과가 무의미한데,
# 그럴듯한 숫자가 박혀 있으면 무의미한 줄을 모른다.
# ---------------------------------------------------------------------------

QUERY_LETTERS = "ABCDEFGHIJKLMNOPQR"
EMOTION_LETTERS = "ABCDEFGHI"

_PLACEHOLDER_QUERY = {
    letter: Label(letter, f"질의유형 {letter}", 50.0, "(자리표시자)")
    for letter in QUERY_LETTERS
}
_PLACEHOLDER_EMOTION = {
    letter: Label(letter, f"감정 {letter}", 50.0, "(자리표시자)")
    for letter in EMOTION_LETTERS
}

QUERY_LABELS: dict[str, Label] = dict(_PLACEHOLDER_QUERY)
EMOTION_LABELS: dict[str, Label] = dict(_PLACEHOLDER_EMOTION)

DEFAULT_QUERY_SCORES = {letter: label.score for letter, label in QUERY_LABELS.items()}
DEFAULT_EMOTION_SCORES = {letter: label.score for letter, label in EMOTION_LABELS.items()}


def is_placeholder() -> bool:
    """실값이 아직 안 들어왔는가.

    이걸 안 보고 라벨·점수 조건을 걸면 필터가 **에러 없이 0건**을 돌려준다.
    로그에 적힌 실제 라벨 이름은 자리표시자 "질의유형 K" 와 절대 안 맞기 때문이다.
    """
    return (QUERY_LABELS == _PLACEHOLDER_QUERY
            and EMOTION_LABELS == _PLACEHOLDER_EMOTION)


def install(query: Optional[dict] = None, emotion: Optional[dict] = None) -> list[str]:
    """실값 테이블을 끼운다. 무엇이 들어왔는지 돌려준다.

    모듈 전역을 바꾸는 것은 config.apply() 와 같은 방식이다 - 필터·조사기가 이미
    이 전역을 읽고 있어서, 그쪽을 전부 인자로 바꾸는 것보다 얕게 끝난다.
    """
    changed = []
    for name, table, target in (("query", query, QUERY_LABELS),
                                ("emotion", emotion, EMOTION_LABELS)):
        if not table:
            continue
        target.clear()
        target.update(table)
        changed.append(f"labels.{name} ({len(table)}개)")
    DEFAULT_QUERY_SCORES.clear()
    DEFAULT_QUERY_SCORES.update({k: v.score for k, v in QUERY_LABELS.items()})
    DEFAULT_EMOTION_SCORES.clear()
    DEFAULT_EMOTION_SCORES.update({k: v.score for k, v in EMOTION_LABELS.items()})
    return changed


# ---------------------------------------------------------------------------
# 조회
# ---------------------------------------------------------------------------

def resolve(spec: str, table: dict[str, Label]) -> Optional[Label]:
    """라벨 지정자를 Label 로 바꾼다.

    받아들이는 형태:
      "I"              글자만
      "I. 어떤라벨"     필터 파일 형태 (붙여쓰기)
      "어떤 라벨"       로그의 result 값 (띄어쓰기)
    """
    if not spec:
        return None
    text = spec.strip()

    match = re.match(r"^([A-Z])[.)]\s*(.*)$", text)
    if match and match.group(1) in table:
        return table[match.group(1)]
    if len(text) == 1 and text.upper() in table:
        return table[text.upper()]

    wanted = normalize_name(match.group(2) if match else text)
    for label in table.values():
        names = (label.name,) + label.aliases
        if any(normalize_name(n) == wanted for n in names):
            return label
    return None


def resolve_all(specs, table: dict[str, Label]) -> tuple[set[str], list[str]]:
    """지정자 목록을 글자 집합으로. 못 찾은 것은 따로 돌려준다.

    조용히 버리면 오타 하나로 필터가 통째로 빗나간다.
    """
    letters, unknown = set(), []
    for spec in specs or []:
        label = resolve(str(spec), table)
        if label:
            letters.add(label.letter)
        else:
            unknown.append(str(spec))
    return letters, unknown


def expected_score(
    alternatives: list[dict],
    scores: dict[str, float],
    fallback: Optional[float] = None,
) -> Optional[float]:
    """확률가중 기대점수. 기록된 *_score 와 같은 계산이다.

    점수표를 바꿔가며 필터를 걸 수 있게 하려면 기록값이 아니라 이 함수로 다시 계산해야
    한다. alternatives 가 비어 있으면 재계산할 수 없으므로 기록값을 그대로 쓴다.
    """
    usable = [
        (a["label"], a["probability"])
        for a in alternatives
        if a.get("label") in scores and a.get("probability")
    ]
    mass = sum(p for _, p in usable)
    if not usable or mass <= 0:
        return fallback
    return sum(scores[letter] * p for letter, p in usable) / mass


# ---------------------------------------------------------------------------
# 문서와의 일치 확인 (테스트에서 사용)
# ---------------------------------------------------------------------------

_LINE = re.compile(r"^([A-Z])\.\s*(.+?)\s*->\s*([\d.]+)", re.M)


def parse_markdown_table(text: str) -> dict[str, Label]:
    """taxonomy .md 를 파싱한다. 코드 테이블이 문서와 어긋나지 않았는지 볼 때 쓴다."""
    table = {}
    group = ""
    for line in text.splitlines():
        if line.startswith("#"):
            group = line.lstrip("# ").strip()
            continue
        match = _LINE.match(line.strip())
        if match:
            letter, name, score = match.groups()
            table[letter] = Label(letter, name.strip(), float(score), group)
    return table


def load_markdown_table(path: str | Path) -> dict[str, Label]:
    return parse_markdown_table(Path(path).read_text(encoding="utf-8"))
