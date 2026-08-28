"""llm_eval / llm_emotion 라벨 테이블.

각 라벨에는 점수가 붙어 있고, 이 점수가 **만족도 대리 지표**다.
명시적 부정 피드백(L)이 0점, 명시적 긍정 피드백(M)이 100점인 걸 보면 방향이 분명하다.
낮은 점수 = 직전 답변이 만족스럽지 않았다는 신호이므로, 필터가 여기에 걸린다.

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

    필터 파일은 "I. 매우부정"(붙여쓰기), taxonomy 문서는 "매우 부정"(띄어쓰기),
    로그의 llm_emotion_result 는 "매우 부정"으로 서로 다르다. 공백을 지우고 맞춘다.
    이걸 안 하면 필터가 에러 없이 0건을 돌려준다 — 가장 찾기 어려운 실패다.
    """
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", name)).lower()


# ---------------------------------------------------------------------------
# 테이블 (query_taxonomy.md / emotion_taxonomy.md 와 일치해야 한다)
# ---------------------------------------------------------------------------

QUERY_LABELS: dict[str, Label] = {
    label.letter: label for label in [
        Label("A", "심화 확장", 80, "정보 요구"),
        Label("B", "맥락 추가", 50, "정보 요구"),
        Label("C", "근거/출처 요구", 45, "정보 요구"),
        Label("D", "예시 요청", 60, "정보 요구", aliases=("예시 요첟",)),
        Label("E", "단순 연속 질문", 60, "정보 요구"),
        Label("F", "조건 변경", 45, "속성 조정"),
        Label("G", "조건 제외", 45, "속성 조정"),
        Label("H", "형식 변경", 40, "속성 조정"),
        Label("I", "범위 좁히기", 65, "속성 조정"),
        Label("J", "범위 넓히기", 40, "속성 조정"),
        Label("K", "명확화 요구", 25, "속성 조정"),
        Label("L", "명시적 부정 피드백", 0, "메타 대화"),
        Label("M", "명시적 긍정 피드백", 100, "메타 대화"),
        Label("N", "후속 행동 요청", 75, "메타 대화"),
        Label("O", "대화 종료", 70, "메타 대화"),
        Label("P", "무관한 화제 전환", 50, "메타 대화"),
        Label("Q", "단순 반복/확인", 40, "메타 대화"),
        Label("R", "기타 메타 대화", 50, "메타 대화"),
    ]
}

EMOTION_LABELS: dict[str, Label] = {
    label.letter: label for label in [
        Label("A", "매우 긍정", 100.0),
        Label("B", "긍정", 87.5),
        Label("C", "약간 긍정", 75.0),
        Label("D", "긍정적 중립", 62.5),
        Label("E", "중립", 50.0),
        Label("F", "부정적 중립", 37.5),
        Label("G", "약간 부정", 25.0),
        Label("H", "부정", 12.5),
        Label("I", "매우 부정", 0.0),
    ]
}

DEFAULT_QUERY_SCORES = {letter: label.score for letter, label in QUERY_LABELS.items()}
DEFAULT_EMOTION_SCORES = {letter: label.score for letter, label in EMOTION_LABELS.items()}


# ---------------------------------------------------------------------------
# 조회
# ---------------------------------------------------------------------------

def resolve(spec: str, table: dict[str, Label]) -> Optional[Label]:
    """라벨 지정자를 Label 로 바꾼다.

    받아들이는 형태:
      "I"              글자만
      "I. 매우부정"     필터 파일 형태
      "매우 부정"       로그의 result 값
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
