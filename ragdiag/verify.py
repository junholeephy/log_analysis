"""인용 검증 - knowledge leakage 차단 장치.

판정자 LLM이 "문서에 답이 있다"고 말할 때, 그게 문서를 읽어서인지 자기가 이미
알던 지식 때문인지 프롬프트로는 구분할 수 없다. 그래서 판정자에게 청크에서
글자 그대로 인용을 뽑게 하고, 그 인용이 실제로 청크 안에 있는지 여기서 대조한다.
지어낸 인용은 원문과 일치하지 않으므로 걸러진다.

"지어내지 마세요"라는 프롬프트와 달리 이건 검증 가능한 장치다.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass, field

from ragdiag.schema import Evidence

# 짧은 인용은 아무 문서에나 우연히 들어맞아 검증을 무력화한다.
MIN_QUOTE_CHARS = 8
# 완전 일치가 아니어도 통과시키는 최소 연속 일치 비율. 모델이 조사나 끝맺음을
# 살짝 다듬는 경우를 흡수하되, 문장을 새로 지어내면 통과하지 못하는 수준.
MATCH_THRESHOLD = 0.9

_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """공백과 유니코드 표기 차이를 제거. 청크 분할 과정에서 공백은 쉽게 달라진다."""
    return _WS.sub("", unicodedata.normalize("NFKC", text)).lower()


def match_ratio(quote: str, chunk: str) -> float:
    """quote가 chunk 안에 얼마나 연속으로 들어있는지. 1.0이면 완전 포함."""
    q, c = normalize(quote), normalize(chunk)
    if not q:
        return 0.0
    if q in c:
        return 1.0
    block = difflib.SequenceMatcher(None, q, c, autojunk=False).find_longest_match(
        0, len(q), 0, len(c)
    )
    return block.size / len(q)


@dataclass
class VerifiedEvidence:
    chunk_index: int
    quote: str
    ratio: float
    index_corrected: bool = False


@dataclass
class CitationCheck:
    kept: list[VerifiedEvidence] = field(default_factory=list)
    dropped: list[dict] = field(default_factory=list)

    @property
    def n_kept(self) -> int:
        return len(self.kept)

    @property
    def any_corrected(self) -> bool:
        return any(e.index_corrected for e in self.kept)


def verify_evidence(evidence: list[Evidence], chunks: list[str]) -> CitationCheck:
    check = CitationCheck()
    for ev in evidence:
        if len(normalize(ev.quote)) < MIN_QUOTE_CHARS:
            check.dropped.append({"quote": ev.quote, "reason": "too_short"})
            continue

        if 0 <= ev.chunk_index < len(chunks):
            ratio = match_ratio(ev.quote, chunks[ev.chunk_index])
            if ratio >= MATCH_THRESHOLD:
                check.kept.append(VerifiedEvidence(ev.chunk_index, ev.quote, ratio))
                continue

        # 인덱스는 틀렸어도 인용 자체가 실제 문서에 있으면 지어낸 게 아니다.
        # leakage를 막는 건 인용의 실재성이지 인덱스의 정확성이 아니므로 살린다.
        best_idx, best_ratio = -1, 0.0
        for i, chunk in enumerate(chunks):
            r = match_ratio(ev.quote, chunk)
            if r > best_ratio:
                best_idx, best_ratio = i, r
        if best_ratio >= MATCH_THRESHOLD:
            check.kept.append(
                VerifiedEvidence(best_idx, ev.quote, best_ratio, index_corrected=True)
            )
        else:
            check.dropped.append(
                {"quote": ev.quote, "reason": "not_found", "best_ratio": round(best_ratio, 3)}
            )
    return check
