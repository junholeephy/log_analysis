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
from typing import Optional
from dataclasses import dataclass, field

from ragdiag.schema import Evidence

from ragdiag.settings import EVIDENCE_MIN_QUOTE_CHARS, MATCH_THRESHOLD

# 옛 이름. checks.py 에도 같은 이름의 다른 값(답변이 제시한 인용의 최소 길이)이
# 있어서 한쪽만 고치고 양쪽을 고쳤다고 착각하기 쉬웠다. settings 에서 이름을
# 갈랐고, 여기 별칭은 기존 호출부를 위해 남긴다.
MIN_QUOTE_CHARS = EVIDENCE_MIN_QUOTE_CHARS

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
    # 무엇을 상대로 대조했는지. 0 이면 검색 결과가 아예 없었다는 뜻이고,
    # 이건 판정이 아니라 로그에 적힌 사실이라 라우팅이 다르게 읽어야 한다.
    # None 은 '모름'이다 - 옛 호출부가 넘기지 않은 경우.
    n_chunks: Optional[int] = None

    @property
    def n_kept(self) -> int:
        return len(self.kept)

    @property
    def any_corrected(self) -> bool:
        return any(e.index_corrected for e in self.kept)


def verify_evidence(evidence: list[Evidence], chunks: list[str]) -> CitationCheck:
    check = CitationCheck(n_chunks=len(chunks))
    for ev in evidence:
        if len(normalize(ev.quote)) < EVIDENCE_MIN_QUOTE_CHARS:
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
