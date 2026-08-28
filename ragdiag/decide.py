"""[회귀 기준선 전용] 새 코드에서 쓰지 말 것.

conv_parse.py 가 현행 경로다. 이 모듈은 실제 LLM 으로 23/23 검증된
구 파이프라인을 그대로 보존하기 위해 남아 있다 — 그 회귀셋이 새 파이프라인의
라우팅 결함(약한 증거가 강한 증거를 가로챈 문제)을 잡아냈다.

최종 라벨 결정 - 진리표.

라벨을 LLM이 고르지 않고 코드가 결정한다. 이유:

1. 한 번의 호출로 원인까지 물으면 모델이 원인을 먼저 직감하고 판정값을 거기에
   역으로 맞춘다(합리화). 판정값만 받아서 코드가 조합하면 그 경로가 막힌다.
2. 라벨 체계를 바꾸고 싶을 때 LLM을 다시 돌리지 않아도 된다. 저장된 판정값에
   이 함수만 다시 적용하면 된다.
3. "왜 이 라벨이 붙었나"에 항상 답할 수 있다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ragdiag.schema import (
    GroundingCheck,
    Label,
    NeedAnalysis,
    SufficiencyJudgment,
    Verdict,
)
from ragdiag.verify import CitationCheck


@dataclass
class Diagnosis:
    case_id: str
    label: Label
    reason: str                      # 이 라벨이 붙은 경로. 감사용.
    complaint_type: str
    context_dependent: bool
    verdict_raw: Optional[Verdict] = None
    verdict_final: Optional[Verdict] = None
    citation_failed: bool = False
    answer_used_rag: Optional[str] = None
    missing: str = ""
    resolved_question: str = ""
    unmet_need: str = ""
    evidence: list[dict] = field(default_factory=list)
    dropped_evidence: list[dict] = field(default_factory=list)
    dept: str = "unknown"
    job_grade: str = "unknown"
    n_chunks: int = 0


def decide(
    case_id: str,
    need: NeedAnalysis,
    judgment: Optional[SufficiencyJudgment],
    check: Optional[CitationCheck],
    grounding: Optional[GroundingCheck],
) -> Diagnosis:
    base = dict(
        case_id=case_id,
        complaint_type=need.complaint_type,
        context_dependent=need.context_dependent,
        resolved_question=need.resolved_question,
        unmet_need=need.unmet_need,
    )

    # 불만이 내용에 대한 게 아니면 sufficiency 판정 자체가 의미 없다.
    if need.complaint_type == "format_or_style":
        return Diagnosis(label="out_of_scope", reason="불만이 형식/스타일", **base)
    if need.complaint_type == "other":
        return Diagnosis(label="unclassified", reason="불만 성격 판별 실패", **base)

    if judgment is None or check is None:
        raise ValueError(f"{case_id}: 내용 불만인데 sufficiency 판정이 없다")

    base.update(
        verdict_raw=judgment.verdict,
        missing=judgment.missing,
        evidence=[
            {"chunk_index": e.chunk_index, "quote": e.quote, "ratio": round(e.ratio, 3),
             "index_corrected": e.index_corrected}
            for e in check.kept
        ],
        dropped_evidence=check.dropped,
    )

    # 문서에 답이 있다고 주장하려면 살아남은 인용이 있어야 한다.
    # 인용이 하나도 검증되지 않았다면 그 주장은 사전지식에서 나온 것으로 보고 강등한다.
    citation_failed = judgment.verdict in ("sufficient", "partial") and check.n_kept == 0
    verdict = "insufficient" if citation_failed else judgment.verdict
    base.update(verdict_final=verdict, citation_failed=citation_failed)

    if verdict == "insufficient":
        reason = "인용 검증 실패로 강등" if citation_failed else "문서에 답 없음"
        return Diagnosis(label="rag_insufficient", reason=reason, **base)

    if verdict == "partial":
        return Diagnosis(label="rag_partial", reason="문서가 일부만 충족", **base)

    if grounding is None:
        raise ValueError(f"{case_id}: verdict가 sufficient인데 grounding 확인이 없다")

    base.update(answer_used_rag=grounding.answer_used_rag)
    if grounding.answer_used_rag in ("ignored", "contradicted"):
        return Diagnosis(
            label="rag_sufficient_generation_failed",
            reason=f"문서는 충분했으나 답변이 {grounding.answer_used_rag}",
            **base,
        )
    return Diagnosis(
        label="rag_sufficient_other", reason="문서도 충분, 활용도 함", **base
    )
