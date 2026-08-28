"""Step 3 — 관측과 검증 결과를 taxonomy case 로 바꾼다. LLM 없음.

case 를 LLM 에게 직접 고르게 하지 않는 이유가 셋 있다.

1. 30지선다는 어떤 모델이든 정확도가 안 나온다. 관측을 8개의 좁은 질문으로 쪼개면
   각각은 쉬운 질문이 된다.
2. 한 호출로 case 까지 물으면 모델이 결론을 먼저 직감하고 관측값을 거기 맞춘다.
3. taxonomy 를 고쳐도 관측값은 그대로 재사용된다. 이 파일만 다시 돌리면 된다.

category 는 case 에서 계산되므로 따로 분류하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ragdiag import taxonomy
from ragdiag.checks import Check
from ragdiag.schema import GroundingCheck, Observation, SufficiencyJudgment
from ragdiag.verify import CitationCheck


@dataclass
class Classification:
    primary_case: str
    confidence: str
    reason: str
    secondary_cases: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        payload = taxonomy.describe(self.primary_case)
        payload.update(
            confidence=self.confidence,
            reason=self.reason,
            secondary_cases=[taxonomy.describe(c) for c in self.secondary_cases],
            notes=self.notes,
        )
        return payload


def _check(checks: dict[str, Check], name: str) -> Optional[Check]:
    return checks.get(name)


def _requested_but_broken(checks: dict[str, Check], name: str) -> Optional[bool]:
    """코드 검증 결과를 삼값으로. None 이면 그런 요구가 없었다는 뜻이다."""
    check = _check(checks, name)
    if check is None or check.verdict == "not_applicable":
        return None
    if check.verdict == "violated":
        return True
    if check.verdict == "ok":
        return False
    return None      # undetermined


def secondary_from(obs: Observation, checks: dict[str, Check]) -> list[str]:
    """주 라벨과 독립적으로 성립하는 케이스들.

    한 턴이 case4(모호한 질문)이면서 동시에 case17(검색 실패)일 수 있다. 억지로
    하나만 고르면 정보가 사라지고 tie-break 가 자의적이 된다.
    """
    extra = []
    if not obs.question_self_contained:
        extra.append("case4")          # 참조가 모호한 질문
    if obs.question_multi_intent:
        extra.append("case3")          # 복합 질문
    pii = _check(checks, "pii")
    if pii is not None and pii.violated:
        extra.append("case6")          # 질문에 개인정보
    quoted = _check(checks, "quoted_spans")
    if quoted is not None and quoted.violated:
        extra.append("case20")         # 인용 표기 오류
    # 실제 질문은 도메인과 계산이 겹치는 경우가 흔하다("5영업일 뒤가 언제냐").
    # question_domain 은 하나만 고를 수 있으므로, 등식 오류는 부가로 따로 잡는다.
    arithmetic = _check(checks, "arithmetic")
    if arithmetic is not None and arithmetic.violated:
        extra.append("case22")         # 계산 오류
    return extra


def route(
    obs: Observation,
    checks: dict[str, Check],
    judgment: Optional[SufficiencyJudgment] = None,
    citation: Optional[CitationCheck] = None,
    grounding: Optional[GroundingCheck] = None,
) -> Classification:
    """관측 + 검증 → case. 값싸고 확실한 갈림길을 앞에 둔다."""
    extra = secondary_from(obs, checks)

    def done(case_id: str, reason: str, notes: Optional[list[str]] = None) -> Classification:
        meta = taxonomy.get(case_id)
        return Classification(
            primary_case=case_id,
            confidence=meta.confidence if meta else "low",
            reason=reason,
            secondary_cases=[c for c in extra if c != case_id],
            notes=notes or [],
        )

    # --- 1. 거절 -------------------------------------------------------------
    if obs.answer_refused:
        # case24(정책)와 case25(권한)는 이 로그로 구분할 수 없다. 권한 조회 결과가
        # 없기 때문이다. 구분한 척하지 않고 case24 로 두되 note 로 남긴다.
        return done("case24", "답변이 정책·권한을 이유로 거절함",
                    ["case25(권한 없는 접근)와 구분 불가 — 권한 조회 결과 필요"])

    # --- 1a. 문서가 모델을 조종했는지 ------------------------------------------
    # 답변 내용을 따지기 전에 볼 문제다. 문서에 심긴 지시를 수행했다면
    # 그 답변의 나머지 판정은 의미가 없다.
    injection = _check(checks, "injection")
    if injection is not None and injection.violated:
        return done("case26", f"검색 문서의 지시를 답변이 수행함 — {injection.detail}")

    # --- 1b. 질문 쪽 문제가 먼저다 --------------------------------------------
    # 챗봇이 낼 수 없는 형태를 요구했으면 답변을 탓할 수 없다.
    if obs.requests_unsupported_output:
        return done("case2", "챗봇이 낼 수 없는 형태를 요구함 (링크·이미지 등)")
    # 질문 자체가 답을 특정할 수 없으면 그 뒤 판정이 전부 의미를 잃는다.
    if not obs.question_answerable_as_asked:
        return done("case1", "질문만으로 답을 특정할 수 없음")

    # --- 2. 답이 없거나 끊김 --------------------------------------------------
    if obs.complaint_target == "no_answer":
        truncated = _check(checks, "truncated")
        if truncated is not None and truncated.violated:
            return done("case9", f"답변이 중간에 끊김 — {truncated.detail}")
        return done(taxonomy.UNCLASSIFIED,
                    "답이 없다는 불만인데 답변은 온전함",
                    ["case7(서비스 끊김)일 수 있으나 로그로는 판정 불가"])

    # --- 3. 형식·언어·길이 요청 불이행 -----------------------------------------
    REQUEST_CASES = [
        ("format", "format", "case12"),
        ("language", "language", "case10"),
        ("length", "length", "case11"),
    ]
    for target, check_name, case_id in REQUEST_CASES:
        if obs.complaint_target != target:
            continue
        broken = _requested_but_broken(checks, check_name)
        if broken is True:
            detail = _check(checks, check_name).detail
            return done(case_id, f"{target} 요구 위반 확인 — {detail}")
        if broken is False:
            # 요구를 지켰는데도 불만이다. 형식 자체가 아니라 기대와 다른 것이다.
            return done("case13", f"{target} 요구는 지켰으나 사용자가 불만",
                        ["코드 검증은 통과 — 기대와 다른 답변 쪽으로 본다"])
        # 요구를 못 찾았거나 판정 불가. 불만은 형식을 가리키므로 case 는 유지하되
        # 코드 근거가 없으므로 신뢰도를 낮춘다.
        result = done(case_id, f"{target} 불만이나 코드 검증 근거 없음")
        result.confidence = "medium"
        result.notes.append("명시적 요구를 찾지 못함 — LLM 판정에만 의존")
        return result

    # --- 4. 일관성 ------------------------------------------------------------
    if obs.complaint_target == "inconsistency":
        return done(taxonomy.UNCLASSIFIED, "이전 답변과 다르다는 불만",
                    ["case16 — 교차 세션 비교가 필요해 턴 단위로는 판정 불가"])

    # --- 4b. 이전 턴 맥락 상실 -------------------------------------------------
    # 답변이 히스토리를 잘못 이어받았으면 문서 충족도를 따지기 전에 그것부터다.
    if obs.answer_used_history == "ignored":
        return done("case14", "답변이 이전 턴의 내용을 잊거나 잘못 연결함")

    # --- 5. 내용 불만: 질문의 성격으로 갈린다 -----------------------------------
    if obs.question_domain == "general_knowledge":
        return done("case21", "상식 질문에 대한 불만",
                    ["판정자의 사전지식에 의존 — 표본 검토 필요"])

    if obs.question_domain == "calculation":
        arithmetic = _check(checks, "arithmetic")
        if arithmetic is not None and arithmetic.violated:
            return done("case22", f"등식 오류 확인 — {arithmetic.detail}")
        result = done("case22", "계산 질문에 대한 불만")
        # 식을 명시하지 않은 계산은 코드로 검증할 수 없다. high 로 두면 안 된다.
        result.confidence = "medium"
        result.notes.append("답변에 검증 가능한 등식이 없음 — 자연어 계산은 판정 불가")
        return result

    if obs.question_domain in ("code", "tool_usage"):
        for name in ("python_syntax", "sql_shape"):
            broken = _check(checks, name)
            if broken is not None and broken.violated:
                return done("case23", f"코드 결함 확인 — {broken.detail}")
        return done("case23", "코드·도구 질문에 대한 불만",
                    ["문법은 통과 — 실행 검증 없이는 정확성을 확인할 수 없음"])

    if obs.question_domain == "domain":
        return _route_domain(obs, judgment, citation, grounding, done)

    # --- 6. 남은 것 -----------------------------------------------------------
    if obs.complaint_target == "other":
        return done(taxonomy.OUT_OF_TAXONOMY, "불만 성격이 taxonomy 어디에도 맞지 않음")
    return done(taxonomy.UNCLASSIFIED,
                f"판별 실패 (complaint={obs.complaint_target}, "
                f"domain={obs.question_domain})")


def _route_domain(obs, judgment, citation, grounding, done) -> Classification:
    """TYPE5 분기 — 검증된 case17/18 판별 로직을 그대로 쓴다."""
    if judgment is None:
        return done(taxonomy.UNCLASSIFIED, "도메인 질문인데 충족도 판정이 없음")

    kept = citation.n_kept if citation else 0
    # 인용이 하나도 검증되지 않은 sufficient/partial 주장은 사전지식에서 나온 것으로 본다.
    downgraded = judgment.verdict in ("sufficient", "partial") and kept == 0
    verdict = "insufficient" if downgraded else judgment.verdict

    if verdict in ("insufficient", "partial"):
        note = ["인용 검증 실패로 강등됨"] if downgraded else []
        note.append("검색 실패와 코퍼스 부재는 구분 불가 — 부서 편중으로만 추정")
        return done("case17", f"문서가 요구를 충족하지 못함 (verdict={verdict})", note)

    if grounding is None:
        return done("case17", "문서는 충분하나 활용 여부를 확인하지 못함")

    if grounding.answer_used_rag == "ignored":
        return done("case18", "문서에 답이 있는데 답변이 쓰지 않음")
    if grounding.answer_used_rag == "contradicted":
        return done("case15", "답변이 문서와 어긋나는 주장을 함",
                    ["문서와 대조 가능한 할루시네이션만 해당 — 문서 밖 허구는 판정 불가"])

    # 문서도 충분하고 답변도 썼는데 불만 → 물은 것과 다른 걸 답한 쪽으로 본다.
    return done("case13", "문서는 충분하고 활용했으나 사용자 기대와 다름")
