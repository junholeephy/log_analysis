"""라우팅 — 관측과 검증 결과를 taxonomy case 로 바꾼다. LLM 없음.

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
from ragdiag.verify import CitationCheck, QuoteCheck


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

    한 턴이 case4(모호한 질문)이면서 동시에 case20(검색 실패)일 수 있다. 억지로
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
        extra.append("case24")         # 인용 표기 오류
    # 실제 질문은 도메인과 계산이 겹치는 경우가 흔하다("5영업일 뒤가 언제냐").
    # question_domain 은 하나만 고를 수 있으므로, 등식 오류는 부가로 따로 잡는다.
    arithmetic = _check(checks, "arithmetic")
    if arithmetic is not None and arithmetic.violated:
        extra.append("case26")         # 계산 오류
    if obs.answer_used_history == "ignored":
        extra.append("case14")         # 이전 턴 맥락 상실
    # 복합 질문인데 일부만 답한 것은 다른 원인과 함께 성립한다.
    if obs.question_multi_intent and not obs.answer_covers_all_intents:
        extra.append("case15")         # 복합 질문 일부만 답변
    return extra



def service_unavailable(check) -> Classification:
    """case9 — 서비스가 자원을 확보하지 못해 안내 문구를 낸 턴.

    관측(Step 1)을 거치지 않고 코드만으로 만든다. 확정 문구와의 대조라
    다른 라우팅과 달리 LLM 판정이 하나도 섞이지 않는다.

    주의를 함께 실어 둔다. 이 라벨이 쌓이면 "챗봇 품질이 나쁘다"가 아니라
    "그 시간대에 자원이 모자랐다"이고, 같은 표에서 다른 case 와 나란히 읽으면
    품질 지표가 인프라 장애에 흔들린다.
    """
    meta = taxonomy.get("case9")
    return Classification(
        primary_case="case9",
        confidence=meta.confidence if meta else "high",
        reason=f"서비스 자원 부족 안내 문구 — {check.detail}",
        secondary_cases=[],
        notes=["모델이 답을 만든 적이 없다. 검색·생성 품질 집계에서 분리할 것.",
               "LLM 판정을 돌리지 않았다 — 관측·충족도·근거 활용이 모두 비어 있다."],
    )


def route(
    obs: Observation,
    checks: dict[str, Check],
    judgment: Optional[SufficiencyJudgment] = None,
    citation: Optional[CitationCheck] = None,
    grounding: Optional[GroundingCheck] = None,
    complaint: Optional[QuoteCheck] = None,
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
        # 권한이 없어 거절한 경우와 구분할 수 없다. 권한 조회 결과가 없기 때문이다.
        # 그 케이스는 taxonomy 에서 제외됐지만 실제로는 여기 섞여 들어온다.
        return done("case28", "답변이 정책·권한을 이유로 거절함",
                    ["권한 부족으로 인한 거절이 섞여 있을 수 있음 — 권한 조회 결과 필요"])

    # --- 1a. 문서가 모델을 조종했는지 ------------------------------------------
    # 답변 내용을 따지기 전에 볼 문제다. 문서에 심긴 지시를 수행했다면
    # 그 답변의 나머지 판정은 의미가 없다.
    injection = _check(checks, "injection")
    if injection is not None and injection.violated:
        return done("case29", f"검색 문서의 지시를 답변이 수행함 — {injection.detail}")

    # --- 1c. 애초에 불만이 아니었나 --------------------------------------------
    # 필터는 재현율 쪽으로 넓게 잡는다. 그래서 그냥 다음을 묻는 턴이 섞여 들어온다.
    # 이걸 실패로 세면 case20 이 부풀고, 그 숫자가 코퍼스 보강 목록이 되어
    # 문서팀이 쓸 필요 없는 문서를 쓴다.
    #
    # 거절·인젝션보다 뒤에 두는 이유: 그 둘은 사용자가 지적했든 아니든 확인된
    # 사실이다. 반대로 질문 모호성(1b)보다는 앞이다 - 사용자가 만족했다면 그
    # 모호함은 실제로 문제가 되지 않았다는 뜻이고, 신호는 secondary(case4·case3)로
    # 남는다.
    if obs.complaint_target == "none":
        # 인용을 못 대면 통과시키지 않는다. "문제 없음"은 가장 쉬운 답이라
        # 열어두면 애매한 턴이 전부 그리로 샌다.
        if complaint is not None and not complaint.verified:
            result = done(taxonomy.UNCLASSIFIED,
                          "불만이 아니라고 했으나 후속 발화에서 근거를 인용하지 못함",
                          ["인용 검증 실패 — 판정자가 관용 쪽으로 기울었을 수 있다"])
            result.confidence = "low"
            return result
        # 불만이 없다고 결함이 없는 건 아니다. 코드가 잡은 위반이 있으면
        # 사용자가 지적하지 않았을 뿐이다.
        violated = sorted(name for name, c in checks.items() if c.violated)
        if violated:
            return done(taxonomy.UNCLASSIFIED,
                        f"불만은 없으나 코드 검증이 위반을 잡음 ({', '.join(violated)})",
                        ["사용자가 지적하지 않은 결함 — 표본 검토 대상"])
        return done("case0", "후속 발화가 앞 답변을 문제 삼지 않음",
                    ["실패율 분모에서 뺄 것.",
                     "쌓이면 챗봇이 아니라 필터를 좁힐 신호다."])

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
            return done("case8", f"답변이 중간에 끊김 — {truncated.detail}")
        return done(taxonomy.UNCLASSIFIED,
                    "답이 없다는 불만인데 답변은 온전함",
                    ["서비스 끊김일 수 있으나 로그로는 판정 불가 — 별도 텔레메트리 필요"])

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

    # --- 3b. 말투·어조 ---------------------------------------------------------
    # 코드로 검증할 수단이 없다. 어조는 규칙으로 재기 어려워 관측에만 의존한다.
    if obs.complaint_target == "tone":
        return done("case16", "말투·어조·용어에 대한 불만")

    # --- 4. 일관성 ------------------------------------------------------------
    if obs.complaint_target == "inconsistency":
        return done(taxonomy.UNCLASSIFIED, "이전 답변과 다르다는 불만",
                    ["case19 — 교차 세션 비교가 필요해 턴 단위로는 판정 불가"])

    # --- 5. 내용 불만: 질문의 성격으로 갈린다 -----------------------------------
    if obs.question_domain == "general_knowledge":
        return done("case25", "상식 질문에 대한 불만",
                    ["판정자의 사전지식에 의존 — 표본 검토 필요"])

    if obs.question_domain == "calculation":
        arithmetic = _check(checks, "arithmetic")
        if arithmetic is not None and arithmetic.violated:
            return done("case26", f"등식 오류 확인 — {arithmetic.detail}")
        result = done("case26", "계산 질문에 대한 불만")
        # 식을 명시하지 않은 계산은 코드로 검증할 수 없다. high 로 두면 안 된다.
        result.confidence = "medium"
        result.notes.append("답변에 검증 가능한 등식이 없음 — 자연어 계산은 판정 불가")
        return result

    if obs.question_domain in ("code", "tool_usage"):
        for name in ("python_syntax", "sql_shape"):
            broken = _check(checks, name)
            if broken is not None and broken.violated:
                return done("case27", f"코드 결함 확인 — {broken.detail}")
        return done("case27", "코드·도구 질문에 대한 불만",
                    ["문법은 통과 — 실행 검증 없이는 정확성을 확인할 수 없음"])

    if obs.question_domain == "domain":
        return _route_domain(obs, judgment, citation, grounding, done)

    # --- 6. 남은 것 -----------------------------------------------------------
    # case14 는 여기서만 주 라벨이 된다. 앞에 두면 case20/case22 을 가로챈다 —
    # 검색이 실패해 답변이 부실하면 모델은 그걸 "히스토리를 못 이어받았다"로도
    # 읽기 때문이다. 인용으로 검증된 문서 증거가 LLM 의 인상보다 강하다.
    if obs.answer_used_history == "ignored":
        return done("case14", "답변이 이전 턴의 내용을 잊거나 잘못 연결함",
                    ["더 구체적인 원인을 찾지 못해 맥락 상실로 판정"])

    if obs.question_multi_intent and not obs.answer_covers_all_intents:
        return done("case15", "복합 질문의 일부만 답변함")

    if obs.complaint_target == "other":
        return done(taxonomy.OUT_OF_TAXONOMY, "불만 성격이 taxonomy 어디에도 맞지 않음")
    return done(taxonomy.UNCLASSIFIED,
                f"판별 실패 (complaint={obs.complaint_target}, "
                f"domain={obs.question_domain})")


def _route_domain(obs, judgment, citation, grounding, done) -> Classification:
    """TYPE5 분기 — 검증된 case20/case22 판별 로직을 그대로 쓴다.

    n_chunks 가 0 이면 검색 결과가 아예 없었다는 뜻이다. 이건 판정이 아니라
    로그에 적힌 사실이라 "가져온 문서가 요구를 충족하지 못했다"와 구분해 적는다.
    고칠 곳이 다르다 - 0건은 검색을 탈지 말지 정하는 쪽이나 검색기 자체이고,
    가져왔는데 빗나간 것은 임베딩·청킹이나 문서 쪽이다.
    """
    if judgment is None:
        return done(taxonomy.UNCLASSIFIED, "도메인 질문인데 충족도 판정이 없음")

    kept = citation.n_kept if citation else 0
    n_chunks = citation.n_chunks if citation else None
    # 인용이 하나도 검증되지 않은 sufficient/partial 주장은 사전지식에서 나온 것으로 본다.
    downgraded = judgment.verdict in ("sufficient", "partial") and kept == 0
    verdict = "insufficient" if downgraded else judgment.verdict

    if verdict in ("insufficient", "partial"):
        note = ["인용 검증 실패로 강등됨"] if downgraded else []
        if n_chunks == 0:
            note.append("서비스가 '검색 없이 답할 수 있다'고 판단했을 수 있다. "
                        "그 판단이 틀린 것이라면 고칠 곳은 검색 트리거다.")
            return done("case21", "검색 결과가 0건 — 대조할 문서가 아예 없음", note)
        note.append("검색 실패와 코퍼스 부재는 구분 불가 — 부서 편중으로만 추정")
        return done("case20", f"가져온 문서가 요구를 충족하지 못함 (verdict={verdict})", note)

    if grounding is None:
        return done("case20", "문서는 충분하나 활용 여부를 확인하지 못함")

    if grounding.answer_used_rag == "ignored":
        return done("case22", "문서에 답이 있는데 답변이 쓰지 않음")
    if grounding.answer_used_rag == "contradicted":
        return done("case18", "답변이 문서와 어긋나는 주장을 함",
                    ["문서와 대조 가능한 할루시네이션만 해당 — 문서 밖 허구는 판정 불가"])

    # 문서도 충분하고 답변도 썼는데 불만이다. 두 갈래로 갈린다.
    if not obs.answer_actionable:
        # 물은 것은 맞게 답했으나 행동으로 이어지지 않는 경우.
        return done("case17", "문서는 충분하나 답변이 두루뭉술해 다음 행동을 알 수 없음")
    return done("case13", "문서는 충분하고 활용했으나 사용자 기대와 다름")
