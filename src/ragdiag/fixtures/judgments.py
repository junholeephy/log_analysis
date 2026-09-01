# -*- coding: utf-8 -*-
"""Step 2·3 골든셋 — 충족도 판정과 근거 활용.

지금까지 직접 채점한 것은 Step 1(관측)뿐이었다. 충족도와 근거 활용은 회귀셋 23건으로
**간접 확인**만 됐다. 층이 다른 검증은 서로를 대체하지 못한다 — 실제로 관측 골든셋이
98%일 때 회귀셋은 15/23 이었고, 반대로 회귀셋이 통과해도 판정 자체의 정확도는 모른다.

충족도(Step 2)에서 재는 것:
  verdict          sufficient / partial / insufficient
  cited_chunks     어느 청크에서 인용을 뽑았는가 (근거를 제대로 짚었는지)
  citation_holds   그 인용이 원문 대조를 통과하는가 (지어내지 않았는지)

근거 활용(Step 3)에서 재는 것:
  used / ignored / contradicted

Step 2 는 챗봇 답변을 보지 않고, Step 3 은 히스토리를 보지 않는다. 실제 파이프라인과
같은 입력만 준다 — 여기서 더 주면 측정이 실전보다 후해진다.
"""

RULES = [
    "국내 출장 식비는 1일 3만원을 상한으로 한다.",
    "국내 출장 숙박비는 1박 8만원을 상한으로 한다.",
    "출장비는 출장 종료 후 5영업일 이내에 정산한다.",
]
LEAVE = [
    "연차유급휴가 신청은 사전에 그룹웨어를 통해 제출한다.",
    "연차 사용 시 팀장의 승인을 받아야 한다.",
    "연차는 반차 단위로도 사용할 수 있다.",
]

# ---------------------------------------------------------------------------
# Step 2 — 충족도 판정
# ---------------------------------------------------------------------------

SUFFICIENCY = [
    dict(
        id="suf01", note="답이 청크에 명확히 있음",
        question="국내 출장 식비의 1일 상한 금액은 얼마인가?",
        unmet_need="국내 출장 식비의 1일 상한 금액",
        chunks=RULES,
        expect_verdict="sufficient", expect_cited={0},
    ),
    dict(
        id="suf02", note="near-miss — 주제는 같고 물어본 것은 없음",
        question="해외 출장 미주 지역의 1일 숙박비 상한은 얼마인가?",
        unmet_need="미주 지역 1일 숙박비 상한 금액",
        chunks=["해외 출장비 정산은 출장 종료 후 5영업일 이내에 제출한다.",
                "숙박비는 실비 정산을 원칙으로 한다.",
                "출장 신청은 출발 7일 전까지 팀장 승인을 받는다."],
        expect_verdict="insufficient",
    ),
    dict(
        id="suf03", note="요구 둘 중 하나만 있음",
        question="국내 출장 식비와 교통비 상한이 각각 얼마인가?",
        unmet_need="국내 출장 식비 상한과 교통비 상한",
        chunks=RULES,
        expect_verdict="partial", expect_cited={0},
    ),
    dict(
        id="suf04", note="distractor — 그럴듯한 인접 문서",
        question="중국 출장 시 비자 수수료를 회사가 지원하는가?",
        unmet_need="비자 수수료의 회사 지원 여부",
        chunks=["해외 출장 시 항공료와 숙박비는 회사가 부담한다.",
                "여권 발급 비용은 개인 부담을 원칙으로 한다.",
                "출장 중 현지 교통비는 실비 정산 대상이다."],
        expect_verdict="insufficient",
    ),
    dict(
        id="suf05", note="leakage probe — 상식으로 아는 답, 문서엔 없음",
        question="근속 1년 미만 근로자에게도 연차가 발생하는가? 사규 근거는?",
        unmet_need="1년 미만 근속자의 연차 발생 요건에 대한 업무 규정",
        chunks=LEAVE,
        expect_verdict="insufficient",
    ),
    dict(
        id="suf06", note="leakage probe — 부가세율",
        question="우리 회사 세금계산서 발행 시 적용 세율은?",
        unmet_need="업무 문서 기준의 적용 세율",
        chunks=["세금계산서는 거래일이 속하는 달의 다음 달 10일까지 발행한다.",
                "세금계산서 발행은 재무팀 승인 후 진행한다."],
        expect_verdict="insufficient",
    ),
    dict(
        id="suf07", note="관련 서술은 있으나 구체성이 못 미침",
        question="미사용 연차를 이월할 수 있는 예외 조건은?",
        unmet_need="연차 이월이 가능한 구체적 예외 조건",
        chunks=["미사용 연차는 발생일로부터 1년이 경과하면 소멸함을 원칙으로 한다.",
                "다만 회사의 귀책사유로 사용하지 못한 경우 예외를 둘 수 있다."],
        expect_verdict="partial", expect_cited={1},
    ),
    dict(
        id="suf08", note="검색 결과가 아예 없음",
        question="운영 환경 헬스장 이용 시간은?",
        unmet_need="운영 환경 헬스장 운영 시간",
        chunks=[],
        expect_verdict="insufficient",
    ),
    dict(
        id="suf09", note="여러 청크에 나뉘어 있지만 온전히 있음",
        question="국내 출장 식비와 숙박비 상한이 각각 얼마인가?",
        unmet_need="국내 출장 식비 상한과 숙박비 상한",
        chunks=RULES,
        expect_verdict="sufficient", expect_cited={0, 1},
    ),
    dict(
        id="suf10", note="숫자가 비슷한 다른 항목이 있어 혼동하기 쉬움",
        question="국내 출장 교통비 상한은 얼마인가?",
        unmet_need="국내 출장 교통비의 상한 금액",
        chunks=RULES,     # 식비 3만 · 숙박비 8만은 있지만 교통비는 없다
        expect_verdict="insufficient",
    ),
]

# ---------------------------------------------------------------------------
# Step 3 — 근거 활용
# ---------------------------------------------------------------------------

GROUNDING = [
    dict(
        id="gnd01", note="문서 내용을 그대로 활용",
        answer="국내 출장 식비는 1일 3만원을 상한으로 합니다.",
        chunks=RULES, expect="used",
    ),
    dict(
        id="gnd02", note="문서에 있는데 일반론으로 때움",
        answer="출장 식비는 회사 규정에 따라 지급되며, 자세한 금액은 부서별로 "
               "다를 수 있습니다.",
        chunks=RULES, expect="ignored",
    ),
    dict(
        id="gnd03", note="회피성 안내 — 거절이 아니라 활용 실패다",
        answer="건강검진 관련 사항은 인사팀에 직접 문의해 주시기 바랍니다.",
        chunks=["임직원 건강검진은 매년 1회 실시하며, 만 35세 이상은 종합검진 대상으로 한다.",
                "만 35세 미만은 일반검진을 실시한다."],
        expect="ignored",
    ),
    dict(
        id="gnd04", note="문서와 다른 숫자를 말함",
        answer="국내 출장 식비는 1일 5만원입니다.",
        chunks=RULES, expect="contradicted",
    ),
    dict(
        id="gnd05", note="문서와 반대되는 결론",
        answer="미사용 연차는 다음 해로 자동 이월됩니다.",
        chunks=["미사용 연차는 발생일로부터 1년이 경과하면 소멸함을 원칙으로 한다."],
        expect="contradicted",
    ),
    dict(
        id="gnd06", note="표현은 다르지만 문서 내용을 반영",
        answer="정산은 출장이 끝난 뒤 5영업일 안에 마치셔야 합니다.",
        chunks=RULES, expect="used",
    ),
    dict(
        id="gnd07", note="문서에 없는 내용이지만 어긋나지도 않음",
        answer="출장 신청은 부서장 승인 후 진행하시면 됩니다.",
        chunks=RULES,
        # 문서를 쓰지 않았다. 다만 어긋나는 주장은 아니므로 contradicted 가 아니다.
        expect="ignored",
    ),
    dict(
        id="gnd08", note="여러 청크를 종합해 답함",
        answer="식비는 1일 3만원, 숙박비는 1박 8만원이며 정산은 5영업일 이내입니다.",
        chunks=RULES, expect="used",
    ),
]
