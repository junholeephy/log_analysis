"""taxonomy 30개 케이스 메타데이터.

case → 이름 · type · category 매핑. **category 는 case 에서 자동으로 결정된다** —
각 case 는 정확히 하나의 type 에, 각 type 은 정확히 하나의 category 에 속하기 때문이다.
그래서 분류를 category → case 순으로 두 번 하지 않는다. case 하나만 정하면 나머지는 계산이다.

confidence 는 무엇이 판정을 결정하는지에 따른다:
  high    코드로 검증 가능 (길이·언어·문법·인용 대조)
  medium  LLM 판정 + 인용 강제
  low     판정자의 사전지식에 의존 — 다른 라벨과 같은 무게로 집계하면 안 된다
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TaxonomyCase:
    case_id: str
    name: str
    type_id: str
    type_name: str
    category: str
    confidence: str = "medium"
    # 이 로그로는 판정할 수 없는 케이스. 라우팅이 절대 여기로 보내지 않는다.
    diagnosable: bool = True


_ROWS = [
    # category_1 · TYPE1 적절하지 않은 질문/요청
    ("case1", "이해하기 어려운 질문", "TYPE1", "적절하지 않은 질문/요청", "category_1", "medium", True),
    ("case2", "지원하지 않는 포맷 요구", "TYPE1", "적절하지 않은 질문/요청", "category_1", "high", True),
    ("case3", "복합 질문 일부만 답변", "TYPE1", "적절하지 않은 질문/요청", "category_1", "medium", True),
    ("case4", "참조가 모호한 질문", "TYPE1", "적절하지 않은 질문/요청", "category_1", "medium", True),
    ("case5", "컨텍스트 길이 초과", "TYPE1", "적절하지 않은 질문/요청", "category_1", "medium", False),
    ("case6", "질문에 개인정보 포함", "TYPE1", "적절하지 않은 질문/요청", "category_1", "high", True),

    # category_2 · TYPE2 서비스 안정성
    ("case7", "서비스 끊김", "TYPE2", "서비스 안정성 문제", "category_2", "medium", False),
    ("case8", "응답 지연으로 이탈", "TYPE2", "서비스 안정성 문제", "category_2", "medium", False),
    ("case9", "출력 잘림", "TYPE2", "서비스 안정성 문제", "category_2", "high", True),

    # category_2 · TYPE3 의도 파악 실패
    ("case10", "요구 언어 불이행", "TYPE3", "사용자의 의도를 파악하지 못함", "category_2", "high", True),
    ("case11", "요구 길이 불이행", "TYPE3", "사용자의 의도를 파악하지 못함", "category_2", "high", True),
    ("case12", "요구 포맷 불이행", "TYPE3", "사용자의 의도를 파악하지 못함", "category_2", "high", True),
    ("case13", "의도와 다른 답변", "TYPE3", "사용자의 의도를 파악하지 못함", "category_2", "medium", True),
    ("case14", "이전 턴 맥락 상실", "TYPE3", "사용자의 의도를 파악하지 못함", "category_2", "medium", True),

    # category_2 · TYPE4 할루시네이션
    ("case15", "사실과 다른 정보", "TYPE4", "할루시네이션 답변", "category_2", "medium", True),
    ("case16", "응답 일관성 문제", "TYPE4", "할루시네이션 답변", "category_2", "medium", False),

    # category_2 · TYPE5 Retrieve Context
    ("case17", "Retrieve 실패", "TYPE5", "도메인 관련 Retrieve Context 문제", "category_2", "medium", True),
    ("case18", "Retrieve 성공, 생성 실패", "TYPE5", "도메인 관련 Retrieve Context 문제", "category_2", "medium", True),
    ("case19", "구 문서 retrieve", "TYPE5", "도메인 관련 Retrieve Context 문제", "category_2", "medium", False),
    ("case20", "출처/인용 표기 오류", "TYPE5", "도메인 관련 Retrieve Context 문제", "category_2", "high", True),

    # category_2 · TYPE6 일반성 질문
    ("case21", "상식 질문 오답", "TYPE6", "일반성 질문", "category_2", "low", True),
    ("case22", "계산 오답", "TYPE6", "일반성 질문", "category_2", "high", True),
    ("case23", "코드/도구 사용법 오답", "TYPE6", "일반성 질문", "category_2", "medium", True),

    # category_2 · TYPE7 보안/정책
    ("case24", "보안 정책상 답변 불가", "TYPE7", "보안/정책 제한", "category_2", "medium", True),
    ("case25", "권한 없는 접근 요청", "TYPE7", "보안/정책 제한", "category_2", "medium", False),
    ("case26", "간접 프롬프트 인젝션", "TYPE7", "보안/정책 제한", "category_2", "medium", True),
    ("case27", "외부 API 호출 실패", "TYPE7", "보안/정책 제한", "category_2", "medium", False),

    # category_3 · TYPE8 인프라/권한
    ("case28", "로그인/인증 실패", "TYPE8", "인프라·권한 시스템 오류", "category_3", "medium", False),
    ("case29", "세션 만료로 중단", "TYPE8", "인프라·권한 시스템 오류", "category_3", "medium", False),
    ("case30", "권한 시스템 오류", "TYPE8", "인프라·권한 시스템 오류", "category_3", "medium", False),
]

CASES: dict[str, TaxonomyCase] = {
    row[0]: TaxonomyCase(*row) for row in _ROWS
}

# 라우팅이 도달할 수 없는 케이스. 목록에서 지우지 않고 남겨두는 이유는
# "우리 분류에는 그런 게 없다"가 되지 않게 하기 위해서다.
UNDIAGNOSABLE = {c.case_id for c in CASES.values() if not c.diagnosable}

# 어느 case 에도 넣을 수 없을 때. "문제 없음"이 아니라 수동 검토 대상이다.
UNCLASSIFIED = "unclassified"
# taxonomy 에 해당 항목이 아예 없을 때. 쌓이면 케이스를 추가하라는 신호다.
OUT_OF_TAXONOMY = "out_of_taxonomy"


def get(case_id: str) -> Optional[TaxonomyCase]:
    return CASES.get(case_id)


def describe(case_id: str) -> dict:
    """출력에 실을 case 메타데이터."""
    case = CASES.get(case_id)
    if case is None:
        return {
            "case_id": case_id,
            "case_name": {
                UNCLASSIFIED: "분류 실패 (수동 검토 대상)",
                OUT_OF_TAXONOMY: "taxonomy 에 없는 유형",
            }.get(case_id, case_id),
            "type_id": "",
            "type_name": "",
            "category": "",
        }
    return {
        "case_id": case.case_id,
        "case_name": case.name,
        "type_id": case.type_id,
        "type_name": case.type_name,
        "category": case.category,
    }
