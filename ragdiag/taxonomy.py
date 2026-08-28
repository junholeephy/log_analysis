"""taxonomy 27개 케이스 메타데이터 (taxonomy_v2.txt 와 일치해야 한다).

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
    ("case2", "지원하지 않는 포맷 요구", "TYPE1", "적절하지 않은 질문/요청", "category_1", "medium", True),
    ("case3", "복합 질문을 함", "TYPE1", "적절하지 않은 질문/요청", "category_1", "medium", True),
    ("case4", "참조가 모호한 질문", "TYPE1", "적절하지 않은 질문/요청", "category_1", "medium", True),
    ("case5", "컨텍스트 길이 초과", "TYPE1", "적절하지 않은 질문/요청", "category_1", "medium", False),
    ("case6", "질문에 개인정보 포함", "TYPE1", "적절하지 않은 질문/요청", "category_1", "high", True),

    # category_2 · TYPE2 서비스 안정성
    ("case7", "응답 지연으로 이탈", "TYPE2", "서비스 안정성 문제", "category_2", "medium", False),
    ("case8", "출력 잘림", "TYPE2", "서비스 안정성 문제", "category_2", "high", True),

    # category_2 · TYPE3 의도 파악 실패
    ("case9", "요구 언어 불이행", "TYPE3", "사용자의 의도를 파악하지 못함", "category_2", "high", True),
    ("case10", "요구 길이 불이행", "TYPE3", "사용자의 의도를 파악하지 못함", "category_2", "high", True),
    ("case11", "요구 포맷 불이행", "TYPE3", "사용자의 의도를 파악하지 못함", "category_2", "high", True),
    ("case12", "의도와 다른 답변", "TYPE3", "사용자의 의도를 파악하지 못함", "category_2", "medium", True),
    ("case13", "이전 턴 맥락 상실", "TYPE3", "사용자의 의도를 파악하지 못함", "category_2", "medium", True),
    ("case14", "복합 질문 일부만 답변", "TYPE3", "사용자의 의도를 파악하지 못함", "category_2", "medium", True),
    ("case15", "말투·어조 불이행", "TYPE3", "사용자의 의도를 파악하지 못함", "category_2", "medium", True),
    ("case16", "실행할 수 없는 수준", "TYPE3", "사용자의 의도를 파악하지 못함", "category_2", "medium", True),

    # category_2 · TYPE4 할루시네이션
    ("case17", "문서와 어긋나는 주장", "TYPE4", "할루시네이션 답변", "category_2", "medium", True),
    ("case18", "응답 일관성 문제", "TYPE4", "할루시네이션 답변", "category_2", "medium", False),

    # category_2 · TYPE5 Retrieve Context
    ("case19", "Retrieve 실패", "TYPE5", "도메인 관련 Retrieve Context 문제", "category_2", "medium", True),
    ("case20", "Retrieve 성공, 생성 실패", "TYPE5", "도메인 관련 Retrieve Context 문제", "category_2", "medium", True),
    ("case21", "구 문서 retrieve", "TYPE5", "도메인 관련 Retrieve Context 문제", "category_2", "medium", False),
    ("case22", "출처/인용 표기 오류", "TYPE5", "도메인 관련 Retrieve Context 문제", "category_2", "high", True),

    # category_2 · TYPE6 일반 질문
    ("case23", "상식 질문 오답", "TYPE6", "일반 질문", "category_2", "low", True),
    ("case24", "계산 오답", "TYPE6", "일반 질문", "category_2", "high", True),
    ("case25", "코드/도구 사용법 오답", "TYPE6", "일반 질문", "category_2", "medium", True),

    # category_2 · TYPE7 보안/정책
    ("case26", "보안 정책상 답변 불가", "TYPE7", "보안/정책 제한", "category_2", "medium", True),
    ("case27", "간접 프롬프트 인젝션", "TYPE7", "보안/정책 제한", "category_2", "medium", True),
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


# ---------------------------------------------------------------------------
# 구 번호 대응
#
# v2 에서 타입 순서대로 번호를 다시 매겼다. 이미 붙인 라벨이 있으면 이 표로 옮긴다.
# case1~14 는 그대로다. TYPE5 의 Retrieve 쌍이 17/18 -> 21/22 로 바뀐 것이
# 가장 자주 걸린다 - 이 도구의 핵심이라 문서와 대화에서 제일 많이 언급된다.
# ---------------------------------------------------------------------------

# 이전 번호 대응. v1(원본 taxonomy.txt) -> v2(재번호) -> v3(판정 불가 제거) 를 거쳤다.
# 관측할 수 없는 케이스는 v3 에서 목록에서 뺐다 - 배정할 수 없는 라벨은 죽은 라벨이다.
# 그 사실 자체는 taxonomy_v2.txt 의 "제외한 실패" 절에 남겼다.

V1_TO_V3 = {
    **{f"case{n}": f"case{n}" for n in range(1, 7)},   # case1~6 그대로
    "case9": "case8",     # 출력 잘림
    "case10": "case9",    # 요구 언어
    "case11": "case10",   # 요구 길이
    "case12": "case11",   # 요구 포맷
    "case13": "case12",   # 의도와 다름
    "case14": "case13",   # 맥락 상실
    "case15": "case17",   # 사실 오류 -> 문서와 어긋나는 주장 (범위 축소)
    "case16": "case18",   # 응답 일관성
    "case17": "case19",   # Retrieve 실패
    "case18": "case20",   # Retrieve 성공, 생성 실패
    "case19": "case21",   # 구 문서
    "case20": "case22",   # 출처/인용
    "case21": "case23",   # 상식
    "case22": "case24",   # 계산
    "case23": "case25",   # 코드/도구
    "case24": "case26",   # 보안 정책
    "case26": "case27",   # 인젝션
    "case8": "case7",     # 응답 지연
}

# v1 에 있었으나 v3 에서 제외된 것. 관측할 수 없어서다.
V1_DROPPED = {
    "case7": "서비스 끊김 - 답변이 없으면 로그에 턴 자체가 안 남음",
    "case25": "권한 없는 접근 - 권한 조회 결과 필요, case26 과 로그상 구분 불가",
    "case27": "외부 API 호출 실패 - 툴 호출 로그 필요",
    "case28": "로그인/인증 실패 - 대화가 시작되지 않음",
    "case29": "세션 만료 - 위와 같음",
    "case30": "권한 시스템 오류 - 위와 같음",
}

# v3 에서 새로 생긴 것. 이전 번호에 대응이 없다.
V3_NEW = {"case14", "case15", "case16"}


def migrate(v1_case_id: str) -> Optional[str]:
    """원본 taxonomy.txt 번호를 현재 번호로. 제외됐거나 대응이 없으면 None."""
    return V1_TO_V3.get(v1_case_id)