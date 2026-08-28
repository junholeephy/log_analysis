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
    # 한 줄 설명. 이름만으로는 옆 케이스와 구분되지 않는 것들이 있어
    # (case3/case15, case4/case14, case13/case17, case20/case21) 가르는 기준을 함께 적는다.
    desc: str = ""


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
    ("case9", "서비스 자원 부족 응답", "TYPE2", "서비스 안정성 문제", "category_2", "high", True),

    # category_2 · TYPE3 의도 파악 실패
    ("case10", "요구 언어 불이행", "TYPE3", "사용자의 의도를 파악하지 못함", "category_2", "high", True),
    ("case11", "요구 길이 불이행", "TYPE3", "사용자의 의도를 파악하지 못함", "category_2", "high", True),
    ("case12", "요구 포맷 불이행", "TYPE3", "사용자의 의도를 파악하지 못함", "category_2", "high", True),
    ("case13", "의도와 다른 답변", "TYPE3", "사용자의 의도를 파악하지 못함", "category_2", "medium", True),
    ("case14", "이전 턴 맥락 상실", "TYPE3", "사용자의 의도를 파악하지 못함", "category_2", "medium", True),
    ("case15", "복합 질문 일부만 답변", "TYPE3", "사용자의 의도를 파악하지 못함", "category_2", "medium", True),
    ("case16", "말투·어조 불이행", "TYPE3", "사용자의 의도를 파악하지 못함", "category_2", "medium", True),
    ("case17", "실행할 수 없는 수준", "TYPE3", "사용자의 의도를 파악하지 못함", "category_2", "medium", True),

    # category_2 · TYPE4 할루시네이션
    ("case18", "문서와 어긋나는 주장", "TYPE4", "할루시네이션 답변", "category_2", "medium", True),
    ("case19", "응답 일관성 문제", "TYPE4", "할루시네이션 답변", "category_2", "medium", False),

    # category_2 · TYPE5 Retrieve Context
    ("case20", "Retrieve 실패", "TYPE5", "도메인 관련 Retrieve Context 문제", "category_2", "medium", True),
    ("case21", "Retrieve 성공, 생성 실패", "TYPE5", "도메인 관련 Retrieve Context 문제", "category_2", "medium", True),
    ("case22", "구 문서 retrieve", "TYPE5", "도메인 관련 Retrieve Context 문제", "category_2", "medium", False),
    ("case23", "출처/인용 표기 오류", "TYPE5", "도메인 관련 Retrieve Context 문제", "category_2", "high", True),

    # category_2 · TYPE6 일반 질문
    ("case24", "상식 질문 오답", "TYPE6", "일반 질문", "category_2", "low", True),
    ("case25", "계산 오답", "TYPE6", "일반 질문", "category_2", "high", True),
    ("case26", "코드/도구 사용법 오답", "TYPE6", "일반 질문", "category_2", "medium", True),

    # category_2 · TYPE7 보안/정책
    ("case27", "보안 정책상 답변 불가", "TYPE7", "보안/정책 제한", "category_2", "medium", True),
    ("case28", "간접 프롬프트 인젝션", "TYPE7", "보안/정책 제한", "category_2", "medium", True),
]


# 케이스별 한 줄 설명. taxonomy_v2.txt 의 정의를 옮겨 왔다.
#
# 원문을 그대로 두지 않고 여기에 둔 이유: taxonomy_v2.txt 는 저장소에 올리지 않는
# 파일이라 대상 장비에 없을 수 있다. 대시보드가 그 파일을 읽게 만들면 없을 때
# 라벨이 case20 같은 번호로만 남고, 번호만 보고 무엇인지 아는 사람은 없다.
#
# 이름만으로 구분되지 않는 쌍이 있어 "무엇이 아닌가"를 함께 적는다. 필터에서 case 를
# 고를 때 실제로 막히는 지점이 그쪽이다.
_DESC = {
    "case1":  "질문이 모호해 무엇을 묻는지 특정되지 않는다. 되묻기를 유도해야 할 자리다.",
    "case2":  "그림·사외 링크처럼 이 서비스가 낼 수 없는 형태를 요구했다. "
              "낼 수 있는 형식인데 안 지킨 것은 case12.",
    "case3":  "한 번에 여러 의도를 섞어 물었다 — 사용자 쪽 문제다. "
              "모델이 그중 일부만 답한 것은 case15.",
    "case4":  "'그거 다시 알려줘'처럼 앞 턴을 전제로 지시 대상이 불명확하다. "
              "질문이 불완전한 것이고, 답변이 맥락을 잃은 것은 case14.",
    "case5":  "입력이 모델 컨텍스트 상한을 넘겨 품질이 떨어졌다. "
              "상한값 필드가 없어 이 로그로는 판정하지 않는다.",
    "case6":  "질문에 주민번호·연락처 등 개인정보가 실려 들어왔다.",
    "case7":  "답변은 왔으나 지연이 심해 사용자가 이탈했다. "
              "서버 응답 시간 필드가 없어 판정하지 않는다 — 턴 시각 차이로는 대체되지 않는다.",
    "case8":  "답변이 문장 중간에서 끊겼다. 토큰 상한 등 출력 잘림.",
    "case9":  "모델 자원을 확보하지 못해 서비스가 정해진 안내 문구를 대신 내보냈다. "
              "모델이 답을 만든 적이 없으므로 검색·생성 품질과 무관하다 — "
              "고칠 곳은 인프라이지 프롬프트가 아니다.",
    "case10":  "특정 언어로 답해 달라고 했는데 다른 언어로 답했다.",
    "case11": "짧게/길게 등 분량 요구를 지키지 않았다.",
    "case12": "표·불릿 등 요구한 형식으로 내지 않았다. "
              "애초에 낼 수 없는 형식을 요구한 것은 case2.",
    "case13": "물은 것과 다른 것을 답했다. 의도 파악 실패. "
              "맞게 답했으나 실행으로 이어지지 않는 것은 case17.",
    "case14": "답변이 앞 턴의 맥락을 잊거나 잘못 이었다. "
              "검색이 실패해도 이렇게 보이므로, 문서 증거가 있는 case20·case21 을 먼저 가른다.",
    "case15": "여러 의도 중 일부만 답했다 — 고칠 곳은 모델이다. "
              "사용자가 복합 질문을 한 것 자체는 case3.",
    "case16": "말투·어조·용어 사용에 대한 요구를 지키지 않았다.",
    "case17": "내용은 맞으나 사용자가 다음에 무엇을 할지 알 수 없다. "
              "아예 다른 것을 답한 것은 case13.",
    "case18": "검색된 문서와 어긋나는 주장을 했다. "
              "문서 밖의 허구는 대조할 것이 없어 판정 대상이 아니다.",
    "case19": "같은 질문에 매번 다르게 답하거나 이전 답변과 상충한다. "
              "턴 하나가 아니라 로그 전체를 훑어야 해 이 파이프라인에서는 판정하지 않는다.",
    "case20": "가져온 청크에 답이 없다. 검색기가 못 찾은 것인지 문서가 애초에 없는 것인지는 "
              "코퍼스 전체를 봐야 갈린다 — 특정 부서에 몰리면 문서 부재 쪽이다.",
    "case21": "청크에 답이 있는데 답변이 쓰지 않았다. "
              "'인사팀에 문의하세요' 같은 회피성 안내도 거절이 아니라 여기다.",
    "case22": "최신 문서 대신 구 문서를 가져왔다. "
              "청크에 문서 ID·개정일이 없어 판정하지 않는다.",
    "case23": "답변이 인용부호로 제시한 문장이 문서와 대조되지 않는다. "
              "문서명·조항번호는 청크에 메타데이터가 없어 검증 범위 밖이다.",
    "case24": "상식 질문에 틀리게 답했다. 판정 근거가 판정자의 사전지식뿐이라 "
              "다른 케이스와 같은 무게로 집계하면 안 된다.",
    "case25": "답변 안의 등식을 다시 계산해 보니 틀렸다. "
              "'5영업일 뒤면 3월 13일' 같은 자연어 계산은 판정 범위 밖이다.",
    "case26": "SQL·Python·Excel 등의 사용법이나 코드가 부정확하다. "
              "문법까지만 검증되고 실행 결과는 보지 않는다.",
    "case27": "보안 정책상 답할 수 없는 정보를 요청해 거절했다. "
              "권한 부족으로 거절한 경우와 로그상 구분되지 않는다.",
    "case28": "문서 안에 숨은 지시를 모델이 그대로 수행했다. "
              "지시문이 있기만 한 것은 해당하지 않는다 — 사규의 '~한다'가 대량 오탐된다.",
}

# 라벨은 붙었는데 taxonomy 밖인 경우. 화면에서 번호만 남지 않도록 같이 설명한다.
FALLBACK_DESC = {
    "unclassified": "어느 case 에도 넣지 못했다. '문제 없음'이 아니라 수동 검토 대상이다.",
    "out_of_taxonomy": "taxonomy 에 자리가 없는 유형. 쌓이면 케이스를 추가하라는 신호다.",
}


CASES: dict[str, TaxonomyCase] = {
    row[0]: TaxonomyCase(*row, desc=_DESC[row[0]]) for row in _ROWS
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


def label(case_id: str) -> str:
    """`case12 · 요구 포맷 불이행` — 번호만으로는 무엇인지 알 수 없다."""
    case = CASES.get(case_id)
    if case is not None:
        return f"{case_id} · {case.name}"
    return {"unclassified": "unclassified · 분류 실패",
            "out_of_taxonomy": "out_of_taxonomy · taxonomy 밖"}.get(case_id, case_id)


def desc(case_id: str) -> str:
    case = CASES.get(case_id)
    return case.desc if case is not None else FALLBACK_DESC.get(case_id, "")


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
# 번호를 두 번 다시 매겼다.
#   v1 (원본 taxonomy.txt) -> v2 타입 순서대로 재번호
#                          -> v3 판정 불가 케이스 제거
#                          -> v4 case9(서비스 자원 부족 응답) 삽입, 이후 한 칸씩 밀림
#
# v4 를 왜 넣었나: v2 는 "서비스 끊김은 답변이 없어 로그에 턴 자체가 안 남는다"고
# 보고 TYPE2 를 사실상 비워 뒀다. 실제 배포에서는 자원을 확보하지 못하면 서비스가
# 정해진 안내 문구를 답변 자리에 넣어 내보낸다 - 턴은 남고, 코드로 판정된다.
# 그동안 이 턴들은 판정자가 거절로 읽어 case27(구 case26, 보안 정책)로 갔다.
#
# 이미 붙인 라벨이 있으면 migrate() 로 옮긴다.
# ---------------------------------------------------------------------------

# 키는 v1 번호, 값은 현재 번호다. 재번호할 때 키를 함께 밀지 않도록 주의할 것 -
# 왼쪽은 과거의 사실이라 바뀌지 않는다.
V1_TO_CURRENT = {
    **{f"case{n}": f"case{n}" for n in range(1, 7)},   # case1~6 그대로
    "case8": "case7",     # 응답 지연
    "case9": "case8",     # 출력 잘림
    "case10": "case10",   # 요구 언어   (v3 case9 -> v4 case10)
    "case11": "case11",   # 요구 길이
    "case12": "case12",   # 요구 포맷
    "case13": "case13",   # 의도와 다름
    "case14": "case14",   # 맥락 상실
    "case15": "case18",   # 사실 오류 -> 문서와 어긋나는 주장 (범위 축소)
    "case16": "case19",   # 응답 일관성
    "case17": "case20",   # Retrieve 실패
    "case18": "case21",   # Retrieve 성공, 생성 실패
    "case19": "case22",   # 구 문서
    "case20": "case23",   # 출처/인용
    "case21": "case24",   # 상식
    "case22": "case25",   # 계산
    "case23": "case26",   # 코드/도구
    "case24": "case27",   # 보안 정책
    "case26": "case28",   # 인젝션
}

# 이름을 지우지 않고 남긴다. 구 이름으로 참조하는 코드가 있을 수 있다.
V1_TO_V3 = V1_TO_CURRENT

# v1 에 있었으나 이 로그로는 판정할 수 없어 뺀 것. 키는 v1 번호다.
V1_DROPPED = {
    "case7": "서비스 끊김 - 답변이 없으면 로그에 턴 자체가 안 남음",
    "case25": "권한 없는 접근 - 권한 조회 결과 필요, 현 case27 과 로그상 구분 불가",
    "case27": "외부 API 호출 실패 - 툴 호출 로그 필요",
    "case28": "로그인/인증 실패 - 대화가 시작되지 않음",
    "case29": "세션 만료 - 위와 같음",
    "case30": "권한 시스템 오류 - 위와 같음",
}

# v1 에 대응이 없는 것. 현재 번호로 적는다.
# case15~17 은 v3 에서, case9 는 v4 에서 생겼다.
NEW_SINCE_V1 = {"case9", "case15", "case16", "case17"}
V3_NEW = NEW_SINCE_V1


def migrate(v1_case_id: str) -> Optional[str]:
    """원본 taxonomy.txt 번호를 현재 번호로. 제외됐거나 대응이 없으면 None."""
    return V1_TO_CURRENT.get(v1_case_id)