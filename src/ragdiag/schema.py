"""데이터 구조 정의.

- Case: 분석 단위(대화 한 턴). 중첩 JSON을 flatten한 결과.
- Stage1/2/3 출력: LLM이 채우는 Pydantic 모델. messages.parse()로 스키마가 강제된다.

Pydantic 모델의 필드 순서에 의미가 있다. JSON은 선언 순서대로 생성되므로
reasoning을 먼저, 결론을 나중에 두어야 모델이 근거를 쓴 뒤 판정하게 된다.
Stage2에서 evidence를 verdict보다 앞에 둔 것도 같은 이유 - 인용을 먼저 뽑게 하면
판정이 인용의 결과가 되고, 반대로 두면 인용이 판정의 사후 정당화가 된다.
"""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

ComplaintType = Literal["content_gap", "wrong_content", "format_or_style", "other"]
Verdict = Literal["sufficient", "partial", "insufficient"]
UsedRag = Literal["used", "ignored", "contradicted"]

Label = Literal[
    "rag_insufficient",                 # 검색된 문서에 답이 없었음 -> 코퍼스/검색 보강 대상
    "rag_partial",                      # 일부만 충족
    "rag_sufficient_generation_failed", # 문서엔 답이 있었으나 생성이 활용 실패
    "rag_sufficient_other",             # 문서도 충분, 활용도 함 -> 깊이/표현 문제
    "out_of_scope",                     # 불만이 형식/스타일 -> rag와 무관
    "unclassified",                     # 불만 성격 판별 실패 -> 수동 검토 대상
]


@dataclass(frozen=True)
class Case:
    """분석 단위. 사용자 메타는 상위 노드에서 상속받는다."""

    case_id: str
    user_id: str
    dept: str
    job_grade: str
    job_name: str
    position_name: str
    conversation_id: str
    turn: int
    pre_queries: list[str]
    llm_ans_on_last_q: str
    current_query: str
    rag_chunks: list[str]

    @property
    def last_query(self) -> str:
        return self.pre_queries[-1] if self.pre_queries else ""


class NeedAnalysis(BaseModel):
    """Stage 1 출력. rag_data를 보지 않고 사용자 쪽 신호만으로 채운다."""

    reasoning: str = Field(description="불만을 어떻게 읽었는지 2~3문장")
    resolved_question: str = Field(
        description="대화 맥락을 반영해 대명사와 생략을 모두 푼, 그 자체로 이해되는 질문"
    )
    unmet_need: str = Field(
        description="사용자가 원했는데 받지 못한 정보를 구체적으로. 형식 불만이면 그렇게 적는다"
    )
    complaint_type: ComplaintType
    context_dependent: bool = Field(
        description="마지막 질문 문장만으로 검색 쿼리를 만들 때 핵심 검색어가 빠지는가"
    )


class Evidence(BaseModel):
    chunk_index: int = Field(description="근거가 있는 청크의 0-기반 인덱스")
    quote: str = Field(description="해당 청크에서 글자 그대로 복사한 문장. 요약·수정 금지")


class SufficiencyJudgment(BaseModel):
    """Stage 2 출력. llm_ans_on_last_q를 보지 않고 문서만으로 채운다."""

    reasoning: str
    evidence: list[Evidence] = Field(
        description="unmet_need를 충족하는 근거. 없으면 빈 배열"
    )
    verdict: Verdict
    missing: str = Field(description="문서에 없어서 답할 수 없었던 것. sufficient면 빈 문자열")


class GroundingCheck(BaseModel):
    """Stage 3 출력. verdict가 sufficient일 때만 실행된다."""

    reasoning: str
    answer_used_rag: UsedRag


# ---------------------------------------------------------------------------
# Step 1 관측 스키마  (taxonomy 30개 확장용)
#
# 기존 NeedAnalysis 를 일반화한 것이다. 핵심 차이는 **case를 고르지 않는다**는 점이다.
# 관측 가능한 사실만 내고, case는 checks.py 의 결정적 검증과 함께 코드가 도출한다.
#
# 이렇게 두면 taxonomy 를 고쳐도 이 값들은 그대로 재사용된다 — 관측은 taxonomy 와
# 무관한 사실이기 때문이다. 라우팅만 다시 돌리면 된다.
#
# nullable 타입을 쓰지 않고 "none" 센티넬과 빈 문자열을 쓴 이유: 로컬 서빙의
# 구조화 출력 강제가 anyOf(null 허용)를 제대로 처리하지 못하는 경우가 있다.
# 평평한 스칼라만 쓰면 어느 백엔드에서도 안전하다.
# ---------------------------------------------------------------------------

ComplaintTarget = Literal[
    "none",              # 불만이 아니다 — 새 질문이거나 수긍   -> case0
    "tone",              # 말투·어조·용어가 마음에 안 듦    -> case16
    "format",            # 형식·구성이 마음에 안 듦        -> case12
    "language",          # 요구한 언어가 아님              -> case10
    "length",            # 너무 길다/짧다                  -> case11
    "content_missing",   # 필요한 정보가 없음              -> case3·case14·case18
    "content_wrong",     # 담긴 정보가 틀림                -> case18·case19·case23~case25
    "no_answer",         # 답이 안 왔거나 끊김             -> case8
    "refusal",           # 거절당함                        -> case28 (권한 부족 거절이 섞여 들어옴)
    "inconsistency",     # 이전 답변과 다름                -> case19
    "other",
]

QuestionDomain = Literal[
    "domain",            # 사내 문서를 찾아야 답할 수 있는 질문
    "general_knowledge", # 상식                            -> case25
    "calculation",       # 수식·날짜·산수                  -> case26
    "code",              # SQL/Python 등                   -> case27
    "tool_usage",        # Excel/Spotfire 등 도구 사용법
    "unclear",
]

HistoryUse = Literal[
    "not_needed",   # 히스토리 없이도 답할 수 있는 질문
    "used",         # 답변이 이전 턴 내용을 반영함
    "ignored",      # 이전 턴에 나온 내용을 잊었거나 잘못 연결함  -> case14
]

LengthRequestKind = Literal[
    "none", "max_chars", "max_sentences", "max_lines", "vague_short"
]

FormatRequest = Literal[
    "none", "numbered_list", "bullet_list", "table", "code_block", "json", "prose"
]


class Observation(BaseModel):
    """Step 1 출력 — 관측만 하고 판정하지 않는다.

    필드 순서에 의미가 있다. reasoning 이 먼저 나와야 뒤의 값들이 근거의 결과가 된다.
    """

    reasoning: str = Field(description="불만과 질문을 어떻게 읽었는지 2~3문장")

    # --- 사용자가 원한 것 (기존 NeedAnalysis 계승) ---
    resolved_question: str = Field(
        description="대화 맥락을 반영해 대명사와 생략을 모두 푼, 그 자체로 이해되는 질문"
    )
    unmet_need: str = Field(
        description="사용자가 원했는데 받지 못한 것. 사용자가 요구하지 않은 항목을 덧붙이지 말 것"
    )

    # --- 불만의 성격 ---
    complaint_target: ComplaintTarget
    complaint_quote: str = Field(
        default="",
        description="후속 발화에서 그대로 따온 구절. complaint_target 을 그렇게 읽은 근거다. "
                    "특히 'none' 일 때 필수 — 그게 가장 쉬운 답이라 근거 없이 통과시키면 "
                    "애매한 턴이 전부 그리로 샌다. 원문 그대로여야 하고 지어내면 검증에서 걸린다",
    )
    question_domain: QuestionDomain

    # --- 질문 쪽 관측 ---
    question_self_contained: bool = Field(
        description="마지막 질문 문장만으로 검색 쿼리를 만들 수 있는가 (case4의 반대)"
    )
    question_multi_intent: bool = Field(
        description="한 질문에 서로 다른 요구가 둘 이상 섞여 있는가 (case3)"
    )

    # --- 답변 쪽 관측 ---
    answer_refused: bool = Field(
        description="답변이 정책·권한을 이유로 거절했는가 (case28)"
    )
    question_answerable_as_asked: bool = Field(
        description="질문이 그 자체로 답을 특정할 수 있을 만큼 분명한가. "
                    "무엇을 묻는지 알 수 없으면 false (case1)"
    )
    answer_covers_all_intents: bool = Field(
        description="복합 질문이었다면 답변이 모든 요구를 다뤘는가. "
                    "단일 요구였으면 true (case15)"
    )
    answer_actionable: bool = Field(
        description="답변만 보고 사용자가 다음에 무엇을 할지 알 수 있는가 (case17)"
    )
    answer_used_history: HistoryUse = Field(
        description="답변이 이전 턴의 내용을 제대로 이어받았는가 (case14)"
    )
    requests_unsupported_output: bool = Field(
        description="챗봇이 낼 수 없는 형태를 요구했는가 — 외부 링크, 이미지·그림 생성, "
                    "파일 첨부 등 (case2)"
    )

    # --- 명시적 요구 (코드 검증기에 넘길 값) ---
    requested_language: str = Field(
        description='요구한 언어의 ISO 코드(ko/en/ja/zh). 요구가 없으면 빈 문자열'
    )
    requested_length_kind: LengthRequestKind
    requested_length_value: int = Field(
        description="수치 요구의 값. 수치가 없으면 0"
    )
    requested_format: FormatRequest
