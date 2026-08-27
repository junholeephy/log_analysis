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
