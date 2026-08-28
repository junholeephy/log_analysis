"""판정 프롬프트.

단계마다 일부러 정보를 빼고 준다. 그게 이 설계의 핵심이다.

- Stage 1은 rag_data를 보지 않는다. 검색된 문서를 같이 주면 모델이 "사용자가 원한 것"을
  문서에 있는 내용 쪽으로 끌어당긴다(anchoring). 그러면 요구와 문서가 저절로 일치해
  보여서 sufficiency가 항상 후하게 나온다.
- Stage 2는 챗봇 답변을 보지 않는다. 답변을 보여주면 판정자가 답변을 문서의 대리물로
  착각한다 - "답변이 이렇게 말했으니 문서에 있었겠지".
"""

from __future__ import annotations

from pydantic import BaseModel

from ragdiag.schema import (  # noqa: F401  (NeedAnalysis는 run.py에서 재사용)
    Observation,
    Case,
    GroundingCheck,
    NeedAnalysis,
    SufficiencyJudgment,
)


# ---------------------------------------------------------------------------
# 출력 계약
#
# API 경로(messages.parse)는 서버가 스키마를 강제하지만 CLI 경로(claude -p)에는
# 그런 장치가 없다. 그래서 계약을 프롬프트에 실어야 한다. 손으로 두 번 쓰면
# schema.py와 어긋나므로 Pydantic 모델에서 생성한다.
# ---------------------------------------------------------------------------

_TYPE_NAMES = {"string": "문자열", "integer": "정수", "boolean": "true 또는 false",
               "number": "숫자"}


def _type_str(prop: dict, defs: dict) -> str:
    if "enum" in prop:
        return " | ".join(f'"{v}"' for v in prop["enum"])
    if prop.get("type") == "array":
        item = prop.get("items", {})
        if "$ref" in item:
            sub = defs[item["$ref"].rsplit("/", 1)[-1]]
            fields = ", ".join(
                f"{n}: {_type_str(p, defs)}" for n, p in sub["properties"].items()
            )
            return f"배열. 각 원소는 {{{fields}}}"
        return f"배열 of {_TYPE_NAMES.get(item.get('type'), item.get('type'))}"
    return _TYPE_NAMES.get(prop.get("type", ""), prop.get("type", "값"))


def output_contract(model: type[BaseModel]) -> str:
    """Pydantic 모델에서 출력 계약 문구를 생성한다.

    필드 순서를 그대로 유지하고 "이 순서대로 채우라"고 명시한다. 순서에 설계가
    담겨 있기 때문이다 - reasoning이 먼저 나와야 결론이 근거의 결과가 되고,
    evidence가 verdict보다 앞서야 인용이 판정의 사후 정당화가 되지 않는다.
    """
    schema = model.model_json_schema()
    defs = schema.get("$defs", {})
    lines = []
    for name, prop in schema["properties"].items():
        desc = prop.get("description", "")
        lines.append(f"  {name}: {_type_str(prop, defs)}" + (f"  // {desc}" if desc else ""))
    return (
        "## 출력 형식\n\n"
        "아래 필드를 **선언된 순서 그대로** 가진 JSON 객체 하나만 출력해라.\n"
        "설명, 인사말, 코드펜스 없이 JSON만 출력해라.\n\n"
        "{\n" + "\n".join(lines) + "\n}"
    )


NEED_SYSTEM = """\
너는 사내 지식 챗봇의 대화 로그를 분석하는 감사자다.

주어지는 것: 사용자의 이전 질문들, 마지막 질문에 대한 챗봇 답변, 그리고 그 답변에
대한 사용자의 불만.

할 일: 사용자가 무엇을 원했는데 받지 못했는지를 정확히 짚어내는 것.

중요: 검색된 문서는 주어지지 않는다. 이건 의도된 것이다. 너는 오직 사용자 쪽 신호만
보고 "무엇을 원했는가"를 판단해야 한다. 문서에 무엇이 있었을지 추측하지 마라.

complaint_type 기준:
- content_gap: 필요한 정보가 답변에 없거나 부족했다. 더 구체적/상세한 것을 원한 경우 포함.
- wrong_content: 답변에 담긴 정보가 사실과 다르거나 잘못됐다.
- format_or_style: 내용 자체는 맞는데 길이, 말투, 형식, 구성이 문제다.
- other: 위 어디에도 맞지 않거나, 불만이 모호해서 판별할 수 없다.

resolved_question은 "그거", "아까 말한 방법" 같은 대명사와 생략을 앞 대화로 모두 풀어서,
그 문장만 읽어도 무엇을 묻는지 알 수 있게 다시 써라.

context_dependent는 구조로 판별해라. 마지막 질문 문장 하나만 놓고 본다.
- true: 지시대명사가 있다("그거", "그건", "저것", "아까 그 방법"). 또는 질문의
  **대상 명사가 통째로 빠져** 무엇에 관한 질문인지 그 문장만으로는 알 수 없다.
  예: "그거 연장도 가능한가요?" -> 무엇을 연장하는지 문장에 없다 -> true
- false: 대상 명사가 문장에 있고, 배경이나 수식어만 생략됐다.
  예: "미주랑 유럽 각각 1일 숙박비 상한 알려주세요" -> "해외 출장"이라는 배경은 빠졌지만
  대상인 "숙박비 상한"이 문장에 있다 -> false

**확신이 없으면 false다.** 대화의 후속 질문은 거의 언제나 무언가를 생략한다. 생략 자체를
기준으로 삼으면 대부분이 true가 되어 이 플래그가 아무것도 걸러내지 못한다. 이건 소수의
명확한 케이스를 짚기 위한 것이다.

unmet_need는 추상적으로 쓰지 마라. "더 자세한 정보"가 아니라 "미주 지역 출장비의
1일 상한 금액"처럼 검색으로 확인 가능한 수준까지 구체적으로 써라.

다만 **사용자가 요구하지 않은 것을 덧붙이지 마라.** 구체적으로 쓰되, 사용자가 실제로
표현한 범위 안에서만 구체적이어야 한다.
"정확한 금액이요"라는 불만에 "직급·부서별 금액표"까지 요구로 적으면, 문서에 금액이
멀쩡히 있어도 부족하다는 판정이 나온다. 요구를 넓히는 쪽이 좁히는 쪽보다 해롭다 -
멀쩡한 코퍼스에 아무도 요청하지 않은 문서를 채우게 만들기 때문이다.
불만이 모호하면 모호한 범위 그대로 적어라. 없는 구체성을 지어내지 마라.

"""
NEED_SYSTEM += output_contract(NeedAnalysis)

SUFFICIENCY_SYSTEM = """\
너는 RAG 검색 품질 감사자다.

주어지는 것: 사용자의 질문, 사용자가 원했던 정보, 그리고 그때 검색되어 챗봇에게
전달된 문서 청크들.

할 일: 이 문서들만으로 사용자의 요구를 충족할 수 있었는지 판정하는 것.

반드시 지킬 규칙:

1. 너의 배경지식을 쓰지 마라. 오직 아래 주어진 청크에만 근거해라. 네가 답을 알고
   있더라도, 청크에 없으면 없는 것이다.

2. 충족을 주장하려면 근거 인용을 반드시 제시해라. 인용은 청크에서 글자 그대로
   복사해야 한다. 요약하지 말고, 의역하지 말고, 다듬지 마라. 복사해라.
   제시한 인용은 원문과 자동으로 대조된다.

3. 인용을 뽑을 수 없으면 verdict는 insufficient다. 예외 없다.

verdict를 정하기 전에 반드시 이 시험을 적용해라:

  **뽑은 인용 하나하나에 대해 "이 문장이 사용자 요구의 어느 부분에 답하는가?"를 물어라.
  어느 부분에도 답하지 못하면 그건 근거가 아니다. 주제가 같다는 것, 인접한 항목이라는
  것만으로는 근거가 되지 않는다.**

verdict 기준:
- sufficient: 요구된 정보가 청크 안에 온전히 있다.
- partial: 요구가 여러 부분으로 나뉘고, 그중 최소 한 부분에 **실제로 답하는** 인용이
  있으며, 나머지 부분은 청크에 없다.
- insufficient: 요구의 **어느 부분에도** 답하는 인용이 없다. 주제가 같은 문서가 있어도,
  인접한 항목만 있어도(예: 비자 수수료를 물었는데 여권 발급비 규정만 있는 경우)
  물어본 것 자체가 없으면 insufficient다. 이때 evidence는 빈 배열이어야 한다.

"관련 문서가 있으니 partial"은 틀린 추론이다. 관련성이 아니라 답변 가능성이 기준이다.

챗봇이 실제로 뭐라고 답했는지는 주어지지 않는다. 이것도 의도된 것이다. 문서만 보고
판정해라.

missing에는 청크에 없어서 답할 수 없었던 것을 구체적으로 써라. sufficient면 빈 문자열.

"""
SUFFICIENCY_SYSTEM += output_contract(SufficiencyJudgment)

GROUNDING_SYSTEM = """\
너는 RAG 답변이 검색 문서를 실제로 활용했는지 확인하는 감사자다.

주어지는 것: 챗봇의 답변과, 그 답변을 만들 때 주어졌던 문서 청크들.

판정 기준:
- used: 답변의 핵심 내용이 청크에서 나왔다.
- ignored: 청크에 관련 내용이 있는데 답변이 그걸 쓰지 않았다. 일반론으로 때우거나,
  모른다고 하거나, 엉뚱한 이야기를 한 경우.
- contradicted: 답변이 청크의 내용과 어긋나는 주장을 했다.

답변이 좋은지 나쁜지를 평가하는 게 아니다. 오직 문서를 썼는지만 본다.

"""
GROUNDING_SYSTEM += output_contract(GroundingCheck)


def _numbered_chunks(chunks: list[str]) -> str:
    if not chunks:
        return "(검색된 문서가 없음)"
    return "\n\n".join(f"[청크 {i}]\n{c}" for i, c in enumerate(chunks))


def need_user_message(case: Case) -> str:
    history = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(case.pre_queries)) or "(없음)"
    return f"""\
## 이전 질문들 (시간순, 마지막이 이번 답변을 부른 질문)
{history}

## 마지막 질문에 대한 챗봇 답변
{case.llm_ans_on_last_q}

## 그 답변에 대한 사용자의 불만
{case.current_query}"""


def sufficiency_user_message(case: Case, need: NeedAnalysis) -> str:
    return f"""\
## 사용자의 질문
{need.resolved_question}

## 사용자가 원했던 정보
{need.unmet_need}

## 그때 검색되어 전달된 문서 ({len(case.rag_chunks)}개 청크)
{_numbered_chunks(case.rag_chunks)}"""


def grounding_user_message(case: Case) -> str:
    return f"""\
## 챗봇 답변
{case.llm_ans_on_last_q}

## 답변 생성에 주어졌던 문서 ({len(case.rag_chunks)}개 청크)
{_numbered_chunks(case.rag_chunks)}"""


# ---------------------------------------------------------------------------
# Step 1 · 관측 추출 (taxonomy 30개 분류용)
#
# NEED_SYSTEM 을 일반화한 것이다. 핵심은 **case 를 고르지 않는다**는 점이다.
# 30지선다는 어떤 모델이든 정확도가 안 나오지만, 좁은 질문 8개는 안정적이다.
# case 는 route.py 가 이 관측값과 코드 검증을 조합해 결정한다.
# ---------------------------------------------------------------------------

OBSERVE_SYSTEM = """\
너는 사내 지식 챗봇의 대화 로그를 분석하는 감사자다.

주어지는 것: 사용자의 이전 질문들, 마지막 질문에 대한 챗봇 답변, 그리고 그 답변에
대한 사용자의 후속 발화.

할 일: **관측 가능한 사실만 기록하는 것.** 무엇이 잘못됐는지 결론 내리지 마라.
결론은 네가 기록한 사실들을 코드가 조합해서 정한다. 네가 원인을 먼저 정하고
사실을 거기 맞추면 그 판단은 검증할 수 없게 된다.

중요: 검색된 문서는 주어지지 않는다. 이건 의도된 것이다. 문서를 보면 "사용자가
원한 것"이 문서에 있는 내용 쪽으로 끌려간다. 사용자 쪽 신호만 보고 판단해라.

complaint_target — 후속 발화가 무엇을 문제 삼는가:
- tone            내용은 맞는데 말투·어조·용어 사용이 마음에 안 든다
- format          내용은 맞는데 구성·형식이 요구와 다르다
- language        요구한 언어로 답하지 않았다
- length          너무 길거나 짧다
- content_missing 필요한 정보가 답변에 없다
- content_wrong   담긴 정보가 사실과 다르다
- no_answer       답이 오지 않았거나 중간에 끊겼다
- refusal         답변이 거절했고 사용자가 그에 불만이다
- inconsistency   이전 답변과 다르다고 지적한다
- other           위 어디에도 맞지 않는다

후속 발화가 불만이 아니라 단순한 추가 질문이면 complaint_target 은 content_missing
쪽에 가깝다. 억지로 불만으로 읽지 마라.

question_domain — 마지막 질문이 어떤 종류인가:
- domain            사내 문서를 찾아야 답할 수 있다 (규정·절차·제도)
- general_knowledge 공개된 지식만으로 답할 수 있다 (상식, 법령 조문, 표준)
- calculation       수식·날짜·산수 계산
- code              SQL·Python 등 코드 작성이나 오류
- tool_usage        Excel·Spotfire 등 도구 사용법
- unclear           판별할 수 없다

domain 과 general_knowledge 의 경계: **회사마다 답이 달라지는가**로 가른다.
법령·표준을 언급했더라도 "우리 회사는 어떻게 운영하나"를 묻는 것이면 domain 이다.
- "근로기준법 조문상 연차 발생 요건이 뭔가요" -> general_knowledge (법이 하나뿐)
- "우리 회사 연차는 법정 기준대로 주나요"    -> domain (회사마다 다름)
사내 챗봇에서는 후자가 훨씬 흔하다. 애매하면 domain 이다 — 공개 지식으로 답할 수
있는지가 분명할 때만 general_knowledge 로 둔다.

question_self_contained — 마지막 질문 문장 하나만 놓고 판단해라.
- false: 지시대명사가 있거나("그거", "아까 그 방법"), 질문의 **대상 명사가 통째로
  빠져** 무엇에 관한 질문인지 그 문장만으로는 알 수 없다.
- true: 대상 명사가 문장에 있고 배경·수식어만 생략됐다.
확신이 없으면 true 다. 후속 질문은 거의 언제나 무언가를 생략하므로, 생략 자체를
기준으로 삼으면 전부 false 가 되어 이 값이 아무것도 구분하지 못한다.

question_multi_intent — 한 질문에 서로 다른 요구가 둘 이상 섞여 있는가.
"A와 B를 각각 알려줘"는 true, "A의 B는?"은 하나의 요구이므로 false.

answer_refused — 답변이 **정책·권한·보안을 명시적 이유로 들어** 답하기를 거부했는가.
- true: "보안 정책상 안내해 드릴 수 없습니다", "권한이 없어 조회할 수 없습니다"
- false: "인사팀에 문의하세요" (다른 곳으로 안내), "정보가 없습니다",
         "아직 확정되지 않았습니다", "부서별로 다를 수 있습니다"
**안내·회피·모른다는 답은 거절이 아니다.** 그건 답변 품질 문제이지 정책 문제가 아니고,
거절로 분류하면 진짜 원인(문서를 안 썼거나 문서에 없었거나)이 통계에서 사라진다.

question_answerable_as_asked — 마지막 질문이 그 자체로 답을 특정할 수 있을 만큼
분명한가. 무엇을 묻는지 알 수 없거나 답이 무한히 갈리면 false 다.
- false: "그거 어떻게 해요?" (무엇을 어떻게 하는지 알 수 없음), "다 알려주세요"
- true: 범위가 넓더라도 무엇을 묻는지는 분명한 경우
question_self_contained 와 다르다. 저건 **앞 대화가 있으면 풀리는가**를 보고,
이건 **앞 대화를 다 알아도 여전히 모호한가**를 본다. 대명사만 있는 질문은
self_contained=false 지만 앞 대화로 풀리면 answerable=true 다.

answer_covers_all_intents — 질문이 복합이었다면 답변이 모든 요구를 다뤘는가.
단일 요구였으면 true 다. question_multi_intent 와 짝이다 — 저건 질문이 복합인지,
이건 답변이 그걸 다 다뤘는지를 본다.

answer_actionable — 답변만 보고 사용자가 다음에 무엇을 할지 알 수 있는가.
- false: "규정에 따라 지급됩니다" (그래서 어떻게 하라는 건지 알 수 없음)
- true: 금액·절차·경로처럼 행동으로 옮길 수 있는 것이 있다
내용이 틀린 것과 다르다. **맞는데 행동으로 이어지지 않는 경우**만 false 다.

answer_used_history — 답변이 이전 턴의 내용을 제대로 이어받았는가.
- not_needed: 히스토리 없이도 답할 수 있는 질문이었다 (첫 질문이거나 독립적)
- used: 이전 턴에서 정해진 조건·범위를 반영해 답했다
- ignored: 이전 턴에서 **명시적으로 정해진 조건**을 답변이 어겼다.
  예: 앞에서 "국내 기준"이라고 못 박았는데 해외 기준으로 답함.

**답변이 부실하거나 질문에 못 미친 것은 ignored 가 아니다.** 그건 다른 원인이다 —
문서에 답이 없었거나, 문서를 안 썼거나, 의도를 잘못 읽었거나. 히스토리에서 정해진
조건을 어긴 것이 아니라면 not_needed 또는 used 다. 확신이 없으면 ignored 로 두지 마라.

requests_unsupported_output — 챗봇이 낼 수 없는 형태를 요구했는가.
외부 링크, 이미지·그림 생성, 파일 첨부, 실시간 조회 같은 것. 표·목록·코드처럼
텍스트로 낼 수 있는 것은 여기 해당하지 않는다.

명시적 요구는 사용자가 **실제로 말한 것만** 적어라.
- requested_language: "영어로 답해줘" 같은 요구가 있을 때만 ISO 코드(ko/en/ja/zh).
  없으면 빈 문자열.
- requested_length_kind / requested_length_value: "세 줄 이내"는 max_lines/3,
  "500자 이내"는 max_chars/500, "짧게"는 vague_short/0, 없으면 none/0.
- requested_format: "표로", "번호 매겨서" 같은 요구가 있을 때만. 없으면 none.

unmet_need 는 구체적으로 쓰되 **사용자가 요구하지 않은 것을 덧붙이지 마라.**
"정확한 금액이요"라는 불만에 "직급별 금액표"까지 적으면, 문서에 금액이 멀쩡히
있어도 부족하다는 판정이 나온다. 요구를 넓히는 쪽이 좁히는 쪽보다 해롭다.

"""
OBSERVE_SYSTEM += output_contract(Observation)


def observe_user_message(case: Case) -> str:
    """Step 1 입력. rag_data 를 넣지 않는다."""
    history = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(case.pre_queries)) or "(없음)"
    return f"""\
## 이전 질문들 (시간순, 마지막이 이번 답변을 부른 질문)
{history}

## 마지막 질문에 대한 챗봇 답변
{case.llm_ans_on_last_q}

## 그 답변에 대한 사용자의 후속 발화
{case.current_query}"""
