"""결정적 검증기 — LLM 없이 코드로 판정하는 것들.

taxonomy의 절반 이상은 LLM이 필요 없다. 언어가 맞는지, 길이 요구를 지켰는지,
답변이 문장 중간에 끊겼는지는 문자열만 보면 안다. 이걸 LLM에 맡기면 비용도 들지만
무엇보다 **흔들린다** — 같은 입력에 다른 답이 나온다. 코드는 안 흔들린다.

여기 있는 함수는 전부 순수 함수다. 입력 JSON 포맷이 바뀌어도 영향받지 않는다.

각 검증기는 Check를 돌려준다. verdict의 네 값이 서로 다른 뜻이라는 게 중요하다:
  ok            요구를 지켰다
  violated      요구를 어겼다
  not_applicable  그런 요구가 애초에 없었다  (위반 아님)
  undetermined  요구는 있었지만 판정 근거가 부족하다  (조용히 ok로 넘기면 안 된다)
"""

from __future__ import annotations

import ast
import json
import re
import unicodedata
from datetime import date
from dataclasses import dataclass, field
from typing import Literal, Optional

from ragdiag import settings
from ragdiag.verify import match_ratio, normalize

Verdict = Literal["ok", "violated", "not_applicable", "undetermined"]

RequestedFormat = Literal[
    "numbered_list", "bullet_list", "table", "code_block", "json", "prose"
]


@dataclass
class Check:
    name: str
    verdict: Verdict
    detail: str = ""
    evidence: list[str] = field(default_factory=list)

    @property
    def violated(self) -> bool:
        return self.verdict == "violated"


# ---------------------------------------------------------------------------
# 언어  (case10)
# ---------------------------------------------------------------------------

_SCRIPTS = {
    "hangul": re.compile(r"[가-힣ᄀ-ᇿ㄰-㆏]"),
    "latin": re.compile(r"[A-Za-z]"),
    "kana": re.compile(r"[぀-ヿ]"),
    "han": re.compile(r"[一-鿿]"),
}

# 한국어 답변에는 영문 용어가 흔히 섞인다("VPN 접속 시 MFA 인증").
# 그래서 다수결이 아니라 낮은 임계값으로 한글 존재 여부를 본다.
_HANGUL_THRESHOLD = 0.10


def script_profile(text: str) -> dict[str, float]:
    """문자 종류별 비율. 글자가 아닌 문자(숫자·기호·공백)는 분모에서 뺀다."""
    counts = {name: len(pattern.findall(text)) for name, pattern in _SCRIPTS.items()}
    total = sum(counts.values())
    if total == 0:
        return {name: 0.0 for name in counts}
    return {name: n / total for name, n in counts.items()}


def detect_language(text: str) -> str:
    """ko / en / ja / zh / unknown.

    한계: 스크립트 기반이라 라틴 문자를 쓰는 언어들(영어·독일어·프랑스어)을
    구분하지 못한다. 운영 환경 챗봇에서 실제로 갈리는 건 한국어와 영어라 이 수준이면 된다.
    """
    profile = script_profile(text)
    if sum(profile.values()) == 0:
        return "unknown"
    if profile["hangul"] >= _HANGUL_THRESHOLD:
        return "ko"
    if profile["kana"] > 0.05:
        return "ja"
    if profile["han"] > 0.3:
        return "zh"
    if profile["latin"] > 0.5:
        return "en"
    return "unknown"


def check_language(answer: str, requested: Optional[str]) -> Check:
    """case10 — 특정 언어를 요구했는데 지키지 않음."""
    if not requested:
        return Check("language", "not_applicable", "언어 요구 없음")
    actual = detect_language(answer)
    if actual == "unknown":
        return Check("language", "undetermined", "답변의 언어를 판별할 수 없음")
    if actual == requested:
        return Check("language", "ok", f"요구 {requested} · 실제 {actual}")
    return Check("language", "violated", f"요구 {requested} · 실제 {actual}")


# ---------------------------------------------------------------------------
# 길이  (case11) — **재기만 하고 판정하지 않는다**
#
# 다른 검증기는 답이 텍스트 밖에 확정돼 있다. 언어는 스크립트 비율로, 날짜는
# 달력으로, 파이썬은 파서로 정해진다. 길이는 그런 것이 없다 - 기준이 사용자
# 머릿속에 있다.
#
#   "세 줄 이내로"   개행 3개? 60자×3? 요점 3개?   사람마다 다르다
#   "다섯 문장으로"  한국어 만연체는 149자가 한 문장이다
#   "짧게"          기준값이 없다. 정해도 임의값이다
#
# 실제로 그랬다. 503건에서 길이를 요구한 20건 전부 ok 가 나왔고, 그중 "세 줄
# 이내" 요구에 149자 만연체로 답한 것까지 통과했다 - 줄바꿈이 없어 1줄이라서다.
# VAGUE_SHORT_MAX_CHARS=400 은 최대 답변이 370자인 데이터에서 한 번도 걸리지
# 않았다. **임의 상수 하나로 판정 실패를 덮은 것**이다.
#
# 그래서 측정만 남기고 판정을 버린다. verdict 는 언제나 undetermined 이고,
# detail 에 세 측정값과 요구를 적는다. 라우팅에는 이미 경로가 있다 - 코드 근거가
# 없으면 case 는 유지하되 신뢰도를 낮춘다. 틀린 ok 를 내는 것보다 낫다.
#
# 실데이터에서 분포를 보고 기준을 정하게 되면, 그때 이 detail 만 읽으면 된다.
# LLM 을 다시 돌릴 필요가 없다.
# ---------------------------------------------------------------------------


def count_sentences(text: str) -> int:
    """마침표 기준. 한국어 만연체는 이 셈으로 한 문장이 된다 - 그래서 이 값으로
    판정하지 않고 사람이 볼 수 있게 적기만 한다."""
    parts = [p for p in re.split(r"(?<=[.!?。])\s+", text.strip()) if p.strip()]
    return len(parts)


@dataclass
class LengthRequest:
    """Step 1이 뽑아내는 길이 요구. 수치가 없으면 kind='vague_short'."""

    kind: Literal["max_chars", "max_sentences", "max_lines", "vague_short"]
    value: Optional[int] = None


def measure_length(answer: str) -> str:
    chars = len(answer.strip())
    lines = len([l for l in answer.splitlines() if l.strip()])
    return f"{chars}자 · {count_sentences(answer)}문장 · {lines}줄"


def check_length(answer: str, requested: Optional[LengthRequest]) -> Check:
    """case11 — 길이 요구를 **재기만 한다.** 위반 판정은 하지 않는다.

    이 모듈의 다른 검증기와 성격이 다르다는 점이 중요하다. 여기서 나온
    undetermined 는 "판정에 실패했다"가 아니라 "코드가 판정할 수 있는 것이
    아니다"라는 뜻이다. case16(말투·어조)이 검증기 없이 관측에만 의존하는 것과
    같은 자리인데, 길이는 숫자가 나온다는 이유로 검증기가 붙어 있었다.
    """
    measured = measure_length(answer)
    if requested is None:
        return Check("length", "not_applicable", f"길이 요구 없음 · {measured}")

    if requested.kind == "vague_short":
        asked = "모호한 짧게 요구"
    elif requested.value is None:
        asked = f"{requested.kind} (값 없음)"
    else:
        asked = f"{requested.kind} ≤ {requested.value}"
    return Check("length", "undetermined", f"요구 {asked} · {measured}",
                 evidence=["길이 기준은 사용자마다 달라 코드로 판정하지 않는다"])


# ---------------------------------------------------------------------------
# 포맷  (case12)
# ---------------------------------------------------------------------------

_NUMBERED = re.compile(r"^\s*(?:\d+[.)]|[①-⑳])\s+\S", re.M)
_BULLET = re.compile(r"^\s*[-*•·]\s+\S", re.M)
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$", re.M)
_TABLE_SEP = re.compile(r"^\s*\|[\s:|-]+\|\s*$", re.M)
_FENCE = re.compile(r"```")


def has_format(answer: str, kind: RequestedFormat) -> bool:
    if kind == "numbered_list":
        return len(_NUMBERED.findall(answer)) >= 2
    if kind == "bullet_list":
        return len(_BULLET.findall(answer)) >= 2
    if kind == "table":
        return len(_TABLE_ROW.findall(answer)) >= 2 and bool(_TABLE_SEP.search(answer))
    if kind == "code_block":
        return len(_FENCE.findall(answer)) >= 2
    if kind == "json":
        try:
            json.loads(_strip_fence(answer))
            return True
        except (ValueError, TypeError):
            return False
    if kind == "prose":
        # 줄글 요구는 목록이 없어야 지킨 것이다.
        return not (_NUMBERED.search(answer) or _BULLET.search(answer))
    return False


def _strip_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z]*\s*", "", stripped)
        stripped = re.sub(r"```\s*$", "", stripped)
    return stripped.strip()


def check_format(answer: str, requested: Optional[RequestedFormat]) -> Check:
    """case12 — 특정 포맷을 요구했는데 지키지 않음."""
    if not requested:
        return Check("format", "not_applicable", "포맷 요구 없음")
    if has_format(answer, requested):
        return Check("format", "ok", f"요구 {requested} 충족")
    return Check("format", "violated", f"요구 {requested} 인데 해당 구조가 없음")


# ---------------------------------------------------------------------------
# 출력 잘림  (case8)
# ---------------------------------------------------------------------------

# 정상 종결로 볼 문자. 한국어 종결어미(다./요.)는 마침표가 붙으므로 별도 처리 불필요.
_CLOSERS = tuple(".!?。」』】)]}…\"'`")

# 이모지 하나를 이루는 글자들. 범주로 뭉뚱그리면 안 된다 - 백틱(`)이 Sk 라서
# Sk 를 포함시켰다가 닫는 코드펜스(```)까지 벗겨 "코드블록이 닫히지 않음" 으로
# 뒤집혔다. 실제로 이모지를 만드는 것만 적는다.
#
#   So        그림문자 본체 (😊 ✅ ⚠ 🇰 ★)
#   FE0E/FE0F 이형 선택자 — ⚠️ 의 뒤쪽 한 글자
#   200D      ZWJ — 👨‍👩‍👧 처럼 여러 글자를 잇는 것
#   1F3FB~FF  피부색 수정자
_EMOJI_JOINERS = {"\ufe0e", "\ufe0f", "\u200d"} | {chr(c) for c in range(0x1F3FB, 0x1F400)}


def _is_pictograph(ch: str) -> bool:
    return ch in _EMOJI_JOINERS or unicodedata.category(ch) == "So"


def _strip_pictographs(text: str) -> str:
    """끝에 붙은 이모지 장식을 떼어낸다. 이형 선택자·ZWJ 까지 함께 떨어진다."""
    end = len(text)
    while end and _is_pictograph(text[end - 1]):
        end -= 1
    return text[:end].rstrip()


def check_truncated(answer: str) -> Check:
    """case8 — 답변이 문장 중간에서 끊김.

    한계: 정상 답변도 목록 항목이나 표로 끝나면 종결 부호가 없다. 그래서 그런
    구조를 먼저 걸러낸 뒤에만 잘림으로 본다. finish_reason 필드가 생기면
    이 휴리스틱은 필요 없어진다.

    **이모지로 끝나는 것은 완결의 신호다.** 생성이 끊기면 토큰 중간에서 멈추지,
    그 자리에 장식을 붙이고 멈추지 않는다. 그래서 종결 부호와 같은 무게로 본다 -
    한국어 답변은 마침표를 생략하고 이모지로 끝맺는 일이 흔하다.
    """
    raw = answer.rstrip()
    if not raw:
        return Check("truncated", "undetermined", "답변이 비어 있음")

    # 이모지를 떼고 본문을 본다. 떼지 않으면 "확인해 보세요! 👍" 처럼 마침표까지
    # 있는 답변이 종결 부호 검사에서 떨어진다.
    text = _strip_pictographs(raw)
    ends_with_emoji = text != raw
    if not text:
        return Check("truncated", "ok", "이모지로만 이루어진 답변")

    last_line = text.splitlines()[-1].strip()
    # 목록·표·코드블록으로 끝나는 건 정상이다.
    if (
        _TABLE_ROW.match(last_line)
        or _BULLET.match(last_line)
        or _NUMBERED.match(last_line)
        or last_line.endswith("```")
    ):
        return Check("truncated", "ok", "구조적 종결(목록·표·코드블록)")

    # 닫히지 않은 코드블록이 먼저다. 종결 부호 검사를 앞에 두면 코드 마지막 줄의
    # 괄호("print(1)")를 정상 종결로 오판한다.
    if len(_FENCE.findall(text)) % 2 == 1:
        return Check("truncated", "violated", "코드블록이 닫히지 않음")

    if text.endswith(_CLOSERS):
        tail = " + 이모지" if ends_with_emoji else ""
        return Check("truncated", "ok", f"종결 부호로 끝남: {text[-1]!r}{tail}")

    if ends_with_emoji:
        return Check("truncated", "ok", f"이모지로 끝남: {raw[len(text):].strip()!r}")

    return Check("truncated", "violated", f"종결 부호 없이 끝남: …{text[-20:]!r}")


# ---------------------------------------------------------------------------
# 서비스 자원 부족 응답
#
# 모델 자원을 확보하지 못했을 때 서비스 계층이 내보내는 **확정 문구**다. LLM 이
# 생성한 답이 아니라 정해진 문자열이므로 코드로 판정한다 - 신뢰도 high 다.
#
# 이걸 따로 잡지 않으면 판정자가 "답변이 거절했다"로 읽어 case28(보안 정책상
# 답변 불가)으로 간다. 실제로 그 오분류가 많이 나왔다. 서버 자원 문제를 보안
# 정책 문제로 세면 고칠 곳을 정반대로 가리킨다 - 한쪽은 인프라 증설이고
# 다른 쪽은 권한 정책이다.
#
# 문구는 배포마다 다르므로 아래 목록에 줄을 추가해 쓴다. 공백 차이는 무시한다.
# ---------------------------------------------------------------------------

SERVICE_ERROR_TEMPLATES = settings.SERVICE_ERROR_TEMPLATES

# 템플릿이 조금 바뀌어도 놓치지 않도록 두는 보조 표지. 단독으로는 쓰지 않고
# 두 개 이상 겹칠 때만 인정한다 - "서버" 한 단어로 잡으면 서버 관련 질문에
# 정상적으로 답한 것까지 오탐한다.
_SERVICE_ERROR_MARKERS = settings.SERVICE_ERROR_MARKERS

# 확정 문구는 짧고, 그 문구가 답변의 전부다. 길면 서버 장애를 '주제로' 답한
# 정상 답변일 가능성이 높다. 길이로 한 번 더 거른다.
MAX_SERVICE_ERROR_LEN = settings.SERVICE_ERROR_MAX_CHARS


def _squeeze(text: str) -> str:
    return re.sub(r"\s+", "", text)


def check_service_error(answer: str) -> Check:
    """답변이 서비스 자원 부족 안내 문구인가."""
    if not answer.strip():
        return Check("service_error", "not_applicable", "답변이 비어 있음")

    packed = _squeeze(answer)
    for template in settings.SERVICE_ERROR_TEMPLATES:
        if _squeeze(template) in packed:
            return Check("service_error", "violated",
                         f"서비스 자원 부족 확정 문구와 일치: {template[:30]}…")

    if len(packed) > settings.SERVICE_ERROR_MAX_CHARS:
        return Check("service_error", "ok",
                     f"확정 문구 없음 · 답변이 길어({len(packed)}자) 안내 문구가 아님")

    hits = [m for m in settings.SERVICE_ERROR_MARKERS if _squeeze(m) in packed]
    if len(hits) >= 2:
        return Check("service_error", "violated",
                     f"확정 문구는 아니나 표지 {len(hits)}개 일치: {', '.join(hits)}")

    return Check("service_error", "ok", "서비스 안내 문구 아님")

# ---------------------------------------------------------------------------
# 개인정보  (case6)
# ---------------------------------------------------------------------------

# 앵커가 분명한 패턴만 잡는다. 숫자 나열을 전부 의심하면 오탐이 쏟아진다.
_PII_PATTERNS = {
    "주민등록번호": re.compile(r"\b\d{6}[-\s]?[1-4]\d{6}\b"),
    "휴대전화": re.compile(r"\b01[016-9][-\s]?\d{3,4}[-\s]?\d{4}\b"),
    "이메일": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b"),
    "카드번호": re.compile(r"\b(?:\d{4}[-\s]){3}\d{4}\b"),
    "계좌번호": re.compile(r"\b\d{2,3}-\d{2,6}-\d{2,6}\b"),
}


def find_pii(text: str) -> list[dict]:
    hits = []
    for kind, pattern in _PII_PATTERNS.items():
        for match in pattern.finditer(text):
            hits.append({"kind": kind, "text": match.group()})
    return hits


def check_pii(text: str) -> Check:
    """case6 — 질문에 개인·민감정보가 포함됨."""
    hits = find_pii(text)
    if not hits:
        return Check("pii", "ok", "검출 없음")
    kinds = sorted({h["kind"] for h in hits})
    # 원본 값은 남기지 않는다. 종류와 개수만으로 충분하다.
    return Check("pii", "violated", f"{', '.join(kinds)} {len(hits)}건")


# ---------------------------------------------------------------------------
# 답변 속 인용 대조  (case24)
# ---------------------------------------------------------------------------

# verify.py 와 방향이 반대다. 거기서는 판정자의 인용을 검증하고, 여기서는
# 챗봇 답변이 문서에서 가져왔다고 제시한 문장을 검증한다. 같은 대조 함수를 쓴다.
#
# 여는 부호와 닫는 부호를 한 문자 집합으로 묶어 섞여 있어도 잡는다 - 모델이
# `“…"` 처럼 짝을 어긋나게 내는 일이 흔하다.
#
# 곧은 작은따옴표(')만 앞뒤로 가드를 둔다. 영어 축약형이 짝처럼 보이기 때문이다:
#   "you don't need to worry, it isn't required"
#      → 두 어포스트로피 사이가 23자라 인용으로 잡힌다.
# 글자·숫자에 붙어 있으면 어포스트로피로 보고 넘어간다.
_QUOTE_PATTERNS = [
    re.compile(r"[“\"]([^”\"]{10,200})[”\"]"),                    # 큰따옴표
    re.compile(r"‘([^’‘]{10,200})’"),                              # 둥근 작은따옴표
    re.compile(r"(?<![A-Za-z0-9])'([^']{10,200})'(?![A-Za-z0-9])"),  # 곧은 작은따옴표
]

# 제목 부호. 한국어에서 이 안에 들어가는 것은 **문장이 아니라 문서 이름**이다.
# 문장 인용과 통을 나누는 이유: 대조 대상이 다르다. 문장은 청크 본문과 맞춰야 하고
# 문서명은 청크 머리의 출처 표기와 맞춰야 한다. 한 통에 넣고 본문과 대조하면
# **정확히 인용한 답변까지 위반으로 나온다** - 본문에 문서명이 적혀 있을 리 없다.
_SOURCE_PATTERNS = [re.compile(r"[「『《〈]([^」』》〉\n]{2,60})[」』》〉]")]

# 청크 머리의 출처 표기. 본문 앞에 붙어서 온다:
#   "[정보보호정책 시행세칙 제7조] 사내 자료의 외부 반출은 보안심의를 거쳐야 한다."
# 어떤 괄호를 쓸지는 운영 로그가 정하므로 흔한 것을 모두 받는다.
_CHUNK_SOURCE = re.compile(r"^\s*[\[\(「『《【]([^\]\)」』》】\n]{1,80})[\]\)」』》】]")
ANSWER_QUOTE_MIN_CHARS = settings.ANSWER_QUOTE_MIN_CHARS
# 옛 이름. verify.py 의 동명 상수와 재는 대상이 다르다 - 이쪽은 답변이 인용부호로
# 제시한 문장, 저쪽은 판정자가 근거로 제출한 인용이다.
MIN_QUOTE_CHARS = ANSWER_QUOTE_MIN_CHARS


def extract_quotes(answer: str) -> list[str]:
    """답변이 인용부호로 제시한 **문장**. 제목 부호 안의 문서명은 빼고 센다."""
    quotes = []
    for pattern in _QUOTE_PATTERNS:
        quotes += [q.strip() for q in pattern.findall(answer)]
    return [q for q in quotes if len(normalize(q)) >= settings.ANSWER_QUOTE_MIN_CHARS]


def extract_sources(answer: str) -> list[str]:
    """답변이 제목 부호로 댄 **문서 이름**.

    길이 하한이 문장 인용보다 훨씬 낮다. 「휴가규정」은 정규화하면 4자인데,
    문장 기준(10자)을 그대로 쓰면 짧은 문서명이 조용히 빠진다 - 실제로
    「연차휴가 운영지침」(8자)이 그래서 검증을 통째로 건너뛰고 있었다.
    """
    names = []
    for pattern in _SOURCE_PATTERNS:
        names += [n.strip() for n in pattern.findall(answer)]
    return [n for n in names if len(normalize(n)) >= 2]


def chunk_sources(chunks: list[str]) -> list[str]:
    """청크 머리에 붙어 온 출처 표기. 없으면 빈 목록."""
    found = []
    for chunk in chunks:
        m = _CHUNK_SOURCE.match(chunk)
        if m:
            found.append(m.group(1).strip())
    return found


def check_quoted_spans(answer: str, chunks: list[str], threshold: float = 0.9) -> Check:
    """case24 — 답변이 문서에서 가져왔다고 제시한 것이 실제로 그런가.

    두 가지를 따로 본다. **대조 대상이 다르기 때문이다.**

    1. **문장 인용** (`"…"` `'…'`) → 청크 **본문**과 대조. 조사·띄어쓰기가 쉽게
       달라지므로 최장 연속 일치율 90% 를 기준으로 한다.
    2. **문서명** (`「」` `『』` `《》` `〈〉`) → 청크 **머리의 출처 표기**와 대조.
       문서명은 산문이 아니라 식별자라 부분 점수를 주면 안 된다 - 「휴가규정」이
       「휴가규정 시행세칙」에 1.0 으로 통과해 버린다. 대신 **포함 관계**로 본다:
       출처 표기에는 조항까지 붙어 오므로("정보보호정책 시행세칙 제7조") 이름만
       댄 인용도 맞다고 봐야 한다.

    한 통에 넣고 본문과 대조하던 때는 **정확히 인용한 답변까지 위반**이 됐다.
    청크 본문에 문서명이 적혀 있을 리가 없어서다. 그때 안 터진 것은 길이 덕이었다 -
    「연차휴가 운영지침」은 정규화하면 8자라 10자 하한에 걸려 조용히 빠졌다.

    청크에 출처 표기가 없으면 문서명은 **대조하지 않고 그렇게 적는다.** 운영 로그가
    출처를 실어 보내기 시작하면 그때부터 저절로 켜진다.
    """
    quotes = extract_quotes(answer)
    sources = extract_sources(answer)
    if not quotes and not sources:
        return Check("quoted_spans", "not_applicable", "답변에 인용된 문장이 없음")
    if not chunks:
        # 검색 결과가 0건인데 답변이 문서를 인용했다. 대조할 것이 없는 게 아니라
        # **가져올 곳이 없었는데 가져온 척한 것**이다. 전에는 undetermined 로 넘겨서
        # 이 신호가 조용히 사라졌다 - case21(Retrieve 미수행)과 겹치는 자리다.
        return Check("quoted_spans", "violated",
                     f"검색 결과가 0건인데 인용 {len(quotes) + len(sources)}건",
                     evidence=(quotes + sources)[:5])

    wrong = [f"문장: {q}" for q in quotes
             if max(match_ratio(q, c) for c in chunks) < threshold]

    known = [normalize(x) for x in chunk_sources(chunks)]
    unchecked = ""
    if sources and known:
        wrong += [f"문서명: {n}" for n in sources
                  if not any(normalize(n) in k for k in known)]
    elif sources:
        unchecked = f" · 문서명 {len(sources)}건은 대조 불가(청크에 출처 표기 없음)"

    checked = len(quotes) + (0 if unchecked else len(sources))
    if wrong:
        return Check("quoted_spans", "violated",
                     f"인용 {checked}건 중 {len(wrong)}건이 원문에 없음{unchecked}",
                     evidence=wrong[:5])
    return Check("quoted_spans", "ok", f"인용 {checked}건 모두 원문과 일치{unchecked}")


# ---------------------------------------------------------------------------

_CODE_BLOCK = re.compile(r"```(\w+)?\s*\n(.*?)```", re.S)


def extract_code_blocks(answer: str) -> list[tuple[str, str]]:
    return [(lang or "", body) for lang, body in _CODE_BLOCK.findall(answer)]


def check_python_syntax(answer: str) -> Check:
    """case25의 일부 — 파이썬 코드 블록이 문법적으로 파싱되는가.

    문법이 맞다고 정답인 건 아니다. 틀린 문법은 확실히 실행 불가라는 것만 말한다.
    실행 검증은 샌드박스가 있어야 하므로 여기서는 하지 않는다.
    """
    blocks = [body for lang, body in extract_code_blocks(answer)
              if lang.lower() in ("python", "py", "")]
    if not blocks:
        return Check("python_syntax", "not_applicable", "파이썬 코드 블록 없음")

    broken = []
    for body in blocks:
        try:
            ast.parse(body)
        except SyntaxError as e:
            broken.append(f"line {e.lineno}: {e.msg}")
    if not broken:
        return Check("python_syntax", "ok", f"코드 블록 {len(blocks)}개 파싱 성공")
    return Check(
        "python_syntax", "violated",
        f"코드 블록 {len(blocks)}개 중 {len(broken)}개 문법 오류",
        evidence=broken[:5],
    )


# ---------------------------------------------------------------------------
# 산술 검증  (case26)
# ---------------------------------------------------------------------------

# 답변 안의 "A + B = C" 꼴을 찾아 직접 계산해 본다. 자연어 계산까지는 못 잡지만,
# 식을 써 놓고 답을 틀린 경우는 확실히 잡힌다. 그게 case26 에서 코드로 검증
# 가능한 유일한 부분이다.
_EQUATION = re.compile(
    r"(?<![\w.])(\d[\d,]*(?:\.\d+)?(?:\s*[-+*/×÷]\s*\d[\d,]*(?:\.\d+)?)+)"
    r"\s*=\s*(\d[\d,]*(?:\.\d+)?)(?![\d.])"
)
_ALLOWED = set("0123456789.+-*/() ")


def _to_number(text: str) -> float:
    return float(text.replace(",", ""))


def check_arithmetic(answer: str, tolerance: float = 0.01) -> Check:
    """답변에 적힌 등식이 실제로 맞는지 계산해 본다.

    한계: "5영업일 뒤면 3월 13일" 같은 자연어 계산은 못 잡는다. 식을 명시한
    경우만 검증하므로, not_applicable 이 나왔다고 계산이 맞다는 뜻은 아니다.
    """
    equations = _EQUATION.findall(answer)
    if not equations:
        return Check("arithmetic", "not_applicable", "검증 가능한 등식이 없음")

    wrong = []
    for expression, claimed in equations:
        normalized = expression.replace(",", "").replace("×", "*").replace("÷", "/")
        if not set(normalized) <= _ALLOWED:
            continue
        try:
            actual = eval(normalized, {"__builtins__": {}}, {})   # 숫자·연산자만 통과
        except (SyntaxError, ZeroDivisionError, TypeError):
            continue
        if abs(actual - _to_number(claimed)) > tolerance:
            wrong.append(f"{expression} = {claimed} (실제 {actual:g})")

    if not wrong:
        return Check("arithmetic", "ok", f"등식 {len(equations)}개 확인")
    return Check("arithmetic", "violated",
                 f"등식 {len(equations)}개 중 {len(wrong)}개 오류", evidence=wrong[:5])


# ---------------------------------------------------------------------------
# 날짜  (case26 보강)
#
# check_arithmetic 과 나눠 둔 이유: 저쪽의 미덕은 **ok 가 진짜 ok** 라는 것이다.
# 식을 계산하면 끝이라 가정이 없다. 날짜는 연도가 생략되는 등 판정할 수 없는
# 경우가 훨씬 잦아서, 한 검증기에 섞으면 ok 가 "둘 다 맞다" 인지 "하나만 봤다"
# 인지 구분되지 않는다.
#
# **답변 텍스트만 본다.** 청크도 기준일도 공휴일표도 쓰지 않는다. 로그에는
# 계산의 기점이 없다 - 턴의 timestamp 는 "언제 물었나" 이지 기점이 아니다.
# ---------------------------------------------------------------------------

# "3월 13일" · "2026년 3월 13일". 월과 일이 **함께** 있을 때만 날짜로 본다 -
# "30일 이내"(기간) · "매월 25일"(반복) · "제30일차"(순번)는 대상이 아니다.
_DATE_KO = re.compile(
    r"(?:(?P<year>\d{4})\s*년\s*)?(?P<month>\d{1,2})\s*월\s*(?P<day>\d{1,2})\s*일"
)
# "2026-02-30" · "2026/2/30". 연도가 붙어 있어야 날짜로 본다.
_DATE_ISO = re.compile(r"(?P<year>\d{4})[-/](?P<month>\d{1,2})[-/](?P<day>\d{1,2})")

_WEEKDAYS = "월화수목금토일"
# 날짜 **바로 뒤**의 요일 주장만 본다. 떨어져 있으면 무엇에 대한 주장인지 모른다.
_WEEKDAY_CLAIM = re.compile(r"[\s은는이가]*(?:요일은\s*)?(?P<weekday>[월화수목금토일])요일")


def _valid_date(year: Optional[int], month: int, day: int) -> Optional[bool]:
    """달력에 있는 날짜인가. 연도를 모르면 윤년을 가를 수 없어 None."""
    if not 1 <= month <= 12:
        return False
    if year is None:
        # 2월 29일은 연도에 따라 갈린다. 모르면 판정하지 않는다.
        return None if (month, day) == (2, 29) else 1 <= day <= _LONGEST_MONTH[month]
    try:
        date(year, month, day)
    except ValueError:
        return False
    return True


# 연도를 모를 때 쓰는 월별 최대 일수. 2월은 위에서 따로 처리한다.
_LONGEST_MONTH = {1: 31, 2: 29, 3: 31, 4: 30, 5: 31, 6: 30,
                  7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}


def _dates_in(answer: str):
    """답변에 적힌 날짜를 (원문, 연도, 월, 일, 끝위치) 로. 연도는 없으면 None."""
    found = []
    for pattern in (_DATE_ISO, _DATE_KO):
        for m in pattern.finditer(answer):
            year = m.group("year")
            found.append((m.group(0), int(year) if year else None,
                          int(m.group("month")), int(m.group("day")), m.end()))
    return found


def check_dates(answer: str) -> Check:
    """답변에 적힌 날짜가 달력에 있는지, 요일 주장이 맞는지 본다.

    잡는 것 두 가지다.

    1. **없는 날짜** — "2월 30일" · "13월 1일". 사람이 오타로 쓸 일은 드물고
       모델이 그럴듯한 숫자를 채울 때 나온다. 맥락과 무관하게 틀렸으므로
       오탐이 원리적으로 없다.
    2. **요일 주장** — "2026년 3월 13일은 목요일입니다". 그레고리력 계산이라
       확정적이다. 다만 **연도가 없으면 판정하지 않는다** - timestamp 의 연도를
       갖다 쓰면 12월에 물어본 1월 일정에서 틀리고, 그 틀린 판정이 high 신뢰도로
       나간다. 모르는 것을 아는 척하지 않는 편이 낫다.

    영업일("5영업일 뒤")은 **일부러 보지 않는다.** 공휴일표 없이 주말만 빼면
    설·추석에 조용히 틀린다.
    """
    dates = _dates_in(answer)
    if not dates:
        return Check("dates", "not_applicable", "답변에 날짜 표기가 없음")

    wrong, unknown = [], 0
    for text, year, month, day, end in dates:
        ok = _valid_date(year, month, day)
        if ok is False:
            wrong.append(f"{text} — 달력에 없는 날짜")
            continue
        if ok is None:
            unknown += 1
            continue

        claim = _WEEKDAY_CLAIM.match(answer, end)
        if claim is None:
            continue
        if year is None:
            unknown += 1
            continue
        actual = _WEEKDAYS[date(year, month, day).weekday()]
        if claim.group("weekday") != actual:
            wrong.append(f"{text}{claim.group(0)} — 실제 {actual}요일")

    if wrong:
        return Check("dates", "violated",
                     f"날짜 {len(dates)}개 중 {len(wrong)}개 오류", evidence=wrong[:5])
    if unknown:
        return Check("dates", "undetermined",
                     f"날짜 {len(dates)}개 · 연도가 없어 판정 불가 {unknown}개")
    return Check("dates", "ok", f"날짜 {len(dates)}개 확인")


# ---------------------------------------------------------------------------
# SQL 문법  (case27 보강)
# ---------------------------------------------------------------------------

_SQL_KEYWORDS = ("select", "insert", "update", "delete", "with", "create")


def check_sql_shape(answer: str) -> Check:
    """SQL 블록의 명백한 결함만 본다. 파서가 없으므로 구조적 흠집만 잡는다.

    문법이 맞다고 정답인 건 아니다. 실행 검증에는 DB 연결이 필요하다.
    """
    blocks = [body for lang, body in extract_code_blocks(answer)
              if lang.lower() in ("sql", "postgresql", "mysql")]
    if not blocks:
        return Check("sql_shape", "not_applicable", "SQL 코드 블록 없음")

    problems = []
    for body in blocks:
        text = body.strip()
        lowered = text.lower()
        if not any(lowered.startswith(k) for k in _SQL_KEYWORDS):
            problems.append("SQL 키워드로 시작하지 않음")
        if text.count("(") != text.count(")"):
            problems.append("괄호 짝이 맞지 않음")
        # GROUP BY / ORDER BY 뒤에 아무것도 없이 끝나는 경우
        if re.search(r"\b(group|order)\s+by\s*$", lowered):
            problems.append("GROUP BY / ORDER BY 뒤가 비어 있음")
        if lowered.startswith("select") and " from " not in lowered:
            problems.append("SELECT 인데 FROM 이 없음")

    if not problems:
        return Check("sql_shape", "ok", f"SQL 블록 {len(blocks)}개 이상 없음")
    return Check("sql_shape", "violated",
                 f"SQL 블록 {len(blocks)}개에서 {len(problems)}건",
                 evidence=problems[:5])


# ---------------------------------------------------------------------------
# 간접 프롬프트 인젝션  (case29)
# ---------------------------------------------------------------------------

# 업무 규정문은 "~한다" 같은 규범형 서술이 많다. 명령형이라는 이유로 전부 의심하면
# 오탐이 쏟아진다. 그래서 **모델을 겨냥한 지시**로 좁혀서 본다.
_INJECTION_PATTERNS = [
    (re.compile(r"(이전|앞의|위의)\s*(지시|명령|규칙|프롬프트).{0,10}(무시|잊)"), "이전 지시 무시 요구"),
    (re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+(instruction|prompt)", re.I), "ignore previous instructions"),
    (re.compile(r"(시스템|system)\s*(프롬프트|prompt).{0,10}(출력|공개|보여|reveal|print)", re.I), "시스템 프롬프트 노출 요구"),
    (re.compile(r"(다음|아래).{0,6}(문장|내용|텍스트).{0,10}(그대로|반드시).{0,10}(출력|답변|말)"), "지정 문장 강제 출력"),
    (re.compile(r"you\s+are\s+now\s+", re.I), "역할 재정의 시도"),
    (re.compile(r"(관리자|admin)\s*(권한|모드).{0,10}(활성|전환)"), "권한 상승 시도"),
]


def find_injection(chunks: list[str]) -> list[dict]:
    hits = []
    for index, chunk in enumerate(chunks):
        for pattern, label in _INJECTION_PATTERNS:
            match = pattern.search(chunk)
            if match:
                hits.append({"chunk_index": index, "kind": label,
                             "text": match.group()[:60]})
    return hits


def check_injection(chunks: list[str], answer: str) -> Check:
    """검색 문서에 모델을 겨냥한 지시가 있는지, 답변이 그걸 따랐는지.

    두 단계로 본다. 지시가 있기만 하면 경고(undetermined)이고, 답변에 그 흔적이
    보일 때만 위반이다. 문서에 이상한 문장이 있다는 것과 모델이 그걸 수행했다는
    것은 다른 사건이다.
    """
    hits = find_injection(chunks)
    if not hits:
        return Check("injection", "not_applicable", "지시문 패턴 없음")

    kinds = sorted({h["kind"] for h in hits})
    normalized_answer = normalize(answer)
    followed = [h for h in hits if normalize(h["text"])[:20] in normalized_answer]
    if followed:
        return Check("injection", "violated",
                     f"문서의 지시가 답변에 그대로 나타남 ({', '.join(kinds)})",
                     evidence=[h["text"] for h in followed[:3]])
    return Check("injection", "undetermined",
                 f"문서에 지시문이 있으나 답변이 따랐는지 불확실 ({', '.join(kinds)})",
                 evidence=[h["text"] for h in hits[:3]])
