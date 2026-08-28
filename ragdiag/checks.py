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
from dataclasses import dataclass, field
from typing import Literal, Optional

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
    구분하지 못한다. 사내 챗봇에서 실제로 갈리는 건 한국어와 영어라 이 수준이면 된다.
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
# 길이  (case11)
# ---------------------------------------------------------------------------

# "짧게 답해줘"처럼 수치가 없는 요구를 판정하기 위한 기준값.
# 임의로 정한 값이므로 실데이터로 보정해야 한다. 그때 LLM을 다시 돌리지 않아도 되도록
# 판정 결과에 실제 측정값을 함께 남긴다.
VAGUE_SHORT_MAX_CHARS = 400

_SENTENCE_END = re.compile(r"[.!?。]|다\.|요\.")


def count_sentences(text: str) -> int:
    parts = [p for p in re.split(r"(?<=[.!?。])\s+", text.strip()) if p.strip()]
    return len(parts)


@dataclass
class LengthRequest:
    """Step 1이 뽑아내는 길이 요구. 수치가 없으면 kind='vague_short'."""

    kind: Literal["max_chars", "max_sentences", "max_lines", "vague_short"]
    value: Optional[int] = None


def check_length(answer: str, requested: Optional[LengthRequest]) -> Check:
    """case11 — 짧게/N자 이내를 요구했는데 지키지 않음."""
    if requested is None:
        return Check("length", "not_applicable", "길이 요구 없음")

    chars = len(answer.strip())
    lines = len([l for l in answer.splitlines() if l.strip()])
    sentences = count_sentences(answer)
    measured = f"{chars}자 · {sentences}문장 · {lines}줄"

    if requested.kind == "vague_short":
        over = chars > VAGUE_SHORT_MAX_CHARS
        return Check(
            "length",
            "violated" if over else "ok",
            f"모호한 짧게 요구 (기준 {VAGUE_SHORT_MAX_CHARS}자) · {measured}",
        )

    if requested.value is None:
        return Check("length", "undetermined", f"수치 요구인데 값이 없음 · {measured}")

    actual = {"max_chars": chars, "max_sentences": sentences, "max_lines": lines}[
        requested.kind
    ]
    over = actual > requested.value
    return Check(
        "length",
        "violated" if over else "ok",
        f"요구 {requested.kind} ≤ {requested.value} · 실제 {actual} · {measured}",
    )


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


def check_truncated(answer: str) -> Check:
    """case8 — 답변이 문장 중간에서 끊김.

    한계: 정상 답변도 목록 항목이나 표로 끝나면 종결 부호가 없다. 그래서 그런
    구조를 먼저 걸러낸 뒤에만 잘림으로 본다. finish_reason 필드가 생기면
    이 휴리스틱은 필요 없어진다.
    """
    text = answer.rstrip()
    if not text:
        return Check("truncated", "undetermined", "답변이 비어 있음")

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
        return Check("truncated", "ok", f"종결 부호로 끝남: {text[-1]!r}")

    return Check("truncated", "violated", f"종결 부호 없이 끝남: …{text[-20:]!r}")


# ---------------------------------------------------------------------------
# 서비스 자원 부족 응답
#
# 모델 자원을 확보하지 못했을 때 서비스 계층이 내보내는 **확정 문구**다. LLM 이
# 생성한 답이 아니라 정해진 문자열이므로 코드로 판정한다 - 신뢰도 high 다.
#
# 이걸 따로 잡지 않으면 판정자가 "답변이 거절했다"로 읽어 case27(보안 정책상
# 답변 불가)으로 간다. 실제로 그 오분류가 많이 나왔다. 서버 자원 문제를 보안
# 정책 문제로 세면 고칠 곳을 정반대로 가리킨다 - 한쪽은 인프라 증설이고
# 다른 쪽은 권한 정책이다.
#
# 문구는 배포마다 다르므로 아래 목록에 줄을 추가해 쓴다. 공백 차이는 무시한다.
# ---------------------------------------------------------------------------

SERVICE_ERROR_TEMPLATES = (
    "서비스에 문제가 있거나, 사용자 분들이 많아서 서버에 부하가 걸리고 있어요",
)

# 템플릿이 조금 바뀌어도 놓치지 않도록 두는 보조 표지. 단독으로는 쓰지 않고
# 두 개 이상 겹칠 때만 인정한다 - "서버" 한 단어로 잡으면 서버 관련 질문에
# 정상적으로 답한 것까지 오탐한다.
_SERVICE_ERROR_MARKERS = (
    "서버에 부하", "사용자 분들이 많아", "일시적인 오류", "잠시 후 다시",
    "요청이 많아", "응답을 생성하지 못", "서비스에 문제가",
)

# 확정 문구는 짧고, 그 문구가 답변의 전부다. 길면 서버 장애를 '주제로' 답한
# 정상 답변일 가능성이 높다. 길이로 한 번 더 거른다.
MAX_SERVICE_ERROR_LEN = 400


def _squeeze(text: str) -> str:
    return re.sub(r"\s+", "", text)


def check_service_error(answer: str) -> Check:
    """답변이 서비스 자원 부족 안내 문구인가."""
    if not answer.strip():
        return Check("service_error", "not_applicable", "답변이 비어 있음")

    packed = _squeeze(answer)
    for template in SERVICE_ERROR_TEMPLATES:
        if _squeeze(template) in packed:
            return Check("service_error", "violated",
                         f"서비스 자원 부족 확정 문구와 일치: {template[:30]}…")

    if len(packed) > MAX_SERVICE_ERROR_LEN:
        return Check("service_error", "ok",
                     f"확정 문구 없음 · 답변이 길어({len(packed)}자) 안내 문구가 아님")

    hits = [m for m in _SERVICE_ERROR_MARKERS if _squeeze(m) in packed]
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
# 답변 속 인용 대조  (case23)
# ---------------------------------------------------------------------------

# verify.py 와 방향이 반대다. 거기서는 판정자의 인용을 검증하고, 여기서는
# 챗봇 답변이 문서에서 가져왔다고 제시한 문장을 검증한다. 같은 대조 함수를 쓴다.
_QUOTE_PATTERNS = [
    re.compile(r"[“\"]([^”\"]{10,200})[”\"]"),
    re.compile(r"[「『]([^」』]{10,200})[」』]"),
]
MIN_QUOTE_CHARS = 10


def extract_quotes(answer: str) -> list[str]:
    quotes = []
    for pattern in _QUOTE_PATTERNS:
        quotes += [q.strip() for q in pattern.findall(answer)]
    return [q for q in quotes if len(normalize(q)) >= MIN_QUOTE_CHARS]


def check_quoted_spans(answer: str, chunks: list[str], threshold: float = 0.9) -> Check:
    """case21의 검증 가능한 부분 — 답변이 인용부호로 제시한 문장이 실제 문서에 있는가.

    한계: 문서명·조항번호 같은 출처 표기는 검증할 수 없다. 청크에 문서 메타데이터가
    없기 때문이다(case20와 같은 이유로 파킹). 여기서 잡는 것은 '문서에서 가져온 척한
    문장'뿐이고, 그게 case21에서 실제로 확인 가능한 유일한 부분이다.
    """
    quotes = extract_quotes(answer)
    if not quotes:
        return Check("quoted_spans", "not_applicable", "답변에 인용된 문장이 없음")
    if not chunks:
        return Check("quoted_spans", "undetermined", "대조할 문서가 없음")

    missing = [q for q in quotes if max(match_ratio(q, c) for c in chunks) < threshold]
    if not missing:
        return Check("quoted_spans", "ok", f"인용 {len(quotes)}건 모두 원문과 일치")
    return Check(
        "quoted_spans",
        "violated",
        f"인용 {len(quotes)}건 중 {len(missing)}건이 원문에 없음",
        evidence=missing[:5],
    )


# ---------------------------------------------------------------------------
# 코드 답변  (case26, 파이썬만)
# ---------------------------------------------------------------------------

_CODE_BLOCK = re.compile(r"```(\w+)?\s*\n(.*?)```", re.S)


def extract_code_blocks(answer: str) -> list[tuple[str, str]]:
    return [(lang or "", body) for lang, body in _CODE_BLOCK.findall(answer)]


def check_python_syntax(answer: str) -> Check:
    """case24의 일부 — 파이썬 코드 블록이 문법적으로 파싱되는가.

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
# 산술 검증  (case25)
# ---------------------------------------------------------------------------

# 답변 안의 "A + B = C" 꼴을 찾아 직접 계산해 본다. 자연어 계산까지는 못 잡지만,
# 식을 써 놓고 답을 틀린 경우는 확실히 잡힌다. 그게 case25 에서 코드로 검증
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
# SQL 문법  (case26 보강)
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
# 간접 프롬프트 인젝션  (case28)
# ---------------------------------------------------------------------------

# 사내 규정문은 "~한다" 같은 규범형 서술이 많다. 명령형이라는 이유로 전부 의심하면
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
