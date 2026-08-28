"""판정 백엔드 - Claude Code CLI와 Anthropic API.

두 경로의 결정적 차이는 **스키마를 누가 강제하느냐**다.

- API(`messages.parse`)는 서버가 스키마를 강제한다. 형식이 어긋난 응답은 나올 수 없다.
- CLI(`claude -p`)는 그런 장치가 없다. 프롬프트가 출력 계약을 짊어지고, 여기서
  JSON을 뽑아 Pydantic으로 검증하고, 실패하면 검증 오류를 되먹여 한 번 재시도한다.

그래서 CLI 경로에서는 프롬프트에 output_contract()가 붙는다(prompts.py).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Optional, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

# 판정에는 도구가 필요 없다. 도구를 쓰기 시작하면 판정이 아니라 조사가 된다.
DISALLOWED_TOOLS = (
    "Bash,Read,Write,Edit,Glob,Grep,WebSearch,WebFetch,Task,TodoWrite,"
    "NotebookEdit,Agent,Skill,Artifact"
)


class JudgeError(RuntimeError):
    pass


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cost_usd += other.cost_usd


# Qwen3 계열 등 하이브리드 추론 모델은 응답 앞에 추론 블록을 붙인다.
# 그 안에 중괄호가 들어있으면 JSON 추출이 엉뚱한 걸 집는다.
_REASON_CLOSE = ("</think>", "</thinking>")


def strip_reasoning(text: str) -> str:
    """추론 블록을 걷어낸다.

    닫는 태그 뒤를 취한다. 여는 태그를 찾지 않는 이유는, 채팅 템플릿이 "<think>"를
    미리 넣어주는 경우 모델 출력이 추론 도중부터 시작해서 닫는 태그만 나오기 때문이다.
    마지막 닫는 태그 기준이라 중첩이나 반복도 함께 처리된다.
    """
    cut = -1
    for tag in _REASON_CLOSE:
        index = text.rfind(tag)
        if index >= 0:
            cut = max(cut, index + len(tag))
    return text[cut:].strip() if cut >= 0 else text


def extract_json(text: str) -> str:
    """모델 응답에서 JSON 객체를 뽑아낸다.

    추론 블록을 먼저 걷어낸 뒤, 첫 '{'부터 짝이 맞는 '}'까지를 잘라낸다.
    문자열 리터럴 안의 중괄호는 세지 않는다. 단순히 rfind('}')를 쓰면 뒤에 붙은
    산문 때문에 깨지고, 추론 블록을 안 걷으면 그 안의 중괄호를 집는다.
    """
    text = strip_reasoning(text)
    start = text.find("{")
    if start < 0:
        raise ValueError(f"JSON 객체를 찾을 수 없음: {text[:200]!r}")

    depth, in_string, escaped = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise ValueError(f"중괄호 짝이 맞지 않음: {text[:200]!r}")


def _repair_note(error: Exception, contract_hint: str) -> str:
    return (
        f"\n\n---\n직전 응답이 형식 검증에 실패했다:\n{error}\n\n"
        f"{contract_hint}\n다른 텍스트 없이 JSON 객체 하나만 다시 출력해라."
    )


def parse_with_repair(call, user: str, out_model: type[T], contract_hint: str,
                      max_attempts: int = 2) -> tuple[T, Usage]:
    """call(message) -> (raw_text, Usage)를 돌리고, 형식이 어긋나면 오류를 되먹여 재시도.

    서버가 스키마를 강제하지 못하는 백엔드에서 마지막 방어선이 된다.
    검증 오류를 그대로 되돌려주는 게 핵심이다 - "다시 해봐"보다 훨씬 잘 고친다.
    """
    total = Usage()
    message, last_error = user, None
    for _ in range(max_attempts):
        raw, usage = call(message)
        total.add(usage)
        try:
            return out_model.model_validate_json(extract_json(raw)), total
        except (ValidationError, ValueError) as e:
            last_error = e
            message = user + _repair_note(e, contract_hint)
    raise JudgeError(f"{max_attempts}회 시도 후에도 형식 검증 실패: {last_error}")


def strict_json_schema(model: type[BaseModel]) -> dict:
    """구조화 출력 강제용 스키마. 모든 객체에 additionalProperties:false를 박는다.

    이게 없으면 서버에 따라 모델이 필드를 덧붙여도 통과시킨다.
    """
    schema = model.model_json_schema()

    def tighten(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                node["additionalProperties"] = False
            for value in node.values():
                tighten(value)
        elif isinstance(node, list):
            for value in node:
                tighten(value)

    tighten(schema)
    return schema


class ClaudeCodeBackend:
    """`claude -p`를 통해 판정한다. API 키 없이 동작한다.

    사용량 주의: Claude Code의 기본 시스템 프롬프트(약 12k 토큰)가 매 호출에 실린다.
    캐시가 더워진 뒤에도 호출당 list 환산 $0.05 안팎으로, API 경로보다 3배쯤 무겁다.

    Usage.cost_usd는 CLI의 total_cost_usd(costBasis="list")를 그대로 담는다.
    구독 인증으로 붙으면 이건 청구액이 아니라 사용량 지표다. 진짜 제약은 요금이 아니라
    5시간/주간 사용량 한도이고, 한도를 넘으면 과금이 아니라 요청이 거절된다.
    """

    def __init__(
        self,
        model: str = "claude-opus-5",
        cli_path: Optional[str] = None,
        timeout: int = 300,
    ):
        self.model = model
        self.cli = cli_path or shutil.which("claude") or os.path.expanduser(
            "~/.local/bin/claude"
        )
        if not os.path.exists(self.cli):
            raise JudgeError(
                f"claude CLI를 찾을 수 없습니다: {self.cli}\n"
                "  --cli-path 로 경로를 지정하거나 PATH에 추가하세요."
            )
        self.timeout = timeout

    def _run(self, system: str, user: str) -> tuple[str, Usage]:
        cmd = [
            self.cli, "-p", user,
            "--output-format", "json",
            "--model", self.model,
            "--system-prompt", system,
            "--disallowedTools", DISALLOWED_TOOLS,
            "--exclude-dynamic-system-prompt-sections",
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.timeout
            )
        except subprocess.TimeoutExpired as e:
            raise JudgeError(f"CLI 타임아웃 ({self.timeout}s)") from e
        if proc.returncode != 0:
            raise JudgeError(f"CLI 종료코드 {proc.returncode}: {proc.stderr[:300]}")

        try:
            envelope = json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            raise JudgeError(f"CLI 응답을 파싱하지 못함: {proc.stdout[:300]!r}") from e
        if envelope.get("is_error"):
            raise JudgeError(f"CLI 오류: {envelope.get('result', '')[:300]}")

        u = envelope.get("usage", {})
        return envelope.get("result", ""), Usage(
            input_tokens=u.get("input_tokens", 0)
            + u.get("cache_read_input_tokens", 0)
            + u.get("cache_creation_input_tokens", 0),
            output_tokens=u.get("output_tokens", 0),
            cost_usd=envelope.get("total_cost_usd", 0.0),
        )

    def complete(
        self, system: str, user: str, out_model: type[T], contract_hint: str = ""
    ) -> tuple[T, Usage]:
        return parse_with_repair(
            lambda msg: self._run(system, msg), user, out_model, contract_hint
        )


class ApiBackend:
    """Anthropic SDK를 통해 판정한다. 서버가 스키마를 강제하므로 재시도가 필요 없다."""

    def __init__(self, model: str = "claude-opus-5", effort: str = "high",
                 use_fallbacks: bool = True, client=None):
        import anthropic

        self.anthropic = anthropic
        self.model = model
        self.effort = effort
        self.use_fallbacks = use_fallbacks
        self.client = client or anthropic.Anthropic()

    def complete(
        self, system: str, user: str, out_model: type[T], contract_hint: str = ""
    ) -> tuple[T, Usage]:
        anthropic = self.anthropic
        kwargs = dict(
            model=self.model, max_tokens=16000, system=system,
            messages=[{"role": "user", "content": user}],
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
            output_format=out_model,
        )
        if self.use_fallbacks:
            kwargs.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        try:
            response = self.client.beta.messages.parse(**kwargs)
        except anthropic.NotFoundError as e:
            raise JudgeError(f"모델/엔드포인트를 찾을 수 없음: {e}") from e
        except anthropic.RateLimitError as e:
            raise JudgeError(f"rate limit: {e}") from e
        except anthropic.APIStatusError as e:
            raise JudgeError(f"API 오류 {e.status_code}: {e}") from e
        except anthropic.APIConnectionError as e:
            raise JudgeError(f"연결 실패: {e}") from e

        if response.stop_reason == "refusal":
            raise JudgeError(
                f"판정자가 거절함 (category={getattr(response.stop_details, 'category', None)})"
            )
        if response.parsed_output is None:
            raise JudgeError("구조화 출력을 파싱하지 못함")
        return response.parsed_output, Usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )


# ---------------------------------------------------------------------------
# 로컬 LLM (OpenAI 호환 HTTP)
#
# 에어갭 장비용. vLLM / Ollama / TGI / LM Studio / llama.cpp 모두 이 규약을 쓴다.
# 표준 라이브러리만으로 붙는다 - 반입할 패키지를 늘리지 않기 위해서다.
# ---------------------------------------------------------------------------

# 구조화 출력을 강제하는 방식이 서버마다 다르다. 에어갭이라 미리 물어볼 수 없으므로
# 첫 호출에서 위에서부터 시도해 통하는 것을 찾고, 이후로는 그것만 쓴다.
JSON_MODES = ("json_schema", "guided_json", "json_object", "none")

# 협상 탐침. 스키마와 무관한 요청이라, 강제가 실제로 걸렸을 때만 스키마에 맞는 응답이 나온다.
_PROBE_SYSTEM = "You are a connectivity test endpoint."
_PROBE_USER = "Reply with a JSON object."


def normalize_base_url(url: str) -> str:
    """붙여넣기로 흔히 생기는 형태를 모두 같은 기준으로 맞춘다.

    http://h:8000, http://h:8000/, http://h:8000/v1,
    http://h:8000/v1/chat/completions -> 전부 http://h:8000

    이걸 안 하면 /v1이 두 번 붙어 404가 나는데, 에어갭 장비에서 이런 걸로
    시간을 버리게 만들 이유가 없다.
    """
    url = url.strip().rstrip("/")
    for suffix in ("/chat/completions", "/completions", "/v1"):
        if url.endswith(suffix):
            url = url[: -len(suffix)].rstrip("/")
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    return url


class OpenAICompatBackend:
    """로컬 LLM 서버에 /v1/chat/completions로 붙는다.

    30B급에서는 구조화 출력 강제가 사실상 필수다. 프롬프트만으로 JSON을 요구하면
    재시도가 잦아지고, 재시도가 잦으면 판정 품질도 흔들린다.

    강제가 아예 안 되는 서버라면 mode="none"으로 떨어지고, 그때는
    parse_with_repair가 유일한 방어선이 된다.
    """

    def __init__(
        self,
        base_url: str,
        model: Optional[str] = None,   # 없으면 /v1/models 로 알아낸다
        api_key: str = "EMPTY",       # 로컬 서버는 보통 검사하지 않지만 헤더는 요구한다
        timeout: int = 600,           # 로컬 서빙은 느릴 수 있다
        json_mode: str = "auto",
        # 추론 모드가 켜져 있으면 생각에만 수천 토큰을 쓴다. 잘리면 JSON이 안 나온다.
        max_tokens: int = 16000,
        temperature: float = 0.0,     # 판정자는 결정적이어야 한다
        max_attempts: int = 3,
        thinking: str = "auto",       # auto: 서버 기본값 그대로 / on / off
    ):
        if thinking not in ("auto", "on", "off"):
            raise JudgeError(f"알 수 없는 thinking: {thinking}")
        if json_mode not in ("auto",) + JSON_MODES:
            raise JudgeError(f"알 수 없는 json_mode: {json_mode}")
        self.base_url = normalize_base_url(base_url)
        self.api_key = api_key
        self.timeout = timeout
        # 사용자가 줘야 할 것을 URL과 키 두 개로 줄인다. 모델은 서버가 알고 있다.
        self.model = model or self.discover_model()
        self.discovered = model is None
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_attempts = max_attempts
        self.thinking = thinking
        self.negotiation_log: list[str] = []   # 어떤 모드가 왜 탈락했는지
        self._mode: Optional[str] = None if json_mode == "auto" else json_mode
        self._lock = threading.Lock()

    # -- HTTP ---------------------------------------------------------------

    def _get(self, path: str) -> dict:
        url = f"{self.base_url}{path}"
        request = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {self.api_key}"}, method="GET"
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise JudgeError(f"HTTP {e.code} {url}") from e
        except urllib.error.URLError as e:
            raise JudgeError(f"{url} 에 연결하지 못했습니다: {e.reason}") from e

    def discover_model(self) -> str:
        """서버가 서빙 중인 모델 이름을 알아낸다."""
        try:
            data = self._get("/v1/models").get("data") or []
        except JudgeError as e:
            raise JudgeError(
                f"모델 목록을 가져오지 못했습니다.\n  {e}\n"
                "  --model 로 직접 지정하면 이 조회를 건너뜁니다."
            ) from e
        names = [m.get("id") for m in data if m.get("id")]
        if not names:
            raise JudgeError(
                f"{self.base_url}/v1/models 가 모델을 하나도 알려주지 않습니다.\n"
                "  --model 로 직접 지정하세요."
            )
        return names[0]

    def list_models(self) -> list[str]:
        return [m.get("id") for m in (self._get("/v1/models").get("data") or []) if m.get("id")]

    def _post(self, payload: dict) -> dict:
        url = f"{self.base_url}/v1/chat/completions"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:400]
            raise JudgeError(f"HTTP {e.code} {url}\n  {body}") from e
        except urllib.error.URLError as e:
            raise JudgeError(f"{url} 에 연결하지 못했습니다: {e.reason}") from e
        except TimeoutError as e:
            raise JudgeError(f"{self.timeout}초 안에 응답이 없습니다") from e

    def _payload(self, system: str, user: str, out_model: type[T], mode: str) -> dict:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.thinking != "auto":
            # Qwen3 계열 채팅 템플릿 스위치. 서버가 모르면 400이 나므로 명시할 때만 보낸다.
            payload["chat_template_kwargs"] = {"enable_thinking": self.thinking == "on"}
        if mode == "json_schema":       # OpenAI 규격, vLLM 최신 / TGI
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": out_model.__name__,
                    "schema": strict_json_schema(out_model),
                    "strict": True,
                },
            }
        elif mode == "guided_json":     # vLLM 고유
            payload["guided_json"] = strict_json_schema(out_model)
        elif mode == "json_object":     # 스키마 없이 JSON만 강제
            payload["response_format"] = {"type": "json_object"}
        return payload

    @staticmethod
    def _text_and_usage(response: dict) -> tuple[str, Usage]:
        try:
            message = response["choices"][0]["message"]
        except (KeyError, IndexError) as e:
            raise JudgeError(f"예상 밖 응답 형태: {str(response)[:300]}") from e
        text = message.get("content") or ""
        if not text.strip():
            # vLLM --reasoning-parser 가 켜져 있으면 추론은 reasoning_content로 가고
            # content가 빌 수 있다. 추론만 하고 답을 못 낸 경우다.
            if message.get("reasoning_content"):
                raise JudgeError(
                    "모델이 추론만 하고 답을 내지 못했습니다. "
                    "max_tokens를 늘리거나 --thinking off 를 시도하세요."
                )
            raise JudgeError(f"빈 응답: {str(response)[:200]}")
        u = response.get("usage") or {}
        return text or "", Usage(
            input_tokens=u.get("prompt_tokens", 0),
            output_tokens=u.get("completion_tokens", 0),
        )

    # -- 모드 협상 -----------------------------------------------------------

    def _enforces(self, mode: str, text: str, out_model: type[T]) -> bool:
        """이 모드가 실제로 출력을 강제했는지 확인한다.

        200 OK만 보고 판단하면 안 된다. LiteLLM 같은 프록시는 모르는 파라미터를
        400으로 거절하지 않고 조용히 버린다. 그러면 강제가 안 걸렸는데 걸린 줄 알고
        재시도를 1회로 줄여버려서, 첫 응답이 어긋나는 순간 그냥 실패한다.

        탐침은 스키마와 무관한 요청이라, 강제가 걸렸을 때만 스키마에 맞는 응답이 나온다.
        """
        if mode == "none":
            return True                      # 마지막 수단. 강제를 기대하지 않는다.
        if mode == "json_object":
            try:                             # 이 모드가 약속하는 건 "유효한 JSON"까지다
                return isinstance(json.loads(extract_json(text)), dict)
            except (ValueError, TypeError):
                return False
        try:                                 # json_schema / guided_json 은 스키마까지 약속한다
            out_model.model_validate_json(extract_json(text))
            return True
        except Exception:
            return False

    def negotiate(self, out_model: type[T]) -> str:
        """서버가 실제로 강제해 주는 구조화 출력 방식을 한 번만 알아낸다."""
        with self._lock:
            if self._mode:
                return self._mode
            errors = []
            for mode in JSON_MODES:
                try:
                    response = self._post(
                        self._payload(_PROBE_SYSTEM, _PROBE_USER, out_model, mode)
                    )
                    text, _ = self._text_and_usage(response)
                except JudgeError as e:
                    errors.append(f"{mode}: {str(e).splitlines()[0]}")
                    self.negotiation_log.append(f"{mode:<12} 거절됨 - {str(e).splitlines()[0]}")
                    continue
                if not self._enforces(mode, text, out_model):
                    reason = "200 OK지만 강제가 걸리지 않음 (프록시가 조용히 무시한 듯)"
                    errors.append(f"{mode}: {reason}")
                    self.negotiation_log.append(f"{mode:<12} {reason}")
                    continue
                self.negotiation_log.append(f"{mode:<12} 채택")
                self._mode = mode
                return mode
            hint = ""
            if self.thinking != "auto":
                hint = ("\n  힌트: --thinking 을 보내고 있습니다. 서버가 "
                        "chat_template_kwargs를 모르면 모든 모드가 실패합니다. "
                        "--thinking auto 로 다시 시도해 보세요.")
            raise JudgeError(
                "어떤 방식으로도 서버가 응답하지 않습니다:\n  "
                + "\n  ".join(errors) + hint
            )

    def complete(
        self, system: str, user: str, out_model: type[T], contract_hint: str = ""
    ) -> tuple[T, Usage]:
        mode = self.negotiate(out_model)

        def call(message: str) -> tuple[str, Usage]:
            return self._text_and_usage(
                self._post(self._payload(system, message, out_model, mode))
            )

        # 스키마가 강제되면 한 번에 맞는다. 강제가 없을 때만 재시도가 의미를 갖는다.
        attempts = 1 if mode in ("json_schema", "guided_json") else self.max_attempts
        return parse_with_repair(call, user, out_model, contract_hint, attempts)


# ---------------------------------------------------------------------------
# 환경변수 → 백엔드
#
# 사용자가 줘야 할 것은 URL과 키 두 개뿐이다. 모델은 서버에 물어본다.
# 진입점이 여러 개라 여기 한 곳에 둔다.
# ---------------------------------------------------------------------------

URL_VARS = ("LLM_API_URL", "API_URL", "RAGDIAG_BASE_URL",
            "OPENAI_BASE_URL", "OPENAI_API_BASE")
KEY_VARS = ("LLM_API_KEY", "API_KEY", "RAGDIAG_API_KEY", "OPENAI_API_KEY")
MODEL_VAR = "RAGDIAG_MODEL"


def env_first(names, default=None):
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


def backend_from_env(
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs,
) -> "OpenAICompatBackend":
    url = base_url or env_first(URL_VARS)
    if not url:
        raise JudgeError(
            "LLM 주소가 없습니다. 필요한 건 이 두 개뿐입니다:\n"
            "  export LLM_API_URL=http://<서버>:8000\n"
            "  export LLM_API_KEY=<키>\n"
            f"  (인식하는 이름: {', '.join(URL_VARS)} / {', '.join(KEY_VARS)})"
        )
    return OpenAICompatBackend(
        base_url=url,
        model=model or os.environ.get(MODEL_VAR),
        api_key=api_key or env_first(KEY_VARS, "EMPTY"),
        **kwargs,
    )
