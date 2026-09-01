"""판정 백엔드 — OpenAI 호환 로컬 서버.

여기 있는 것은 **운영 환경에서 실제로 도는 경로 하나뿐이다.** claude CLI 와 Anthropic
API 백엔드는 운영 환경에서 호출이 전부 실패하므로 `tools/` 로 뺐다 (규격 §1.4 · C8).
import 방향은 한쪽이다 — `tools/` 는 여기를 import 하지만 여기는 `tools/` 를
import 하지 않는다. 그래서 그 파일들이 없어도 이 모듈은 그대로 돈다.

두 경로의 결정적 차이는 **스키마를 누가 강제하느냐**다.

- 서버가 강제하는 경우(json_schema·guided_json, API messages.parse): 형식이 어긋난
  응답은 나올 수 없다. 재시도가 필요 없다.
- 강제가 안 되는 경우(json_object·none, 그리고 프록시가 파라미터를 조용히 버릴 때):
  프롬프트가 출력 계약을 짊어지고, 여기서 JSON을 뽑아 Pydantic으로 검증하고,
  실패하면 검증 오류를 되먹여 재시도한다.

그래서 강제가 없는 경로에서는 프롬프트에 output_contract()가 붙는다(prompts.py).
negotiate() 가 서버가 실제로 강제하는지를 탐침으로 확인하는 것도 같은 이유다.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Optional, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class JudgeError(RuntimeError):
    pass


class Truncated(JudgeError):
    """모델이 답을 내기 전에 생성이 끝났다.

    형식 오류와 반드시 구분해야 한다. 형식 오류는 같은 요청을 다시 보내면서
    무엇이 틀렸는지 알려주면 고쳐지지만(parse_with_repair), 잘린 응답은 같은
    요청을 그대로 다시 보내면 **같은 자리에서 또 잘린다**. 조건(추론 스위치,
    토큰 한도)을 바꿔야만 다른 결과가 나온다.
    """

    def __init__(self, message: str, *, finish_reason: Optional[str] = None,
                 usage: Optional["Usage"] = None):
        super().__init__(message)
        self.finish_reason = finish_reason
        # 잘린 요청도 토큰을 다 태웠다. 사용량에서 빠지면 안 된다.
        self.usage = usage or Usage()


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


def _truncated_message(finish_reason: Optional[str], reasoned_chars: int,
                       max_tokens: int) -> str:
    """무엇이 듣는지는 왜 끝났는지에 달렸다. 짐작하지 말고 finish_reason 을 쓴다.

    예전 문구는 어느 경우든 "max_tokens 를 늘리거나"라고 했는데, 한도에 걸린 게
    아닐 때는 늘려도 똑같다. 틀린 조치를 권하면 사이클 하나를 그냥 버린다.
    """
    if finish_reason == "length":
        return (f"토큰 한도({max_tokens:,})를 다 쓰고도 답이 나오지 않았습니다 "
                f"(추론 {reasoned_chars:,}자). 추론이 한도를 먹었습니다.")
    return (f"모델이 추론만 하고 답을 내지 못했습니다 "
            f"(finish_reason={finish_reason}, 추론 {reasoned_chars:,}자). "
            f"한도({max_tokens:,})에 걸린 게 아니라 스스로 멈췄으므로 "
            f"max_tokens 를 늘려도 같습니다.")


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
        # 잘려서 조건을 바꿔 다시 물은 횟수. RUN SUMMARY 로 나간다.
        # negotiate() 가 _lock 을 쥔 채 _attempt 를 부르므로 락을 따로 둔다.
        self.fallbacks: list[str] = []
        self._fallback_lock = threading.Lock()

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

    def _payload(self, system: str, user: str, out_model: type[T], mode: str,
                 *, thinking: Optional[str] = None,
                 max_tokens: Optional[int] = None) -> dict:
        # 폴백 사다리가 이 둘을 갈아끼운다. self.thinking 을 직접 바꾸면 같은
        # 백엔드를 공유하는 다른 스레드의 판정 조건까지 바뀌므로 인자로만 받는다.
        thinking = thinking or self.thinking
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        if thinking != "auto":
            # Qwen3 계열 채팅 템플릿 스위치. 서버가 모르면 400이 나므로 명시할 때만 보낸다.
            payload["chat_template_kwargs"] = {"enable_thinking": thinking == "on"}
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

    def _text_and_usage(self, response: dict,
                        sent_max_tokens: Optional[int] = None) -> tuple[str, Usage]:
        try:
            choice = response["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError) as e:
            raise JudgeError(f"예상 밖 응답 형태: {str(response)[:300]}") from e

        u = response.get("usage") or {}
        usage = Usage(input_tokens=u.get("prompt_tokens", 0),
                      output_tokens=u.get("completion_tokens", 0))
        finish = choice.get("finish_reason")
        text = message.get("content") or ""
        cap = sent_max_tokens or self.max_tokens

        if not text.strip():
            # vLLM --reasoning-parser 가 켜져 있으면 추론은 reasoning_content 로 가고
            # content 가 빈다. 답에 도달하기 전에 생성이 끝난 것이다.
            reasoning = message.get("reasoning_content") or ""
            if reasoning:
                raise Truncated(_truncated_message(finish, len(reasoning), cap),
                                finish_reason=finish, usage=usage)
            raise JudgeError(f"빈 응답: {str(response)[:200]}")

        # 파서가 없는 서버는 추론을 content 에 그대로 담는다. 같은 사고인데 모양만
        # 다르고, 이쪽은 "형식 검증 실패"로 위장해서 재시도만 태우다 끝난다.
        # 잘렸어도 JSON 이 온전히 나왔다면(뒤에 붙던 산문만 잘린 경우) 그냥 쓴다.
        if finish == "length":
            try:
                extract_json(text)
            except ValueError:
                raise Truncated(_truncated_message(finish, len(text), cap),
                                finish_reason=finish, usage=usage) from None
        return text, usage

    # -- 잘린 응답 되살리기 ---------------------------------------------------

    def _rungs(self) -> list[tuple[str, dict]]:
        """폴백 사다리. 순서가 중요하다.

        추론을 끄는 쪽을 먼저 밟는다. 더 잘 듣고 더 싸다 - 토큰을 늘리는 건 이미
        한도만큼 태운 요청을 두 배로 태우는 것이라, 안 들으면 손해가 두 배다.
        """
        rungs = []
        if self.thinking != "off":
            rungs.append(("--thinking off", {"thinking": "off"}))
        rungs.append((f"max_tokens {self.max_tokens * 2:,}",
                      {"max_tokens": self.max_tokens * 2}))
        return rungs

    def _note_fallback(self, label: str) -> None:
        with self._fallback_lock:
            first = not self.fallbacks
            self.fallbacks.append(label)
        if first:
            # 마지막에 몰아서 알리면 이미 다 태운 뒤다. 처음 한 번은 즉시 알린다.
            print(f"\n[!] 추론이 답까지 도달하지 못해 {label} 로 다시 물었습니다.\n"
                  f"    계속 나오면 처음부터 {label} 로 돌리는 편이 빠릅니다.",
                  file=sys.stderr)

    def _attempt(self, system: str, user: str, out_model: type[T],
                 mode: str) -> tuple[str, Usage]:
        """한 번 보내고, 잘렸으면 조건을 바꿔 다시 보낸다.

        parse_with_repair 는 이걸 못 고친다. 그쪽은 같은 조건으로 다시 물으면서
        무엇이 틀렸는지 알려주는 장치인데, 잘린 응답은 조건이 같으면 같은 자리에서
        또 잘리기 때문이다. 조건을 바꾸는 건 여기서만 한다.
        """
        spent, tried, last = Usage(), [], None
        for label, override in [("", {})] + self._rungs():
            try:
                text, usage = self._text_and_usage(
                    self._post(self._payload(system, user, out_model, mode, **override)),
                    override.get("max_tokens"))
            except Truncated as e:
                spent.add(e.usage)
                tried.append(label or "그대로")
                last = e
                continue
            except JudgeError:
                if last is None:
                    raise      # 첫 시도의 실패는 폴백 대상이 아니다. 그대로 올린다.
                # 이 칸을 서버가 못 받는다 (chat_template_kwargs 미지원 등). 다음 칸으로.
                tried.append(f"{label}(서버가 거절)")
                continue
            spent.add(usage)
            if label:
                self._note_fallback(label)
            return text, spent

        raise Truncated(
            f"{last}\n  시도: {' → '.join(tried)} — 모두 답에 도달하지 못했습니다.\n"
            f"  이 서버·모델에서는 --thinking off 를 기본으로 두는 편이 낫습니다.",
            finish_reason=last.finish_reason, usage=spent)

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
            errors, overran = [], []
            for mode in JSON_MODES:
                try:
                    # 탐침이 잘려서 멀쩡한 모드를 떨어뜨리면 그 뒤가 전부 어긋난다.
                    text, _ = self._attempt(
                        _PROBE_SYSTEM, _PROBE_USER, out_model, mode)
                except Truncated as e:
                    # 이 모드가 거절된 게 아니다. 모델이 답까지 못 갔을 뿐이다.
                    # 다음 모드는 볼 값어치가 있다 - guided_json 처럼 첫 토큰부터
                    # 문법을 강제하는 모드는 추론 자체를 못 하게 막아서 안 잘린다.
                    overran.append(mode)
                    errors.append(f"{mode}: {e}")
                    self.negotiation_log.append(f"{mode:<12} 답까지 도달 못 함")
                    continue
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
            if overran:
                # "서버가 응답하지 않는다"고 하면 네트워크·URL 을 뒤지러 간다.
                # 서버는 멀쩡히 응답했다. 고칠 곳은 정반대다.
                raise Truncated(
                    f"서버는 응답하는데 모델이 답까지 도달하지 못합니다 "
                    f"({', '.join(overran)}). 연결 문제가 아닙니다.\n  "
                    + "\n  ".join(errors) + hint)
            raise JudgeError(
                "어떤 방식으로도 서버가 응답하지 않습니다:\n  "
                + "\n  ".join(errors) + hint
            )

    def complete(
        self, system: str, user: str, out_model: type[T], contract_hint: str = ""
    ) -> tuple[T, Usage]:
        mode = self.negotiate(out_model)

        def call(message: str) -> tuple[str, Usage]:
            return self._attempt(system, message, out_model, mode)

        # 스키마가 강제되면 한 번에 맞는다. 강제가 없을 때만 재시도가 의미를 갖는다.
        attempts = 1 if mode in ("json_schema", "guided_json") else self.max_attempts
        return parse_with_repair(call, user, out_model, contract_hint, attempts)


# ---------------------------------------------------------------------------
# 환경변수 → 백엔드
#
# 사용자가 줘야 할 것은 URL과 키 두 개뿐이다. 모델은 서버에 물어본다.
# 진입점이 여러 개라 여기 한 곳에 둔다.
# ---------------------------------------------------------------------------

from ragdiag.settings import KEY_VARS, MODEL_VAR, URL_VARS  # noqa: F401  (재수출)


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
