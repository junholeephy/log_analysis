"""개발 장비 전용 판정 백엔드 — Anthropic API.

**규격 §1.4 · C8.** 사내에서는 이 호출이 전부 실패한다. 그래서 `src/` 에 두지
않는다 - `src/` 안에 있으면 sync.sh 의 이식 표면 점검이 `anthropic` import 를
잡아 preflight 가 FAILED 로 끝난다.

`.gitattributes` 의 export-ignore 로 archive 결과에서 빠지므로 저장소에는 있지만
사내에는 도착하지 않는다. import 방향은 한쪽이다 — tools/ 는 src/ 를 import 하지만
src/ 는 tools/ 를 import 하지 않는다.
"""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from ragdiag.backends import JudgeError, Usage

T = TypeVar("T", bound=BaseModel)


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
