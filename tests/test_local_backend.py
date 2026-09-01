"""로컬 LLM 백엔드 테스트. 실제 HTTP 스텁 서버를 띄워서 검증한다.

이 백엔드는 에어갭 장비에서 처음 돌아간다. 거기서 처음 깨지면 고칠 방법이 없으므로,
서버 규약 협상과 폴백 경로를 여기서 다 밟아본다.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from ragdiag.backends import JudgeError, OpenAICompatBackend, strict_json_schema
from ragdiag.prompts import output_contract
from ragdiag.schema import GroundingCheck

GOOD = '{"reasoning": "문서를 쓰지 않았다", "answer_used_rag": "ignored"}'


class _Stub:
    """지정한 모드만 받아주는 가짜 OpenAI 호환 서버."""

    def __init__(self, accepts, replies=(GOOD,), models=("served-model",), ignores=(),
                 responder=None, rejects_thinking=False):
        self.accepts = set(accepts)
        # chat_template_kwargs 를 모르는 서버. 400 으로 거절한다.
        self.rejects_thinking = rejects_thinking
        self.ignores = set(ignores)   # 200은 주지만 강제는 안 하는 모드
        self.replies = list(replies)
        # 요청 내용에 따라 응답을 바꾸는 서버. 추론 스위치를 껐을 때만 답이
        # 나오는 모델을 흉내내는 데 쓴다.
        self.responder = responder
        self.models = list(models)
        self.seen = []          # 받은 (mode, payload)
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                body = json.dumps(
                    {"data": [{"id": m} for m in outer.models]}
                ).encode()
                self.send_response(200 if outer.models is not None else 500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if "guided_json" in body:
                    mode = "guided_json"
                elif body.get("response_format", {}).get("type") == "json_schema":
                    mode = "json_schema"
                elif body.get("response_format", {}).get("type") == "json_object":
                    mode = "json_object"
                else:
                    mode = "none"
                outer.seen.append((mode, body))

                if mode not in outer.accepts or (
                        outer.rejects_thinking and "chat_template_kwargs" in body):
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b'{"error":"unsupported"}')
                    return

                if mode in outer.ignores:
                    reply = "Hello! How can I help you today?"   # 강제 안 걸린 응답
                elif outer.responder:
                    reply = outer.responder(body)
                else:
                    reply = outer.replies[min(len(outer.seen) - 1, len(outer.replies) - 1)]
                # 문자열이면 평범한 응답, dict 면 choice 를 통째로 준 것이다.
                choice = reply if isinstance(reply, dict) else {"message": {"content": reply}}
                payload = {
                    "choices": [choice],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20},
                }
                data = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def __enter__(self):
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *a):
        self.server.shutdown()
        self.server.server_close()

    def modes_tried(self):
        return [m for m, _ in self.seen]


def _backend(url, **kw):
    return OpenAICompatBackend(base_url=url, model="local-30b", timeout=10, **kw)


def _as_response(choice):
    return {"choices": [choice], "usage": {"completion_tokens": 16000}}


def test_prefers_json_schema_when_server_supports_it():
    with _Stub(accepts=["json_schema", "guided_json", "json_object", "none"]) as stub:
        assert _backend(stub.url).negotiate(GroundingCheck) == "json_schema"
        assert stub.modes_tried() == ["json_schema"]


def test_falls_back_to_vllm_guided_json():
    # vLLM 구버전은 response_format을 모르고 guided_json만 받는다.
    with _Stub(accepts=["guided_json", "none"]) as stub:
        assert _backend(stub.url).negotiate(GroundingCheck) == "guided_json"
        assert stub.modes_tried() == ["json_schema", "guided_json"]


def test_falls_back_to_plain_json_object():
    with _Stub(accepts=["json_object", "none"]) as stub:
        assert _backend(stub.url).negotiate(GroundingCheck) == "json_object"


def test_falls_back_to_prompt_only():
    with _Stub(accepts=["none"]) as stub:
        assert _backend(stub.url).negotiate(GroundingCheck) == "none"
        assert stub.modes_tried() == ["json_schema", "guided_json", "json_object", "none"]


def test_negotiation_happens_once():
    with _Stub(accepts=["json_object", "none"]) as stub:
        backend = _backend(stub.url)
        backend.negotiate(GroundingCheck)
        tried = len(stub.seen)
        for _ in range(3):
            backend.negotiate(GroundingCheck)
        assert len(stub.seen) == tried  # 재협상하지 않는다


def test_explicit_mode_skips_negotiation():
    with _Stub(accepts=["guided_json"]) as stub:
        backend = _backend(stub.url, json_mode="guided_json")
        backend.complete("sys", "usr", GroundingCheck)
        assert stub.modes_tried() == ["guided_json"]


def test_strict_schema_is_sent_to_the_server():
    with _Stub(accepts=["json_schema"]) as stub:
        _backend(stub.url).complete("sys", "usr", GroundingCheck)
        _, body = stub.seen[0]
        sent = body["response_format"]["json_schema"]
        assert sent["strict"] is True
        assert sent["schema"] == strict_json_schema(GroundingCheck)
        assert sent["schema"]["additionalProperties"] is False


def test_parses_result_and_usage():
    with _Stub(accepts=["json_schema"]) as stub:
        result, usage = _backend(stub.url).complete("sys", "usr", GroundingCheck)
        assert result.answer_used_rag == "ignored"
        assert (usage.input_tokens, usage.output_tokens) == (100, 20)


def test_temperature_is_zero_for_determinism():
    with _Stub(accepts=["json_schema"]) as stub:
        _backend(stub.url).complete("sys", "usr", GroundingCheck)
        assert stub.seen[0][1]["temperature"] == 0.0


def test_repairs_malformed_output_when_schema_is_not_enforced():
    # 강제가 없는 서버에서는 재시도가 유일한 방어선이다.
    # json_mode를 명시해 협상 호출을 없애야 스텁 응답이 실제 호출과 1:1로 맞는다.
    with _Stub(accepts=["none"], replies=["형식이 틀린 응답입니다", GOOD]) as stub:
        result, _ = _backend(stub.url, json_mode="none").complete(
            "sys", "usr", GroundingCheck, output_contract(GroundingCheck)
        )
        assert len(stub.seen) == 2  # 첫 시도 + 복구 시도
        assert result.answer_used_rag == "ignored"
        # 재시도 메시지에 검증 오류가 실려야 한다
        assert "형식 검증에 실패" in stub.seen[-1][1]["messages"][1]["content"]


def test_gives_up_with_a_clear_error_after_retries():
    with _Stub(accepts=["none"], replies=["끝까지 틀린 응답"]) as stub:
        with pytest.raises(JudgeError, match="2회 시도 후에도 형식 검증 실패"):
            _backend(stub.url, json_mode="none", max_attempts=2).complete(
                "sys", "usr", GroundingCheck
            )
        assert len(stub.seen) == 2


def test_no_retry_when_schema_is_enforced():
    """스키마가 강제되면 재시도는 낭비다. 로컬 서빙에서는 한 번이 비싸다.

    협상 탐침은 통과시키고(GOOD) 본 호출만 어긋나게 해서, 재시도가 없는지 본다.
    """
    with _Stub(accepts=["json_schema"], replies=[GOOD, "틀린 응답"]) as stub:
        with pytest.raises(JudgeError):
            _backend(stub.url).complete("sys", "usr", GroundingCheck)
        assert len(stub.seen) == 2  # 협상 1 + 본 호출 1, 재시도 없음


def test_unreachable_server_message_names_the_url():
    backend = OpenAICompatBackend(base_url="http://127.0.0.1:1", model="m", timeout=2)
    with pytest.raises(JudgeError, match="연결하지 못했습니다"):
        backend.negotiate(GroundingCheck)


def test_rejects_unknown_json_mode():
    with pytest.raises(JudgeError, match="알 수 없는 json_mode"):
        OpenAICompatBackend(base_url="http://x", model="m", json_mode="엉뚱한값")


# ---------------------------------------------------------------------------
# 하이브리드 추론 모델 (Qwen3 계열)
# ---------------------------------------------------------------------------

THINKING = ('<think>\n사용자 요구는 {식비 상한}이다. 청크 0에 3만원이 있다.\n</think>\n'
            '{"reasoning": "청크 0에 금액이 있다", "answer_used_rag": "used"}')


def test_reasoning_block_containing_braces_does_not_hijack_extraction():
    """추론 블록 안의 중괄호를 JSON으로 집으면 케이스가 통째로 날아간다.

    검증 실패 -> 재시도 -> 같은 실패로 호출만 태우고 에러가 된다.
    """
    with _Stub(accepts=["json_schema"], replies=[THINKING]) as stub:
        result, _ = _backend(stub.url).complete("sys", "usr", GroundingCheck)
        assert result.answer_used_rag == "used"


def test_dangling_close_tag_is_handled():
    # 채팅 템플릿이 "<think>"를 미리 넣으면 모델 출력에는 닫는 태그만 나온다.
    raw = '추론 도중 {중괄호} 언급</think>{"reasoning": "r", "answer_used_rag": "ignored"}'
    with _Stub(accepts=["json_schema"], replies=[raw]) as stub:
        result, _ = _backend(stub.url).complete("sys", "usr", GroundingCheck)
        assert result.answer_used_rag == "ignored"


def test_thinking_kwarg_is_only_sent_when_requested():
    with _Stub(accepts=["json_schema"]) as stub:
        _backend(stub.url).complete("sys", "usr", GroundingCheck)
        assert "chat_template_kwargs" not in stub.seen[-1][1]

    with _Stub(accepts=["json_schema"]) as stub:
        _backend(stub.url, thinking="off").complete("sys", "usr", GroundingCheck)
        assert stub.seen[-1][1]["chat_template_kwargs"] == {"enable_thinking": False}

    with _Stub(accepts=["json_schema"]) as stub:
        _backend(stub.url, thinking="on").complete("sys", "usr", GroundingCheck)
        assert stub.seen[-1][1]["chat_template_kwargs"] == {"enable_thinking": True}


# ---------------------------------------------------------------------------
# 잘린 응답
#
# 추론 모델은 생각에만 수천 토큰을 쓴다. 답에 도달하기 전에 생성이 끝나면 그 턴은
# 통째로 버려졌다 - 재시도조차 없었다. parse_with_repair 는 "무엇이 틀렸는지"를
# 되먹이는 장치라 같은 조건으로 다시 물을 뿐이고, 조건이 같으면 같은 자리에서 또
# 잘리기 때문이다. 조건을 바꾸는 사다리가 있어야 한다.
# ---------------------------------------------------------------------------

def _thinking_off(body):
    return body.get("chat_template_kwargs", {}).get("enable_thinking") is False


def _reasoning_only(finish_reason="length"):
    """reasoning_parser 가 켜진 서버가 추론만 하다 끝냈을 때의 응답."""
    return {"message": {"content": "", "reasoning_content": "음, 이건 " * 300},
            "finish_reason": finish_reason}


def test_truncation_recovers_by_turning_thinking_off():
    """생각만 하다 끝난 턴을 버리지 않는다. 조건을 바꿔 한 번 더 묻는다."""
    def server(body):
        return GOOD if _thinking_off(body) else _reasoning_only()

    with _Stub(accepts=["json_schema"], responder=server) as stub:
        backend = _backend(stub.url)
        result, usage = backend.complete("sys", "usr", GroundingCheck)
        assert result.answer_used_rag == "ignored"
        assert backend.fallbacks, "무엇으로 되살렸는지 기록이 남아야 한다"
        assert all("thinking off" in f for f in backend.fallbacks)
        # 잘린 요청도 토큰을 다 태웠다. 사용량에서 빠지면 비용을 과소평가한다.
        assert usage.output_tokens >= 40, usage


def test_truncation_falls_through_to_a_bigger_budget():
    """추론 스위치를 못 받는 서버도 있다. 그때는 한도를 늘려서 다시 묻는다."""
    def server(body):
        if "chat_template_kwargs" in body:
            raise AssertionError("여기 오면 안 된다")   # 아래 rejects 가 먼저 막는다
        return GOOD if body["max_tokens"] > 1000 else _reasoning_only()

    with _Stub(accepts=["json_schema"], responder=server,
               rejects_thinking=True) as stub:
        backend = _backend(stub.url, max_tokens=1000)
        result, _ = backend.complete("sys", "usr", GroundingCheck)
        assert result.answer_used_rag == "ignored"
        assert any("max_tokens" in f for f in backend.fallbacks), backend.fallbacks


def test_giving_up_says_what_was_tried():
    with _Stub(accepts=["json_schema"],
               responder=lambda body: _reasoning_only()) as stub:
        backend = _backend(stub.url)
        with pytest.raises(JudgeError) as e:
            backend.complete("sys", "usr", GroundingCheck)
        msg = str(e.value)
        assert "시도:" in msg, msg
        assert "thinking off" in msg, msg
        # 연결을 의심하게 만들면 정반대 방향으로 시간을 버린다.
        assert "연결 문제가 아닙니다" in msg, msg


def test_the_error_does_not_prescribe_the_wrong_knob():
    """한도에 걸린 게 아니면 max_tokens 를 늘려도 소용없다.

    예전 문구는 어느 경우든 "max_tokens를 늘리거나"라고 했다. 사내에서는 그 한 줄이
    사이클 하나다 - 늘려서 다시 돌리고, 똑같이 실패하는 데 몇 시간이 간다.
    """
    with _Stub(accepts=["json_schema"]) as stub:
        backend = _backend(stub.url)

        # 스스로 멈춘 경우 - 늘려도 같다
        with pytest.raises(JudgeError, match="추론만 하고") as e:
            backend._text_and_usage(_as_response(_reasoning_only("stop")))
        assert "늘려도 같습니다" in str(e.value)

        # 한도에 걸린 경우 - 이때는 늘리는 게 맞다
        with pytest.raises(JudgeError, match="토큰 한도") as e:
            backend._text_and_usage(_as_response(_reasoning_only("length")))


def test_truncated_content_is_caught_even_without_a_reasoning_parser():
    """파서가 없는 서버는 추론을 content 에 담는다. 같은 사고인데 모양만 다르다.

    이쪽은 "형식 검증 실패"로 위장해서, 재시도만 태우고 끝났다.
    """
    cut = {"message": {"content": "<think>어디 보자, 이 문서는" * 50},
           "finish_reason": "length"}

    def server(body):
        return GOOD if _thinking_off(body) else cut

    with _Stub(accepts=["json_schema"], responder=server) as stub:
        backend = _backend(stub.url)
        result, _ = backend.complete("sys", "usr", GroundingCheck)
        assert result.answer_used_rag == "ignored"
        assert backend.fallbacks


def test_a_complete_answer_cut_short_is_still_used():
    """JSON 이 온전히 나온 뒤 뒤에 붙던 산문만 잘린 건 잘린 게 아니다."""
    with _Stub(accepts=["json_schema"],
               replies=[{"message": {"content": GOOD + "\n덧붙이자면 이 답변은"},
                         "finish_reason": "length"}]) as stub:
        backend = _backend(stub.url)
        result, _ = backend.complete("sys", "usr", GroundingCheck)
        assert result.answer_used_rag == "ignored"
        assert not backend.fallbacks, "멀쩡한 응답에 폴백을 태우면 호출이 두 배가 된다"


def test_a_truncated_probe_does_not_disqualify_a_good_mode():
    """탐침이 잘려서 멀쩡한 모드를 떨어뜨리면 그 뒤가 전부 어긋난다.

    json_schema 가 탈락하면 none 까지 떨어지고, 강제가 없으니 판정 품질이 흔들린다.
    """
    def server(body):
        return GOOD if _thinking_off(body) else _reasoning_only()

    with _Stub(accepts=["json_schema"], responder=server) as stub:
        assert _backend(stub.url).negotiate(GroundingCheck) == "json_schema"


def test_rejects_unknown_thinking_mode():
    with pytest.raises(JudgeError, match="알 수 없는 thinking"):
        OpenAICompatBackend(base_url="http://x", model="m", thinking="가끔")


# ---------------------------------------------------------------------------
# 설정 최소화: 사용자가 줘야 할 것은 URL과 키 두 개뿐
# ---------------------------------------------------------------------------

def test_model_is_discovered_from_the_server():
    with _Stub(accepts=["json_schema"], models=["qwen3.5-397b-a17b"]) as stub:
        backend = OpenAICompatBackend(base_url=stub.url, timeout=10)
        assert backend.model == "qwen3.5-397b-a17b"
        assert backend.discovered is True


def test_explicit_model_skips_discovery():
    with _Stub(accepts=["json_schema"], models=["다른것"]) as stub:
        backend = OpenAICompatBackend(base_url=stub.url, model="내가정한것", timeout=10)
        assert backend.model == "내가정한것"
        assert backend.discovered is False


def test_first_model_wins_and_others_are_listable():
    with _Stub(accepts=["json_schema"], models=["a", "b", "c"]) as stub:
        backend = OpenAICompatBackend(base_url=stub.url, timeout=10)
        assert backend.model == "a"
        assert backend.list_models() == ["a", "b", "c"]


def test_empty_model_list_gives_an_actionable_error():
    with _Stub(accepts=["json_schema"], models=[]) as stub:
        with pytest.raises(JudgeError, match="--model 로 직접 지정"):
            OpenAICompatBackend(base_url=stub.url, timeout=10)


def test_discovered_model_is_actually_sent_in_requests():
    with _Stub(accepts=["json_schema"], models=["서버가정한모델"]) as stub:
        OpenAICompatBackend(base_url=stub.url, timeout=10).complete(
            "sys", "usr", GroundingCheck
        )
        assert stub.seen[-1][1]["model"] == "서버가정한모델"


@pytest.mark.parametrize("suffix", ["", "/", "/v1", "/v1/", "/v1/chat/completions"])
def test_url_variants_all_work(suffix):
    # 붙여넣기로 흔히 생기는 형태들. /v1이 두 번 붙어 404가 나면 안 된다.
    with _Stub(accepts=["json_schema"]) as stub:
        backend = OpenAICompatBackend(base_url=stub.url + suffix, timeout=10)
        result, _ = backend.complete("sys", "usr", GroundingCheck)
        assert result.answer_used_rag == "ignored"


def test_scheme_is_added_when_missing():
    from ragdiag.backends import normalize_base_url

    assert normalize_base_url("host:8000") == "http://host:8000"
    assert normalize_base_url("https://host/v1") == "https://host"


# ---------------------------------------------------------------------------
# 프록시가 파라미터를 조용히 버리는 경우 (LiteLLM 등)
# ---------------------------------------------------------------------------

def test_mode_that_returns_200_but_does_not_enforce_is_rejected():
    """200 OK만 보고 확정하면 안 된다.

    강제가 안 걸렸는데 걸린 줄 알면 재시도를 1회로 줄여버려서,
    첫 응답이 어긋나는 순간 케이스가 그냥 실패한다.
    """
    with _Stub(accepts=["json_schema", "none"], ignores=["json_schema"]) as stub:
        assert _backend(stub.url).negotiate(GroundingCheck) == "none"
        assert stub.modes_tried() == ["json_schema", "guided_json", "json_object", "none"]


def test_falls_through_silent_drop_to_a_mode_that_really_enforces():
    with _Stub(accepts=["json_schema", "guided_json"], ignores=["json_schema"]) as stub:
        assert _backend(stub.url).negotiate(GroundingCheck) == "guided_json"


def test_json_object_only_needs_valid_json_not_the_full_schema():
    # 이 모드가 약속하는 건 "유효한 JSON"까지다. 스키마까지 요구하면 부당하게 탈락한다.
    class _Partial(_Stub):
        pass

    with _Stub(accepts=["json_object", "none"], replies=['{"아무":"객체"}']) as stub:
        assert _backend(stub.url).negotiate(GroundingCheck) == "json_object"


def test_silent_drop_is_named_in_the_failure_message():
    with _Stub(accepts=list(("json_schema", "guided_json", "json_object")),
               ignores=["json_schema", "guided_json", "json_object"]) as stub:
        with pytest.raises(JudgeError, match="조용히 무시"):
            _backend(stub.url).negotiate(GroundingCheck)
