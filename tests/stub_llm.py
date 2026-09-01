"""스키마를 읽어 답하는 가짜 OpenAI 호환 서버.

이걸 두는 이유: 파이프라인을 **끝까지** 도는 테스트(진입점, 출력 파일 이름,
디렉터리 생성)가 판정 백엔드를 필요로 하는데, claude CLI 백엔드는 개발 장비
전용이라 저장소에 없다. 거기에 묶어두면 그 테스트들이 깨끗한 사본에서 조용히
건너뛰어지고, 정작 사내에서 처음 도는 경로가 검증되지 않은 채 나간다.

3스텝이 각기 다른 스키마를 요구하므로 응답을 손으로 적어둘 수 없다. 요청에 실린
JSON 스키마를 읽어 최소 유효 객체를 만들어 돌려준다. 판정 **내용**은 아무 의미가
없다 - 여기서 재는 것은 배관이지 판정 품질이 아니다.

    python tests/stub_llm.py      # 첫 줄에 URL 을 찍고 계속 떠 있는다
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

MODEL = "stub-model"


def sample(schema: dict, defs: dict):
    """스키마를 만족하는 최소 객체. enum 은 첫 값, 배열은 빈 배열."""
    if "$ref" in schema:
        return sample(defs[schema["$ref"].split("/")[-1]], defs)
    if "enum" in schema:
        return schema["enum"][0]
    kind = schema.get("type")
    if kind == "object":
        return {k: sample(v, defs) for k, v in schema.get("properties", {}).items()}
    if kind == "array":
        return []
    if kind == "integer":
        return 0
    if kind == "boolean":
        return False
    if kind == "number":
        return 0.0
    # 인용 검증의 최소 길이를 넘겨야 한다. 짧으면 too_short 로 버려진다.
    return "x" * 12


def make_server() -> HTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send({"data": [{"id": MODEL}]})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            schema = (body.get("guided_json")
                      or body.get("response_format", {})
                             .get("json_schema", {}).get("schema"))
            answer = sample(schema, schema.get("$defs", {})) if schema else {}
            self._send({
                "choices": [{"message": {"content": json.dumps(answer, ensure_ascii=False)},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 30},
            })

    return HTTPServer(("127.0.0.1", 0), Handler)


class StubLLM:
    """with 블록 안에서만 사는 서버. url 속성을 쓴다."""

    def __enter__(self) -> "StubLLM":
        self.server = make_server()
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
        return False


if __name__ == "__main__":
    import time

    server = make_server()
    print(f"http://127.0.0.1:{server.server_port}", flush=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    while True:
        time.sleep(3600)
