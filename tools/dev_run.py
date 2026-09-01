#!/usr/bin/env python
"""개발 장비 전용 실행기 — claude CLI / Anthropic API 로 판정한다.

**왜 별도 진입점인가.** 사내에서는 이 두 백엔드의 호출이 전부 실패한다. 규격
§1.4 는 그런 코드를 `src/` 밖에 두라고 하고, sync.sh 의 이식 표면 점검이
`anthropic` import 를 실제로 잡는다. import 방향은 한쪽이다 — 여기서 `src/` 를
불러 쓰고, `src/` 는 여기를 모른다.

**같은 코드 경로를 돈다.** 백엔드만 만들어서 `ragdiag.__main__.main()` 에 넣는다.
검증하는 코드와 배포되는 코드가 갈라지면 여기서 통과한 것이 사내에서 통과한다는
보장이 사라진다 - 그래서 인자도 출력도 전부 같다.

    python tools/dev_run.py --conv-data data/conv_eval.json
    python tools/dev_run.py --golden
    python tools/dev_run.py --backend api --legacy-regression
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 파이썬이 sys.path[0] 에 넣는 것은 이 파일이 있는 tools/ 다. 두 개가 더 필요하다:
# ragdiag 를 찾을 src/, 그리고 tools.backend_cli 를 패키지로 찾을 저장소 루트.
# src/run.py 는 이 줄들이 필요 없다 - 자기가 src/ 안에 있기 때문이다.
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from ragdiag.__main__ import main as ragdiag_main  # noqa: E402
from ragdiag.backends import JudgeError  # noqa: E402

DEFAULT_MODEL = "claude-opus-5"


def make_dev_backend(kind: str, model: str | None, timeout: int, cli_path: str | None):
    if kind == "cli":
        try:
            from tools.backend_cli import ClaudeCodeBackend
        except ImportError as e:
            raise JudgeError(
                "tools/backend_cli.py 가 없습니다.\n"
                "  claude CLI 백엔드는 저장소에 올리지 않습니다 (개발 장비 전용).\n"
                "  --backend api 를 쓰거나, 로컬 LLM 서버로 src/run.py 를 쓰세요."
            ) from e
        return ClaudeCodeBackend(model=model or DEFAULT_MODEL,
                                 cli_path=cli_path, timeout=timeout)

    from tools.backend_api import ApiBackend

    return ApiBackend(model=model or DEFAULT_MODEL)


def main() -> int:
    p = argparse.ArgumentParser(
        description="개발 장비에서 claude CLI / Anthropic API 로 판정한다",
        epilog="나머지 인자는 그대로 src/run.py 에 넘어간다.")
    p.add_argument("--backend", choices=["cli", "api"], default="cli")
    p.add_argument("--model")
    p.add_argument("--cli-path")
    p.add_argument("--timeout", type=int, default=300)
    args, rest = p.parse_known_args()

    try:
        backend = make_dev_backend(args.backend, args.model, args.timeout, args.cli_path)
    except JudgeError as e:
        print(e, file=sys.stderr)
        return 2

    # 실제로 친 것을 RUN SUMMARY 에 남기기 위해 백엔드 인자도 함께 넘긴다.
    return ragdiag_main(rest, backend=backend)


if __name__ == "__main__":
    raise SystemExit(main())
