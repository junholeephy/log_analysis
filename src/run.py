#!/usr/bin/env python
"""진입점.

  python <저장소>/src/run.py --conv-data <로그> --turns <턴 목록> --output-dir <출력>

**PYTHONPATH 도 설치도 필요 없다.** 파이썬이 스크립트가 있는 디렉터리를
sys.path[0] 에 넣으므로, 이 파일이 src/ 에 있는 것만으로 옆의 ragdiag/ 가
그대로 import 된다. 공용 venv 에 우리 패키지를 남기지 않고, 사본 통째 교체가
무연산이 된다.

진입점만 여기 두고 나머지는 전부 src/ragdiag/ 안에 있다. src/ 를 평평하게 쓰면
contracts · report 같은 흔한 이름이 최상위 모듈이 되어 서드파티를 가린다.

상대 경로는 전부 **실행 위치 기준**으로 해석된다. 운영 환경에서는 작업 폴더에서
실행하므로 output/ 이 저장소 사본이 어디 있든 맞아떨어진다.

여기서 하는 일이 하나 더 있다 — 설정의 `paths.venv` 로 갈아타는 것. **ragdiag 를
import 하기 전이어야 한다.** activate 를 잊고 시스템 파이썬으로 실행하면 pydantic
import 에서 먼저 죽어서, 그 뒤에 있는 어떤 안내도 화면에 못 나온다.
"""

import os
import sys
from pathlib import Path

# 갈아탄 파이썬이 또 다른 prefix 를 보고하면 무한히 돈다. 한 번만 갈아탄다.
_SWITCH_FLAG = "RAGDIAG_VENV_SWITCHED"


def _config_path(argv: list[str]) -> str:
    for i, arg in enumerate(argv):
        if arg == "--config" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--config="):
            return arg.split("=", 1)[1]
    return ""


def _peek_venv(path: str) -> str:
    """설정에서 paths.venv **하나만** 미리 읽는다.

    config.py 를 쓸 수 없다. 그쪽은 ragdiag 를 끌어오고, 아직 갈아타지 않은
    파이썬에는 그 의존이 없을 수 있다 - 그걸 해결하려고 여기 있는 코드다.
    PyYAML 이 있으면 그걸 쓰고, 없으면 두 줄짜리 스캔으로 떨어진다.
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return ""

    try:
        import yaml

        tree = yaml.safe_load(text) or {}
        return str((tree.get("paths") or {}).get("venv") or "")
    except Exception:
        pass

    in_paths = False
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line[:1].isspace():
            in_paths = line.split(":")[0].strip() == "paths"
            continue
        if in_paths and line.strip().split(":")[0].strip() == "venv":
            value = line.split(":", 1)[1].split("#")[0].strip().strip("\"'")
            return "" if value in ("", "null", "~") else value
    return ""


def switch_venv(argv: list[str]) -> None:
    """설정에 적은 파이썬으로 갈아타고 같은 명령을 다시 시작한다.

    적어만 두고 쓰지 않으면 "설정했는데 무시된다" 가 된다. 설정 파일의 값이
    아무것도 바꾸지 않는 것은 그 자체로 결함이다.

    exec 라 이 프로세스가 그대로 대체된다. 인자와 cwd 가 보존되므로 같은 명령이
    새 파이썬으로 도는 것과 같고, 아직 아무 계산도 하지 않았으므로 잃을 것이 없다.
    """
    if os.environ.get(_SWITCH_FLAG):
        return
    config = _config_path(argv)
    want = _peek_venv(config) if config else ""
    if not want:
        return

    venv = Path(want).expanduser()
    if venv.resolve() == Path(sys.prefix).resolve():
        return                                   # 이미 그 venv 다

    for python in (venv / "bin" / "python", venv / "Scripts" / "python.exe"):
        if python.exists():
            break
    else:
        print(f"설정({config})의 paths.venv 에 파이썬이 없습니다: {venv}\n"
              f"  찾은 곳: {venv / 'bin' / 'python'}\n"
              f"  경로를 고치거나, paths.venv 를 비우고 그 venv 를 activate 한 뒤 "
              f"실행하세요.", file=sys.stderr)
        raise SystemExit(2)

    print(f"[venv] {sys.prefix}\n    -> {venv}   (설정 paths.venv)", file=sys.stderr)
    os.environ[_SWITCH_FLAG] = "1"
    os.execv(str(python), [str(python), *sys.argv])


if __name__ == "__main__":
    switch_venv(sys.argv[1:])

    from ragdiag.__main__ import main

    raise SystemExit(main())
