"""RUN SUMMARY — 화면이 유일한 출력이다.

결과 파일도 플롯도 로그도 반출할 수 없다. 사내 실험에서 이쪽으로 돌아오는 것은
사람이 화면을 보고 옮겨 적은 것뿐이다. 그래서 이 블록은 **성공·실패 무관하게**
마지막에 찍고, 한 줄에 한 항목, 옮겨 적기 쉽게 짧게 쓴다.

계약 위반이 이 출력의 핵심이다. `validation failed` 같은 줄은 여기서 결함이다 —
옮겨 적을 것이 없기 때문이다. 무엇이 어떻게 어긋났는지가 그대로 나와야 다음
사이클에서 contracts.py 를 고칠 수 있다.

실데이터의 개별 값·식별자는 찍지 않는다. 건수와 타입 이름까지다.
"""

from __future__ import annotations

import subprocess
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

WIDTH = 78
RULE = "=" * WIDTH


def display_width(text: str) -> int:
    """터미널에서 차지하는 칸 수. 한글·한자는 두 칸이다.

    len() 으로 맞추면 라벨이 한글일 때 표가 전부 어긋난다. 이 출력은 사람이
    화면에서 읽고 옮겨 적는 것이라 정렬이 곧 가독성이다.
    """
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def pad_to(text: str, width: int) -> str:
    return text + " " * max(0, width - display_width(text))


def _clip(text: str, width: int) -> str:
    """화면 width 칸에서 자른다. 글자 수가 아니라 칸 수다."""
    if display_width(text) <= width:
        return text
    out, used = [], 0
    for ch in text:
        w = display_width(ch)
        if used + w > width:
            break
        out.append(ch)
        used += w
    return "".join(out)


def version() -> str:
    """BB/VERSION(이식본) 또는 git(개발 중)에서 읽는다.

    "어떤 코드로 돌렸는지"는 결과 파일이 반출되지 않는 상황에서 사내에 남는
    유일한 단서다. 알 수 없으면 그렇다고 적는다 - 빈칸으로 두면 옮겨 적을 때
    빠진다.
    """
    stamp = Path(__file__).resolve().parents[2] / "VERSION"
    if stamp.exists():
        return stamp.read_text(encoding="utf-8").strip() or "(VERSION 비어 있음)"
    try:
        root = Path(__file__).resolve().parents[2]
        tag = subprocess.run(["git", "-C", str(root), "describe", "--tags", "--always",
                              "--dirty"], capture_output=True, text=True, timeout=5)
        if tag.returncode == 0 and tag.stdout.strip():
            return f"{tag.stdout.strip()} (git)"
    except (OSError, subprocess.SubprocessError):
        pass
    return "(알 수 없음 — 태그 없이 실행했다)"


def peak_memory_gb() -> Optional[float]:
    try:
        import resource
        import sys

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # 단위가 플랫폼마다 다르다 - 리눅스는 KB, macOS 는 바이트.
        # 사내는 리눅스지만 여기서 시험하면서 40GB 로 찍히면 못 믿게 된다.
        divisor = 1024 ** 3 if sys.platform == "darwin" else 1024 ** 2
        return peak / divisor
    except (ImportError, OSError):
        return None


@dataclass
class Source:
    """값 하나와 **그 값이 어디서 왔는지**.

    값만 찍으면 "왜 저게 저 값이지"를 못 푼다. 사내에서는 .bashrc 의 환경변수,
    AA/configs/local.yaml, CLI 플래그가 겹쳐 있고 셋 다 안 보인다. 어느 쪽이
    이겼는지가 안 보이면 설정을 고쳐도 안 먹는 이유를 알 수 없다.
    """

    label: str
    value: str
    origin: str = ""      # --flag / 설정 llm.model / 환경변수 LLM_API_URL / 기본값
    note: str = ""

    def line(self, pad: int) -> str:
        text = "  " + pad_to(self.label, pad) + self.value
        tail = " · ".join(x for x in (self.origin, self.note) if x)
        return pad_to(text, 46) + f"  ← {tail}" if tail else text


@dataclass
class Conditions:
    """무엇을 읽어 무엇으로 돌리는지. 계산을 시작하기 전에 찍는다.

    시작 전이어야 의미가 있다 - 다 돌고 나서 "모델이 그거였네"를 알면 늦는다.
    """

    rows: list[Source] = field(default_factory=list)
    hint: str = ""

    def add(self, label: str, value: str, origin: str = "", note: str = "") -> None:
        self.rows.append(Source(label, value, origin, note))

    def compact(self) -> list[str]:
        """RUN SUMMARY 용 압축본. 옮겨 적을 사람 기준으로 접는다.

        전체 블록을 그대로 넣으면 RUN SUMMARY 가 두 배가 된다. 재현에 필요한
        것만 남긴다 - 경로는 args 줄에 이미 있으므로 뺀다.
        """
        import re

        keep = {"설정", "백엔드", "주소", "모델", "판정", "동시"}
        # 값이 이미 key=value 꼴이면 라벨을 덧붙이지 않는다 (판정=json_mode=... 방지).
        # 출처 표시는 뺀다 - 화면 블록에 이미 있고, 여기서 필요한 것은 재현에 쓸
        # **값**이다. 넣으면 줄이 길어져 잘리고, 잘린 줄은 옮겨 적을 수 없다.
        parts = [re.sub(r"\((?:설정 [\w.]+|--[\w-]+)\)", "",
                        r.value if "=" in r.value else f"{r.label}={r.value}")
                 for r in self.rows if r.label in keep]

        lines, current = [], ""
        for part in parts:
            # len() 으로 접으면 한글이 두 칸이라 80 자 자르기에서 단어가 끊긴다.
            if display_width(current) + display_width(part) + 2 > WIDTH - 14:
                lines.append(current)
                current = part
            else:
                current = f"{current}  {part}" if current else part
        if current:
            lines.append(current)
        return lines

    def render(self) -> str:
        # "바꾸려면"도 같은 칸에 맞춘다. 빼면 힌트 줄만 붙어 나온다.
        pad = max([display_width(r.label) for r in self.rows]
                  + [display_width("바꾸려면")]) + 2
        lines = ["─" * WIDTH, "실행 조건"]
        lines += [r.line(pad) for r in self.rows]
        if self.hint:
            lines += ["", "  " + pad_to("바꾸려면", pad) + self.hint]
        lines.append("─" * WIDTH)
        return "\n".join(lines)


@dataclass
class RunSummary:
    version: str = ""
    # 실행 인자를 그대로 한 줄. 반출이 안 되므로 "그때 뭘로 돌렸는지"가 셸
    # 히스토리에만 남으면 사라진다. 이 한 줄만 옮겨 적으면 재현된다.
    args: str = ""
    # 인자만으로는 재현이 안 된다. 설정 파일과 환경변수가 값을 덮고, 모델은
    # 서버에서 자동 탐지될 수 있다. 그 결과를 한두 줄로 적어 둔다.
    setup: list[str] = field(default_factory=list)
    input_shape: str = ""
    contract_ok: int = 0
    contract_mismatches: list = field(default_factory=list)
    # 안 쓰는 필드에서 나온 어긋남. 옮겨 적을 값어치는 있지만 판정에는 영향이 없다.
    contract_notes: list = field(default_factory=list)
    metrics: list[tuple[str, str]] = field(default_factory=list)
    runtime_sec: float = 0.0
    peak_gb: Optional[float] = None
    status: str = "OK"
    notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [f"{'=' * 16} RUN SUMMARY {'=' * (WIDTH - 30)}"]
        lines.append(f"version   : {self.version}")
        if self.args:
            lines.append(f"args      : {self.args}")
        for i, line in enumerate(self.setup):
            lines.append(f"{'setup     : ' if i == 0 else ' ' * 12}{line}")
        lines.append(f"input     : {self.input_shape}")

        n_bad = len(self.contract_mismatches)
        lines.append(f"contract  : {self.contract_ok} ok / "
                     f"{n_bad} MISMATCH" if n_bad else
                     f"contract  : {self.contract_ok} ok")
        for m in self.contract_mismatches:
            lines.append(f"  - {m.line()}"[:WIDTH])
        # 판정에 영향이 없는 것은 기호를 달리한다. `-` 줄만 보면 되게 한다.
        for m in self.contract_notes:
            lines.append(f"  · {m.line()} (안 쓰는 필드)"[:WIDTH])

        for i, (name, value) in enumerate(self.metrics):
            label = "metrics   : " if i == 0 else " " * 12
            lines.append(f"{label}{name} {value}"[:WIDTH])

        runtime = f"{self.runtime_sec:.0f}s"
        if self.peak_gb:
            runtime += f", peak {self.peak_gb:.1f}GB"
        lines.append(f"runtime   : {runtime}")
        lines.append(f"status    : {self.status}")
        for note in self.notes:
            lines.append(f"  ! {note}"[:WIDTH])
        lines.append("=" * WIDTH)
        # 옮겨 적는 사람 기준으로 한 줄에 한 항목, 화면 80칸 이내.
        # 글자 수가 아니라 **칸 수**로 자른다 - 한글이 두 칸이라 [:80] 은 단어
        # 중간에서 끊고, 중간에서 끊긴 줄은 옮겨 적을 수 없다.
        return "\n".join(_clip(l, 80) for l in lines)


class Timer:
    def __enter__(self):
        self.start = time.monotonic()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.monotonic() - self.start
        return False

    @property
    def seconds(self) -> float:
        return getattr(self, "elapsed", time.monotonic() - self.start)
