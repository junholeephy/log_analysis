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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

WIDTH = 78
RULE = "=" * WIDTH


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
class RunSummary:
    version: str = ""
    # 실행 인자를 그대로 한 줄. 반출이 안 되므로 "그때 뭘로 돌렸는지"가 셸
    # 히스토리에만 남으면 사라진다. 이 한 줄만 옮겨 적으면 재현된다.
    args: str = ""
    input_shape: str = ""
    contract_ok: int = 0
    contract_mismatches: list = field(default_factory=list)
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
        lines.append(f"input     : {self.input_shape}")

        n_bad = len(self.contract_mismatches)
        lines.append(f"contract  : {self.contract_ok} ok / "
                     f"{n_bad} MISMATCH" if n_bad else
                     f"contract  : {self.contract_ok} ok")
        for m in self.contract_mismatches:
            lines.append(f"  - {m.line()}"[:WIDTH])

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
        # 옮겨 적는 사람 기준으로 한 줄에 한 항목, 80자 이내.
        return "\n".join(l[:80] for l in lines)


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
