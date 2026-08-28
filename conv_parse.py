#!/usr/bin/env python
"""실행 스크립트. PYTHONPATH 없이 그냥 돌아간다.

  python conv_parse.py --conv-data <로그> --filter-data <필터> --output-dir <출력>

사내에서는 사본 안의 이 파일을 가리킨다:

  python log_analysis/conv_parse.py \\
      --conv-data data/conv_eval.json \\
      --filter-data data/filter.json \\
      --output-dir outputs

규격이 정한 `PYTHONPATH={BB}/src python -m ragdiag` 와 **완전히 같은 일**을 한다.
하는 일은 sys.path 에 옆의 src/ 를 끼워 넣는 것뿐이다 - 로직은 여기 없다.

둘 중 아무거나 써도 된다.
  PYTHONPATH=log_analysis/src python -m ragdiag ...   # 규격 형태
  python log_analysis/conv_parse.py ...               # 같은 것, 손이 덜 감

패키지를 venv 에 설치하지 않는 것은 그대로다. 공용 venv 를 건드리지 않고
사본 통째 교체가 무연산이 된다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ragdiag.__main__ import main  # noqa: E402  (경로를 먼저 세워야 한다)

if __name__ == "__main__":
    raise SystemExit(main())
