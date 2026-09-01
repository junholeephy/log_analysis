#!/usr/bin/env python
"""진입점.

  python <저장소>/src/run.py --conv-data <로그> --filter-data <필터> --output-dir <출력>

**PYTHONPATH 도 설치도 필요 없다.** 파이썬이 스크립트가 있는 디렉터리를
sys.path[0] 에 넣으므로, 이 파일이 src/ 에 있는 것만으로 옆의 ragdiag/ 가
그대로 import 된다. 공용 venv 에 우리 패키지를 남기지 않고, 사본 통째 교체가
무연산이 된다.

진입점만 여기 두고 나머지는 전부 src/ragdiag/ 안에 있다. src/ 를 평평하게 쓰면
contracts · report 같은 흔한 이름이 최상위 모듈이 되어 서드파티를 가린다.

상대 경로는 전부 **실행 위치 기준**으로 해석된다. 운영 환경에서는 작업 폴더에서
실행하므로 outputs/ 가 저장소 사본이 어디 있든 맞아떨어진다.
"""

from ragdiag.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
