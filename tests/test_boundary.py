"""반입 경계를 실제 import 로 확인한다.

운영 장비에는 로그 파서와 필터가 이미 있다. 이 저장소에서 가져갈 것은 판정과
출력, 즉 **Case 를 받아 case_id 를 붙이는 부분**이다. 그 부분이 입력 계층을
붙들고 있으면 "필요한 것만 복사"가 성립하지 않는다.

문서로만 적어 두면 썩는다. 누가 output.py 에 conv 를 하나 import 하는 순간
경계는 조용히 무너지고, 그걸 알아채는 건 운영 장비에서 ImportError 가 났을
때다. 그래서 여기서 정적 분석이 아니라 **실제로 import 해서 sys.modules 를**
본다 - TYPE_CHECKING 블록이나 함수 안 import 는 런타임 의존이 아니고,
정적 분석은 그 둘을 가려내지 못한다.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# 운영 장비으로 가져갈 모듈. 이 목록이 README 의 복사 목록과 같아야 한다.
CORE = [
    "ragdiag.settings",     # 배포마다 바뀌는 값
    "ragdiag.schema",       # Case · 판정 결과 모델
    "ragdiag.taxonomy",     # 케이스 29개
    "ragdiag.prompts",      # 시스템 프롬프트 4종
    "ragdiag.backends",     # LLM 접속
    "ragdiag.judge",        # 3스텝 호출
    "ragdiag.decide",
    "ragdiag.verify",       # 인용 대조
    "ragdiag.checks",       # 코드 검증기
    "ragdiag.route",        # 진리표
    "ragdiag.classify",     # 오케스트레이션
    "ragdiag.output",       # 출력 JSON 모양
    "ragdiag.pipeline",     # 단계별 함수
]

# 여기 남는 것. 운영 장비에는 그쪽 구현이 있다.
LOCAL_ONLY = [
    "ragdiag.load",         # conv_eval 파일 읽기
    "ragdiag.conv",         # conv_eval → Case
    "ragdiag.filters",      # 필터 적용
    "ragdiag.labels",       # query · emotion 라벨표
    "ragdiag.org",          # 조직 분류 (대시보드용)
    "ragdiag.survey",       # 로그 훑어보기
    "ragdiag.golden",       # 골든셋 채점
]


def imported_modules(module: str) -> set[str]:
    """새 인터프리터에서 하나만 import 하고 딸려온 ragdiag 모듈을 돌려준다."""
    code = (
        f"import {module}, sys, json; "
        "print(json.dumps(sorted(m for m in sys.modules "
        "if m.startswith('ragdiag.'))))"
    )
    # 운영 환경에서 붙이는 방식과 같게 한다 - PYTHONPATH=BB/src 로만 붙이고
    # 패키지를 venv 에 설치하지 않는다.
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                          capture_output=True, text=True, env=env)
    assert proc.returncode == 0, f"{module} import 실패:\n{proc.stderr}"
    import json as _json
    return set(_json.loads(proc.stdout.strip().splitlines()[-1]))


@pytest.mark.parametrize("module", CORE)
def test_core_does_not_pull_in_the_input_layer(module):
    """코어를 import 할 때 입력 계층이 딸려오면 안 된다."""
    pulled = imported_modules(module)
    leaked = pulled & set(LOCAL_ONLY)
    assert not leaked, (
        f"{module} 이 입력 계층을 끌어온다: {sorted(leaked)}\n"
        "반입 목록이 늘어나거나, 운영 장비에서 ImportError 가 난다.\n"
        "타입 주석용이면 TYPE_CHECKING 블록으로, 일부 경로에서만 쓰면 "
        "함수 안으로 옮길 것."
    )


def test_core_closure_is_exactly_the_copy_list():
    """코어 전체를 import 해도 목록 밖 모듈이 딸려오지 않는다.

    개별 모듈이 각각 깨끗해도 합집합이 목록을 넘을 수 있다.
    """
    pulled = set()
    for module in CORE:
        pulled |= imported_modules(module)
    pulled = {m for m in pulled if m.count(".") == 1}   # 서브모듈 제외
    extra = pulled - set(CORE)
    assert not extra, (
        f"복사 목록에 없는 모듈이 딸려온다: {sorted(extra)}\n"
        "README 의 복사 목록과 이 테스트의 CORE 를 함께 고칠 것."
    )


def test_readme_copy_list_matches_this_test():
    """README 가 코드보다 앞서 나가지 않도록.

    taxonomy 문서에서 한 번 겪었다 - 문서에 '판정 가능'이라 써두고 라우팅은
    그 case 를 만들지 못하는 상태였는데 문서만 읽으면 알 수 없었다.
    """
    readme = ROOT / "README.md"
    if not readme.exists():
        pytest.skip("README.md 없음")
    text = readme.read_text(encoding="utf-8")
    if "<!-- copy-list -->" not in text:
        pytest.skip("README 에 복사 목록 표시가 없음")
    block = text.split("<!-- copy-list -->")[1].split("<!-- /copy-list -->")[0]
    listed = {f"ragdiag.{name}" for name in
              __import__("re").findall(r"(?:src/)?ragdiag/(\w+)\.py", block)}
    assert listed == set(CORE), (
        f"README 에만 있음: {sorted(listed - set(CORE))} / "
        f"테스트에만 있음: {sorted(set(CORE) - listed)}"
    )


# ---------------------------------------------------------------------------
# 붙이는 법이 실제로 도는가
# ---------------------------------------------------------------------------

def test_core_accepts_a_foreign_conversation_object():
    """운영 장비의 파서가 만든 객체로도 출력이 조립돼야 한다.

    build_output 은 conversation_id 와 user 두 속성만 읽는다. 그 계약이 깨지면
    반입한 쪽에서 AttributeError 가 나고, 여기서는 conv.Conversation 을 쓰고
    있어서 끝까지 드러나지 않는다. README 의 예제를 그대로 박아 둔다.
    """
    from dataclasses import dataclass

    from ragdiag.pipeline import build_outcome, judge_cases, make_judge
    from ragdiag.schema import Case

    @dataclass
    class Meta:
        user_id: str
        raw_user_id: str
        db_login_id: str
        job_grade: str
        dept: str
        job_name: str
        position_name: str

    @dataclass
    class Conv:
        conversation_id: str
        user: Meta

    # 서비스 오류 문구는 코드로 판정되므로 LLM 없이 끝까지 간다.
    cases = [Case(
        case_id="C-9001:3", user_id="u1", dept="인사팀", job_grade="사원",
        job_name="인사", position_name="", conversation_id="C-9001", turn=3,
        pre_queries=["연차 이월 예외 조건 알려줘"],
        llm_ans_on_last_q="서비스에 문제가 있거나, 사용자 분들이 많아서 "
                          "서버에 부하가 걸리고 있어요.",
        current_query="답이 왜 안 나와요?",
        rag_chunks=["연차는 반차 단위로도 사용할 수 있다."],
    )]
    owners = [Conv("C-9001", Meta("u1", "raw1", "login1", "사원", "인사팀", "인사", ""))]

    # 백엔드가 None 인 것이 이 테스트의 절반이다. 확정 문구는 코드로 판정되므로
    # LLM 이 한 번도 불리면 안 되고, None 을 넣으면 불리는 순간 터져서 그게 드러난다.
    outcome = build_outcome(
        owners, judge_cases(cases, make_judge(None), workers=1))

    assert outcome.n_llm_calls == 0, "확정 문구는 LLM 없이 판정돼야 한다"
    assert outcome.n_failed == 0
    turn = outcome.payload["analysis_results"][0]["conversations"][0]["turns"][0]
    assert turn["classification"]["case_id"] == "case9"
