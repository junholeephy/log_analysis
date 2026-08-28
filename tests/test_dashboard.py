"""대시보드가 실제로 렌더되는가.

**서버가 뜨는 것으로는 확인이 안 된다.** streamlit 은 스크립트를 브라우저가
붙을 때 실행하고 예외를 화면에만 보여준다. 그래서 헬스체크는 통과하고 로그도
깨끗한데 화면만 죽어 있는 상태가 된다 - 실제로 한 번 그렇게 깨졌다.

깨진 원인은 배치였다. streamlit 은 스크립트가 있는 디렉터리를 sys.path[0] 에
넣으므로, dashboard.py 가 src/ragdiag/ 안에 있으면 그 디렉터리가 올라가고
src/ 는 안 올라가서 `import ragdiag` 가 자기 자신을 못 찾는다.

여기서는 AppTest 로 스크립트를 끝까지 실행해 예외를 본다.
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "src" / "dashboard.py"

AppTest = pytest.importorskip(
    "streamlit.testing.v1", reason="streamlit 이 없으면 대시보드는 선택 기능이다").AppTest


@pytest.fixture(scope="module")
def result_file(tmp_path_factory):
    """합성 데이터를 분류한 결과. 파일로 저장소에 두지 않는다."""
    from ragdiag.backends import ClaudeCodeBackend
    from ragdiag.pipeline import build_outcome, judge_cases, load_and_select, make_judge
    from ragdiag.fixtures.synth import generate

    tmp = tmp_path_factory.mktemp("dash")
    log = tmp / "conv_eval.json"
    log.write_text(json.dumps(generate(n=4, seed=0), ensure_ascii=False), encoding="utf-8")

    selection = load_and_select(log)
    results = judge_cases(selection.cases[:8], make_judge(ClaudeCodeBackend()), workers=2)
    outcome = build_outcome(selection.owners[:8], results)
    out = tmp / "conv_parsed.json"
    outcome.save(out)
    return out


def render(result_file, *extra, monkeypatch=None):
    """대시보드를 끝까지 실행하고 결과를 돌려준다.

    인자는 sys.argv 로 준다. AppTest 에는 args 속성이 없어서 at.args = [...] 는
    그냥 새 속성을 만들 뿐 아무 일도 하지 않는다 - 그렇게 짰다가 모든 테스트가
    기본 경로(outputs/conv_parsed.json)를 읽고 있었다. 통과했지만 재고 있던
    것이 아니었다.
    """
    import sys

    import streamlit as st

    saved = sys.argv
    sys.argv = ["dashboard.py", "--result", str(result_file), *extra]
    try:
        st.cache_data.clear()   # 같은 프로세스에서 여러 번 돈다
        at = AppTest.from_file(str(DASHBOARD), default_timeout=120)
        at.run()
        return at
    finally:
        sys.argv = saved


def test_dashboard_lives_directly_under_src():
    """src/ragdiag/ 안으로 옮기면 import 가 깨진다. 자리를 고정한다."""
    assert DASHBOARD.exists(), (
        "대시보드는 src/ 직하에 있어야 한다. streamlit 이 스크립트 디렉터리를 "
        "sys.path[0] 에 넣으므로 src/ragdiag/ 안에 두면 ragdiag 를 못 찾는다.")
    assert not (ROOT / "src" / "ragdiag" / "dashboard.py").exists()


def test_dashboard_renders_without_pythonpath(result_file, monkeypatch):
    """PYTHONPATH 를 지운 상태에서도 import 가 풀려야 한다."""
    monkeypatch.delenv("PYTHONPATH", raising=False)
    at = render(result_file)
    assert at.metric, "아무것도 렌더되지 않았다 - 인자가 안 먹었을 수 있다"
    assert not at.exception, [f"{e.type}: {e.message}" for e in at.exception]


def test_dashboard_shows_the_health_metrics_first(result_file):
    """라벨 분포보다 판정 건강이 먼저다. 여기가 나쁘면 아래 집계를 믿을 수 없다."""
    at = render(result_file)
    labels = [m.label for m in at.metric]
    for expected in ("분류된 턴", "지어낸 인용", "신뢰도 낮음", "미분류", "서비스 오류"):
        assert expected in labels, f"{expected} 지표가 없다. 있는 것: {labels}"


def test_dashboard_survives_an_empty_result(tmp_path):
    """분류된 턴이 없을 때 죽지 않고 안내해야 한다."""
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"analysis_results": []}), encoding="utf-8")
    at = render(empty)
    assert not at.exception, [f"{e.type}: {e.message}" for e in at.exception]
    assert any("분류된 턴이 없" in w.value for w in at.warning), (
        "빈 결과에 안내가 없다")


def test_dashboard_does_not_need_matplotlib(result_file):
    """히트맵에 matplotlib 을 쓰면 에어갭 반입 부담이 늘어난다.

    Styler.background_gradient 가 matplotlib 을 요구해 한 번 깨졌다.
    """
    import sys

    at = render(result_file)
    assert not at.exception
    assert "matplotlib" not in sys.modules, (
        "matplotlib 이 import 됐다. 대시보드 하나 때문에 반입할 패키지를 늘리지 않는다.")


def test_dashboard_works_without_org_classification(result_file):
    """조직 분류 JSON 을 안 줘도 돌아야 한다. 사내에 그 파일이 없을 수 있다."""
    at = render(result_file)
    assert not at.exception, [f"{e.type}: {e.message}" for e in at.exception]
