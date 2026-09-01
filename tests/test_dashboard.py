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
    import sys

    sys.path.insert(0, str(ROOT / "tests"))
    from stub_llm import StubLLM

    from ragdiag.backends import OpenAICompatBackend
    from ragdiag.pipeline import build_outcome, judge_cases, load_and_select, make_judge
    from ragdiag.fixtures.synth import generate

    tmp = tmp_path_factory.mktemp("dash")
    log = tmp / "conv_eval.json"
    log.write_text(json.dumps(generate(n=4, seed=0), ensure_ascii=False), encoding="utf-8")

    selection = load_and_select(log)
    # 개발 장비 전용 백엔드에 묶어두면 깨끗한 사본에서 이 테스트가 통째로
    # 건너뛰어진다. 대시보드는 운영 환경에서 쓰는 화면이라 그러면 안 된다.
    with StubLLM() as stub:
        backend = OpenAICompatBackend(base_url=stub.url, model="stub-model",
                                      api_key="stub", timeout=30)
        results = judge_cases(selection.cases[:8], make_judge(backend), workers=2)
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
    """조직 분류 JSON 을 안 줘도 돌아야 한다. 운영 환경에 그 파일이 없을 수 있다."""
    at = render(result_file)
    assert not at.exception, [f"{e.type}: {e.message}" for e in at.exception]


# ---------------------------------------------------------------------------
# 잘못 실행했을 때 무엇을 하라고 하는가
#
# 운영 환경에서는 맨 트레이스백 하나가 사이클을 먹는다. 인터넷도 없고 물어볼 곳도
# 없어서 화면에 적힌 것이 전부다.
# ---------------------------------------------------------------------------

def test_running_with_plain_python_says_to_use_streamlit():
    """python src/dashboard.py 는 경고만 쏟고 화면이 안 뜬다. 실패로도 안 끝난다."""
    import subprocess
    import sys

    proc = subprocess.run([sys.executable, str(DASHBOARD)],
                          capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "-m streamlit run" in proc.stderr, (
        "PATH 를 타지 않는 형태로 안내해야 한다. `streamlit run` 만 쓰면 venv 를 "
        "activate 하지 않았을 때 command not found 가 난다.\n" + proc.stderr)
    assert sys.executable in proc.stderr, "지금 도는 인터프리터를 그대로 알려준다"
    assert "--" in proc.stderr, "`--` 가 필요하다는 것도 알려야 한다"
    assert str(DASHBOARD) in proc.stderr, (
        "사용자가 친 경로를 그대로 돌려줘야 한다. 사본 위치가 배포마다 다르다.")


def test_missing_dependency_names_the_install_command(tmp_path):
    """pandas·streamlit 이 없을 때 무엇을 깔라는지 적어야 한다."""
    import subprocess
    import sys
    import textwrap

    # 의존이 없는 상태를 흉내낸다 - 실제로 지울 수는 없다.
    stub = tmp_path / "block.py"
    stub.write_text(textwrap.dedent("""
        import builtins
        _real = builtins.__import__
        def _blocked(name, *a, **k):
            if name.split(".")[0] == "pandas":
                raise ModuleNotFoundError("No module named 'pandas'", name="pandas")
            return _real(name, *a, **k)
        builtins.__import__ = _blocked
        import runpy, sys
        sys.argv = [%r]
        runpy.run_path(sys.argv[0], run_name="__main__")
    """) % str(DASHBOARD), encoding="utf-8")

    proc = subprocess.run([sys.executable, str(stub)],
                          capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "-m pip install -r" in proc.stderr, (
        "PATH 를 타지 않는 형태로, 그리고 어떤 파일을 쓸지까지 알려야 한다\n"
        + proc.stderr)
    assert "requirements-dashboard.txt" in proc.stderr, proc.stderr
    assert "분류 파이프라인" in proc.stderr, (
        "파이프라인에는 필요 없다는 것도 알려야 반입 부담을 안 늘린다")


def test_corpus_gaps_group_by_need_not_department(result_file):
    """같은 요구를 부서로 쪼개면 가장 강한 신호가 숨는다.

    "연차 이월 예외 조건"이 세 부서에서 나왔는데 부서로 먼저 묶으면 세 줄로
    흩어져 "여러 부서가 같은 것에 막혔다"가 안 보인다. 문서팀이 받는 것은
    "쓸 문서 목록"이므로 요구가 축이어야 한다.
    """
    at = render(result_file)
    assert not at.exception
    body = "\n".join(m.value for m in at.markdown)
    header = [s.value for s in at.subheader if "코퍼스 보강" in s.value]
    assert header, "코퍼스 보강 목록 절이 없다"
    assert "종)" in header[0], "건수가 아니라 요구 '종' 수를 세야 한다"
    # 같은 요구가 여러 번 나왔다면 묶여서 ×N 이 붙어야 한다
    import re
    if re.search(r"`×\d+`", body):
        assert "묶였다" in "\n".join(c.value for c in at.caption), (
            "몇 건이 몇 종으로 묶였는지 알려줘야 한다")


def test_detail_panel_does_not_claim_zero_llm_involvement(result_file):
    """llm_calls 는 이번 실행의 호출 수다. 캐시가 맞으면 0 이 된다.

    그대로 "LLM 호출 0" 이라 적으면 case22 처럼 LLM 이 세 번 필요한 판정도
    LLM 이 관여 안 한 것처럼 읽힌다.
    """
    at = render(result_file)
    labels = [m.label for m in at.metric]
    assert "LLM 호출" not in labels, (
        "이번 실행의 호출 수를 판정 근거처럼 보여주고 있다. "
        "어디까지 봤는지(판정 단계)를 보여줄 것.")
    assert "판정 단계" in labels, f"판정 단계 지표가 없다: {labels}"


def test_dashboard_picks_the_newest_result(tmp_path):
    """파일명에 시각이 붙어 고정 경로를 기본값으로 둘 수 없다.

    --result 를 안 주면 --output-dir 에서 가장 최근 것을 골라야 한다.
    """
    import json
    import sys

    import streamlit as st

    out = tmp_path / "output"
    out.mkdir()
    payload = json.loads((ROOT / "outputs" / "conv_parsed.json").read_text(encoding="utf-8")) \
        if (ROOT / "outputs" / "conv_parsed.json").exists() else {"analysis_results": []}
    for stamp in ("20260101-000000", "20260831-153018", "20260501-120000"):
        (out / f"conv_parsed_{stamp}.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    saved = sys.argv
    sys.argv = ["dashboard.py", "--output-dir", str(out)]
    try:
        st.cache_data.clear()
        at = AppTest.from_file(str(DASHBOARD), default_timeout=120)
        at.run()
    finally:
        sys.argv = saved

    assert not at.exception, [f"{e.type}: {e.message}" for e in at.exception]
    shown = " ".join(c.value for c in at.caption)
    assert "20260831-153018" in shown, f"최신을 안 골랐다: {shown[:120]}"
    assert "다른 실행 2건" in shown, "몇 건이 더 있는지 알려줘야 한다"


def test_sidebar_offers_three_levels_when_a_classification_is_given(result_file, tmp_path):
    """대 → 중 → 소 로 좁혀 고른다. 팀이 수십 개면 평평한 목록에서는 못 고른다."""
    import json
    import sys

    import streamlit as st

    # 결과에 실제로 있는 부서로 체계를 만든다. 안 그러면 매칭률이 낮아
    # detect_field 가 필드를 못 고르고 층이 아예 안 뜬다.
    payload = json.loads(result_file.read_text(encoding="utf-8"))
    depts = sorted({u.get("db_dept_name", "") for u in payload["analysis_results"]})
    depts = [d for d in depts if d]
    scheme = tmp_path / "dept.json"
    scheme.write_text(json.dumps({"dept_classes": [
        {"id": 1, "name": "본부A", "subclasses": [
            {"name": "중분류A", "items": depts}]}]}, ensure_ascii=False),
        encoding="utf-8")

    saved = sys.argv
    sys.argv = ["dashboard.py", "--result", str(result_file),
                "--dept-class", str(scheme)]
    # 다시 돌리는 것까지 argv 를 물고 있어야 한다. finally 뒤에서 at.run() 을
    # 부르면 --dept-class 가 사라져 분류 체계가 안 붙고, 그러면 이 테스트는
    # 계층이 아니라 평평한 선택기를 재게 된다.
    try:
        st.cache_data.clear()
        at = AppTest.from_file(str(DASHBOARD), default_timeout=120)
        at.run()
        assert not at.exception, [f"{e.type}: {e.message}" for e in at.exception]
        _check_levels(at)
    finally:
        sys.argv = saved


def _check_levels(at):
    # 축마다 상자 하나다. 라벨은 "대분류"로 같으므로 key 로 찾는다 - 그래서
    # 위젯에 key 를 준 것이기도 하다 (같은 라벨 둘은 streamlit 이 거절한다).
    def keys():
        return {w.key for w in at.sidebar.multiselect}

    def pick(key):
        return next(w for w in at.sidebar.multiselect if w.key == key)

    # 아래층은 위층을 고른 뒤에만 보인다. 셋을 한꺼번에 늘어놓으면 무엇이
    # 무엇에 매이는지 안 보인다.
    assert "부서:대분류" in keys()
    assert "부서:중분류" not in keys(), "고르기 전부터 중분류가 보인다"

    pick("부서:대분류").set_value([pick("부서:대분류").options[0]]); at.run()
    assert "부서:중분류" in keys()
    assert "부서:소분류" not in keys(), "중분류를 고르기 전부터 소분류가 보인다"

    pick("부서:중분류").set_value([pick("부서:중분류").options[0]]); at.run()
    assert "부서:소분류" in keys()

    # 다른 축은 영향을 받지 않는다 - 상자를 나눈 이유다.
    assert "직급:중분류" not in keys(), "부서를 골랐는데 직급 층이 열렸다"

    # 위층을 비우면 아래층도 접힌다 - 안 그러면 안 보이는 조건이 남아 거른다.
    pick("부서:대분류").set_value([]); at.run()
    assert "부서:중분류" not in keys() and "부서:소분류" not in keys()


def test_switching_the_major_drops_a_middle_that_no_longer_exists(result_file, tmp_path):
    """대분류를 A 에서 B 로 갈면 A 밑에서 고른 중분류는 후보에 없다.

    그대로 두면 아무 데이터에도 안 맞아 0건이 나오는데, 화면에는 여전히 선택된
    것처럼 보인다. 조건을 걸었는데 결과가 비는 것보다 나쁜 건 왜 비는지 모르는 것이다.
    """
    import json
    import sys

    import streamlit as st

    # 합성 생성기는 사용자를 부서별로 몰아서 내므로 앞쪽 몇 건은 한 부서다.
    # 갈아치우는 상황을 보려면 부서가 둘이어야 해서, 한 사용자를 복제해 부서만 바꾼다.
    payload = json.loads(result_file.read_text(encoding="utf-8"))
    users = payload["analysis_results"]
    twin = json.loads(json.dumps(users[0]))
    twin["user_id"] = twin.get("user_id", "u") + "-twin"
    twin["db_dept_name"] = "복제부서"
    payload["analysis_results"] = users + [twin]

    two_dept = tmp_path / "two_dept.json"
    two_dept.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    first = users[0].get("db_dept_name", "")

    scheme = tmp_path / "dept2.json"
    scheme.write_text(json.dumps({"dept_classes": [
        {"id": 1, "name": "본부A", "subclasses": [{"name": "중A", "items": [first]}]},
        {"id": 2, "name": "본부B", "subclasses": [{"name": "중B", "items": ["복제부서"]}]},
    ]}, ensure_ascii=False), encoding="utf-8")

    saved = sys.argv
    sys.argv = ["dashboard.py", "--result", str(two_dept), "--dept-class", str(scheme)]
    try:
        st.cache_data.clear()
        at = AppTest.from_file(str(DASHBOARD), default_timeout=120)
        at.run()

        def pick(k):
            return next(w for w in at.sidebar.multiselect if w.key == k)


        pick("부서:대분류").set_value(["본부A"]); at.run()
        pick("부서:중분류").set_value(["중A"]); at.run()
        pick("부서:대분류").set_value(["본부B"]); at.run()

        assert not at.exception, [f"{e.type}: {e.message}" for e in at.exception]
        assert pick("부서:중분류").options == ["중B"]
        assert at.metric[0].value != 0, "사라진 선택이 남아 0건이 됐다"
    finally:
        sys.argv = saved


def test_sidebar_stays_flat_without_a_classification(result_file):
    """체계가 없는 축에 없는 층을 만들어 보여주면 고를 수 있는 것처럼 보인다."""
    at = render(result_file)
    assert not at.exception
    labels = [w.label for w in at.sidebar.multiselect]
    keys = {w.key for w in at.sidebar.multiselect}
    assert "부서" in labels, labels
    assert "부서:대분류" not in keys
