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

    **at.run() 은 한 번이 아니다.** 테스트가 버튼을 누를 때마다 스크립트가 처음부터
    다시 돌고, 그때도 인자를 다시 읽는다. 첫 실행 뒤에 sys.argv 를 되돌리면 두
    번째부터는 인자 없이 돌아서 output/ 의 최신 파일을 집는다 - 표는 fixture 인데
    버튼은 다른 파일을 넘기는 상태가 된다. 실제로 그렇게 통과하고 있었고, output/
    에 482행짜리 결과가 생기고 나서야 드러났다. 그래서 run 마다 인자를 끼운다.
    """
    import sys

    import streamlit as st

    argv = ["dashboard.py", "--result", str(result_file), *extra]
    st.cache_data.clear()   # 같은 프로세스에서 여러 번 돈다
    at = AppTest.from_file(str(DASHBOARD), default_timeout=120)
    bare_run = at.run

    def run_with_args(*a, **kw):
        saved, sys.argv = sys.argv, argv
        try:
            return bare_run(*a, **kw)
        finally:
            sys.argv = saved

    at.run = run_with_args
    at.run()
    return at


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


# ---------------------------------------------------------------------------
# 설정 파일 — 적어둔 값이 실제로 쓰여야 한다
#
# configs/env.yaml 에 paths.dept_class 를 두고도 대시보드가 CLI 인자만 읽었다.
# 설정 파일의 값이 아무것도 바꾸지 않는 것은 그 자체로 결함이다.
# ---------------------------------------------------------------------------

def _run_in(cwd, argv):
    import os
    import sys

    import streamlit as st

    saved_dir, saved_argv = os.getcwd(), sys.argv
    os.chdir(cwd)
    sys.argv = ["dashboard.py", *argv]
    try:
        st.cache_data.clear()
        at = AppTest.from_file(str(DASHBOARD), default_timeout=120)
        at.run()
        return at
    finally:
        os.chdir(saved_dir)
        sys.argv = saved_argv


def _work_folder(tmp_path, result_file, **paths):
    """{AA} 를 흉내낸다 - output/ 과 configs/env.yaml 이 있는 실행 위치."""
    import json
    import shutil

    (tmp_path / "output").mkdir()
    shutil.copy(result_file, tmp_path / "output" / "conv_parsed_20260101-000000.json")
    (tmp_path / "configs").mkdir()

    payload = json.loads(result_file.read_text(encoding="utf-8"))
    depts = [d for d in {u.get("db_dept_name", "") for u in payload["analysis_results"]} if d]
    (tmp_path / "configs" / "dept.json").write_text(json.dumps({"dept_classes": [
        {"id": 1, "name": "본부A", "subclasses": [{"name": "중A", "items": depts}]}]},
        ensure_ascii=False), encoding="utf-8")

    lines = ["paths:"] + [f"  {k}: {v}" for k, v in paths.items()]
    (tmp_path / "configs" / "env.yaml").write_text("\n".join(lines) + "\n",
                                                   encoding="utf-8")
    return tmp_path


def test_classification_path_can_come_from_the_config(result_file, tmp_path):
    """화면을 열 때마다 조직 분류 경로를 다시 치게 만들 이유가 없다."""
    work = _work_folder(tmp_path, result_file, dept_class="configs/dept.json")
    at = _run_in(work, [])

    assert not at.exception, [f"{e.type}: {e.message}" for e in at.exception]
    keys = {w.key for w in at.sidebar.multiselect}
    assert "부서:대분류" in keys, f"설정의 dept_class 가 무시됐다: {keys}"


def test_cli_beats_the_config(result_file, tmp_path):
    """우선순위는 파이프라인과 같다 — CLI > 설정 > 기본값."""
    import json

    work = _work_folder(tmp_path, result_file, dept_class="configs/dept.json")
    # 아무 것에도 안 붙는 체계를 CLI 로 준다. 그게 이기면 계층이 안 뜬다.
    (work / "configs" / "none.json").write_text(
        json.dumps({"dept_classes": [{"id": 1, "name": "X", "subclasses": [
            {"name": "y", "items": ["어디에도없는팀"]}]}]}, ensure_ascii=False),
        encoding="utf-8")

    at = _run_in(work, ["--dept-class", "configs/none.json"])
    keys = {w.key for w in at.sidebar.multiselect}
    assert "부서:대분류" not in keys, "CLI 가 설정을 못 이겼다"


def test_broken_config_does_not_stop_the_screen(result_file, tmp_path):
    """파이프라인은 설정이 깨지면 죽는 것이 맞지만, 여기서 죽으면 결과를 볼
    방법 자체가 사라진다. 무엇이 잘못됐는지는 화면에 남긴다."""
    work = _work_folder(tmp_path, result_file)
    (work / "configs" / "env.yaml").write_text("paths:\n  없는키: 1\n", encoding="utf-8")

    at = _run_in(work, [])
    assert not at.exception, [f"{e.type}: {e.message}" for e in at.exception]
    assert at.metric, "설정이 깨졌다고 화면이 안 뜨면 안 된다"
    assert any("설정을 읽지 못해" in w.value for w in at.warning), (
        "조용히 기본값으로 돌면 왜 다른지 알 수 없다")


def test_case_columns_carry_their_meaning_as_a_tooltip():
    """표 머리글이 case3 뿐이면 무슨 문제인지 알 수 없다.

    이름을 다 적으면 열이 넓어져 한 화면에 안 들어온다 - 머리글은 짧게 두고
    마우스를 올렸을 때 펼친다.
    """
    from ragdiag.taxonomy import tooltip

    text = tooltip("case3")
    assert text.startswith("case3 · 복합 질문을 함")
    # "무엇이 아닌가"까지 담아야 헷갈리는 쌍에서 갈린다.
    assert "case15" in text, text

    assert tooltip("case20").startswith("case20 · Retrieve 실패")
    assert tooltip("unclassified"), "분류 실패에도 무언가는 나와야 한다"


def test_heatmap_headers_get_the_tooltip(result_file):
    """히트맵 열이 case 다. 번호만 있으면 이 표를 못 읽는다.

    dashboard.py 는 streamlit 밖에서 import 하면 종료하므로(그게 맞다) 실제로
    그려진 것을 본다. 함수를 직접 부르면 배선이 빠져도 통과한다.
    """
    import json

    at = render(result_file)
    assert not at.exception

    tips = {}
    for frame in at.dataframe:
        columns = getattr(frame.proto, "columns", "") or ""
        if not columns:
            continue
        for name, spec in json.loads(columns).items():
            if spec.get("help"):
                tips[name] = spec["help"]

    # "case명" 도 case 로 시작한다. 실제 판정 라벨만 본다.
    import re

    labelled = [c for c in tips
                if re.fullmatch(r"case\d+", c) or c in ("unclassified", "out_of_taxonomy")]
    assert labelled, f"판정 라벨 열에 툴팁이 없다. 붙은 것: {sorted(tips)}"
    for cid in labelled:
        assert tips[cid].startswith(f"{cid} ·"), tips[cid]

    # 제일 헷갈리는 열이다. "분류 실패"를 "문제 없음"으로 읽으면 집계를 잘못 읽는다.
    if "unclassified" in tips:
        assert "문제 없음" in tips["unclassified"], tips["unclassified"]


def test_ratio_table_shows_the_count_too(result_file):
    """35% 가 2건 중 1건인지 40건 중 20건인지 모르면 그 성향을 믿을 수 없다."""
    import re

    at = render(result_file)
    assert not at.exception

    # 열 이름으로 표를 고르지 않는다 - 데이터에 따라 case 열이 하나도 없고
    # unclassified 만 있을 수 있다. 모든 표의 셀을 훑는다.
    cells = []
    for frame in at.dataframe:
        data = frame.value
        data = data.data if hasattr(data, "data") else data
        cells += [str(v) for row in data.values for v in row]

    ratios = [c for c in cells if re.fullmatch(r"\d+% \(\d+\)", c)]
    assert ratios, f"비율에 건수가 없다. 본 것: {sorted(set(cells))[:12]}"
    # 0 건은 0% 로 채우지 않는다 - 빈 칸이 많으면 눈이 갈 데를 못 찾는다.
    assert "0% (0)" not in cells


# ---------------------------------------------------------------------------
# 화면 배치 — 태블릿에서 읽히는가
# ---------------------------------------------------------------------------

def test_sections_are_split_into_tabs(result_file):
    """한 화면에 다 쌓으면 계속 스크롤해야 하고 어디가 끝인지 알 수 없다."""
    at = render(result_file)
    assert not at.exception
    assert [t.label for t in at.tabs] == ["분포", "조직", "코퍼스 보강", "개별 케이스"]


def test_health_metrics_stay_outside_the_tabs(result_file):
    """탭 안에 넣으면 건너뛸 수 있다. 이 숫자가 나쁘면 나머지를 믿을 수 없다는
    것이 이 화면의 전제다."""
    at = render(result_file)
    labels = [m.label for m in at.metric]
    assert "분류된 턴" in labels and "지어낸 인용" in labels

    # 탭 안에서 렌더된 지표는 그 탭의 자식으로 잡힌다. 건강 지표는 아니어야 한다.
    in_tabs = {m.label for t in at.tabs for m in t.metric}
    assert "지어낸 인용" not in in_tabs, "건강 지표가 탭 안으로 들어갔다"


def test_individual_cases_open_by_row_selection(result_file):
    """행을 눌러 펼친다. 열 열 개를 늘어놓으면 질문이 끝까지 밀려 안 보인다."""
    at = render(result_file)
    assert not at.exception

    frames = [f for t in at.tabs for f in t.dataframe]
    picked = [f for f in frames if "질문" in getattr(f.value, "columns", [])]
    assert picked, "개별 케이스 표를 못 찾았다"

    columns = list(picked[0].value.columns)
    assert len(columns) <= 7, f"열이 너무 많다: {columns}"
    assert "충족도" not in columns, "상세가 펼 것을 표에 다시 넣었다"

    # 아무것도 안 골라도 상세는 펼쳐져 있어야 한다 - 빈 화면이면 여기서
    # 무엇을 하는 곳인지 알 수 없다.
    assert any("판정 근거" in i.value for t in at.tabs for i in t.info), (
        "첫 행 상세가 안 펼쳐졌다")


def test_corpus_list_is_a_table_not_a_wall_of_text(result_file):
    """마크다운 목록이면 항목 스무 개가 회색 잔글씨 마흔 줄이 된다.

    문서팀에 그대로 넘기는 목록이라 정렬·검색·복사가 되는 표가 맞다.
    """
    at = render(result_file)
    frames = [f for t in at.tabs for f in t.dataframe]
    gap_tables = [f for f in frames
                  if "부서수" in getattr(f.value, "columns", [])]
    # 표가 있거나(케이스가 있을 때), 없으면 안내가 있어야 한다.
    assert gap_tables or any("해당 케이스가 없" in i.value
                             for t in at.tabs for i in t.info)

    # 원시 HTML 을 쓰지 않는다. 잔글씨로 밀어넣기 시작하면 좁은 화면에서
    # 읽히지 않는 것이 늘어난다 - st.caption 이 같은 일을 한다.
    source = (ROOT / "src" / "dashboard.py").read_text(encoding="utf-8")
    assert "unsafe_allow_html" not in source, "원시 HTML 이 남아 있다"


def test_detail_shows_what_was_judged_without_a_click(result_file):
    """이 패널은 "왜 이 라벨이지"에 답하는 자리다.

    판정 대상인 답변과, 라벨을 정한 사용자 발화가 접힘 상자 안에 있으면
    한 번 더 눌러야 나온다. 그 둘 없이는 답이 안 된다.
    """
    at = render(result_file)
    assert not at.exception

    texts = [b.value for t in at.tabs
             for b in list(t.info) + list(t.warning) + list(t.markdown)]
    joined = "\n".join(texts)
    # 한 턴의 이야기가 시간순으로 서 있어야 한다: 물었다 → 답했다 → 불만이다.
    assert "앞 질문" in joined, "답변이 무엇에 대한 답이었는지 안 보인다"
    assert "비판받은 답변" in joined, "판정 대상이 안 보인다"
    assert "사용자의 불만" in joined, "라벨을 정한 발화가 안 보인다"
    assert "판정 근거" in joined

    # 세 스텝이 다 자기 자리를 갖는다. Step 3 만 지표 하나로 끝나면
    # case22 와 case18 이 왜 갈렸는지 알 수 없다.
    for step in ("Step 1 · 관측", "Step 2 · 충족도", "Step 3 · 근거 활용"):
        assert step in joined, f"{step} 이 없다"


def test_bulky_things_stay_folded(result_file):
    """검색된 문서는 부피가 크다. 펼쳐 두면 그 아래가 안 보인다."""
    at = render(result_file)
    labels = [e.label for t in at.tabs for e in t.expander]
    assert any("검색된 문서" in l for l in labels), labels


def test_prev_and_next_walk_the_cases(result_file):
    """표에서 고르는 것만으로는 스무 건을 훑을 수 없다.

    한 건 보고 다음 건으로 가는 것이 이 화면에서 가장 잦은 동작이다.
    """
    at = render(result_file)
    tab = at.tabs[3]

    def position():
        return next(m.value for m in at.tabs[3].markdown
                    if m.value.startswith("**") and "/" in m.value)

    def button(key):
        return next(b for b in at.tabs[3].button if b.key == key)

    assert position().startswith("**1 /")
    assert button("prev-top").disabled, "첫 건에서 이전이 눌린다"

    button("next-top").click(); at.run()
    assert position().startswith("**2 /")
    button("next-top").click(); at.run()
    assert position().startswith("**3 /")
    button("prev-top").click(); at.run()
    assert position().startswith("**2 /")

    # 상세를 다 읽고 나면 화면 맨 아래다. 거기서도 옮길 수 있어야 한다.
    button("next-bottom").click(); at.run()
    assert position().startswith("**3 /")
    # 위아래가 같은 자리를 가리킨다 - 따로 놀면 어느 것이 맞는지 알 수 없다.
    shown = [m.value for m in at.tabs[3].markdown
             if m.value.startswith("**") and "/" in m.value]
    assert len(shown) == 2 and shown[0] == shown[1], shown


def test_navigation_stays_on_the_file_under_test(result_file):
    """버튼을 누른 뒤에도 같은 파일을 보고 있어야 한다.

    at.run() 은 스크립트를 처음부터 다시 돌리고 인자를 다시 읽는다. 인자가
    빠지면 대시보드는 output/ 의 최신 결과로 갈아타는데, 표는 fixture 를 그린
    채라 화면만 보면 알아채기 어렵다 - 실제로 그 상태로 통과하고 있었고,
    output/ 에 큰 결과가 생기고서야 드러났다.
    """
    at = render(result_file)

    def total():
        shown = next(m.value for m in at.tabs[3].markdown
                     if m.value.startswith("**") and "/" in m.value)
        return shown.split("/")[1].strip(" *")

    before = total()
    next(b for b in at.tabs[3].button if b.key == "next-top").click()
    at.run()
    assert total() == before, (
        f"버튼 한 번에 대상이 {before}건에서 {total()}건으로 바뀌었다 - "
        "재실행에서 --result 가 빠져 다른 파일을 읽고 있다")


def test_last_case_disables_next(result_file):
    """끝에서 다음을 누르면 없는 행을 가리킨다."""
    at = render(result_file)

    def button(key):
        return next(b for b in at.tabs[3].button if b.key == key)

    for _ in range(50):
        if button("next-top").disabled:
            break
        button("next-top").click(); at.run()
    assert button("next-top").disabled, "끝인데 다음이 계속 눌린다"
    assert button("next-bottom").disabled, "위아래 버튼 상태가 다르다"
    assert not at.exception


def test_result_file_can_be_picked_from_the_sidebar(result_file, tmp_path):
    """실행마다 결과가 쌓인다. 최신만 볼 수 있으면 설정을 바꿔가며 돌린 것을
    나란히 못 본다 - 대시보드를 다시 띄우지 않고 고른다."""
    import shutil
    import sys

    import streamlit as st

    out = tmp_path / "output"
    out.mkdir()
    for stamp in ("20260101-000000", "20260901-120000", "20260501-090000"):
        shutil.copy(result_file, out / f"conv_parsed_{stamp}.json")

    saved = sys.argv
    sys.argv = ["dashboard.py", "--output-dir", str(out)]
    try:
        st.cache_data.clear()
        at = AppTest.from_file(str(DASHBOARD), default_timeout=120)
        at.run()
    finally:
        sys.argv = saved

    assert not at.exception, [f"{e.type}: {e.message}" for e in at.exception]
    picker = next(w for w in at.sidebar.selectbox if w.label == "실행 결과")
    # 최신이 위에 오고 기본값이다 - 대개 방금 돌린 것을 본다.
    assert picker.options[0] == "20260901-120000", picker.options
    assert picker.value.endswith("conv_parsed_20260901-120000.json")


def test_explicit_result_is_not_replaced_by_the_picker(result_file, tmp_path):
    """--result 로 지목한 것이 --output-dir 밖일 수 있다.

    목록에 없다고 최신으로 갈아치우면 인자가 조용히 무시된다 - 실제로 그랬고,
    빈 결과를 넘겼는데 다른 실행이 떠서 테스트 둘이 깨졌다.
    """
    import json
    import sys

    import streamlit as st

    outside = tmp_path / "따로_둔_결과.json"
    outside.write_text(result_file.read_text(encoding="utf-8"), encoding="utf-8")

    saved = sys.argv
    sys.argv = ["dashboard.py", "--result", str(outside)]
    try:
        st.cache_data.clear()
        at = AppTest.from_file(str(DASHBOARD), default_timeout=120)
        at.run()
    finally:
        sys.argv = saved

    assert not at.exception, [f"{e.type}: {e.message}" for e in at.exception]
    picker = next(w for w in at.sidebar.selectbox if w.label == "실행 결과")
    assert picker.value == str(outside), f"인자가 무시됐다: {picker.value}"
    assert any(str(outside) in c.value for c in at.caption), "무엇을 보고 있는지 안 적혔다"


def _checked_row(at):
    frame = next(f.value for f in at.tabs[3].dataframe
                 if "보기" in getattr(f.value, "columns", []))
    return [i for i, on in enumerate(frame["보기"]) if on]


def test_buttons_move_the_check_mark(result_file):
    """버튼으로 옮겼는데 체크가 옛 행에 남으면 어느 쪽이 지금인지 알 수 없다.

    st.dataframe 의 선택은 프로그램으로 못 옮긴다 - session_state 에 넣어도
    조용히 무시된다. 그래서 체크 상태를 우리가 쥐는 data_editor 를 쓴다.
    """
    at = render(result_file)

    def button(key):
        return next(b for b in at.tabs[3].button if b.key == key)

    assert _checked_row(at) == [0]

    button("next-top").click(); at.run()
    assert _checked_row(at) == [1], "위 버튼이 체크를 안 옮겼다"

    button("next-bottom").click(); at.run()
    assert _checked_row(at) == [2], "아래 버튼이 체크를 안 옮겼다"

    button("prev-top").click(); at.run()
    assert _checked_row(at) == [1]

    # 체크·위치·펼쳐진 케이스가 한 곳을 가리켜야 한다.
    position = next(m.value for m in at.tabs[3].markdown
                    if m.value.startswith("**") and "/" in m.value)
    assert position.startswith("**2 /"), position


def test_only_one_row_is_checked(result_file):
    """여러 개가 켜져 있으면 무엇이 펼쳐진 것인지 알 수 없다."""
    at = render(result_file)

    def button(key):
        return next(b for b in at.tabs[3].button if b.key == key)

    for _ in range(3):
        if button("next-top").disabled:
            break
        button("next-top").click(); at.run()
        assert len(_checked_row(at)) == 1, _checked_row(at)


def test_the_judged_answer_gets_the_widest_column(result_file):
    """셋 중 답변이 제일 길고 판정 대상이기도 하다.

    같은 폭을 주면 답변만 여러 줄로 접히고 양옆은 비어서, 정작 봐야 할 것이
    제일 읽기 나쁘다.
    """
    at = render(result_file)
    assert not at.exception

    weights = [round(c.proto.weight, 3) for c in at.tabs[3].get("column")
               if getattr(getattr(c, "proto", None), "weight", None)]
    assert [0.2, 0.6, 0.2] in [weights[i:i + 3] for i in range(len(weights) - 2)], (
        f"1:3:1 인 줄이 없다: {weights}")
