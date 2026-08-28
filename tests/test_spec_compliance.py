"""구현 규격(IMPLEMENTATION_SPEC.md)을 코드로 지킨다.

규격의 규칙은 전부 제약에서 따라나온 것이고, 어긋나면 사내 사이클 하나를 버린다.
문서로만 두면 다음에 파일 하나 추가하면서 조용히 깨진다.

여기서 재는 것:
  C6  가짜 데이터가 파일로 저장소에 있으면 이식을 통해 사내로 흘러간다
  C3  합성 데이터가 실데이터의 값을 흉내내면 그것도 유출 경로다
  1.3 바뀔 만한 값이 설정에 없으면 사내에서 "코드 한 줄만" 이 된다
  3.2 계약 위반 메시지가 옮겨 적을 수 없으면 포맷 회수가 끊긴다
"""

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

DATA_SUFFIXES = (".csv", ".tsv", ".parquet", ".xlsx", ".pkl", ".npy", ".npz",
                 ".h5", ".feather", ".sqlite", ".sqlite3", ".jsonl")


def tracked() -> list[str]:
    proc = subprocess.run(["git", "-C", str(ROOT), "ls-files"],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.skip("git 저장소가 아님")
    return proc.stdout.splitlines()


# ---------------------------------------------------------------------------
# C6 · 2.3 — 데이터가 저장소에 없다
# ---------------------------------------------------------------------------

def test_no_data_files_are_tracked():
    """규격 2.3 의 점검 명령을 그대로 돌린다."""
    bad = [f for f in tracked()
           if f.endswith(DATA_SUFFIXES) or f.startswith(".staging")]
    assert not bad, (
        "데이터성 파일이 추적되고 있다. 이식하면 사내 저장소로 넘어간다:\n"
        + "\n".join(f"  {f}" for f in bad))


def test_no_json_fixtures_are_tracked():
    """가짜 데이터는 파일이 아니라 코드다 (규격 1.2).

    .gitignore 는 `git add -f` 한 번에 뚫린다. 없는 파일은 올라갈 수 없다.
    """
    bad = [f for f in tracked() if f.endswith(".json") and "package.json" not in f]
    assert not bad, (
        "JSON 데이터가 추적되고 있다:\n" + "\n".join(f"  {f}" for f in bad))


def test_local_config_is_not_tracked():
    """사내 실값이 밖으로 나가면 안 된다."""
    bad = [f for f in tracked() if "local.yaml" in f or f.endswith(".env")]
    assert not bad, f"사내 설정이 추적되고 있다: {bad}"


def test_synth_generates_at_runtime_not_from_a_file():
    """합성 데이터가 파일을 읽지 않아야 한다."""
    source = (ROOT / "src/ragdiag/fixtures/synth.py").read_text(encoding="utf-8")
    for banned in ("open(", "read_text", "json.load(", "Path("):
        assert banned not in source, (
            f"synth.py 가 {banned} 을 쓴다. 파일에서 읽으면 그 파일이 "
            "저장소에 있어야 하고, 그러면 사내로 흘러간다.")


def test_synth_is_deterministic():
    from ragdiag.fixtures.synth import generate

    assert generate(seed=0) == generate(seed=0)
    assert generate(seed=0) != generate(seed=1)


def test_synth_output_satisfies_the_contract():
    """계약이 바뀌면 합성 데이터도 따라 바뀌어야 한다.

    어긋나면 여기서는 도는 코드가 사내에서 죽는다.
    """
    from ragdiag.contracts import check_log
    from ragdiag.fixtures.synth import generate

    report = check_log(generate(seed=0))
    assert report.ok, "\n".join(m.line() for m in report.mismatches)


# ---------------------------------------------------------------------------
# C3 · 3.2 — 실데이터 값이 화면으로 새지 않는다
# ---------------------------------------------------------------------------

def test_contract_mismatch_reports_types_not_values():
    """계약 위반 메시지에 실제 값이 실리면 그게 유출 경로다.

    타입 이름과 건수까지만 적는다.
    """
    from ragdiag.contracts import TURN_SCHEMA, validate

    rows = [{"turn": "비밀사번-12345", "user_question": "김철수 연봉이 얼마야"}]
    for m in validate(rows, TURN_SCHEMA, "turn"):
        assert "비밀사번" not in m.line()
        assert "김철수" not in m.line()


def test_contract_mismatch_is_transcribable():
    """`validation failed` 같은 메시지는 규격에서 결함이다.

    무엇이 어떻게 어긋났는지가 나와야 사람이 옮겨 적고 contracts.py 를 고친다.
    """
    from ragdiag.contracts import TURN_SCHEMA, validate

    rows = [{"turn": "1", "user_question": "질문", "새필드": 1}]
    lines = [m.line() for m in validate(rows, TURN_SCHEMA, "turn")]
    assert any("turn.turn" in l and "str" in l for l in lines), lines
    assert any("새필드" in l for l in lines), "새 필드는 인사이트가 된다"


def test_run_summary_prints_every_required_row():
    """규격 3.2 의 블록. 성공·실패 무관하게 이 줄들이 있어야 한다."""
    from ragdiag.contracts import Mismatch
    from ragdiag.summary import RunSummary

    text = RunSummary(
        version="v0.4 (a1b2c3d)", input_shape="1,204,331 rows",
        contract_ok=24,
        contract_mismatches=[Mismatch("turn", "grade", "allowed", "허용값 밖 12건")],
        metrics=[("auc", "0.8123")], runtime_sec=412, peak_gb=6.2,
        status="OK",
    ).render()
    for row in ("version", "input", "contract", "metrics", "runtime", "status"):
        assert f"{row:<10}:" in text, f"{row} 줄이 없다:\n{text}"
    assert "MISMATCH" in text
    assert "turn.grade" in text


def test_run_summary_lines_fit_80_columns():
    """옮겨 적는 사람을 위해 한 줄에 한 항목, 80자 이내."""
    from ragdiag.contracts import Mismatch
    from ragdiag.summary import RunSummary

    text = RunSummary(
        version="v" * 40, input_shape="x" * 80,
        contract_mismatches=[Mismatch("turn", "f" * 40, "k", "d" * 80)],
        metrics=[("m" * 40, "v" * 60)],
    ).render()
    long = [l for l in text.splitlines() if len(l) > 80]
    assert not long, "80자를 넘는 줄:\n" + "\n".join(long)


def test_version_is_never_blank():
    """빈칸이면 옮겨 적을 때 빠진다. 모르면 모른다고 적는다."""
    from ragdiag.summary import version

    assert version().strip()


# ---------------------------------------------------------------------------
# 1.3 — 바뀔 만한 값이 전부 설정에 있다
# ---------------------------------------------------------------------------

def test_example_config_is_committed():
    assert "configs/example.yaml" in tracked()


def test_sync_script_is_committed_and_executable():
    assert "scripts/sync.sh" in tracked()
    assert (ROOT / "scripts/sync.sh").stat().st_mode & 0o111


def test_sync_derives_names_instead_of_hardcoding_them():
    """규격: {BB} 는 스크립트 위치에서, <pkg> 는 src/ 아래에서 유도한다.

    이름을 박아 두면 저장소나 패키지 이름이 바뀔 때 사내에서 조용히 엉뚱한
    경로를 만든다. 거기서는 고칠 수 없다.
    """
    text = (ROOT / "scripts/sync.sh").read_text(encoding="utf-8")
    assert 'STAGING=".staging/BB"' not in text, "저장소 이름이 박혀 있다"
    assert 'DEST="BB"' not in text, "사본 경로가 박혀 있다"
    assert "BASH_SOURCE" in text, "스크립트 위치에서 이름을 유도해야 한다"
    assert '"$DEST"/src/*.py' in text, "src/ 에서 진입점을 찾아야 한다"


def test_sync_refuses_outside_the_work_root():
    """AA 루트가 아닌 곳에서 돌면 엉뚱한 자리에 사본을 만든다."""
    proc = subprocess.run(["bash", str(ROOT / "scripts/sync.sh"), "v0.0"],
                          capture_output=True, text=True, cwd="/tmp")
    assert proc.returncode != 0
    assert "AA 루트" in proc.stderr


def test_sync_refuses_without_a_tag():
    """태그 없이 실행하지 않는다 (규격 '하지 말 것' 4)."""
    proc = subprocess.run(["bash", str(ROOT / "scripts/sync.sh")],
                          capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode != 0
    assert "태그" in proc.stderr


def test_pyproject_declares_runtime_deps():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "pydantic" in text
    assert "PyYAML" in text, "--config 를 쓰려면 필요하다"
    # 대시보드는 선택이어야 한다. 사내 반입 부담을 늘리지 않는다.
    assert "streamlit" in text.split("optional-dependencies")[1]


def _fake_repo(tmp_path, with_launcher=True, extra_req=""):
    """sync.sh 만 시험하기 위한 최소 저장소. 이 저장소의 git 상태에 기대지 않는다."""
    import subprocess

    bb = tmp_path / "toolkit"
    (bb / "scripts").mkdir(parents=True)
    (bb / "src" / "somepkg").mkdir(parents=True)
    (bb / "src" / "somepkg" / "__init__.py").write_text("", encoding="utf-8")
    (bb / "configs").mkdir()
    (bb / "configs" / "example.yaml").write_text("a: 1\n", encoding="utf-8")
    (bb / "requirements.txt").write_text("pydantic>=2.0\n" + extra_req, encoding="utf-8")
    (bb / "scripts" / "sync.sh").write_text(
        (ROOT / "scripts/sync.sh").read_text(encoding="utf-8"), encoding="utf-8")
    if with_launcher:
        (bb / "src" / "run.py").write_text("print('hi')\n", encoding="utf-8")
    run = lambda *a: subprocess.run(["git", "-C", str(bb), *a],
                                    capture_output=True, text=True, check=True)
    run("init", "-q")
    run("config", "user.email", "t@t"); run("config", "user.name", "t")
    run("add", "-A"); run("commit", "-q", "-m", "x")
    return bb


def _sync(aa, bb, tag):
    import subprocess
    return subprocess.run(["bash", ".staging/toolkit/scripts/sync.sh", tag],
                          capture_output=True, text=True, cwd=aa)


def _fresh_aa(tmp_path, bb):
    import subprocess

    aa = tmp_path / "work"
    (aa / ".staging").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(aa)], check=True)
    subprocess.run(["git", "clone", "-q", str(bb), str(aa / ".staging" / "toolkit")],
                   check=True)
    return aa


def test_sync_hint_skips_pip_when_dependencies_did_not_change(tmp_path):
    """규격 부록에서 이 블록만 갈라져 있다. 갈라진 이유가 지워지지 않게 고정한다.

    부록은 매번 pip 와 PYTHONPATH 를 찍는다. 둘 다 늘 필요한 것이 아닌데,
    사람은 여기 찍힌 줄을 그대로 따라간다. 필요 없는 줄이 섞이면 매번 안 해도
    될 일을 하거나 - 더 나쁘게는 - 이 안내를 통째로 안 믿게 된다.
    """
    import subprocess

    bb = _fake_repo(tmp_path)
    subprocess.run(["git", "-C", str(bb), "tag", "v1"], check=True)
    aa = _fresh_aa(tmp_path, bb)

    first = _sync(aa, bb, "v1")
    assert first.returncode == 0, first.stderr
    assert "pip install" in first.stdout, "최초에는 의존 설치를 안내해야 한다"

    again = _sync(aa, bb, "v1")
    assert again.returncode == 0, again.stderr
    assert "pip install" not in again.stdout, (
        "의존이 그대로인데 pip 를 또 안내한다:\n" + again.stdout)


def test_sync_hint_warns_when_dependencies_changed(tmp_path):
    import subprocess

    bb = _fake_repo(tmp_path)
    subprocess.run(["git", "-C", str(bb), "tag", "v1"], check=True)
    aa = _fresh_aa(tmp_path, bb)
    _sync(aa, bb, "v1")

    (bb / "requirements.txt").write_text("pydantic>=2.0\nrequests>=2\n", encoding="utf-8")
    for a in (["add", "-A"], ["commit", "-q", "-m", "dep"], ["tag", "v2"]):
        subprocess.run(["git", "-C", str(bb), *a], check=True)

    out = _sync(aa, bb, "v2")
    assert out.returncode == 0, out.stderr
    assert "requirements.txt 가 바뀌었습니다" in out.stdout, out.stdout
    assert "pip install" in out.stdout


def test_sync_hint_points_at_the_entry_script(tmp_path):
    """규격이 정한 src/run.py 를 안내한다. PYTHONPATH 는 필요 없다."""
    import subprocess

    bb = _fake_repo(tmp_path, with_launcher=True)
    subprocess.run(["git", "-C", str(bb), "tag", "v1"], check=True)
    out = _sync(_fresh_aa(tmp_path, bb), bb, "v1")
    assert "python toolkit/src/run.py" in out.stdout, out.stdout
    assert "PYTHONPATH" not in out.stdout


def test_sync_hint_marks_the_entry_when_it_cannot_be_found(tmp_path):
    """진입점을 못 찾으면 자리표시자를 찍는다. 틀린 명령을 주는 것보다 낫다."""
    import subprocess

    bb = _fake_repo(tmp_path, with_launcher=False)
    subprocess.run(["git", "-C", str(bb), "tag", "v1"], check=True)
    out = _sync(_fresh_aa(tmp_path, bb), bb, "v1")
    assert "toolkit/src/<entry>.py" in out.stdout, out.stdout
