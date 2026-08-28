"""설정이 실제로 먹히는가.

사내에서는 코드를 한 줄도 못 고친다. 설정에 적은 값이 조용히 무시되면 그 사이클을
통째로 버린다 - 임계값을 바꿨는데 안 바뀐 채로 30분이 지나가고, 결과만 보고는
알 수 없다.

여기서 두 가지를 지킨다.
  1. 잘못된 설정은 **계산을 시작하기 전에** 죽는다
  2. 맞는 설정은 **검증기와 파서까지 실제로 도달한다** (import 시점 스냅샷이
     아니라 호출 시점에 읽히는가)

2번이 조용히 깨지는 게 특히 위험하다. 파이썬 기본 인자는 def 시점에 굳어서
`def f(n=settings.X)` 는 나중에 settings.X 를 바꿔도 안 먹는다.
"""

import textwrap

import pytest

from ragdiag import settings
from ragdiag.backends import env_first
from ragdiag.config import Config, ConfigError, apply, flatten, load, validate
from ragdiag.settings import KEY_VARS, URL_VARS

yaml = pytest.importorskip("yaml")


@pytest.fixture
def restore_settings():
    """설정 적용은 모듈 전역을 바꾼다. 테스트끼리 새지 않게 되돌린다."""
    names = ["MATCH_THRESHOLD", "EVIDENCE_MIN_QUOTE_CHARS", "ANSWER_QUOTE_MIN_CHARS",
             "VAGUE_SHORT_MAX_CHARS", "SERVICE_ERROR_TEMPLATES", "SERVICE_ERROR_MARKERS",
             "SERVICE_ERROR_MAX_CHARS", "MAX_HISTORY_TURNS", "DEFAULT_WORKERS",
             "ORG_CANDIDATE_FIELDS", "FILTER_ANY_VALUES", "CACHE_DIR"]
    saved = {n: getattr(settings, n) for n in names}
    yield
    for n, v in saved.items():
        setattr(settings, n, v)


def write(tmp_path, text: str):
    path = tmp_path / "local.yaml"
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 계산 전에 죽는가
# ---------------------------------------------------------------------------

def test_missing_config_file_names_the_path(tmp_path):
    with pytest.raises(ConfigError) as e:
        load(tmp_path / "없는파일.yaml")
    assert "없는파일.yaml" in str(e.value)
    assert "example.yaml" in str(e.value), "어떻게 만들지 알려줘야 한다"


def test_unknown_key_is_rejected_with_a_suggestion(tmp_path):
    """오타를 통과시키면 조용히 기본값으로 돈다. 그게 가장 나쁜 결과다."""
    path = write(tmp_path, """
        run:
          worker: 8
    """)
    with pytest.raises(ConfigError) as e:
        load(path)
    assert "run.worker" in str(e.value)
    assert "run.workers" in str(e.value), "가까운 키를 알려줘야 옮겨 적을 수 있다"


def test_wrong_type_says_what_came_instead(tmp_path):
    path = write(tmp_path, """
        run:
          workers: "넷"
    """)
    with pytest.raises(ConfigError) as e:
        load(path)
    assert "run.workers" in str(e.value) and "str" in str(e.value)


def test_bool_does_not_sneak_through_as_int(tmp_path):
    """파이썬에서 bool 은 int 다. workers: true 가 1 로 통과하면 안 된다."""
    path = write(tmp_path, """
        run:
          workers: true
    """)
    with pytest.raises(ConfigError):
        load(path)


def test_choice_fields_list_what_is_allowed(tmp_path):
    path = write(tmp_path, """
        llm:
          backend: vllm
    """)
    with pytest.raises(ConfigError) as e:
        load(path)
    assert "local" in str(e.value) and "cli" in str(e.value)


def test_out_of_range_threshold_is_rejected(tmp_path):
    path = write(tmp_path, """
        thresholds:
          match_threshold: 90
    """)
    with pytest.raises(ConfigError) as e:
        load(path)
    assert "match_threshold" in str(e.value)


def test_empty_service_error_template_is_rejected(tmp_path):
    """빈 문자열은 모든 답변에 일치해 전부 case9 가 된다."""
    path = write(tmp_path, """
        service_error:
          templates:
            - ""
    """)
    with pytest.raises(ConfigError) as e:
        load(path)
    assert "case9" in str(e.value)


def test_all_problems_are_reported_at_once(tmp_path):
    """하나씩 던지면 사내 실험 왕복이 그만큼 늘어난다."""
    path = write(tmp_path, """
        run:
          workers: 0
          nope: 1
        llm:
          backend: 없는것
    """)
    with pytest.raises(ConfigError) as e:
        load(path)
    text = str(e.value)
    assert "workers" in text and "nope" in text and "backend" in text
    assert "계산을 시작하지 않았습니다" in text


# ---------------------------------------------------------------------------
# 실제로 도달하는가
# ---------------------------------------------------------------------------

def test_service_error_templates_reach_the_checker(tmp_path, restore_settings):
    """설정에 적은 문구로 판정이 바뀌어야 한다."""
    from ragdiag.checks import check_service_error

    캔드 = "지금은 응답할 수 없습니다. 관리자에게 문의하세요"
    assert not check_service_error(캔드).violated, "바꾸기 전에는 안 잡혀야 한다"

    path = write(tmp_path, f"""
        service_error:
          templates:
            - "{캔드}"
    """)
    apply(load(path))
    assert check_service_error(캔드).violated, "설정이 검증기까지 도달하지 않았다"


def test_match_threshold_reaches_the_verifier(tmp_path, restore_settings):
    from ragdiag.schema import Evidence
    from ragdiag.verify import verify_evidence

    chunks = ["국내 출장 식비는 1일 3만원을 상한으로 한다."]
    # 실제 연속 일치율이 0.45 인 인용. 기본 0.9 에서는 떨어진다.
    ev = [Evidence(chunk_index=0, quote="국내 출장 식비는 1일 5만원을 상한으로 본다.")]
    assert verify_evidence(ev, chunks).n_kept == 0

    apply(load(write(tmp_path, """
        thresholds:
          match_threshold: 0.4
    """)))
    assert verify_evidence(ev, chunks).n_kept == 1, "임계값이 대조기까지 도달하지 않았다"


def test_history_turns_reaches_the_parser(tmp_path, restore_settings):
    """기본 인자가 def 시점에 굳으면 여기서 걸린다."""
    import inspect

    from ragdiag import conv

    sig = inspect.signature(conv.to_case)
    assert sig.parameters["history_turns"].default is None, (
        "기본 인자에 settings 값을 박으면 def 시점에 굳어 --config 가 안 먹는다")

    apply(load(write(tmp_path, """
        run:
          history_turns: 1
    """)))
    assert settings.MAX_HISTORY_TURNS == 1


def test_no_default_argument_binds_a_settings_value():
    """전수 검사 — 새로 짠 함수가 같은 함정을 밟지 않게."""
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "ragdiag"
    offenders = []
    for path in root.rglob("*.py"):
        if path.name == "settings.py":
            continue
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"=\s*settings\.[A-Z_]+\s*[,)]", line):
                offenders.append(f"{path.name}:{n}  {line.strip()}")
    assert not offenders, (
        "기본 인자에 settings 값이 박혀 있다. def 시점에 굳어 설정이 안 먹는다:\n"
        + "\n".join(offenders))


def test_apply_reports_what_it_changed(tmp_path, restore_settings):
    changed = apply(load(write(tmp_path, """
        run:
          workers: 9
        thresholds:
          match_threshold: 0.8
    """)))
    assert any("DEFAULT_WORKERS" in c for c in changed)
    assert any("MATCH_THRESHOLD" in c for c in changed)


def test_example_yaml_passes_its_own_validation():
    """커밋된 예시가 스스로 통과하지 못하면 복사해 쓸 수 없다."""
    import pathlib

    example = pathlib.Path(__file__).resolve().parents[1] / "configs" / "example.yaml"
    assert example.exists(), "configs/example.yaml 이 있어야 한다"
    tree = yaml.safe_load(example.read_text(encoding="utf-8"))
    problems = validate(flatten(tree))
    assert not problems, "\n".join(problems)


def test_example_yaml_lists_every_key():
    """규격: 모든 키가 example.yaml 에 등장한다.

    없는 키는 사내에서 "코드 한 줄만 고치면 되는데" 가 되는 자리다.
    """
    import pathlib

    from ragdiag.config import SPEC

    example = pathlib.Path(__file__).resolve().parents[1] / "configs" / "example.yaml"
    tree = yaml.safe_load(example.read_text(encoding="utf-8"))
    listed = set(flatten(tree))
    missing = set(SPEC) - listed
    assert not missing, f"example.yaml 에 없는 키: {sorted(missing)}"


def test_example_defaults_match_settings():
    """예시에 적힌 기본값이 코드의 기본값과 달라선 안 된다.

    다르면 --config 를 쓰는 사람과 안 쓰는 사람이 다른 결과를 본다.
    """
    import pathlib

    from ragdiag.config import TO_SETTINGS

    example = pathlib.Path(__file__).resolve().parents[1] / "configs" / "example.yaml"
    tree = yaml.safe_load(example.read_text(encoding="utf-8"))
    values = flatten(tree)
    mismatched = []
    for key, name in TO_SETTINGS.items():
        want = values.get(key)
        if want is None:
            continue
        have = getattr(settings, name)
        if isinstance(have, (tuple, frozenset)):
            want = type(have)(want)
        if want != have:
            mismatched.append(f"{key}={want!r} vs settings.{name}={have!r}")
    assert not mismatched, "\n".join(mismatched)


# ---------------------------------------------------------------------------
# 환경변수 (설정에 없으면 여기서 찾는다)
# ---------------------------------------------------------------------------

def test_env_var_names_are_searched_in_order(monkeypatch):
    for name in URL_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_API_URL", "http://server:8000")
    assert env_first(URL_VARS) == "http://server:8000"


def test_documented_names_come_first():
    """README 와 사내 .bashrc 가 쓰는 이름이 1순위여야 한다."""
    assert URL_VARS[0] == "LLM_API_URL"
    assert KEY_VARS[0] == "LLM_API_KEY"


def test_env_first_falls_back(monkeypatch):
    for name in KEY_VARS:
        monkeypatch.delenv(name, raising=False)
    assert env_first(KEY_VARS, "EMPTY") == "EMPTY"


def test_no_config_means_defaults():
    config = load(None)
    assert config.values == {}
    assert apply(config) == []


def test_backend_falls_back_to_the_environment(monkeypatch):
    """--backend 도 설정도 없으면 환경으로 고른다.

    argparse 기본값을 None 으로 바꾸면서 이 자동 선택이 사라진 적이 있다.
    그러면 LLM_API_URL 이 없는 장비에서 --golden 이 안 돈다.
    """
    import argparse

    from ragdiag.__main__ import make_backend

    for name in URL_VARS:
        monkeypatch.delenv(name, raising=False)
    args = argparse.Namespace(backend=None, model=None, timeout=None,
                              base_url=None, api_key=None, json_mode=None,
                              thinking=None, max_tokens=None)
    backend = make_backend(args, Config())
    assert type(backend).__name__ == "ClaudeCodeBackend", (
        "환경변수가 없으면 CLI 백엔드로 떨어져야 한다")


# ---------------------------------------------------------------------------
# CLI 형태 — 사내에서 실제로 칠 명령
# ---------------------------------------------------------------------------

def test_launcher_runs_without_pythonpath(tmp_path):
    """python conv_parse.py --conv-data ... --filter-data ... --output-dir ...

    사내에서 PYTHONPATH 를 매번 붙이지 않아도 되게 둔 진입점이다.
    규격의 `PYTHONPATH={BB}/src python -m ragdiag` 와 같은 일을 한다.
    """
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    from ragdiag.fixtures.synth import generate

    log = tmp_path / "conv_eval.json"
    log.write_text(json.dumps(generate(n=2, seed=0), ensure_ascii=False),
                   encoding="utf-8")
    out = tmp_path / "outputs"

    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    proc = subprocess.run(
        [sys.executable, str(root / "conv_parse.py"),
         "--conv-data", str(log), "--output-dir", str(out), "--dry-run"],
        capture_output=True, text=True, env=env, cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "RUN SUMMARY" in proc.stdout


def test_output_dir_is_created_and_holds_both_artifacts(tmp_path):
    """디렉터리가 없으면 만든다.

    30분 돌린 뒤 디렉터리가 없어서 못 쓰면 그 사이클을 버린다.
    RUN SUMMARY 를 파일로도 남긴다 - 손으로 옮겨 적을 때 스크롤을 뒤지지 않게.
    """
    import json
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    from ragdiag.fixtures.synth import generate

    log = tmp_path / "conv_eval.json"
    log.write_text(json.dumps(generate(n=1, seed=0), ensure_ascii=False),
                   encoding="utf-8")
    out = tmp_path / "없던" / "디렉터리"

    proc = subprocess.run(
        [sys.executable, str(root / "conv_parse.py"), "--backend", "cli",
         "--conv-data", str(log), "--output-dir", str(out)],
        capture_output=True, text=True, cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert (out / "conv_parsed.json").exists()
    assert (out / "run_summary.txt").exists()
    assert "RUN SUMMARY" in (out / "run_summary.txt").read_text(encoding="utf-8")


def test_filter_keeps_the_old_flag_name(tmp_path):
    """--filter 로 적어둔 스크립트가 있을 수 있다. 둘 다 받는다."""
    import argparse
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run([sys.executable, str(root / "conv_parse.py"), "--help"],
                          capture_output=True, text=True)
    assert "--filter-data" in proc.stdout
    assert "--filter" in proc.stdout
