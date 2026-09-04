"""설정이 실제로 먹히는가.

운영 환경에서는 코드를 한 줄도 못 고친다. 설정에 적은 값이 조용히 무시되면 그 사이클을
통째로 버린다 - 임계값을 바꿨는데 안 바뀐 채로 30분이 지나가고, 결과만 보고는
알 수 없다.

여기서 두 가지를 지킨다.
  1. 잘못된 설정은 **계산을 시작하기 전에** 죽는다
  2. 맞는 설정은 **검증기와 파서까지 실제로 도달한다** (import 시점 스냅샷이
     아니라 호출 시점에 읽히는가)

2번이 조용히 깨지는 게 특히 위험하다. 파이썬 기본 인자는 def 시점에 굳어서
`def f(n=settings.X)` 는 나중에 settings.X 를 바꿔도 안 먹는다.
"""

import json
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

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
    path = tmp_path / "env.yaml"
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 계산 전에 죽는가
# ---------------------------------------------------------------------------

def test_missing_config_file_names_the_path(tmp_path):
    with pytest.raises(ConfigError) as e:
        load(tmp_path / "없는파일.yaml")
    assert "없는파일.yaml" in str(e.value)
    assert "env.example.yaml" in str(e.value), "어떻게 만들지 알려줘야 한다"


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
    """하나씩 던지면 운영 실험 왕복이 그만큼 늘어난다."""
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
            # 비교·복합대입 연산자를 기본 인자로 오인하지 않게 한다.
            # `ratio >= settings.MATCH_THRESHOLD)` 는 런타임 읽기라 옳은 코드다.
            if re.search(r"(?<![<>!=+\-*/%&|^])=\s*settings\.[A-Z_]+\s*[,)]", line):
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

    example = pathlib.Path(__file__).resolve().parents[1] / "configs" / "env.example.yaml"
    assert example.exists(), "configs/env.example.yaml 이 있어야 한다"
    tree = yaml.safe_load(example.read_text(encoding="utf-8"))
    problems = validate(flatten(tree))
    assert not problems, "\n".join(problems)


def test_example_yaml_lists_every_key():
    """규격: 모든 키가 env.example.yaml 에 등장한다.

    없는 키는 운영 환경에서 "코드 한 줄만 고치면 되는데" 가 되는 자리다.
    """
    import pathlib

    from ragdiag.config import SPEC

    example = pathlib.Path(__file__).resolve().parents[1] / "configs" / "env.example.yaml"
    tree = yaml.safe_load(example.read_text(encoding="utf-8"))
    listed = set(flatten(tree))
    missing = set(SPEC) - listed
    assert not missing, f"env.example.yaml 에 없는 키: {sorted(missing)}"


def test_example_defaults_match_settings():
    """예시에 적힌 기본값이 코드의 기본값과 달라선 안 된다.

    다르면 --config 를 쓰는 사람과 안 쓰는 사람이 다른 결과를 본다.
    """
    import pathlib

    from ragdiag.config import TO_SETTINGS

    example = pathlib.Path(__file__).resolve().parents[1] / "configs" / "env.example.yaml"
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
    """README 와 운영 환경 .bashrc 가 쓰는 이름이 1순위여야 한다."""
    assert URL_VARS[0] == "LLM_API_URL"
    assert KEY_VARS[0] == "LLM_API_KEY"


def test_env_first_falls_back(monkeypatch):
    for name in KEY_VARS:
        monkeypatch.delenv(name, raising=False)
    assert env_first(KEY_VARS, "EMPTY") == "EMPTY"


def test_label_file_from_config_replaces_the_placeholder(tmp_path, placeholder_labels):
    """운영 taxonomy 문서를 그대로 가리키면 된다. 형식은 `A. 이름 -> 점수`."""
    from ragdiag import labels as mod

    doc = tmp_path / "q.md"
    doc.write_text("# 그룹\nA. 어떤 라벨 -> 80\nB. 다른 라벨 -> 20\n", encoding="utf-8")
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"labels:\n  query: {doc}\n", encoding="utf-8")

    changed = apply(load(cfg))
    assert any("labels.query" in c for c in changed), changed
    assert mod.QUERY_LABELS["A"].name == "어떤 라벨"
    assert mod.DEFAULT_QUERY_SCORES["A"] == 80
    assert not mod.is_placeholder()


def test_missing_label_file_dies_before_computing(tmp_path):
    """조용히 자리표시자로 도는 것이 최악이다 - 필터가 에러 없이 0건을 돌려준다."""
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"labels:\n  query: {tmp_path}/없는파일.md\n", encoding="utf-8")
    with pytest.raises(ConfigError) as e:
        apply(load(cfg))
    assert "라벨 파일이 없습니다" in str(e.value)


def test_unparseable_label_file_is_rejected(tmp_path):
    cfg = tmp_path / "c.yaml"
    doc = tmp_path / "q.md"
    doc.write_text("이건 라벨 형식이 아니다\n", encoding="utf-8")
    cfg.write_text(f"labels:\n  query: {doc}\n", encoding="utf-8")
    with pytest.raises(ConfigError) as e:
        apply(load(cfg))
    assert "하나도 읽지 못했습니다" in str(e.value)


def test_filter_with_labels_refuses_to_run_on_placeholders(tmp_path, placeholder_labels):
    """가장 위험한 실패는 에러가 아니라 조용한 0건이다."""
    import json

    from ragdiag.filters import LabelTableMissing, load_filter

    path = tmp_path / "f.json"
    path.write_text(json.dumps(
        {"state": {"emotion_labels": ["I. 어떤라벨"], "eval_range": [0, 60]}}),
        encoding="utf-8")
    with pytest.raises(LabelTableMissing) as e:
        load_filter(path)
    assert "labels:" in str(e.value), "무엇을 채우라는지 적어야 한다"


def test_filter_without_labels_runs_on_placeholders(tmp_path, placeholder_labels):
    """라벨을 안 쓰는 필터는 실값 없이도 돈다. 필요 이상으로 막지 않는다."""
    import json

    from ragdiag.filters import load_filter

    path = tmp_path / "f.json"
    path.write_text(json.dumps({"state": {"turn": "2-"}}), encoding="utf-8")
    assert load_filter(path).turn_buckets, "턴 조건만 쓰는 필터는 통과해야 한다"


def test_must_fill_keys_are_marked_in_the_example():
    """반드시 채울 넷만 표시한다. 경로는 실행할 때 --conv-data 로 줘도 된다.

    30개 키가 대부분 null 인 파일에서 반드시 채울 것이 묻히면, 안 채운 채
    돌리고 30분 뒤에 알게 된다.
    """
    text = (ROOT / "configs/env.example.yaml").read_text(encoding="utf-8")
    marked = [l.split(":")[0].strip() for l in text.splitlines()
              if "← 채우세요" in l and ":" in l and not l.strip().startswith("#")]
    assert marked == ["venv", "url", "key", "model"], marked


# ---------------------------------------------------------------------------
# paths.venv — 적어둔 값이 실제로 쓰여야 한다
#
# 확인만 하고 쓰지 않으면 "설정했는데 무시된다" 가 된다. 설정 파일의 값이
# 아무것도 바꾸지 않는 것은 그 자체로 결함이다.
# ---------------------------------------------------------------------------

def _run_entry(argv, cwd, python=None):
    import subprocess
    import sys as _sys

    return subprocess.run([str(python or _sys.executable),
                           str(ROOT / "src" / "run.py"), *argv],
                          capture_output=True, text=True, cwd=cwd)


def test_configured_venv_is_switched_into(tmp_path):
    """activate 를 잊고 다른 파이썬으로 쳐도 설정한 venv 로 넘어가야 한다.

    **ragdiag 를 import 하기 전에** 일어나야 한다 - 의존이 없는 파이썬으로
    실행하면 pydantic import 에서 먼저 죽어서 어떤 안내도 화면에 못 나온다.
    """
    import sys as _sys

    from ragdiag.fixtures.synth import generate

    other = Path(_sys.base_prefix) / "bin" / "python3"
    if not other.exists() or _sys.prefix == _sys.base_prefix:
        pytest.skip("이 장비에서는 venv 밖 파이썬을 특정할 수 없다")

    log = tmp_path / "conv.json"
    log.write_text(json.dumps(generate(n=1, seed=0), ensure_ascii=False), encoding="utf-8")
    cfg = tmp_path / "env.yaml"
    cfg.write_text(f"paths:\n  venv: {_sys.prefix}\n", encoding="utf-8")

    proc = _run_entry(["--conv-data", str(log), "--config", str(cfg), "--dry-run"],
                      cwd=tmp_path, python=other)
    assert proc.returncode == 0, proc.stderr[-800:]
    assert "[venv]" in proc.stderr, "갈아탄 사실을 알려야 한다\n" + proc.stderr[-400:]
    assert "갈아탐" in proc.stderr, "실행 조건에도 남아야 한다"
    assert "ModuleNotFoundError" not in proc.stderr


def test_no_switch_when_already_in_the_configured_venv(tmp_path):
    import sys as _sys

    from ragdiag.fixtures.synth import generate

    log = tmp_path / "conv.json"
    log.write_text(json.dumps(generate(n=1, seed=0), ensure_ascii=False), encoding="utf-8")
    cfg = tmp_path / "env.yaml"
    cfg.write_text(f"paths:\n  venv: {_sys.prefix}\n", encoding="utf-8")

    proc = _run_entry(["--conv-data", str(log), "--config", str(cfg), "--dry-run"],
                      cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr[-500:]
    assert "[venv]" not in proc.stderr, "이미 그 venv 인데 갈아탔다 (루프 위험)"


def test_bad_venv_path_dies_before_computing(tmp_path):
    cfg = tmp_path / "env.yaml"
    cfg.write_text("paths:\n  venv: /어디에도/없는/venv\n", encoding="utf-8")
    proc = _run_entry(["--config", str(cfg), "--dry-run"], cwd=tmp_path)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "파이썬이 없습니다" in proc.stderr
    assert "activate" in proc.stderr, "다른 방법도 알려줘야 한다"


def test_venv_peek_works_without_pyyaml(tmp_path):
    """갈아타기 전에는 PyYAML 도 없을 수 있다. 그때도 이 한 키는 읽어야 한다."""
    import builtins
    import importlib.util

    spec = importlib.util.spec_from_file_location("_entry", ROOT / "src" / "run.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)

    cfg = tmp_path / "env.yaml"
    cfg.write_text("# 주석\nllm:\n  url: http://x\npaths:\n  venv: /opt/some/venv\n",
                   encoding="utf-8")

    real = builtins.__import__

    def blocked(name, *a, **k):
        if name.split(".")[0] == "yaml":
            raise ModuleNotFoundError("no yaml", name="yaml")
        return real(name, *a, **k)

    builtins.__import__ = blocked
    try:
        assert entry._peek_venv(str(cfg)) == "/opt/some/venv"
    finally:
        builtins.__import__ = real


# ---------------------------------------------------------------------------
# 사본 안에 둔 설정 — 조용히 지워지는 자리
#
# {AA}/{BB} 는 sync 때마다 rm -rf 로 통째 교체된다. 거기에 env.yaml 을 만들면
# 채워 넣은 값이 사라지는데, 화면에는 "그대로 둡니다" 가 찍힌다 - 그건
# {AA}/configs 쪽 이야기인데 자기 파일이 지켜진 줄 알게 된다.
# ---------------------------------------------------------------------------

def _fake_copy(tmp_path, under=""):
    """sync 로 만들어진 사본을 흉내낸다. 표식은 VERSION 파일이다.

    under 로 중간 폴더를 끼울 수 있다. {AA}/{BB} 만 상정하면 {AA}/tools/vendor/{BB}
    에서 깨지는 안내를 못 잡는다.
    """
    work = tmp_path / "work"
    copy = work / under / "log_analysis" if under else work / "log_analysis"
    (copy / "configs").mkdir(parents=True)
    (copy / "VERSION").write_text("v9.9 abc1234\n", encoding="utf-8")
    (work / "configs").mkdir()
    return work, copy


def test_config_missing_inside_a_copy_points_at_the_work_folder(tmp_path):
    from ragdiag.config import synced_copy_root

    work, copy = _fake_copy(tmp_path)
    assert synced_copy_root(copy / "configs" / "env.yaml", root=copy) == copy
    assert synced_copy_root(work / "configs" / "env.yaml", root=copy) is None


def test_dev_repo_is_not_treated_as_a_copy(tmp_path):
    """개발 저장소에 configs/env.yaml 을 만드는 것은 정상이다. 표식은 VERSION 이다."""
    from ragdiag.config import synced_copy_root

    repo = tmp_path / "repo"
    (repo / "configs").mkdir(parents=True)
    assert synced_copy_root(repo / "configs" / "env.yaml", root=repo) is None


@pytest.mark.parametrize("under", ["", "tools/vendor"])
def test_config_inside_a_copy_is_refused(tmp_path, monkeypatch, under):
    """설정은 언제나 {AA} 에 둔다. 경고로 끝내면 이번 실행은 돌고 다음에 사라진다.

    작업 폴더는 **실행 위치**다. 전에는 사본의 부모를 {AA} 로 쳤는데, 그건
    {AA}/{BB} 일 때만 맞다 - {AA}/tools/vendor/{BB} 면 중간 폴더에 설정을
    만들라고 안내하고, 거기서 실행하라고 한다.
    """
    import ragdiag.config as mod

    work, copy = _fake_copy(tmp_path, under)
    cfg = copy / "configs" / "env.yaml"
    cfg.write_text("run:\n  workers: 2\n", encoding="utf-8")
    monkeypatch.chdir(work)          # {AA} 에서 실행한다

    real = mod.synced_copy_root
    monkeypatch.setattr(mod, "synced_copy_root",
                        lambda path, root=None: real(path, root=copy))

    with pytest.raises(ConfigError) as e:
        mod.load(cfg)
    message = str(e.value)
    assert "사본 안에 있습니다" in message
    assert f"mv {cfg} {work / 'configs' / 'env.yaml'}" in message, (
        "어디로 옮기라는지 명령 그대로 적어야 한다\n" + message)
    assert f"cd {work}" in message, "실행 위치도 알려줘야 한다"
    # 사본까지의 경로는 실행 위치 기준이어야 한다. 사본 이름만 찍으면
    # 중간 폴더가 낀 구조에서 그 경로가 존재하지 않는다.
    entry = f"{under}/log_analysis" if under else "log_analysis"
    assert f"python {entry}/src/run.py" in message, message


def test_work_folder_config_is_found_without_the_flag(tmp_path):
    """--config 를 안 줘도 작업 폴더의 설정을 쓴다.

    운영 환경에서는 sync.sh 가 {AA}/configs/env.yaml 을 만들어 둔다. 그걸 매번
    --config 로 가리키게 하면 한 번 빼먹는 순간 조용히 기본값으로 돈다.
    """
    from ragdiag.fixtures.synth import generate

    log = tmp_path / "conv.json"
    log.write_text(json.dumps(generate(n=1, seed=0), ensure_ascii=False), encoding="utf-8")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "env.yaml").write_text("run:\n  workers: 6\n",
                                                   encoding="utf-8")

    proc = _run_entry(["--conv-data", "conv.json", "--dry-run"], cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr[-500:]
    assert "자동 (작업 폴더)" in proc.stderr, proc.stderr[-400:]
    assert "동시" in proc.stderr and "6" in proc.stderr


def test_missing_work_folder_config_is_visible(tmp_path):
    """없으면 기본값으로 도는 것은 그대로다. 다만 없다는 사실이 화면에 남아야 한다."""
    from ragdiag.fixtures.synth import generate

    log = tmp_path / "conv.json"
    log.write_text(json.dumps(generate(n=1, seed=0), ensure_ascii=False), encoding="utf-8")

    proc = _run_entry(["--conv-data", "conv.json", "--dry-run"], cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr[-500:]
    assert "configs/env.yaml 이 없다" in proc.stderr, proc.stderr[-400:]


def test_both_entry_points_look_in_the_same_place():
    """run.py 가 못 찾으면 venv 를 안 갈아탄 채로 그 설정이 적용된다."""
    import importlib.util

    from ragdiag.__main__ import DEFAULT_CONFIG as from_main

    spec = importlib.util.spec_from_file_location("_entry", ROOT / "src" / "run.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    assert str(from_main) == entry.DEFAULT_CONFIG


def test_cli_flag_beats_the_config(tmp_path):
    """CLI 가 최상위다. 한 번만 다르게 돌려보는 길이 막히면 안 된다."""
    from ragdiag.fixtures.synth import generate

    log = tmp_path / "conv.json"
    log.write_text(json.dumps(generate(n=1, seed=0), ensure_ascii=False), encoding="utf-8")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "env.yaml").write_text("run:\n  workers: 6\n",
                                                   encoding="utf-8")

    proc = _run_entry(["--conv-data", "conv.json", "--workers", "3", "--dry-run"],
                      cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr[-500:]
    assert "--workers" in proc.stderr, "어느 쪽이 이겼는지 화면에 남아야 한다"


def test_no_config_means_defaults():
    config = load(None)
    assert config.values == {}
    assert apply(config) == []


def _bare_args(**kw):
    import argparse

    base = dict(backend=None, model=None, timeout=None, base_url=None,
                api_key=None, json_mode=None, thinking=None, max_tokens=None)
    base.update(kw)
    return argparse.Namespace(**base)


def test_entry_point_knows_only_the_local_backend(monkeypatch):
    """규격 §1.4 · C8 — 운영 환경에서 실패할 호출은 src/ 에 없다.

    예전에는 LLM_API_URL 이 없으면 claude CLI 로 떨어졌다. 그 경로가 tools/ 로
    나가면서 자동 선택도 없앴다. 주소가 없으면 무엇을 export 하라고 알려주는
    쪽이 맞다 - 운영 환경에는 claude CLI 자체가 없어서 폴백이 성립하지 않는다.
    """
    from ragdiag.__main__ import make_backend
    from ragdiag.backends import JudgeError

    for name in URL_VARS:
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(JudgeError) as e:
        make_backend(_bare_args(), Config())
    assert "LLM_API_URL" in str(e.value), str(e.value)


@pytest.mark.parametrize("kind", ["cli", "api"])
def test_dev_backends_point_at_the_tools_runner(kind, monkeypatch):
    """트레이스백 하나가 사이클을 먹는다. 어디로 가라는지까지 적어야 한다."""
    from ragdiag.__main__ import make_backend
    from ragdiag.backends import JudgeError

    for name in URL_VARS:
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(JudgeError) as e:
        make_backend(_bare_args(backend=kind), Config())
    assert "tools/dev_run.py" in str(e.value), str(e.value)
    assert "--backend local" in str(e.value), str(e.value)


def test_src_does_not_import_tools():
    """import 방향은 한쪽이다 (규격 §1.4). 반대로 가면 tools/ 없는 사본이 죽는다."""
    import pathlib as _p
    import re

    offenders = []
    for path in (ROOT / "src").rglob("*.py"):
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.match(r"\s*(from|import)\s+tools\b", line):
                offenders.append(f"{path.relative_to(ROOT)}:{n}  {line.strip()}")
    assert not offenders, "src/ 가 tools/ 를 import 한다:\n" + "\n".join(offenders)


def test_shipped_source_has_no_llm_api_imports():
    """sync.sh 가 잡기 전에 여기서 잡는다. 거기서 걸리면 태그를 다시 내야 한다."""
    import re

    offenders = []
    for path in (ROOT / "src").rglob("*.py"):
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.match(r"\s*(import|from)\s+(anthropic|openai)\b", line):
                offenders.append(f"{path.relative_to(ROOT)}:{n}  {line.strip()}")
    assert not offenders, (
        "운영 환경에서 죽을 의존이 src/ 에 있다 (C8):\n" + "\n".join(offenders))


# ---------------------------------------------------------------------------
# CLI 형태 — 운영 환경에서 실제로 칠 명령
# ---------------------------------------------------------------------------

def _run_against_stub(args: list[str], cwd) -> "subprocess.CompletedProcess":
    """가짜 LLM 서버를 띄우고 진입점을 끝까지 돌린다.

    claude CLI 백엔드에 묶어두면 그게 없는 사본에서 이 테스트들이 건너뛰어진다 -
    그러면 운영 환경에서 처음 도는 경로(출력 디렉터리 생성, 파일 이름)가 검증되지 않은
    채로 나간다. 배관을 재는 테스트에 판정 품질은 필요 없다.
    """
    import os
    import subprocess
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parent))
    from stub_llm import StubLLM

    root = _Path(__file__).resolve().parents[1]
    with StubLLM() as stub:
        env = dict(os.environ, LLM_API_URL=stub.url, LLM_API_KEY="stub")
        return subprocess.run(
            [sys.executable, str(root / "src" / "run.py"), "--backend", "local",
             "--no-cache", *args],
            capture_output=True, text=True, env=env, cwd=cwd)


def test_entry_script_runs_without_pythonpath(tmp_path):
    """python <저장소>/src/run.py --conv-data ... --filter-data ... --output-dir ...

    운영 환경에서 PYTHONPATH 를 매번 붙이지 않아도 되게 둔 진입점이다.
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
        [sys.executable, str(root / "src" / "run.py"),
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

    proc = _run_against_stub([
         "--conv-data", str(log), "--output-dir", str(out)],
        cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    results = list(out.glob("conv_parsed_*.json"))
    summaries = list(out.glob("run_summary_*.txt"))
    assert results, f"결과 파일이 없다: {list(out.iterdir())}"
    assert summaries, "RUN SUMMARY 사본이 없다"
    assert "RUN SUMMARY" in summaries[0].read_text(encoding="utf-8")


def test_filter_keeps_the_old_flag_name(tmp_path):
    """--filter 로 적어둔 스크립트가 있을 수 있다. 둘 다 받는다."""
    import argparse
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run([sys.executable, str(root / "src" / "run.py"), "--help"],
                          capture_output=True, text=True)
    assert "--filter-data" in proc.stdout
    assert "--filter" in proc.stdout


def test_output_filename_carries_the_finish_time(tmp_path):
    """같은 데이터를 여러 번 돌리면 어느 것이 언제 것인지 알 수 없다.

    운영 환경에서는 결과를 반출할 수 없어 이 파일들이 그 자리에 계속 쌓인다.
    파일 이름에 시각이 없으면 덮어써지거나 뒤섞인다.
    """
    import json
    import re
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    from ragdiag.fixtures.synth import generate

    log = tmp_path / "conv_eval.json"
    log.write_text(json.dumps(generate(n=1, seed=0), ensure_ascii=False), encoding="utf-8")

    proc = _run_against_stub([
         "--conv-data", str(log)],
        cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr

    # --output-dir 을 안 줘도 ./output 에 생긴다
    out = tmp_path / "output"
    assert out.is_dir(), f"기본 출력 디렉터리가 없다: {list(tmp_path.iterdir())}"
    results = list(out.glob("conv_parsed_*.json"))
    summaries = list(out.glob("run_summary_*.txt"))
    assert results, f"시각이 붙은 결과 파일이 없다: {list(out.iterdir())}"
    assert summaries, "RUN SUMMARY 사본이 없다"

    stamp = re.search(r"conv_parsed_(\d{8}-\d{6})\.json", results[0].name)
    assert stamp, f"파일명에 시각이 없다: {results[0].name}"
    # 결과와 요약이 같은 시각을 쓴다 — 짝을 찾을 수 있어야 한다
    assert (out / f"run_summary_{stamp.group(1)}.txt").exists(), (
        "결과와 RUN SUMMARY 의 시각이 다르다")


def test_out_flag_overrides_the_timestamp(tmp_path):
    """--out 을 주면 그 경로 그대로 쓴다. 자동화에서 경로를 고정해야 할 때가 있다."""
    import json
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    from ragdiag.fixtures.synth import generate

    log = tmp_path / "conv_eval.json"
    log.write_text(json.dumps(generate(n=1, seed=0), ensure_ascii=False), encoding="utf-8")
    fixed = tmp_path / "고정경로.json"

    proc = _run_against_stub([
         "--conv-data", str(log), "--out", str(fixed)],
        cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert fixed.exists(), "--out 경로에 안 썼다"


# ---------------------------------------------------------------------------
# llm.env_file — 다른 프로젝트의 env.yaml 참조
# ---------------------------------------------------------------------------

def _serving(tmp_path, **over):
    """vllm 주소·모델·키를 들고 있는 남의 설정 파일."""
    block = {"base_url": "http://gpu-01:8000", "model": "qwen3.5-32b",
             "api_key": "sk-serving"}
    block.update(over)
    file = tmp_path / "serving.yaml"
    file.write_text(yaml.safe_dump({"vllm": block, "other": {"x": 1}},
                                   allow_unicode=True), encoding="utf-8")
    return file


def test_env_file_fills_empty_llm_values(tmp_path, monkeypatch):
    """여러 프로젝트가 같은 vllm 을 쓴다. 주소를 두 벌로 두면 한쪽만 고치고
    "왜 옛 서버로 가지" 를 한참 찾게 된다."""
    monkeypatch.chdir(tmp_path)
    serving = _serving(tmp_path)
    cfg = tmp_path / "env.yaml"
    cfg.write_text(yaml.safe_dump({"llm": {"env_file": serving.name}}),
                   encoding="utf-8")

    loaded = load(cfg)
    assert loaded.get("llm.url") == "http://gpu-01:8000"
    assert loaded.get("llm.model") == "qwen3.5-32b"
    assert loaded.get("llm.key") == "sk-serving"


def test_a_direct_value_beats_the_referenced_one(tmp_path, monkeypatch):
    """참조는 "따로 안 적으면 저기 걸 쓴다" 이지 "저기 걸로 덮어쓴다" 가 아니다."""
    monkeypatch.chdir(tmp_path)
    serving = _serving(tmp_path)
    cfg = tmp_path / "env.yaml"
    cfg.write_text(yaml.safe_dump({"llm": {
        "env_file": serving.name, "url": "http://직접:9000"}}), encoding="utf-8")

    loaded = load(cfg)
    assert loaded.get("llm.url") == "http://직접:9000"
    # 안 적은 것만 끌어온다.
    assert loaded.get("llm.model") == "qwen3.5-32b"


def test_the_referenced_value_beats_the_environment(tmp_path, monkeypatch):
    """우선순위: CLI > 설정의 직접값 > env_file 참조값 > 환경변수.

    참조값이 **설정 자리**에 앉으므로 make_backend 의 pick() 순서가 그대로
    이 순서가 된다 - 배선을 따로 하지 않는다.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LLM_API_URL", "http://환경변수:7000")
    serving = _serving(tmp_path)
    cfg = tmp_path / "env.yaml"
    cfg.write_text(yaml.safe_dump({"llm": {"env_file": serving.name}}),
                   encoding="utf-8")

    loaded = load(cfg)
    assert loaded.get("llm.url") == "http://gpu-01:8000"


def test_env_file_records_where_the_value_came_from(tmp_path, monkeypatch):
    """"설정 llm.url" 로만 찍으면 어느 파일의 값인지 알 수 없다.

    실행 조건 블록을 두는 이유가 "어느 쪽이 이겼는지" 를 보여주는 것이라,
    출처를 잃으면 설정을 고쳐도 안 먹는 이유를 못 찾는다.
    """
    monkeypatch.chdir(tmp_path)
    serving = _serving(tmp_path)
    cfg = tmp_path / "env.yaml"
    cfg.write_text(yaml.safe_dump({"llm": {"env_file": serving.name}}),
                   encoding="utf-8")

    origin = load(cfg).origins["llm.url"]
    assert "serving.yaml" in origin and "vllm.base_url" in origin, origin


def test_a_named_section_is_honoured(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    file = tmp_path / "serving.yaml"
    file.write_text(yaml.safe_dump({"sglang": {"base_url": "http://s:8000"}}),
                    encoding="utf-8")
    cfg = tmp_path / "env.yaml"
    cfg.write_text(yaml.safe_dump({"llm": {
        "env_file": file.name, "env_file_section": "sglang"}}), encoding="utf-8")

    assert load(cfg).get("llm.url") == "http://s:8000"


def test_a_missing_env_file_dies_before_computing(tmp_path, monkeypatch):
    """30분 돌린 뒤에 죽으면 사이클 하나를 버린다."""
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "env.yaml"
    cfg.write_text(yaml.safe_dump({"llm": {"env_file": "없는파일.yaml"}}),
                   encoding="utf-8")

    with pytest.raises(ConfigError) as e:
        load(cfg)
    assert "없는파일.yaml" in str(e.value)
    assert "실행 위치" in str(e.value), str(e.value)


def test_a_missing_section_says_what_is_in_the_file(tmp_path, monkeypatch):
    """오타 하나로 조용히 빈 값이 되면 서버가 없다는 에러를 나중에 본다."""
    monkeypatch.chdir(tmp_path)
    serving = _serving(tmp_path)
    cfg = tmp_path / "env.yaml"
    cfg.write_text(yaml.safe_dump({"llm": {
        "env_file": serving.name, "env_file_section": "vllmm"}}), encoding="utf-8")

    with pytest.raises(ConfigError) as e:
        load(cfg)
    assert "vllmm" in str(e.value)
    assert "vllm" in str(e.value) and "other" in str(e.value), str(e.value)


def test_no_env_file_means_nothing_changes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "env.yaml"
    cfg.write_text(yaml.safe_dump({"llm": {"url": "http://직접:9000"}}),
                   encoding="utf-8")
    loaded = load(cfg)
    assert loaded.origins == {}
    assert loaded.get("llm.model") is None
