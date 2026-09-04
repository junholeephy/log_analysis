"""설정 읽기와 검증.

운영 환경에서는 코드를 한 줄도 못 고친다. 바뀔 만한 값이 코드에 박혀 있으면 그 사이클은
거기서 끝난다 - 고칠 수 없고, 고치면 규칙을 깬 것이다. 그래서 값은 YAML 로 받는다.

**검증은 시작 즉시 하고 계산 전에 죽는다.** 30분 돌린 뒤에 키 하나 때문에 죽으면
사이클 하나를 버린다. 운영 실험은 왕복이 비싸다.

에러 메시지는 사람이 그대로 옮겨 적을 수 있게 쓴다. 결과 파일을 반출할 수 없으므로
화면에 찍히는 문장이 유일한 회수 채널이다.

settings.py 와의 관계: settings 는 기본값을, 여기는 그 위에 덮어쓸 값을 담는다.
apply() 가 settings 의 모듈 전역을 실제로 바꾼다 - 검증기와 파서가 이미 settings 를
읽고 있어서, 그쪽을 전부 config 인자로 바꾸는 것보다 얕게 끝난다.
"""

from __future__ import annotations

import difflib

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ragdiag import settings


class ConfigError(Exception):
    """설정 문제. 계산을 시작하기 전에 던진다."""


# 키 → (기대 타입, 필수인가). 여기 없는 키는 오타로 본다.
#
# 오타를 통과시키면 조용히 기본값으로 돈다. 운영 환경에서 임계값을 바꿨는데 안 바뀐
# 채로 30분이 지나가는 것이 가장 나쁜 결과다.
SPEC: dict[str, tuple[type | tuple, bool]] = {
    # 여기 적은 파이썬으로 갈아타서 실행한다 (src/run.py 의 switch_venv).
    # 갈아타기는 ragdiag 를 import 하기 전에 일어나야 한다 - activate 를 잊으면
    # pydantic import 에서 먼저 죽어서 그 뒤의 어떤 안내도 화면에 못 나온다.
    "paths.venv": (str, False),
    "paths.conv_data": (str, False),
    "paths.filter_data": (str, False),
    "paths.turns": (str, False),
    "paths.output_dir": (str, False),
    "paths.out": (str, False),
    "paths.cache": (str, False),
    "paths.dept_class": (str, False),
    "paths.job_class": (str, False),

    "llm.backend": (str, False),

    # 다른 프로젝트의 env.yaml 을 가리킨다. vllm 주소·모델·키를 이미 그쪽에
    # 적어 두었으면 여기에 다시 적지 않는다 - 두 벌이 되면 한쪽만 고치고
    # "왜 옛 서버로 가지" 를 한참 찾게 된다.
    #
    #   llm:
    #     env_file: ../serving/configs/env.yaml
    #     env_file_section: vllm        # 기본값
    #
    # 그 파일에서 <section>.base_url / model / api_key 를 가져온다.
    # **비어 있을 때만 채운다** - 여기 직접 적은 값이 언제나 이긴다.
    "llm.env_file": (str, False),
    "llm.env_file_section": (str, False),

    "llm.url": (str, False),
    "llm.key": (str, False),
    "llm.model": (str, False),
    "llm.json_mode": (str, False),
    "llm.thinking": (str, False),
    "llm.max_tokens": (int, False),
    "llm.timeout_sec": (int, False),

    "run.workers": (int, False),
    "run.limit": (int, False),
    "run.history_turns": (int, False),

    "thresholds.match_threshold": ((int, float), False),
    "thresholds.evidence_min_quote_chars": (int, False),
    "thresholds.answer_quote_min_chars": (int, False),
    "thresholds.vague_short_max_chars": (int, False),

    "service_error.templates": (list, False),
    "service_error.markers": (list, False),
    "service_error.max_chars": (int, False),

    # 라벨 실값 파일 경로. 운영 코드값이라 저장소에 두지 않는다 (§1.1 · C3).
    "labels.query": (str, False),
    "labels.emotion": (str, False),

    "org.candidate_fields": (list, False),
    "filter.any_values": (list, False),
}

CHOICES = {
    "llm.backend": ("local", "cli", "api"),
    "llm.json_mode": ("auto", "json_schema", "guided_json", "json_object", "none"),
    "llm.thinking": ("auto", "on", "off"),
}

# 설정 키 → settings 모듈의 전역 이름.
TO_SETTINGS = {
    "paths.cache": "CACHE_DIR",
    "run.workers": "DEFAULT_WORKERS",
    "run.history_turns": "MAX_HISTORY_TURNS",
    "llm.max_tokens": "DEFAULT_MAX_TOKENS",
    "llm.timeout_sec": "DEFAULT_TIMEOUT_SEC",
    "thresholds.match_threshold": "MATCH_THRESHOLD",
    "thresholds.evidence_min_quote_chars": "EVIDENCE_MIN_QUOTE_CHARS",
    "thresholds.answer_quote_min_chars": "ANSWER_QUOTE_MIN_CHARS",
    "thresholds.vague_short_max_chars": "VAGUE_SHORT_MAX_CHARS",
    "service_error.templates": "SERVICE_ERROR_TEMPLATES",
    "service_error.markers": "SERVICE_ERROR_MARKERS",
    "service_error.max_chars": "SERVICE_ERROR_MAX_CHARS",
    "org.candidate_fields": "ORG_CANDIDATE_FIELDS",
    "filter.any_values": "FILTER_ANY_VALUES",
}


def flatten(tree: dict, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in tree.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, f"{path}."))
        else:
            out[path] = value
    return out


# llm.env_file 이 채워 주는 것. 설정 키 ← 그 파일의 키.
ENV_FILE_KEYS = {
    "llm.url": "base_url",
    "llm.model": "model",
    "llm.key": "api_key",
}

DEFAULT_ENV_FILE_SECTION = "vllm"


@dataclass
class Config:
    values: dict[str, Any] = field(default_factory=dict)
    source: str = "(기본값)"
    # 키 → 그 값이 어디서 왔는지. llm.env_file 로 채운 것만 담는다.
    #
    # 없으면 화면에 "설정 llm.url" 로만 찍혀서 **외부 파일에서 온 값인지 알 수
    # 없다.** 실행 조건 블록을 두는 이유가 그것이라(어느 쪽이 이겼는지가 안
    # 보이면 설정을 고쳐도 안 먹는 이유를 알 수 없다) 출처를 잃으면 안 된다.
    origins: dict[str, str] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        value = self.values.get(key)
        return default if value is None else value

    def __contains__(self, key: str) -> bool:
        return self.values.get(key) is not None


def validate(values: dict[str, Any]) -> list[str]:
    """설정 문제를 전부 모아 돌려준다. 하나씩 던지면 왕복이 늘어난다."""
    problems: list[str] = []

    for key, value in sorted(values.items()):
        if key not in SPEC:
            near = difflib.get_close_matches(key, SPEC, n=1, cutoff=0.6)
            hint = f" — 혹시 {near[0]} ?" if near else ""
            problems.append(f"{key}: 모르는 키다{hint}")
            continue
        if value is None:
            continue
        want, _ = SPEC[key]
        # bool 은 int 의 서브클래스라 따로 막는다. workers: true 가 1 로 통과하면 안 된다.
        if isinstance(value, bool) and want is not bool:
            problems.append(f"{key}: 참/거짓이 아니라 {want} 가 와야 한다")
            continue
        if not isinstance(value, want):
            names = getattr(want, "__name__", None) or "/".join(t.__name__ for t in want)
            problems.append(
                f"{key}: {names} 를 기대했으나 {type(value).__name__} 가 왔다")
            continue
        if key in CHOICES and value not in CHOICES[key]:
            problems.append(f"{key}: {value!r} 는 허용되지 않는다. "
                            f"쓸 수 있는 값 — {', '.join(CHOICES[key])}")

    for key, _ in SPEC.items():
        want, required = SPEC[key]
        if required and values.get(key) is None:
            problems.append(f"{key}: 반드시 있어야 한다")

    # 값 자체의 앞뒤가 맞는지.
    workers = values.get("run.workers")
    if isinstance(workers, int) and workers < 1:
        problems.append("run.workers: 1 이상이어야 한다")
    threshold = values.get("thresholds.match_threshold")
    if isinstance(threshold, (int, float)) and not 0 < threshold <= 1:
        problems.append("thresholds.match_threshold: 0 과 1 사이여야 한다 "
                        "(연속 일치 비율이다)")
    templates = values.get("service_error.templates")
    if isinstance(templates, list) and any(not str(t).strip() for t in templates):
        problems.append("service_error.templates: 빈 문자열이 있다. "
                        "빈 문자열은 모든 답변에 일치해 전부 case9 가 된다")
    return problems


def _work_folder_hint(copy: Path, file: Path) -> str:
    """설정을 어디에 두고 어떻게 실행하라는 안내.

    작업 폴더({AA})는 **실행 위치**다. 이 저장소의 다른 모든 경로가 그 규칙을
    따르고(run.py · DEFAULT_CONFIG · sync.sh 의 pwd), 코드가 {AA} 를 알 다른
    방법도 없다 - 아는 것은 사본({BB})의 위치뿐이다.

    전에는 사본의 **부모**를 {AA} 로 쳤다. {AA}/{BB} 일 때만 맞고, 중간에 폴더가
    끼면({AA}/tools/vendor/{BB}) 엉뚱한 자리에 설정을 만들라고 안내한다.
    """
    import os

    work = Path.cwd()
    try:
        entry = os.path.relpath(copy, work)
    except ValueError:              # 다른 드라이브. 상대 경로가 없다
        entry = str(copy)
    return (f"    mv {file} {work / 'configs' / file.name}\n"
            f"    cd {work}\n"
            f"    python {entry}/src/run.py --config configs/{file.name} ...")


def synced_copy_root(path: Path, root: Optional[Path] = None) -> Optional[Path]:
    """이 경로가 sync 로 만들어진 사본 안에 있나. 있으면 사본 루트를 돌려준다.

    사본({AA}/{BB})은 다음 sync 때 `rm -rf` 로 통째 교체된다. 거기에 설정을 두면
    채워 넣은 값이 **조용히 사라지고**, 화면에는 "configs/env.yaml exists — kept"
    가 찍힌다 - 그건 {AA}/configs 쪽 이야기인데 지켜진 줄 알게 된다.

    사본에는 sync.sh 가 VERSION 을 남긴다. 개발 저장소에는 그게 없으므로,
    거기서 configs/env.yaml 을 만드는 것은 정상이고 경고하지 않는다.
    """
    root = root or Path(__file__).resolve().parents[2]
    if not (root / "VERSION").exists():
        return None
    try:
        path.resolve().relative_to(root)
    except ValueError:
        return None
    return root


def _read_yaml(file: Path, what: str) -> dict:
    """YAML 하나를 읽어 dict 로. 문제는 계산 전에 던진다."""
    try:
        import yaml
    except ImportError:
        raise ConfigError(
            "PyYAML 이 필요합니다.\n"
            "  pip install PyYAML\n"
            "  (설정 파일을 안 쓸 거면 --config 를 빼고 실행하세요)")
    try:
        tree = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"{file} 를 읽을 수 없습니다:\n  {e}")
    if not isinstance(tree, dict):
        raise ConfigError(f"{what}의 최상위는 키:값 이어야 합니다: {file}")
    return tree


def merge_env_file(values: dict[str, Any]) -> dict[str, str]:
    """llm.env_file 이 가리키는 YAML 에서 주소·모델·키를 끌어온다.

    **비어 있을 때만 채운다.** 여기 직접 적은 llm.url 이 언제나 이긴다 - 참조는
    "따로 안 적으면 저기 걸 쓴다" 이지 "저기 걸로 덮어쓴다" 가 아니다.

    make_backend 의 우선순위(CLI > 설정 > 환경변수)에서 이 값은 **설정 자리**에
    앉는다. 그래서 배선을 따로 하지 않아도 요구한 순서가 그대로 나온다:

        CLI 플래그 > 설정의 llm.* 직접값 > llm.env_file 참조값 > 환경변수

    상대 경로는 **실행 위치 기준**이다. 이 저장소의 다른 경로가 전부 그러므로
    (run.py 의 규칙) 여기만 다르게 두면 규칙이 둘이 된다.

    돌려주는 것은 채운 키의 출처다. 화면에 "설정 llm.url" 로만 찍히면 외부
    파일에서 온 값인지 알 수 없다.
    """
    reference = values.get("llm.env_file")
    if not reference:
        return {}

    file = Path(reference).expanduser()
    if not file.exists():
        raise ConfigError(
            f"llm.env_file 이 가리키는 파일이 없습니다: {file}\n"
            f"  찾은 곳: {file.resolve().parent}\n"
            f"  상대 경로는 실행 위치({Path.cwd()}) 기준입니다. "
            f"헷갈리면 절대 경로로 적으세요.")

    section = values.get("llm.env_file_section") or DEFAULT_ENV_FILE_SECTION
    tree = _read_yaml(file, "llm.env_file")
    block = tree.get(section)
    if block is None:
        raise ConfigError(
            f"llm.env_file 에 '{section}' 섹션이 없습니다: {file}\n"
            f"  그 파일의 최상위 키: {', '.join(map(str, tree)) or '(없음)'}\n"
            f"  다른 이름이면 llm.env_file_section 으로 지정하세요.")
    if not isinstance(block, dict):
        raise ConfigError(
            f"llm.env_file 의 '{section}' 은 키:값 이어야 합니다: {file}")

    origins: dict[str, str] = {}
    for key, name in ENV_FILE_KEYS.items():
        if values.get(key):
            continue                      # 직접 적은 값이 이긴다
        found = block.get(name)
        if found is None or found == "":
            continue
        if not isinstance(found, str):
            raise ConfigError(
                f"llm.env_file 의 {section}.{name}: 문자열이어야 하는데 "
                f"{type(found).__name__} 가 왔습니다 ({file})")
        values[key] = found
        origins[key] = f"{file} 의 {section}.{name}"
    return origins


def load(path: Optional[str | Path]) -> Config:
    """설정을 읽고 검증한다. 문제가 있으면 계산을 시작하기 전에 던진다."""
    if path is None:
        return Config()

    file = Path(path)
    if not file.exists():
        copy = synced_copy_root(file)
        if copy is not None:
            # 사본 안을 가리키고 있다. 여기에 만들라고 하면 다음 sync 때 지워진다.
            work = Path.cwd()
            raise ConfigError(
                f"설정 파일이 없습니다: {file}\n"
                f"  여기는 sync 로 만들어진 사본 안이라 다음 교체 때 지워집니다.\n"
                f"  설정은 작업 폴더에 둡니다: {work / 'configs' / file.name}\n"
                f"  거기서 실행하세요:\n"
                + _work_folder_hint(copy, file))
        raise ConfigError(
            f"설정 파일이 없습니다: {file}\n"
            f"  configs/env.example.yaml 을 복사해 쓰세요.")

    values = flatten(_read_yaml(file, "설정"))
    problems = validate(values)
    if problems:
        raise ConfigError(
            f"{file} 에 문제가 {len(problems)}건 있습니다. "
            "계산을 시작하지 않았습니다.\n"
            + "\n".join(f"  - {p}" for p in problems))

    copy = synced_copy_root(file)
    if copy is not None:
        # 경고로 끝내면 이번 실행은 돌고 다음 sync 에 사라진다. 그 사이에 값이
        # {AA} 쪽과 갈라져도 알 방법이 없다 - 설정은 언제나 {AA} 에 둔다.
        work = Path.cwd()
        raise ConfigError(
            f"설정이 사본 안에 있습니다: {file}\n"
            f"  {copy} 는 sync 때마다 통째로 교체됩니다. 여기 둔 설정은 다음\n"
            f"  교체에 사라지고, 그때까지 {work / 'configs' / file.name} 와\n"
            f"  갈라져 있어도 드러나지 않습니다.\n\n"
            f"  설정은 언제나 작업 폴더에 둡니다:\n"
            + _work_folder_hint(copy, file))
    # 검증을 통과한 뒤에 끌어온다. 오타가 있는 설정으로 남의 파일까지 읽으면
    # 에러가 두 파일에 걸쳐 나와서 어느 쪽을 고칠지 흐려진다.
    origins = merge_env_file(values)
    return Config(values=values, source=str(file), origins=origins)


def _install_labels(config: "Config") -> list[str]:
    """라벨 실값 파일을 읽어 테이블에 끼운다.

    경로가 틀렸으면 여기서 죽는다. 조용히 자리표시자로 도는 것이 최악이다 -
    필터가 에러 없이 0건을 돌려주고, 30분 뒤에 빈 결과를 보게 된다.
    """
    from ragdiag import labels as label_mod

    tables = {}
    for key, name in (("labels.query", "query"), ("labels.emotion", "emotion")):
        path = config.get(key)
        if not path:
            continue
        file = Path(path)
        if not file.exists():
            raise ConfigError(
                f"{key}: 라벨 파일이 없습니다: {file}\n"
                f"  운영 taxonomy 문서를 그 자리에 두세요 (형식: `A. 이름 -> 점수`).\n"
                f"  이 파일은 저장소에 올라가지 않습니다.")
        table = label_mod.load_markdown_table(file)
        if not table:
            raise ConfigError(
                f"{key}: {file} 에서 라벨을 하나도 읽지 못했습니다.\n"
                f"  한 줄이 `A. 이름 -> 점수` 형태여야 합니다.")
        tables[name] = table
    return label_mod.install(**tables)


def apply(config: Config) -> list[str]:
    """설정을 settings 모듈에 반영한다. 바꾼 것들을 돌려준다.

    전역을 바꾸는 것이라 조심스럽지만, 대안은 검증기·파서 20개에 config 를
    실어 나르는 것이다. 설정은 실행 시작 시 한 번만 적용되고 그 뒤로 바뀌지
    않으므로 이쪽이 얕게 끝난다.
    """
    changed = _install_labels(config)
    for key, name in TO_SETTINGS.items():
        if key not in config:
            continue
        value = config.get(key)
        if isinstance(value, list):
            value = tuple(value)
        if key == "filter.any_values":
            value = frozenset(value)
        if getattr(settings, name, None) != value:
            setattr(settings, name, value)
            changed.append(f"{name} ← {key}")
    return changed
