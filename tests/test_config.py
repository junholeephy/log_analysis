"""환경변수 인식 테스트.

대상 장비의 .bashrc 에 어떤 이름으로 들어있느냐로 실행 여부가 갈린다.
이름 하나가 목록에서 빠지면 그 장비에서 "주소가 없습니다"로 멈춘다.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run import KEY_VARS, URL_VARS, env_first


@pytest.mark.parametrize("name", ["LLM_API_URL", "API_URL", "RAGDIAG_BASE_URL",
                                  "OPENAI_BASE_URL", "OPENAI_API_BASE"])
def test_every_documented_url_var_is_recognized(name, monkeypatch):
    for other in URL_VARS:
        monkeypatch.delenv(other, raising=False)
    monkeypatch.setenv(name, "http://server:8000")
    assert env_first(URL_VARS) == "http://server:8000"


@pytest.mark.parametrize("name", ["LLM_API_KEY", "API_KEY", "RAGDIAG_API_KEY",
                                  "OPENAI_API_KEY"])
def test_every_documented_key_var_is_recognized(name, monkeypatch):
    for other in KEY_VARS:
        monkeypatch.delenv(other, raising=False)
    monkeypatch.setenv(name, "sk-abc")
    assert env_first(KEY_VARS) == "sk-abc"


def test_llm_prefixed_names_win_when_several_are_set():
    # 대상 장비가 쓰는 이름이 가장 앞이어야 한다.
    assert URL_VARS[0] == "LLM_API_URL"
    assert KEY_VARS[0] == "LLM_API_KEY"


def test_empty_string_is_treated_as_unset(monkeypatch):
    for name in URL_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_API_URL", "")
    monkeypatch.setenv("API_URL", "http://fallback:8000")
    assert env_first(URL_VARS) == "http://fallback:8000"


def test_returns_default_when_nothing_is_set(monkeypatch):
    for name in KEY_VARS:
        monkeypatch.delenv(name, raising=False)
    assert env_first(KEY_VARS, "EMPTY") == "EMPTY"
