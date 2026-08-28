"""conv_eval 포맷 로더.

이전 포맷(data_format_ex.json)과 두 가지가 결정적으로 다르다.

**1. 모든 턴이 들어있다.** 이전에는 불만 턴만 걸러져 있었다. 이제는 대화 전체가 오고,
어느 턴을 볼지는 별도 필터 파일이 정한다. 덕분에 대화 히스토리를 재구성할 수 있어
Stage 1의 대명사 해소가 정확해진다.

**2. 한 턴이 (질문, 답변) 쌍이다.** 이전에는 한 레코드에 "히스토리 + 답변 + 불만"이
같이 있었다. 이제 불만은 **다음 턴의 user_question**이다. 그래서 진단 케이스 하나는
연속한 두 턴의 쌍이 된다:

    turn N   ├─ user_question   사용자가 물은 것
             ├─ retrieved_data  그 질문으로 검색된 문서   ← 충족도 판정 대상
             └─ llm_response    불만을 부른 답변
    turn N+1 └─ user_question   그 답변에 대한 불만       ← 요구를 읽어내는 신호

`retrieved_data`를 turn N에서 가져오는 게 중요하다. 우리가 묻는 건 "비판받은 답변을
만든 문서가 충분했나"이지 "다음 질문으로 검색된 문서"가 아니다.

`llm_eval_result` / `llm_emotion_result`는 **직전 턴의 질문과 답변을 참고해 계산한**
값이다. 그래서 turn 1에는 없고 turn 2부터 존재한다. 이 값이 붙은 턴이 곧 후속 질문이고,
필터는 여기에 거는 것이 자연스럽다.

`trace_matched`는 **그 대화에 턴이 2개 이상 있는지**를 나타낸다. 즉 turns 배열에서
계산할 수 있는 파생값이다. 독립 정보가 아니므로 분포를 세는 건 의미가 없고, 대신
**무결성 검사**로 쓴다 — 선언값과 실제 턴 수가 어긋나면 파일에서 턴이 누락된 것이다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ragdiag.load import mask
from ragdiag.schema import Case


# ---------------------------------------------------------------------------
# 정규화 헬퍼 — 실데이터의 흔한 흠집을 흡수한다
# ---------------------------------------------------------------------------

def _as_bool(value: Any) -> Optional[bool]:
    """trace_matched 가 "True"/"False" 문자열로 온다."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "y", "yes"):
            return True
        if lowered in ("false", "0", "n", "no"):
            return False
    return None


def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value)
        except ValueError:
            return None
    return None


def parse_retrieved(value: Any) -> list[str]:
    """retrieved_data 를 청크 리스트로.

    관측된 형태: "[]" · '["a", "b"]' (JSON을 담은 문자열) · 실제 리스트 · null.
    JSON 문자열이 정상 형태지만, 이스케이프가 깨진 채 오는 경우도 있어
    파싱 실패 시 통문자열로 취급해 경계를 복원한다(이전 포맷과 같은 처리).
    """
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    elif isinstance(value, str):
        text = value.strip()
        if not text or text == "[]":
            return []
        try:
            parsed = json.loads(text)
            items = parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError:
            from ragdiag.load import split_concatenated

            return split_concatenated(text)
    else:
        return [str(value)]

    chunks = []
    for item in items:
        if isinstance(item, str):
            text = item
        elif isinstance(item, dict):
            text = item.get("text") or item.get("content") or item.get("chunk") or ""
        else:
            text = str(item)
        if text.strip():
            chunks.append(text)
    return chunks


def _normalize_alternatives(raw: Any) -> list[dict]:
    """대안 목록의 키 오타를 흡수한다.

    실제로 관측된 것: " probability"(앞 공백), "lable"(label 오타).
    이런 걸 그대로 두면 확률 기반 필터가 조용히 빈 값을 읽는다.
    """
    if not isinstance(raw, list):
        return []
    fixed = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        entry = {}
        for key, value in item.items():
            clean = key.strip()
            if clean == "lable":
                clean = "label"
            entry[clean] = value
        fixed.append({
            "label": entry.get("label", ""),
            "name": entry.get("name", ""),
            "probability": _as_float(entry.get("probability")) or 0.0,
        })
    return fixed


# ---------------------------------------------------------------------------
# 자료구조
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UserMeta:
    user_id: str          # 해시 마스킹된 값. 집계 리포트는 이것만 쓴다.
    dept: str
    job_grade: str
    job_name: str
    position_name: str
    # 출력 파일은 원본 로그 옆에 놓이므로 원본 식별자를 실어 조인할 수 있게 한다.
    # 리포트·집계에는 쓰지 않는다.
    raw_user_id: str = ""
    db_login_id: str = ""


@dataclass(frozen=True)
class Turn:
    turn: int
    timestamp: str
    user_question: str
    llm_response: str
    retrieved: list[str]
    prev_question: str
    trace_matched: Optional[bool]
    # 이 턴이 직전 턴과 어떤 관계인지에 대한 기존 분류. 필터가 거는 대상.
    eval_result: str
    eval_score: Optional[float]
    eval_score_top1: Optional[float]
    eval_alternatives: list[dict] = field(default_factory=list)
    emotion_result: str = ""
    emotion_score: Optional[float] = None
    emotion_score_top1: Optional[float] = None
    emotion_alternatives: list[dict] = field(default_factory=list)

    @property
    def is_followup(self) -> bool:
        """직전 턴에 이어진 질문인가. turn 1은 eval_result 가 비어 있다."""
        return bool(self.eval_result)


@dataclass(frozen=True)
class Conversation:
    conversation_id: str
    user: UserMeta
    turns: list[Turn]

    @property
    def declared_multi_turn(self) -> Optional[bool]:
        """turns 가 선언한 trace_matched. 턴들이 서로 다른 값을 가지면 None."""
        flags = {t.trace_matched for t in self.turns if t.trace_matched is not None}
        return flags.pop() if len(flags) == 1 else None

    @property
    def actual_multi_turn(self) -> bool:
        return len(self.turns) >= 2

    def turn_at(self, number: int) -> Optional[Turn]:
        for t in self.turns:
            if t.turn == number:
                return t
        return None


# ---------------------------------------------------------------------------
# 파싱
# ---------------------------------------------------------------------------

def _parse_turn(raw: dict) -> Turn:
    return Turn(
        turn=int(raw.get("turn", -1)),
        timestamp=raw.get("timestamp") or "",
        user_question=raw.get("user_question") or "",
        llm_response=raw.get("llm_response") or "",
        retrieved=parse_retrieved(raw.get("retrieved_data")),
        prev_question=raw.get("prev_question") or "",
        trace_matched=_as_bool(raw.get("trace_matched")),
        eval_result=raw.get("llm_eval_result") or "",
        eval_score=_as_float(raw.get("llm_eval_score")),
        eval_score_top1=_as_float(raw.get("llm_eval_score_top1")),
        eval_alternatives=_normalize_alternatives(raw.get("llm_alternatives")),
        emotion_result=raw.get("llm_emotion_result") or "",
        emotion_score=_as_float(raw.get("llm_emotion_score")),
        emotion_score_top1=_as_float(raw.get("llm_emotion_score_top1")),
        emotion_alternatives=_normalize_alternatives(raw.get("llm_emotion_alternatives")),
    )


def parse_conversations(raw: dict) -> list[Conversation]:
    conversations: list[Conversation] = []
    for user_raw in raw.get("users", []):
        user = UserMeta(
            user_id=mask(str(user_raw.get("user_id", ""))),
            dept=user_raw.get("db_dept_name") or "unknown",
            job_grade=user_raw.get("job_grade") or "unknown",
            job_name=user_raw.get("db_job_name") or "unknown",
            position_name=user_raw.get("db_position_name") or "unknown",
            raw_user_id=str(user_raw.get("user_id") or ""),
            db_login_id=str(user_raw.get("db_login_id") or ""),
        )
        for index, conv_raw in enumerate(user_raw.get("conversations", [])):
            # conversation_id 가 빠져 있는 대화가 실제로 있다. 케이스 식별자가
            # 겹치지 않도록 순번으로 채운다.
            conv_id = conv_raw.get("conversation_id") or f"{user.user_id}#{index}"
            turns = sorted(
                (_parse_turn(t) for t in conv_raw.get("turns", [])),
                key=lambda t: t.turn,
            )
            conversations.append(Conversation(conv_id, user, turns))
    return conversations


def load_conversations(path: str | Path) -> list[Conversation]:
    return parse_conversations(json.loads(Path(path).read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# 진단 케이스로 변환
# ---------------------------------------------------------------------------

def to_case(conv: Conversation, followup_turn: int) -> Optional[Case]:
    """후속 턴 번호를 받아 진단 케이스를 만든다.

    followup_turn 이 불만이 표현된 턴이고, 그 직전 턴이 비판받은 답변이다.
    직전 턴이 없으면(첫 턴을 지목한 경우) 판정할 대상이 없으므로 None.
    """
    followup = conv.turn_at(followup_turn)
    if followup is None:
        return None
    prior = [t for t in conv.turns if t.turn < followup_turn]
    if not prior:
        return None
    answered = prior[-1]

    return Case(
        case_id=f"{conv.user.user_id}:{conv.conversation_id}:{followup_turn}",
        user_id=conv.user.user_id,
        dept=conv.user.dept,
        job_grade=conv.user.job_grade,
        job_name=conv.user.job_name,
        position_name=conv.user.position_name,
        conversation_id=conv.conversation_id,
        turn=followup_turn,
        # 히스토리는 비판받은 답변까지의 질문 전부. 이전 포맷은 prev_question 하나뿐이라
        # 대명사 해소가 얕았는데, 이제 전체를 넘길 수 있다.
        pre_queries=[t.user_question for t in prior if t.user_question],
        llm_ans_on_last_q=answered.llm_response,
        current_query=followup.user_question,
        # 비판받은 답변을 만든 문서여야 한다. 후속 턴의 검색 결과가 아니다.
        rag_chunks=answered.retrieved,
    )
