"""중첩 JSON -> Case 리스트.

입력 형태(data_format_ex.json):
    {"analysis_results": [ {user meta..., "conversations": [ {"turns": [...]} ]} ]}

turns에는 이미 "불만 턴"만 필터링되어 들어있다고 가정한다.

user_id / db_login_id는 여기서 해시로 치환한다. 마스킹을 리포트 단계가 아니라
로딩 단계에 두면 원본 식별자가 어떤 산출물에도 들어가지 않는다. 해시는 salt 없는
결정적 값이라 그룹핑은 그대로 되고, 필요하면 원본에서 역조회도 가능하다.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from ragdiag.schema import Case


def mask(value: str) -> str:
    if not value:
        return "unknown"
    return "u_" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


_PARA_BREAK = re.compile(r"\n\s*\n")


def split_concatenated(text: str) -> list[str]:
    """청크를 이어붙인 통문자열에서 경계를 복원한다.

    빈 줄(\n\n)을 먼저 시도하고, 그걸로 안 쪼개질 때만 단일 개행으로 내려간다.
    순서가 중요하다 - 청크 내부에도 개행이 있을 수 있으므로, 단일 개행부터
    쪼개면 한 청크가 여러 조각으로 찢어진다.

    한계: 청크가 단일 개행으로 이어붙여져 있고 청크 내부에도 개행이 있으면
    경계는 원리적으로 복원 불가능하다. 그래도 인용 검증은 전 청크를 훑으므로
    (verify.verify_evidence) 잘못 쪼개진 경계는 index_corrected로 흡수된다.
    """
    parts = [p.strip() for p in _PARA_BREAK.split(text) if p.strip()]
    if len(parts) > 1:
        return parts
    return [p.strip() for p in text.split("\n") if p.strip()]


def _as_chunks(rag_data: Any) -> list[str]:
    """rag_data를 청크 문자열 리스트로 정규화.

    실데이터는 청크를 \n\n 또는 \n으로 이어붙인 통문자열로 온다. 이걸 통째로
    청크 1개로 두면 chunk_index가 무의미해지고 판정자에게 주는 청크 번호도 쓸모없어진다.
    배열이나 dict 배열로 오는 경우도 함께 받아준다.
    """
    if rag_data is None:
        return []
    if isinstance(rag_data, str):
        return split_concatenated(rag_data)
    if isinstance(rag_data, list):
        chunks = []
        for item in rag_data:
            if isinstance(item, str):
                text = item
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content") or item.get("chunk") or ""
            else:
                text = str(item)
            if text.strip():
                chunks.append(text)
        return chunks
    raise TypeError(f"지원하지 않는 rag_data 형태: {type(rag_data)}")


def _as_queries(pre_queries: Any) -> list[str]:
    if pre_queries is None:
        return []
    if isinstance(pre_queries, str):
        return [pre_queries]
    return [q if isinstance(q, str) else str(q) for q in pre_queries]


def load_cases(path: str | Path) -> list[Case]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return parse_cases(raw)


def parse_cases(raw: dict[str, Any]) -> list[Case]:
    cases: list[Case] = []
    for user in raw.get("analysis_results", []):
        user_id = mask(str(user.get("user_id", "")))
        meta = dict(
            user_id=user_id,
            dept=user.get("db_dept_name") or "unknown",
            job_grade=user.get("job_grade") or "unknown",
            job_name=user.get("db_job_name") or "unknown",
            position_name=user.get("db_position_name") or "unknown",
        )
        for conv in user.get("conversations", []):
            conv_id = str(conv.get("conversation_id", ""))
            for turn in conv.get("turns", []):
                turn_no = int(turn.get("turn", -1))
                cases.append(
                    Case(
                        case_id=f"{user_id}:{conv_id}:{turn_no}",
                        conversation_id=conv_id,
                        turn=turn_no,
                        pre_queries=_as_queries(turn.get("pre_queries")),
                        llm_ans_on_last_q=turn.get("llm_ans_on_last_q") or "",
                        current_query=turn.get("current_query") or "",
                        rag_chunks=_as_chunks(turn.get("rag_data")),
                        **meta,
                    )
                )
    return cases
