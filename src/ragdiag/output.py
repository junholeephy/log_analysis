"""분류 결과를 pre_data_format 형태로 낸다.

원본 포맷에 분류 결과를 덧붙인 모양이다. 기존 필드는 이름과 의미를 그대로 두고,
새로 붙는 것은 전부 `classification` 아래에 모은다 — 섞어 놓으면 나중에
"이 필드가 원본인가 우리가 붙인 건가"를 매번 확인해야 한다.

pre_queries / llm_ans_on_last_q / current_query / chunk_data 는 turn N+1(불만)과
turn N(비판받은 답변)을 짝지은 결과다. 원본 conv_eval 의 한 턴이 아니라는 점에 주의.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from typing import Optional

from ragdiag.classify import TurnResult
if TYPE_CHECKING:                      # 런타임 의존을 만들지 않는다.
    # 쓰는 것은 .user 와 .conversation_id 두 개뿐이다. 운영 장비에서는 그쪽
    # 파서가 만든 객체가 들어올 수 있으므로 런타임에 conv 를 붙들지 않는다.
    from ragdiag.conv import Conversation


def _check_payload(result: TurnResult) -> list[dict]:
    """코드 검증 결과. 위반과 미해당을 구분해서 싣는다."""
    return [
        {"name": check.name, "verdict": check.verdict, "detail": check.detail,
         **({"evidence": check.evidence} if check.evidence else {})}
        for check in result.checks.values()
        if check.verdict != "not_applicable"      # 요구가 없던 항목은 싣지 않는다
    ]


def _evidence_payload(result: TurnResult) -> dict:
    """판정 근거. 왜 그 case 가 나왔는지 사후에 추적할 수 있어야 한다."""
    payload: dict = {}
    if result.observation:
        obs = result.observation
        payload["observation"] = {
            "resolved_question": obs.resolved_question,
            "unmet_need": obs.unmet_need,
            "complaint_target": obs.complaint_target,
            "question_domain": obs.question_domain,
            "question_self_contained": obs.question_self_contained,
            "question_multi_intent": obs.question_multi_intent,
            "answer_refused": obs.answer_refused,
            # complaint_target 을 그렇게 읽은 근거. 어느 값이든 남긴다 - 라벨이
            # 이상할 때 판정자가 무엇을 보고 그랬는지가 첫 단서다.
            "complaint_quote": obs.complaint_quote,
        }
        if obs.complaint_target == "none":
            # 왜 "문제 없음"으로 넘어갔는지(또는 왜 못 넘어갔는지)가 사후에
            # 확인돼야 한다. 이 라벨은 필터를 고치는 근거로 쓰인다.
            if result.complaint:
                payload["observation"]["quote_verified"] = result.complaint.verified
                payload["observation"]["quote_ratio"] = round(result.complaint.ratio, 3)
    if result.judgment:
        payload["sufficiency"] = {
            "verdict": result.judgment.verdict,
            "missing": result.judgment.missing,
            "evidence": [
                {"chunk_index": e.chunk_index, "quote": e.quote,
                 "ratio": round(e.ratio, 3), "index_corrected": e.index_corrected}
                for e in (result.citation.kept if result.citation else [])
            ],
            "dropped_evidence": result.citation.dropped if result.citation else [],
        }
    if result.grounding:
        payload["grounding"] = {"answer_used_rag": result.grounding.answer_used_rag}
    checks = _check_payload(result)
    if checks:
        payload["checks"] = checks
    return payload


def build_turn(result: TurnResult, source_turn_no: int) -> dict:
    case = result.case
    turn: dict = {
        "turn": case.turn,
        "pre_queries": case.pre_queries,
        "llm_ans_on_last_q": case.llm_ans_on_last_q,
        "current_query": case.current_query,
        "chunk_data": case.rag_chunks,
    }
    if result.error:
        turn["classification"] = {"error": result.error}
        return turn

    payload = result.classification.as_dict() if result.classification else {}
    payload["evidence"] = _evidence_payload(result)
    payload["llm_calls"] = result.n_calls
    # 답변을 만든 원본 턴. 짝짓기를 사후에 확인할 수 있어야 한다.
    payload["answered_turn"] = source_turn_no
    turn["classification"] = payload
    return turn


def build_output(pairs: list[tuple["Conversation", TurnResult]]) -> dict:
    """(대화, 결과) 쌍들을 사용자 → 대화 → 턴 으로 다시 묶는다."""
    users: dict[str, dict] = {}
    for conv, result in pairs:
        meta = conv.user
        # 원본 식별자가 있으면 그것으로 묶는다. 출력은 원본 로그 옆에 놓여
        # 조인에 쓰이므로 마스킹본만 남기면 되돌릴 수 없다.
        key = meta.raw_user_id or meta.user_id
        user = users.setdefault(key, {
            "user_id": meta.raw_user_id or meta.user_id,
            "user_id_hashed": meta.user_id,
            "db_login_id": meta.db_login_id,
            "job_grade": meta.job_grade,
            "db_dept_name": meta.dept,
            "db_job_name": meta.job_name,
            "db_position_name": meta.position_name,
            "conversations": {},
        })
        conversation = user["conversations"].setdefault(
            conv.conversation_id, {"conversation_id": conv.conversation_id, "turns": []}
        )
        conversation["turns"].append(build_turn(result, result.case.turn - 1))

    results = []
    for user in users.values():
        conversations = list(user.pop("conversations").values())
        for conversation in conversations:
            conversation["turns"].sort(key=lambda t: t["turn"])
        results.append({**user, "conversations": conversations})
    return {"analysis_results": results}


def summarize(pairs: list[tuple["Conversation", TurnResult]]) -> str:
    """분류 분포. 어디를 고쳐야 하는지 보이게 하는 게 목적이다."""
    from collections import Counter

    from ragdiag.report import _pad, _w

    ok = [r for _, r in pairs if r.classification]
    errors = [r for _, r in pairs if r.error]
    if not ok:
        return f"분류된 턴이 없습니다. (실패 {len(errors)}건)"

    lines = [
        "=" * 78,
        f"분류 결과  |  성공 {len(ok)}건 / 실패 {len(errors)}건",
        "=" * 78,
        "",
        "[1] case 분포",
    ]
    counts = Counter(r.classification.primary_case for r in ok)
    width = max(_w(c) for c in counts) + 2
    for case_id, count in counts.most_common():
        from ragdiag import taxonomy

        meta = taxonomy.describe(case_id)
        lines.append(
            f"  {_pad(case_id, width)}{_pad(meta['case_name'], 26)}"
            f"{count:>5}  {count / len(ok):>5.1%}"
        )

    lines += ["", "[2] type 분포"]
    from ragdiag import taxonomy

    types = Counter(
        taxonomy.describe(r.classification.primary_case)["type_name"] or "(미분류)"
        for r in ok
    )
    for name, count in types.most_common():
        lines.append(f"  {_pad(name, 34)}{count:>5}  {count / len(ok):>5.1%}")

    lines += ["", "[3] 신뢰도"]
    conf = Counter(r.classification.confidence for r in ok)
    for level in ("high", "medium", "low"):
        if conf[level]:
            lines.append(f"  {_pad(level, 10)}{conf[level]:>5}  {conf[level] / len(ok):>5.1%}")
    if conf["low"]:
        lines.append("  └ low 는 판정자의 사전지식에 의존한다. 표본 검토 없이 집계하지 말 것.")

    secondary = Counter(
        c for r in ok for c in r.classification.secondary_cases
    )
    if secondary:
        lines += ["", "[4] 부가 케이스 (주 라벨과 별개로 성립)"]
        for case_id, count in secondary.most_common():
            meta = taxonomy.describe(case_id)
            lines.append(f"  {_pad(case_id, width)}{_pad(meta['case_name'], 26)}{count:>5}")

    if errors:
        lines += ["", f"[!] 실패 {len(errors)}건"]
        for result in errors[:5]:
            lines.append(f"  {result.case.case_id}: {result.error}")

    calls = sum(r.n_calls for _, r in pairs)
    tokens_in = sum(r.usage.input_tokens for _, r in pairs)
    tokens_out = sum(r.usage.output_tokens for _, r in pairs)
    lines += ["", f"LLM 호출 {calls}회 | 입력 {tokens_in:,} | 출력 {tokens_out:,}"]
    return "\n".join(lines)
