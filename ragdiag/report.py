"""[회귀 기준선 전용] 새 코드에서 쓰지 말 것.

conv_parse.py 가 현행 경로다. 이 모듈은 실제 LLM 으로 23/23 검증된
구 파이프라인을 그대로 보존하기 위해 남아 있다 — 그 회귀셋이 새 파이프라인의
라우팅 결함(약한 증거가 강한 증거를 가로챈 문제)을 잡아냈다.

집계 리포트.

중심 산출물은 라벨 분포가 아니라 **라벨 x 부서 교차표**다.
"검색 실패 40%"는 retriever를 갈아엎으라는 막연한 말이지만, "검색 실패가 특정
부서에 몰려 있다"는 그 부서 도메인 문서가 코퍼스에 없다는 구체적 진단이다.

rag_data만으로는 '검색 실패'와 '코퍼스에 애초에 문서 없음'을 구분할 수 없다.
부서 편중은 그 구분을 간접적으로 되살리는 유일한 신호다.
"""

from __future__ import annotations

import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from ragdiag.judge import CaseResult

# 문서가 충분했다고 본 라벨들. 방향성 오류를 재는 데 쓴다.
SUFFICIENT_FAMILY = {"rag_sufficient_generation_failed", "rag_sufficient_other"}
# 문서가 모자랐다는 판정. 둘 다 "코퍼스에 문서를 채워라"로 이어지므로 같은 방향이다.
# rag_partial을 빼놓으면 과소평가 오류를 놓친다.
UNDERSTATES_RAG = {"rag_insufficient", "rag_partial"}
LABEL_ORDER = [
    "rag_insufficient",
    "rag_partial",
    "rag_sufficient_generation_failed",
    "rag_sufficient_other",
    "out_of_scope",
    "unclassified",
]


def _w(text: str) -> int:
    """표시 폭. 한글/한자는 터미널에서 두 칸을 차지하므로 len()으로는 정렬이 깨진다."""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def _pad(text: str, width: int, right: bool = False) -> str:
    gap = " " * max(0, width - _w(text))
    return gap + text if right else text + gap


def write_jsonl(results: list[CaseResult], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        for r in results:
            row = {"case_id": r.case_id, "error": r.error,
                   "input_tokens": r.usage.input_tokens,
                   "output_tokens": r.usage.output_tokens,
                   "cost_usd": round(r.usage.cost_usd, 4), "n_calls": r.n_calls}
            if r.diagnosis:
                row.update(vars(r.diagnosis))
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _crosstab(rows: list, key: str) -> str:
    table: dict[str, Counter] = defaultdict(Counter)
    for d in rows:
        table[getattr(d, key)][d.label] += 1
    if not table:
        return "  (데이터 없음)"

    labels = [l for l in LABEL_ORDER if any(c[l] for c in table.values())]
    col = max(len(l) for l in labels) + 3
    width = max([_w(k) for k in table] + [_w(key)]) + 2
    head = _pad(key, width) + "".join(_pad(l, col, right=True) for l in labels) + _pad("총계", 8, right=True)
    lines = [head, "-" * _w(head)]
    for name in sorted(table, key=lambda k: -sum(table[k].values())):
        counts = table[name]
        total = sum(counts.values())
        cells = "".join(
            _pad(f"{counts[l]} ({counts[l] / total:.0%})" if counts[l] else "-", col, right=True)
            for l in labels
        )
        lines.append(_pad(name, width) + cells + _pad(str(total), 8, right=True))
    return "\n".join(lines)


def _accuracy(rows: list, expected: dict) -> str:
    scored = [(d, expected[d.case_id]) for d in rows if d.case_id in expected]
    if not scored:
        return "  (기대 라벨 없음)"

    hits = [d.label in e["expect"] for d, e in scored]
    lines = [f"  전체 일치: {sum(hits)}/{len(hits)} ({sum(hits) / len(hits):.0%})", ""]

    per_trap: dict[str, list[bool]] = defaultdict(list)
    for (d, e), ok in zip(scored, hits):
        per_trap[e["trap"]].append(ok)
    lines.append("  함정 유형별:")
    for trap in sorted(per_trap):
        v = per_trap[trap]
        lines.append(f"    {trap:<20} {sum(v)}/{len(v)}")

    # 방향성 오류. 두 방향의 비용이 다르다.
    false_insuff = [d.case_id for d, e in scored
                    if d.label in UNDERSTATES_RAG and set(e["expect"]) <= SUFFICIENT_FAMILY]
    false_suff = [d.case_id for d, e in scored
                  if d.label in SUFFICIENT_FAMILY and e["expect"] == ["rag_insufficient"]]
    lines += [
        "",
        f"  false insufficient (충분했는데 부족/부분이라 판정): {len(false_insuff)}",
        "    -> 멀쩡한 코퍼스에 문서를 더 채우게 만드는 오류. unmet_need 부풀리기가 주원인",
        f"  false sufficient (부족한데 충분하다고 판정): {len(false_suff)}",
        "    -> 사전지식 오염(knowledge leakage) 의심. 검색 실패가 통계에서 사라진다",
    ]
    for cid in false_insuff + false_suff:
        lines.append(f"      {cid}")

    # 대명사 의존 판정이 맞았는지. 쿼리 재작성 부재를 짚는 근거가 된다.
    ctx = [(d.context_dependent, e["context_dependent"]) for d, e in scored]
    ctx_ok = sum(1 for a, b in ctx if a == b)
    lines.append(f"\n  context_dependent 일치: {ctx_ok}/{len(ctx)}")
    return "\n".join(lines)


def render(results: list[CaseResult], expected: Optional[dict] = None) -> str:
    ok = [r.diagnosis for r in results if r.diagnosis]
    errors = [r for r in results if r.error]
    out = [
        "=" * 78,
        f"RAG 충족도 진단 리포트  |  성공 {len(ok)}건 / 실패 {len(errors)}건",
        "=" * 78,
        "",
        "[1] 라벨 분포",
    ]

    dist = Counter(d.label for d in ok)
    for label in LABEL_ORDER:
        if dist[label]:
            out.append(f"  {_pad(label, 36)}{dist[label]:>4}  ({dist[label] / len(ok):.0%})")

    out += ["", "[2] 라벨 x 부서", _crosstab(ok, "dept"),
            "", "[3] 라벨 x 직급", _crosstab(ok, "job_grade")]

    # 판정자 자체의 건강 지표. 인용 검증 실패가 잦으면 프롬프트나 청크 품질을 의심해야 한다.
    n_cit = sum(1 for d in ok if d.citation_failed)
    n_ctx = sum(1 for d in ok if d.context_dependent)
    n_drop = sum(len(d.dropped_evidence) for d in ok)
    out += [
        "",
        "[4] 판정 건강 지표",
        f"  인용 검증 실패로 강등된 케이스: {n_cit}  (지어낸 인용 = leakage 시도)",
        f"  폐기된 인용 개수:              {n_drop}",
        f"  맥락 의존 질문:                {n_ctx}  -> 쿼리 재작성 부재 후보군",
    ]

    # rag_data가 통문자열로 오면 \n\n / \n로 청크 경계를 복원한다. 그게 실패하면
    # 전체가 청크 1개가 되는데, 이건 조용히 일어나므로 여기서 눈에 띄게 만든다.
    chunk_counts = [d.n_chunks for d in ok]
    if chunk_counts:
        single = sum(1 for n in chunk_counts if n <= 1)
        avg = sum(chunk_counts) / len(chunk_counts)
        out.append(f"  청크 수 평균:                  {avg:.1f}")
        out.append(
            f"  청크 1개 이하인 케이스:        {single}"
            + ("  <- 청크 경계 복원 실패 의심" if single > len(chunk_counts) * 0.2 else "")
        )

    insuff = [d for d in ok if d.label in ("rag_insufficient", "rag_partial")]
    out += ["", f"[5] 코퍼스 보강 목록 ({len(insuff)}건) - 문서에 없어서 답할 수 없었던 것"]
    for d in sorted(insuff, key=lambda x: x.dept):
        out.append(f"  [{d.dept}] {d.missing or '(미기재)'}")
        out.append(f"      질문: {d.resolved_question}")

    if expected:
        out += ["", "[6] 합성 데이터 검증 (회귀 테스트용 - 실전 정확도가 아님)",
                _accuracy(ok, expected)]

    if errors:
        out += ["", f"[!] 실패한 케이스 {len(errors)}건"]
        out += [f"  {r.case_id}: {r.error}" for r in errors]

    tin = sum(r.usage.input_tokens for r in results)
    tout = sum(r.usage.output_tokens for r in results)
    cost = sum(r.usage.cost_usd for r in results)
    calls = sum(r.n_calls for r in results)
    line = f"LLM 호출 {calls}회 | 입력 {tin:,} 토큰 | 출력 {tout:,} 토큰"
    if cost:
        # CLI가 주는 total_cost_usd는 costBasis="list", 즉 API 정가 환산치다.
        # 구독(OAuth) 인증으로 붙으면 청구액이 아니라 사용량 무게를 재는 지표일 뿐이고,
        # 실제 제약은 돈이 아니라 5시간/주간 사용량 한도다.
        line += f" | list 환산 ${cost:.2f} (청구액 아님)"
    out += ["", line]
    return "\n".join(out)
