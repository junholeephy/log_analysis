"""리포트 렌더링 테스트. API 없이 배선과 정렬만 검증한다."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragdiag.fixtures.synthetic import build
from ragdiag.decide import Diagnosis
from ragdiag.backends import Usage
from ragdiag.judge import CaseResult
from ragdiag.load import parse_cases
from ragdiag.report import _pad, _w, render

VERDICT_OF = {"rag_insufficient": "insufficient", "rag_partial": "partial"}


def _stub_results():
    """기대 라벨을 그대로 판정 결과인 것처럼 넣은 결과 집합.

    100% 일치가 나오는 게 정상이다. 판정 품질이 아니라 리포트 배선을 보는 것이다.
    """
    data, expected = build()
    cases = {c.case_id: c for c in parse_cases(data)}
    results = []
    for cid, e in expected.items():
        c, label = cases[cid], e["expect"][0]
        results.append(CaseResult(case_id=cid, n_calls=2, usage=Usage(1500, 400, 0.1), diagnosis=Diagnosis(
            case_id=cid, label=label, reason="stub",
            complaint_type="format_or_style" if label == "out_of_scope" else "content_gap",
            context_dependent=e["context_dependent"],
            verdict_final=VERDICT_OF.get(label, "sufficient"),
            missing="구체 금액" if label in VERDICT_OF else "",
            resolved_question=c.pre_queries[-1], dept=c.dept, job_grade=c.job_grade,
            n_chunks=len(c.rag_chunks))))
    return results, expected


def test_render_contains_every_section():
    results, expected = _stub_results()
    out = render(results, expected)
    for section in ["[1] 라벨 분포", "[2] 라벨 x 부서", "[3] 라벨 x 직급",
                    "[4] 판정 건강 지표", "[5] 코퍼스 보강 목록", "[6] 합성 데이터 검증"]:
        assert section in out


def test_crosstab_rows_align_by_display_width():
    # 한글은 터미널에서 두 칸을 차지한다. len()으로 패딩하면 표가 깨진다.
    results, expected = _stub_results()
    block = render(results, expected).split("[2] 라벨 x 부서")[1].split("\n\n")[0]
    rows = [l for l in block.strip().splitlines() if l and not l.startswith("-")]
    assert len({_w(r) for r in rows}) == 1


def _metric(out: str, name: str) -> int:
    """지표 값만 뽑아낸다. 문구가 바뀌어도 테스트가 깨지지 않게."""
    line = next(l for l in out.splitlines() if l.strip().startswith(name))
    return int(line.rsplit(":", 1)[1].strip())


def test_perfect_stub_scores_perfectly():
    results, expected = _stub_results()
    out = render(results, expected)
    assert f"전체 일치: {len(expected)}/{len(expected)} (100%)" in out
    assert _metric(out, "false insufficient") == 0
    assert _metric(out, "false sufficient") == 0


def test_partial_counts_as_understating_the_corpus():
    """rag_partial도 "문서를 채워라"로 이어지므로 과소평가 오류로 세야 한다.

    이걸 rag_insufficient만 세면, 충분한 문서를 partial로 깎아내린 케이스가
    지표에서 사라진다. 실제로 이 사각지대 때문에 실패를 놓친 적이 있다.
    """
    results, expected = _stub_results()
    target = next(r for r in results
                  if expected[r.case_id]["expect"] == ["rag_sufficient_generation_failed"])
    target.diagnosis.label = "rag_partial"
    assert _metric(render(results, expected), "false insufficient") == 1


def test_insufficient_on_sufficient_case_also_counts():
    results, expected = _stub_results()
    target = next(r for r in results
                  if expected[r.case_id]["expect"] == ["rag_sufficient_generation_failed"])
    target.diagnosis.label = "rag_insufficient"
    assert _metric(render(results, expected), "false insufficient") == 1


def test_errors_are_surfaced_not_swallowed():
    results, expected = _stub_results()
    results.append(CaseResult(case_id="broken", error="rate limit"))
    out = render(results, expected)
    assert "실패한 케이스 1건" in out and "broken: rate limit" in out


def test_pad_accounts_for_wide_characters():
    assert _w("해외영업팀") == 10
    assert _w(_pad("해외영업팀", 14)) == 14
    assert _w(_pad("abc", 14, right=True)) == 14


def test_chunk_split_failure_is_flagged():
    # rag_data가 통문자열로 오는데 경계 복원이 실패하면 전부 청크 1개가 된다.
    # 조용히 일어나는 실패라 리포트가 눈에 띄게 만들어야 한다.
    results, expected = _stub_results()
    assert "청크 경계 복원 실패 의심" not in render(results, expected)

    for r in results:
        r.diagnosis.n_chunks = 1
    assert "청크 경계 복원 실패 의심" in render(results, expected)
