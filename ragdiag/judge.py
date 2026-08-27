"""판정 오케스트레이션.

호출을 3단계로 쪼갠 이유는 prompts.py에, 백엔드 차이는 backends.py에 적어두었다.
여기서는 단계 순서와 건너뛰기 규칙, 디스크 캐시, 케이스 단위 병렬만 다룬다.

캐시가 있는 이유: 리포트 코드를 고칠 때마다 판정을 다시 사는 건 낭비다.
CLI 경로는 호출당 $0.05 안팎이라 특히 그렇다.
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol, TypeVar

from pydantic import BaseModel

from ragdiag import prompts
from ragdiag.backends import JudgeError, Usage
from ragdiag.decide import Diagnosis, decide
from ragdiag.schema import Case, GroundingCheck, NeedAnalysis, SufficiencyJudgment
from ragdiag.verify import verify_evidence

T = TypeVar("T", bound=BaseModel)
DEFAULT_MODEL = "claude-opus-5"


class Backend(Protocol):
    model: str

    def complete(
        self, system: str, user: str, out_model: type[T], contract_hint: str = ""
    ) -> tuple[T, Usage]: ...


@dataclass
class CaseResult:
    case_id: str
    diagnosis: Optional[Diagnosis] = None
    error: Optional[str] = None
    usage: Usage = field(default_factory=Usage)
    n_calls: int = 0


class Judge:
    def __init__(self, backend: Optional[Backend] = None,
                 cache_dir: Optional[str | Path] = ".cache"):
        self.backend = backend
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, stage: str, system: str, user: str) -> Optional[Path]:
        if not self.cache_dir:
            return None
        # backend가 없어도(추적 전용) 캐시 경로는 계산할 수 있어야 한다.
        model = self.backend.model if self.backend else DEFAULT_MODEL
        key = hashlib.sha256(
            "\x00".join([stage, model, system, user]).encode("utf-8")
        ).hexdigest()[:24]
        return self.cache_dir / f"{stage}-{key}.json"

    def _call(
        self, stage: str, system: str, user: str, out_model: type[T]
    ) -> tuple[T, Usage]:
        """판정 결과와 **그 호출의** 사용량을 함께 반환한다.

        공유 카운터를 두고 케이스마다 그 차이를 재면 스레드가 섞여서 사용량이
        엉뚱하게 부풀려진다. 호출자가 지역 변수에 쌓게 하면 락도 필요 없다.
        캐시 적중은 실제 호출이 아니므로 빈 Usage를 돌려준다.
        """
        path = self._cache_path(stage, system, user)
        if path and path.exists():
            return out_model.model_validate_json(path.read_text(encoding="utf-8")), Usage()

        parsed, usage = self.backend.complete(
            system, user, out_model, prompts.output_contract(out_model)
        )
        if path:
            path.write_text(parsed.model_dump_json(), encoding="utf-8")
        return parsed, usage

    def analyze_need(self, case: Case) -> tuple[NeedAnalysis, Usage]:
        return self._call(
            "need", prompts.NEED_SYSTEM, prompts.need_user_message(case), NeedAnalysis
        )

    def judge_sufficiency(
        self, case: Case, need: NeedAnalysis
    ) -> tuple[SufficiencyJudgment, Usage]:
        return self._call(
            "sufficiency", prompts.SUFFICIENCY_SYSTEM,
            prompts.sufficiency_user_message(case, need), SufficiencyJudgment,
        )

    def check_grounding(self, case: Case) -> tuple[GroundingCheck, Usage]:
        return self._call(
            "grounding", prompts.GROUNDING_SYSTEM,
            prompts.grounding_user_message(case), GroundingCheck,
        )


def diagnose(case: Case, judge: Judge) -> CaseResult:
    """한 케이스를 끝까지 진단. 단계 건너뛰기 규칙이 여기 있다."""
    usage, n_calls = Usage(), 0

    def track(result: tuple) -> object:
        nonlocal n_calls
        value, u = result
        usage.add(u)
        n_calls += bool(u.input_tokens or u.output_tokens or u.cost_usd)
        return value

    try:
        need = track(judge.analyze_need(case))

        judgment = check = grounding = None
        # 내용에 대한 불만일 때만 sufficiency를 묻는다. 형식 불만에 문서 충족도를
        # 따지는 건 무의미하고, 호출만 낭비한다.
        if need.complaint_type in ("content_gap", "wrong_content"):
            judgment = track(judge.judge_sufficiency(case, need))
            check = verify_evidence(judgment.evidence, case.rag_chunks)
            # 인용이 살아남아 sufficient가 유지될 때만 생성 활용 여부가 의미를 갖는다.
            if judgment.verdict == "sufficient" and check.n_kept > 0:
                grounding = track(judge.check_grounding(case))

        diag = decide(case.case_id, need, judgment, check, grounding)
        diag.dept = case.dept
        diag.job_grade = case.job_grade
        diag.n_chunks = len(case.rag_chunks)
        return CaseResult(
            case_id=case.case_id, diagnosis=diag, n_calls=n_calls, usage=usage,
        )
    except Exception as e:
        # 배치 실행 중 한 케이스의 실패가 나머지를 날리면 안 된다.
        # 타입명을 남겨서 예상 못 한 예외가 조용히 묻히지 않게 한다.
        return CaseResult(case_id=case.case_id, error=f"{type(e).__name__}: {e}")


def run_pipeline(cases: list[Case], judge: Judge, max_workers: int = 4) -> list[CaseResult]:
    results: list[CaseResult] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(diagnose, c, judge): c for c in cases}
        for fut in as_completed(futures):
            results.append(fut.result())
    order = {c.case_id: i for i, c in enumerate(cases)}
    results.sort(key=lambda r: order.get(r.case_id, 1 << 30))
    return results
