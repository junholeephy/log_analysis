"""분류 결과를 훑어보는 대시보드.

  python -m streamlit run <저장소>/src/dashboard.py

--result 를 생략하면 ./output 에서 가장 최근 conv_parsed_*.json 을 고른다.
조직 분류를 붙이려면:

  python -m streamlit run <저장소>/src/dashboard.py -- \
      --dept-class configs/dept_class.json \
      --job-class  configs/job_class.json

**src/ 직하에 있어야 한다.** streamlit 은 스크립트가 있는 디렉터리를
sys.path[0] 에 넣는다. src/ragdiag/ 안에 두면 그 디렉터리가 올라가고 src/ 는
안 올라가서 `import ragdiag` 가 자기 자신을 못 찾는다. 실제로 한 번 그렇게
깨졌는데, 서버는 정상으로 뜨고 헬스체크도 통과해서 로그만 봐서는 멀쩡해 보였다 -
streamlit 이 스크립트를 브라우저 접속 시에 실행하고 예외를 화면에만 보이기
때문이다. src/run.py 와 같은 이유로 여기 둔다.

에어갭 장비에서 돌아야 하므로 streamlit·pandas 둘만 추가로 필요하다. 차트는
내장 st.bar_chart 로 그리고 히트맵은 CSS 를 직접 만든다 - plotly 나 matplotlib
을 쓰면 반입할 패키지가 늘어난다.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def _bail(message: str) -> None:
    """무엇을 하라는지까지 적고 멈춘다.

    운영 환경에서는 맨 트레이스백 하나가 사이클을 먹는다. 인터넷도 없고 물어볼
    곳도 없어서, 화면에 적힌 것이 전부다.
    """
    print(f"\n{message}\n", file=sys.stderr)
    raise SystemExit(2)


try:
    import pandas as pd
    import streamlit as st
except ModuleNotFoundError as e:
    _bail(f"대시보드에 필요한 {e.name} 이 없습니다.\n\n"
          f"  {sys.executable} -m pip install -r "
          f"{Path(__file__).resolve().parents[1] / 'requirements-dashboard.txt'}\n\n"
          "분류 파이프라인(src/run.py)에는 필요 없습니다. 그래서 "
          "requirements.txt 와 나눠 뒀습니다 —\n"
          "에어갭 장비에 반입할 것을 늘리지 않기 위해서입니다.")

# streamlit 서버 없이 이 파일을 직접 실행하면 경고만 쏟고 아무 화면도 안 나온다.
# 실패로 끝나지도 않아서 뭐가 잘못됐는지 알기 어렵다.
if not st.runtime.exists():
    # 경로는 사용자가 친 것을, 인터프리터는 지금 도는 것을 그대로 돌려준다.
    # `streamlit run` 은 PATH 에 실행 파일이 있어야 하는데 venv 를 activate 하지
    # 않았거나 공용 venv 를 쓰면 command not found 가 난다. python -m 은 지금
    # 이 인터프리터를 그대로 쓰므로 PATH 를 타지 않는다.
    _bail("이 파일은 streamlit 이 실행합니다. python 으로 직접 돌리면\n"
          "경고만 나오고 화면이 뜨지 않습니다.\n\n"
          f"  {sys.executable} -m streamlit run {sys.argv[0]}\n\n"
          "결과 경로를 생략하면 ./output 의 최신 것을 봅니다. 인자를 주려면\n"
          "`--` 뒤에 붙입니다 — 앞은 streamlit 이 먹습니다.\n"
          "`python -m streamlit` 은 PATH 를 타지 않습니다 — `streamlit` 만 쓰면\n"
          "venv 를 activate 하지 않았을 때 command not found 가 납니다.")

from ragdiag import taxonomy as tx

CONF_ORDER = ["high", "medium", "low"]
CONF_LABEL = {"high": "높음 (코드 검증)", "medium": "중간 (LLM+인용)",
              "low": "낮음 (사전지식 의존)"}


# ---------------------------------------------------------------------------
# 로딩
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="설정 YAML. 생략하면 configs/env.yaml")
    parser.add_argument("--result", help="분류 결과 JSON. 생략하면 --output-dir 의 최신 것")
    parser.add_argument("--output-dir",
                        help="여기서 가장 최근 conv_parsed_*.json 을 고른다")
    parser.add_argument("--dept-class", help="부서 분류 체계 JSON")
    parser.add_argument("--job-class", help="직급 분류 체계 JSON")
    known, _ = parser.parse_known_args(sys.argv[1:])

    # 설정 파일도 읽는다. 여기 적어둔 값이 무시되면 "설정했는데 안 먹는다"가
    # 되고, 대시보드는 매번 인자를 붙여야 하는 도구가 된다 - 화면을 열 때마다
    # 조직 분류 경로를 다시 치게 만들 이유가 없다.
    # 우선순위는 파이프라인과 같다: CLI > 설정 > 기본값.
    config = _load_config(known.config)
    known.output_dir = (known.output_dir or config.get("paths.output_dir")
                        or DEFAULT_OUTPUT_DIR)
    known.dept_class = known.dept_class or config.get("paths.dept_class")
    known.job_class = known.job_class or config.get("paths.job_class")
    known.config_source = config.source

    if not known.result:
        known.result = newest_result(known.output_dir)
    return known


DEFAULT_OUTPUT_DIR = "output"


def _load_config(path: str | None):
    """설정을 읽는다. 못 읽어도 대시보드는 떠야 한다 - 화면을 보는 도구다.

    파이프라인은 설정이 깨지면 계산 전에 죽는 것이 맞지만, 여기서 죽으면
    결과를 볼 방법 자체가 사라진다. 무엇이 잘못됐는지는 화면에 남긴다.
    """
    from ragdiag.config import Config, ConfigError, load

    target = path or ("configs/env.yaml" if Path("configs/env.yaml").exists() else None)
    if not target:
        return Config()
    try:
        return load(target)
    except ConfigError as e:
        st.warning(f"설정을 읽지 못해 기본값으로 띄웁니다.\n\n{e}")
        return Config()


def newest_result(out_dir: str) -> str:
    """--output-dir 에서 가장 최근 결과를 고른다.

    파일 이름에 끝난 시각이 박혀 있어(conv_parsed_20260831-150422.json) 고정된
    경로를 기본값으로 둘 수 없다. 이름이 시각순으로 정렬되므로 마지막이 최신이다.
    """
    found = sorted(Path(out_dir).glob("conv_parsed_*.json"))
    if found:
        return str(found[-1])
    # 시각 스탬프가 없던 시절의 파일이나 --out 으로 직접 지정한 것
    legacy = Path(out_dir) / "conv_parsed.json"
    return str(legacy)


@st.cache_data
def load(path: str) -> pd.DataFrame:
    """중첩 JSON 을 턴 하나 = 한 행으로 편다."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = []
    for user in raw.get("analysis_results", []):
        for conv in user.get("conversations", []):
            for turn in conv.get("turns", []):
                cls = turn.get("classification") or {}
                evidence = cls.get("evidence") or {}
                obs = evidence.get("observation") or {}
                suf = evidence.get("sufficiency") or {}
                rows.append({
                    "부서": user.get("db_dept_name", "-"),
                    "직급": user.get("job_grade", "-"),
                    "직무": user.get("db_job_name", "-"),
                    "대화": conv.get("conversation_id", "-"),
                    "턴": turn.get("turn"),
                    "case": cls.get("case_id", "error"),
                    "case명": cls.get("case_name", ""),
                    "type": cls.get("type_name", ""),
                    "신뢰도": cls.get("confidence", ""),
                    "판정근거": cls.get("reason", ""),
                    "부가": [c.get("case_id") for c in cls.get("secondary_cases", [])],
                    "주의": cls.get("notes", []),
                    "질문": obs.get("resolved_question", ""),
                    "원한것": obs.get("unmet_need", ""),
                    "불만유형": obs.get("complaint_target", ""),
                    "질문성격": obs.get("question_domain", ""),
                    "충족도": suf.get("verdict", ""),
                    "없던것": suf.get("missing", ""),
                    "인용수": len(suf.get("evidence", [])),
                    "폐기인용": len(suf.get("dropped_evidence", [])),
                    "근거활용": (evidence.get("grounding") or {}).get("answer_used_rag", ""),
                    "검증": evidence.get("checks", []),
                    "LLM호출": cls.get("llm_calls", 0),
                    "_원본": turn,
                })
    return pd.DataFrame(rows)


@st.cache_data
def load_org(dept_path: str | None, job_path: str | None, records: list[dict]):
    """조직 분류를 읽고 어느 필드에 붙는지 판별한다.

    파일 이름으로 추측하지 않는다 - job_class 가 직무일 수도 직급일 수도 있고,
    잘못 붙이면 에러 없이 전부 (미분류)가 되어 알아채기 어렵다.
    """
    from ragdiag.org import detect_field, load_classification

    result = {}
    for label, path in (("부서", dept_path), ("직급", job_path)):
        if not path or not Path(path).exists():
            continue
        table = load_classification(path)
        chosen, scores = detect_field(records, table)
        result[label] = {"table": table, "field": chosen, "scores": scores}
    return result


def flat_records(path: str) -> list[dict]:
    """조직 필드 판별용 - 사용자 레코드만 뽑는다."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [{k: v for k, v in u.items() if k != "conversations"}
            for u in raw.get("analysis_results", [])]


LOG_FIELD_TO_COLUMN = {"db_dept_name": "부서", "job_grade": "직급",
                       "db_job_name": "직무", "db_position_name": "직위"}



def heat(value: float) -> str:
    """0~1 을 배경색으로. matplotlib 을 쓰지 않는다.

    Styler.background_gradient 는 matplotlib 을 요구하는데, 대시보드 하나 때문에
    에어갭 장비에 반입할 패키지를 늘릴 이유가 없다. 값에 따라 CSS 를 직접 만든다.
    비율이 높을수록 진해지고, 글자는 배경이 어두워지면 흰색으로 바꾼다.
    """
    if not isinstance(value, (int, float)) or pd.isna(value):
        return ""
    ratio = max(0.0, min(1.0, float(value)))
    if ratio < 0.02:
        return "color: #b0b0b0"
    # 흰색 -> 청록 계열. 낮은 값도 옅게 보이도록 감마를 준다.
    weight = ratio ** 0.7
    r = int(255 - 200 * weight)
    g = int(255 - 90 * weight)
    b = int(255 - 105 * weight)
    text = "#ffffff" if weight > 0.55 else "#1a1a1a"
    return f"background-color: rgb({r},{g},{b}); color: {text}"


def case_tooltips(columns, numeric: bool = True) -> dict:
    """case 열 머리글에 설명을 매단다.

    표에는 case3 만 찍힌다. 이름을 다 적으면 열이 넓어져 한 화면에 안 들어오고,
    번호만 있으면 무엇인지 알 수 없다 - 머리글은 짧게 두고 마우스를 올렸을 때
    펼친다.
    """
    # unclassified · out_of_taxonomy 도 붙인다. 오히려 이 둘이 제일 헷갈린다 -
    # "분류 실패"를 "문제 없음"으로 읽으면 집계 전체를 잘못 읽는다.
    known = set(tx.CASES) | {tx.UNCLASSIFIED, tx.OUT_OF_TAXONOMY}
    column = st.column_config.NumberColumn if numeric else st.column_config.TextColumn
    return {c: column(c, help=tx.tooltip(c))
            for c in columns if isinstance(c, str) and c in known}


def crosstab(df: pd.DataFrame, axis: str) -> pd.DataFrame:
    table = pd.crosstab(df[axis], df["case"])
    order = sorted(table.columns,
                   key=lambda c: (0, int(c[4:])) if c.startswith("case") and c[4:].isdigit()
                   else (1, 0))
    table = table[order]
    table["계"] = table.sum(axis=1)
    return table.sort_values("계", ascending=False)


# ---------------------------------------------------------------------------
# 화면
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="대화 로그 실패 분류", layout="wide")
    args = parse_args()

    st.title("대화 로그 실패 분류")

    # 결과는 실행마다 쌓인다 (파일 이름에 끝난 시각). 최신만 볼 수 있으면 설정을
    # 바꿔가며 돌린 것들을 나란히 못 본다 - 대시보드를 다시 띄우지 않고 고른다.
    runs = sorted(Path(args.output_dir).glob("conv_parsed_*.json"), reverse=True)
    path = Path(args.result)
    with st.sidebar:
        names = [str(r) for r in runs]
        # --result 로 지목한 것이 이 디렉터리 밖일 수 있다. 목록에 없다고 최신으로
        # 갈아치우면 인자가 조용히 무시된다 - 맨 앞에 넣어 그것이 기본이 되게 한다.
        if str(path) not in names:
            names.insert(0, str(path))
        if names:
            picked = st.selectbox(
                "실행 결과", names, index=0,
                format_func=lambda n: Path(n).stem.replace("conv_parsed_", ""),
                help="파일 이름의 시각이 분류가 끝난 시각이다. 최신이 위에 온다")
            path = Path(picked)
        st.divider()

    if not path.exists():
        st.error(f"결과 파일이 없습니다: {path}\n\n"
                 "먼저 src/run.py 로 분류를 돌리세요.")
        st.stop()

    df = load(str(path))
    org = load_org(args.dept_class, args.job_class, flat_records(str(path)))
    if df.empty:
        st.warning("분류된 턴이 없습니다.")
        st.stop()

    others = len(runs) - 1
    stamp = f"  ·  다른 실행 {others}건 더 있음" if others > 0 else ""
    source = getattr(args, "config_source", "(기본값)")
    st.caption(f"{path}  ·  {len(df)}건{stamp}  ·  설정 {source}")

    # --- 필터 ---------------------------------------------------------------
    def _keep_valid(chosen: list[str], options: list[str]) -> list[str]:
        """위층을 바꿔 후보에서 사라진 선택을 흘려보낸다.

        대분류를 A 에서 B 로 갈면 A 밑에서 고른 중분류는 후보에 없다. 그대로
        두면 아무 데이터에도 안 맞아 0건이 나오는데, 화면에는 여전히 선택된
        것처럼 보인다.
        """
        return [c for c in chosen if c in options]

    def cascade(label: str) -> "set[str] | None":
        """대분류 → 중분류 → 소분류 로 좁혀 고른다. 각 층은 복수 선택.

        위층을 고르면 아래층 후보가 거기 매인다. 팀이 수십 개일 때 소분류만
        평평하게 늘어놓으면 고를 수 없기 때문이다.

        분류 체계가 안 붙은 축은 원래대로 값 하나만 고르게 둔다 - 없는 층을
        만들어 보여주면 고를 수 있는 것처럼 보인다.
        """
        from ragdiag.org import allowed_values, level_options, observed_tree

        info = org.get(label)
        table = info["table"] if info and info["field"] else None
        values = [str(v) for v in df[label].unique()]

        if table is None:
            chosen = st.multiselect(label, sorted(values))
            return set(chosen) if chosen else None

        tree = observed_tree(table, values)

        # 한 축의 세 층을 한 상자에 담는다. 층이 늘어날수록 사이드바가 길어지는데,
        # 상자가 없으면 "직급 소분류" 가 부서 것인지 직급 것인지 스크롤하며 세게 된다.
        # 라벨에서 축 이름을 뺄 수 있는 것도 이득이다 - 좁은 화면에서 줄바꿈이 준다.
        with st.container(border=True):
            st.markdown(f"**{label}**")

            # 아래층은 위층을 고른 뒤에만 보인다. 셋을 한꺼번에 늘어놓으면 무엇이
            # 무엇에 매이는지 안 보이고, 소분류만 골랐을 때 위층이 비어 있는
            # 조합이 생겨 "왜 이것만 나오지"가 된다.
            majors = st.multiselect("대분류", level_options(tree, [], [])[0],
                                    key=f"{label}:대분류")

            middles: list[str] = []
            if majors:
                options = level_options(tree, majors, [])[1]
                middles = _keep_valid(
                    st.multiselect("중분류", options, key=f"{label}:중분류"), options)

            items: list[str] = []
            if middles:
                options = level_options(tree, majors, middles)[2]
                items = _keep_valid(
                    st.multiselect("소분류", options, key=f"{label}:소분류"), options)

        return allowed_values(tree, majors, middles, items)

    with st.sidebar:
        st.header("좁히기")
        depts = cascade("부서")
        grades = cascade("직급")
        # 번호만 보고 무엇인지 아는 사람은 없다. 고르는 자리에 이름을 붙이고,
        # 고른 뒤에는 옆 케이스와 가르는 기준까지 펼쳐 준다.
        cases = st.multiselect("case", sorted(df["case"].unique()),
                               format_func=tx.label)
        for cid in cases:
            if tx.desc(cid):
                st.caption(f"**{cid}** {tx.desc(cid)}")
        confs = st.multiselect(
            "신뢰도", [c for c in CONF_ORDER if c in set(df["신뢰도"])],
            format_func=lambda c: CONF_LABEL.get(c, c))
        st.divider()
        if org:
            st.divider()
            st.caption("조직 분류")
            for label, info in org.items():
                if info["field"]:
                    cov = info["scores"][info["field"]]
                    st.caption(f"· {label} ← `{info['field']}` {cov.rate:.0%}")
                    if cov.unmapped:
                        st.caption(f"  체계에 없는 값 {len(cov.unmapped)}종 → (미분류)")
                else:
                    st.caption(f"· {label} 매칭 실패 — 원래 값으로 표시")
        st.divider()
        hide_low = st.checkbox(
            "신뢰도 낮음 제외", value=False,
            help="판정자의 사전지식에 의존하는 케이스(case25 상식 등)를 뺀다. "
                 "다른 케이스와 같은 무게로 집계하면 안 되는 값이다.")

    view = df
    # None 은 "안 좁혔다", 빈 집합은 "골랐는데 해당 없음"이다. 둘을 같게 다루면
    # 조건을 걸었는데 전체가 나온다.
    for column, chosen in [("부서", depts), ("직급", grades),
                           ("case", cases or None), ("신뢰도", confs or None)]:
        if chosen is not None:
            view = view[view[column].isin(chosen)]
    if hide_low:
        view = view[view["신뢰도"] != "low"]

    if view.empty:
        st.warning("조건에 맞는 케이스가 없습니다. 좁히기를 완화하세요.")
        st.stop()

    # --- 판정 건강 지표 (라벨 분포보다 먼저) ---------------------------------
    st.subheader("판정 건강")
    st.caption("라벨 분포보다 먼저 볼 숫자다. 여기가 나쁘면 아래 집계를 믿을 수 없다.")
    dropped = int(view["폐기인용"].sum())
    low = int((view["신뢰도"] == "low").sum())
    unclassified = int(view["case"].isin(["unclassified", "out_of_taxonomy"]).sum())

    # 서비스 자원 부족(case9)은 모델이 답을 만든 적이 없는 턴이다. 품질 분포에
    # 섞어 두면 "챗봇이 나쁘다"로 읽히는데 실제로는 인프라가 모자랐던 것이다.
    service = int((view["case"] == "case9").sum())

    # case0 은 실패가 아니다. 필터가 넓게 잡아 들어온 정상 턴이라 실패율 분모에서
    # 빼야 하고, 이 숫자가 가리키는 것은 챗봇이 아니라 필터다.
    normal = int((view["case"] == "case0").sum())

    cols = st.columns(6)
    cols[0].metric("분류된 턴", len(view),
                   delta=f"-{normal} 정상" if normal else None, delta_color="off",
                   help="case0(정상)을 빼면 실제 실패 건수가 된다.")
    cols[1].metric("지어낸 인용", dropped,
                   help="판정자가 제시한 인용이 원문과 대조되지 않은 건수. "
                        "크면 사전지식 오염을 의심해야 한다.")
    cols[2].metric("신뢰도 낮음", low,
                   help="판정 근거가 판정자의 사전지식뿐인 케이스")
    cols[3].metric("미분류", unclassified,
                   help="분류 실패. '문제 없음'이 아니라 수동 검토 대상이다.")
    cols[4].metric("서비스 오류", service,
                   help="case9 — 모델 자원 부족으로 서비스가 안내 문구를 낸 턴. "
                        "검색·생성 품질과 무관하므로 아래 분포를 읽을 때 빼고 봐야 한다.")
    cols[5].metric("정상", normal,
                   help="case0 — 후속 발화가 앞 답변을 문제 삼지 않은 턴. "
                        "필터가 넓게 잡아 들어온 것이라 챗봇이 아니라 필터를 가리킨다.")
    if normal:
        st.info(f"필터가 고른 {len(view)}건 중 {normal}건({normal/len(view):.0%})은 "
                "불만이 아니었다. 실패율에서 빼고 읽고, 이 비율이 크면 필터를 "
                "좁힐 신호다 — 다만 0 이라고 좋은 것은 아니다. 너무 좁아서 놓치고 "
                "있을 수도 있어 필터 리포트와 짝으로 봐야 한다.", icon="✅")
    if service:
        st.info(f"서비스 자원 부족 응답이 {service}건이다 "
                f"(전체의 {service/len(view):.0%}). 모델이 답을 만든 적이 없는 턴이라 "
                "검색·생성 품질 지표에서 빼고 읽어야 한다 — 고칠 곳은 인프라다.", icon="🔌")
    if dropped:
        st.warning(f"지어낸 인용이 {dropped}건 있다. 해당 케이스의 판정을 먼저 확인할 것.")

    # --- 분포 ---------------------------------------------------------------
    # 절을 탭으로 가른다. 한 화면에 다 쌓으면 태블릿에서 계속 스크롤해야 하고,
    # 어디가 끝이고 어디가 시작인지 알 수 없다. 사이드바 필터는 탭과 무관하게
    # 계속 걸리므로 탭을 옮겨도 보고 있던 범위가 유지된다.
    #
    # **판정 건강은 탭 밖에 둔다.** 탭 안에 넣으면 건너뛸 수 있는데, 그 숫자가
    # 나쁘면 나머지 집계를 믿을 수 없다는 것이 이 화면의 전제다.
    tab_dist, tab_org, tab_gap, tab_case = st.tabs(
        ["분포", "조직", "코퍼스 보강", "개별 케이스"])

    with tab_dist:
        _distribution(view)
    with tab_org:
        _by_org(view, org)
    with tab_gap:
        _corpus_gaps(view)
    with tab_case:
        _cases(view)


def _distribution(view: pd.DataFrame) -> None:
    st.subheader("무엇이 얼마나")
    left, right = st.columns(2)
    with left:
        st.caption("case 별")
        counts = view.groupby(["case", "case명"]).size().reset_index(name="건수")
        counts["라벨"] = counts["case"] + " " + counts["case명"]
        st.bar_chart(counts.set_index("라벨")["건수"], horizontal=True)
    with right:
        st.caption("type 별")
        st.bar_chart(view["type"].replace("", "(미분류)").value_counts(), horizontal=True)


def _by_org(view: pd.DataFrame, org: dict) -> None:
    st.subheader("어느 팀이 무엇을 겪나")
    st.caption(
        "**이 표가 이 도구의 핵심이다.** case20(Retrieve 실패)는 '검색기가 못 찾음'과 "
        "'코퍼스에 문서가 없음'을 합친 라벨이라 로그만으로는 가를 수 없다. "
        "특정 부서에 몰려 있으면 후자, 즉 그 도메인 문서가 비어 있다는 뜻이다.")
    picker = st.columns([2, 3])
    with picker[0]:
        axis = st.radio("기준", ["부서", "직급", "직무"], horizontal=True,
                        label_visibility="collapsed")
    org_for_axis = org.get(axis)
    level_label = {"major": "대분류", "middle": "중분류", "item": "소분류"}
    if org_for_axis and org_for_axis["field"]:
        with picker[1]:
            level = st.radio(
                "층", ["major", "middle", "item"], index=0, horizontal=True,
                format_func=lambda x: level_label[x], label_visibility="collapsed")
        table_source = view.assign(
            **{axis: view[axis].map(
                lambda v: org_for_axis["table"].rollup(v, level))})
        st.caption(f"{level_label[level]} 기준 · "
                   f"{org_for_axis['field']} 필드로 매칭")
    else:
        table_source = view
        if org_for_axis:
            st.caption("이 축에 붙는 분류 체계를 찾지 못해 원래 값으로 표시한다.")

    table = crosstab(table_source, axis)
    st.dataframe(table, use_container_width=True,
                 column_config=case_tooltips(table.columns))

    counts_only = table.drop(columns="계")
    share = counts_only.div(table["계"], axis=0)

    # 비율만 있으면 "50%" 가 2건 중 1건인지 40건 중 20건인지 알 수 없다. 성향을
    # 보려고 비율을 쓰지만, 몇 건 위에서 나온 비율인지 모르면 그 성향을 믿을
    # 수 없다. 색은 비율이 정하고 괄호가 표본 크기를 말한다.
    def with_count(ratio, count):
        return pd.Series(
            ["—" if not c else f"{p:.0%} ({int(c)})" for p, c in zip(ratio, count)],
            index=ratio.index)

    st.caption("행 내 비율 · 괄호는 건수 — 몇 건 위에서 나온 비율인지 함께 본다 · "
               "열 머리글에 마우스를 올리면 무슨 문제인지 나온다")
    st.dataframe(
        share.combine(counts_only, with_count).style.apply(
            lambda _: share.map(heat), axis=None),
        use_container_width=True, column_config=case_tooltips(share.columns, numeric=False))


def _corpus_gaps(view: pd.DataFrame) -> None:
    """문서에 없어서 답할 수 없었던 것.

    부서로 먼저 묶으면 같은 요구가 부서 수만큼 쪼개진다. 그런데 **여러 부서가
    같은 것에 막혔다는 사실이 가장 강한 신호다** - "연차 이월 예외 조건"이 세
    부서에서 나왔는데 부서별로 나누면 세 줄로 흩어져 안 보인다.
    문서팀이 받는 것은 "쓸 문서 목록"이므로 요구를 축으로 세운다.
    """
    gaps = view[view["case"].isin(["case20"]) & (view["없던것"] != "")]
    grouped = (gaps.groupby("원한것")
               .agg(건수=("질문", "size"),
                    부서=("부서", lambda x: " · ".join(sorted(set(x)))),
                    부서수=("부서", lambda x: len(set(x))),
                    질문=("질문", "first"))
               .reset_index()
               .sort_values(["부서수", "건수", "원한것"], ascending=[False, False, True]))

    st.subheader(f"코퍼스 보강 목록 ({len(grouped)}종)")
    st.caption("문서에 없어서 답할 수 없었던 것. 문서팀에 그대로 넘길 수 있는 목록이다. "
               "**막힌 부서 수**로 정렬돼 있다 — 여러 부서가 같은 것에 막혔으면 "
               "특정 부서 업무가 아니라 공통 문서가 비어 있다는 뜻이고, 그게 우선순위다.")
    if grouped.empty:
        st.info("해당 케이스가 없습니다.")
        return

    if len(gaps) > len(grouped):
        st.caption(f"턴 {len(gaps)}건이 요구 {len(grouped)}종으로 묶였다. "
                   "판정자가 같은 요구를 조금 다르게 적으면 따로 잡히므로, "
                   "비슷한 항목이 이웃해 있으면 한 문서로 묶어 볼 것.")

    # 마크다운 목록이었다. 항목이 스무 개면 회색 잔글씨 마흔 줄이 되어 읽히지
    # 않고, 정렬도 검색도 복사도 안 된다. 문서팀에 그대로 넘기는 목록이라
    # 표가 맞다 - 정렬 축(막힌 부서 수)이 곧 우선순위다.
    st.dataframe(
        grouped[["원한것", "부서수", "건수", "부서", "질문"]],
        use_container_width=True, hide_index=True,
        column_config={
            "원한것": st.column_config.TextColumn("필요한 문서", width="medium"),
            "부서수": st.column_config.NumberColumn(
                "막힌 부서", help="여러 부서가 같은 것에 막혔으면 공통 문서가 비어 있다"),
            "건수": st.column_config.NumberColumn("턴"),
            "부서": st.column_config.TextColumn("어느 부서", width="medium"),
            "질문": st.column_config.TextColumn("대표 질문", width="large")})


def _cases(view: pd.DataFrame) -> None:
    with st.expander(f"케이스 설명 — 이 데이터에 나온 {view['case'].nunique()}종"):
        st.caption("무엇이 아닌가까지 적었다. 헷갈리는 쌍(case3/case15, case4/case14, "
                   "case13/case17, case20/case22, case2/case12)에서 라벨이 갈린다.")
        counts = view["case"].value_counts()
        st.dataframe(pd.DataFrame([
            {"case": cid,
             "이름": case.name if (case := tx.get(cid)) else cid,
             "건수": int(n),
             "신뢰도": CONF_LABEL.get(case.confidence, "—") if case else "—",
             "설명": tx.desc(cid)}
            for cid, n in counts.items()
        ]), use_container_width=True, hide_index=True)

    st.subheader("개별 케이스")
    st.caption("**행을 누르면** 그 케이스의 판정 근거가 아래에 펼쳐진다. "
               "집계만 보고 근거를 못 보면 '왜 이 라벨이지'에서 막힌다.")

    # 열 열 개를 늘어놓으면 좁은 화면에서 질문이 끝까지 밀려 안 보인다.
    # 고르는 데 필요한 것만 남기고 나머지는 아래 상세가 편다.
    total = len(view)
    state = st.session_state
    state.setdefault("case_idx", 0)
    # 표를 새 위젯으로 만들면 선택이 지워진다. 버튼으로 옮겼는데 표에는 옛 행이
    # 계속 칠해져 있으면 어느 쪽이 지금 보고 있는 것인지 알 수 없다.
    state.setdefault("case_nonce", 0)
    # 필터가 바뀌면 행 수가 준다. 손대지 않으면 없는 행을 가리킨다.
    state.case_idx = max(0, min(state.case_idx, total - 1))

    # st.dataframe 의 선택은 **프로그램으로 못 옮긴다** - session_state 에 넣어도
    # 조용히 무시된다(실제로 확인했다). 버튼으로 옮긴 자리에 표시가 따라오게
    # 하려면 체크 상태를 우리가 쥐고 있어야 해서 data_editor 를 쓴다.
    compact = view[["부서", "턴", "case명", "신뢰도", "질문"]].copy()
    compact.insert(0, "보기", False)
    compact.iloc[state.case_idx, 0] = True

    edited = st.data_editor(
        compact, use_container_width=True, height=280, hide_index=True,
        key=f"cases-{state.case_nonce}",
        disabled=[c for c in compact.columns if c != "보기"],
        column_config={
            "보기": st.column_config.CheckboxColumn(
                "보기", help="체크한 행이 아래에 펼쳐진다. 이전/다음 버튼도 여기를 옮긴다"),
            "case명": st.column_config.TextColumn("무엇이", width="medium"),
            "신뢰도": st.column_config.TextColumn(
                "신뢰도", help="낮음은 판정자의 사전지식에 의존한다"),
            "질문": st.column_config.TextColumn(
                "질문(정리)", width="large",
                help="대명사와 생략을 푼 것. 사용자가 실제로 친 말은 아래 상세에 있다")})

    # 새로 체크한 행이 있으면 그쪽으로 간다. 한 번에 하나만 켜지도록, 다음 실행에
    # 데이터를 다시 만들면서 나머지가 꺼진다.
    checked = [i for i, on in enumerate(edited["보기"]) if on]
    moved_by_click = [i for i in checked if i != state.case_idx]
    if moved_by_click:
        state.case_idx = moved_by_click[0]
        state.case_nonce += 1
        st.rerun()

    # 표에서 고르는 것만으로는 스무 건을 훑을 수 없다. 한 건 보고 다음 건으로
    # 가는 것이 이 화면에서 가장 잦은 동작이라 버튼을 둔다.
    if _navigate(state, total, "top"):
        st.rerun()

    detail(view.iloc[state.case_idx])

    # 상세를 다 읽고 나면 화면 맨 아래다. 다음 건으로 가려고 표까지 다시 올라가는
    # 것이 이 화면에서 제일 잦은 왕복이라 여기도 둔다.
    st.divider()
    if _navigate(state, total, "bottom"):
        st.rerun()


def _navigate(state, total: int, where: str) -> bool:
    """이전 / 위치 / 다음. 눌렸으면 True.

    같은 버튼을 위아래에 두므로 key 를 나눈다 - 라벨이 같은 위젯 둘은
    streamlit 이 거절한다.
    """
    back, position, forward = st.columns([1, 2, 1])
    moved = False
    if back.button("← 이전", key=f"prev-{where}", use_container_width=True,
                   disabled=state.case_idx <= 0):
        state.case_idx -= 1
        moved = True
    position.markdown(f"**{state.case_idx + 1} / {total}**")
    if forward.button("다음 →", key=f"next-{where}", use_container_width=True,
                      disabled=state.case_idx >= total - 1):
        state.case_idx += 1
        moved = True
    if moved:
        # 표를 새 위젯으로 만든다. data_editor 는 사용자가 만진 것을 위젯 상태에
        # 들고 있어서, 새로 그린 데이터보다 그쪽이 이긴다 - 그러면 버튼으로
        # 옮겨도 체크가 옛 행에 남는다.
        state.case_nonce += 1
    return moved


def detail(row: pd.Series) -> None:
    """한 케이스의 판정 경로를 원본 값 그대로 보여준다."""
    st.markdown(f"### {row['case']} · {row['case명']}")
    if tx.desc(row["case"]):
        st.caption(tx.desc(row["case"]))
    meta = st.columns(4)
    meta[0].metric("신뢰도", CONF_LABEL.get(row["신뢰도"], row["신뢰도"]))
    meta[1].metric("충족도", row["충족도"] or "—")
    meta[2].metric("근거 활용", row["근거활용"] or "—")
    # llm_calls 는 **이번 실행에서 실제로 부른 횟수**다. 캐시가 맞으면 0 이 된다.
    # 그대로 "LLM 호출 0" 이라 적으면 case22 처럼 LLM 이 세 번 필요한 판정도
    # LLM 이 관여 안 한 것처럼 읽힌다. 어디까지 봤는지를 evidence 로 보여준다.
    ran = [name for name, key in [("관측", "observation"), ("충족도", "sufficiency"),
                                  ("근거활용", "grounding")]
           if (row["_원본"].get("classification", {}).get("evidence") or {}).get(key)]
    meta[3].metric("판정 단계", " · ".join(ran) if ran else "코드만",
                   help=f"이번 실행의 LLM 호출 {row['LLM호출']}회. "
                        "0 이면 캐시에 있었거나 코드만으로 판정된 것이다.")

    # 판정 대상인 답변과 사용자의 불만은 접어두면 안 된다. 이 패널은 "왜 이
    # 라벨이지"에 답하는 자리인데, 무엇을 보고 판정했는지가 없으면 답이 안 된다.
    # 전에는 "원본 로그" 접힘 상자 안에 있어서 한 번 더 눌러야 나왔다.
    original = row["_원본"]
    prior = original.get("pre_queries") or []

    # 한 턴의 이야기를 시간순으로 늘어놓는다: 물었다 → 답했다 → 불만이다.
    # 앞 질문을 caption 으로 흘리면 배경으로 읽혀서, 답변이 무엇에 대한
    # 답이었는지가 안 보인다. 셋 다 상자에 넣되 색으로 역할을 가른다.
    # 셋 중 답변이 제일 길고, 판정 대상이기도 하다. 같은 폭을 주면 답변만
    # 여러 줄로 접히고 양옆은 비어서, 정작 봐야 할 것이 제일 읽기 나쁘다.
    asked, said, complained = st.columns([1, 3, 1])
    with asked:
        st.markdown("**앞 질문**")
        st.caption("이 질문에 대한 답이 판정 대상이다")
        with st.container(border=True):
            st.write(prior[-1] if prior else "—")
        if len(prior) > 1:
            st.caption(f"그 앞에 {len(prior) - 1}개 더 (아래 접힘 상자)")
    with said:
        st.markdown("**비판받은 답변**")
        st.caption("이것이 판정 대상이다")
        st.info(original.get("llm_ans_on_last_q", "") or "—")
    with complained:
        st.markdown("**사용자의 불만**")
        st.caption("이 발화가 라벨을 정한다")
        st.warning(original.get("current_query", "") or "—")

    st.info(f"**판정 근거** — {row['판정근거']}")
    if row["부가"]:
        st.caption(f"부가 케이스: {', '.join(row['부가'])} "
                   "(주 라벨과 별개로 성립한다)")
    for note in row["주의"]:
        st.warning(note, icon="⚠️")

    left, right, third = st.columns(3)
    with left:
        st.markdown("**Step 1 · 관측** — 문서를 주지 않고 사용자 쪽 신호만 본다")
        st.json({"정리된 질문": row["질문"], "원한 것": row["원한것"],
                 "불만 유형": row["불만유형"], "질문 성격": row["질문성격"]}, expanded=True)
    with right:
        st.markdown("**Step 2 · 충족도** — 챗봇 답변을 주지 않고 문서만 본다")
        suf = (row["_원본"]["classification"].get("evidence") or {}).get("sufficiency")
        if suf:
            for ev in suf.get("evidence", []):
                st.success(f"청크 {ev['chunk_index']} · 원문 일치 {ev['ratio']}\n\n"
                           f"> {ev['quote']}")
            for bad in suf.get("dropped_evidence", []):
                st.error(f"폐기 ({bad.get('reason')}) — {bad.get('quote', '')[:80]}")
            if row["없던것"]:
                st.caption(f"없던 것: {row['없던것']}")
        else:
            st.caption("도메인 질문이 아니거나 내용 불만이 아니어서 판정하지 않았다.")
    with third:
        # Step 1·2 는 자기 칸이 있는데 Step 3 만 지표 하나로 끝났다. 무엇을
        # 봤고 무슨 뜻인지가 없으면 case22 와 case18 이 왜 갈렸는지 알 수 없다.
        st.markdown("**Step 3 · 근거 활용** — 질문과 불만을 주지 않고 답변과 문서만 본다")
        used = row["근거활용"]
        meaning = {"used": "문서를 썼다. 그런데도 불만이면 기대와 다른 것이다",
                   "ignored": "문서에 답이 있는데 답변이 쓰지 않았다 (case22)",
                   "contradicted": "답변이 문서와 어긋나는 주장을 했다 (case18)"}
        if used:
            st.metric("answer_used_rag", used)
            st.caption(meaning.get(used, ""))
        else:
            st.caption("충족도가 sufficient 이고 인용이 살아남았을 때만 돈다. "
                       "여기까지 오지 않았다.")

    if row["검증"]:
        st.markdown("**코드 검증** — LLM 없이 판정된 것들")
        st.dataframe(pd.DataFrame(row["검증"]), use_container_width=True)

    # 답변·불만은 위로 올렸다. 여기 남는 것은 부피가 큰 것들이다.
    chunks = original.get("chunk_data", [])
    with st.expander(f"그때 검색된 문서 {len(chunks)}개 · 이전 질문 {len(prior)}개"):
        if prior:
            st.markdown("**이전 질문들**")
            for q in prior:
                st.markdown(f"- {q}")
        if chunks:
            st.markdown("**검색된 문서** — Step 2 가 이것만 보고 충족도를 판정했다")
            for i, chunk in enumerate(chunks):
                st.markdown(f"`{i}` {chunk}")
        else:
            st.caption("검색 결과가 0건이다. 서비스가 '검색 없이 답할 수 있다'고 "
                       "판단했을 수 있다 (case21).")


if __name__ == "__main__":
    main()
