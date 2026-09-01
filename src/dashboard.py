"""분류 결과를 훑어보는 대시보드.

  python -m streamlit run <저장소>/src/dashboard.py

--result 를 생략하면 ./output 에서 가장 최근 conv_parsed_*.json 을 고른다.
조직 분류를 붙이려면:

  python -m streamlit run <저장소>/src/dashboard.py -- \
      --dept-class output/class_dept.json \
      --job-class  output/class_job.json

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
    parser.add_argument("--result", help="분류 결과 JSON. 생략하면 --output-dir 의 최신 것")
    parser.add_argument("--output-dir", default="output",
                        help="여기서 가장 최근 conv_parsed_*.json 을 고른다")
    parser.add_argument("--dept-class", help="부서 분류 체계 JSON")
    parser.add_argument("--job-class", help="직급 분류 체계 JSON")
    known, _ = parser.parse_known_args(sys.argv[1:])
    if not known.result:
        known.result = newest_result(known.output_dir)
    return known


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

    파일 이름으로 추측하지 않는다 - class_job 이 직무일 수도 직급일 수도 있고,
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

    path = Path(args.result)
    if not path.exists():
        st.error(f"결과 파일이 없습니다: {path}\n\n"
                 "먼저 src/run.py 로 분류를 돌리세요.")
        st.stop()

    df = load(str(path))
    org = load_org(args.dept_class, args.job_class, flat_records(str(path)))
    if df.empty:
        st.warning("분류된 턴이 없습니다.")
        st.stop()

    st.title("대화 로그 실패 분류")
    others = len(sorted(Path(args.output_dir).glob("conv_parsed_*.json"))) - 1
    stamp = f"  ·  다른 실행 {others}건 더 있음" if others > 0 else ""
    st.caption(f"{path}  ·  {len(df)}건{stamp}")

    # --- 필터 ---------------------------------------------------------------
    with st.sidebar:
        st.header("좁히기")
        depts = st.multiselect("부서", sorted(df["부서"].unique()))
        grades = st.multiselect("직급", sorted(df["직급"].unique()))
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
    for column, chosen in [("부서", depts), ("직급", grades),
                           ("case", cases), ("신뢰도", confs)]:
        if chosen:
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

    # --- 교차표 -------------------------------------------------------------
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
    st.dataframe(table, use_container_width=True)

    share = table.drop(columns="계").div(table["계"], axis=0)
    st.caption("행 내 비율 — 절대 건수가 아니라 성향을 본다")
    st.dataframe(share.style.format("{:.0%}").map(heat), use_container_width=True)

    # --- 코퍼스 보강 목록 ---------------------------------------------------
    # --- 코퍼스 보강 목록 ---------------------------------------------------
    #
    # 부서로 먼저 묶으면 같은 요구가 부서 수만큼 쪼개진다. 그런데 **여러 부서가
    # 같은 것에 막혔다는 사실이 가장 강한 신호다** - 실제로 "연차 이월 예외 조건"이
    # 세 부서에서 나왔는데 부서별로 나누면 세 줄로 흩어져 안 보인다.
    # 문서팀이 받는 것은 "쓸 문서 목록"이므로 요구를 축으로 세운다.
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
               "**여러 부서가 막힌 것이 위로 온다** — 그게 우선순위다.")
    if grouped.empty:
        st.info("해당 케이스가 없습니다.")
    else:
        if len(gaps) > len(grouped):
            st.caption(f"턴 {len(gaps)}건이 요구 {len(grouped)}종으로 묶였다. "
                       "판정자가 같은 요구를 조금 다르게 적으면 따로 잡히므로, "
                       "비슷한 항목이 이웃해 있으면 한 문서로 묶어 볼 것.")
        for _, row in grouped.iterrows():
            tally = f" `×{row['건수']}`" if row["건수"] > 1 else ""
            multi = " 🔥" if row["부서수"] > 1 else ""
            st.markdown(
                f"- **{row['원한것']}**{tally}{multi}  \n"
                f"  <sub>{row['부서']} · 질문: {row['질문']}</sub>",
                unsafe_allow_html=True)
        if (grouped["부서수"] > 1).any():
            st.caption("🔥 는 여러 부서가 같은 것에 막혔다는 뜻이다. "
                       "특정 부서 업무가 아니라 **공통 문서가 비어 있다**는 신호다.")

    # --- 케이스 사전 --------------------------------------------------------
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

    # --- 개별 케이스 --------------------------------------------------------
    st.subheader("개별 케이스")
    st.caption("집계만 보고 근거를 못 보면 '왜 이 라벨이지'에서 막힌다.")
    table_view = view[["부서", "직급", "대화", "턴", "case", "case명",
                       "신뢰도", "충족도", "근거활용", "질문"]]
    st.dataframe(table_view, use_container_width=True, height=280)

    labels = [f"{r['대화']}:{r['턴']}  {tx.label(r['case'])}  {r['질문'][:40]}"
              for _, r in view.iterrows()]
    picked = st.selectbox("근거 펼쳐보기", range(len(labels)),
                          format_func=lambda i: labels[i])
    detail(view.iloc[picked])


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

    st.info(f"**판정 근거** — {row['판정근거']}")
    if row["부가"]:
        st.caption(f"부가 케이스: {', '.join(row['부가'])} "
                   "(주 라벨과 별개로 성립한다)")
    for note in row["주의"]:
        st.warning(note, icon="⚠️")

    left, right = st.columns(2)
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

    if row["검증"]:
        st.markdown("**코드 검증** — LLM 없이 판정된 것들")
        st.dataframe(pd.DataFrame(row["검증"]), use_container_width=True)

    with st.expander("원본 로그"):
        original = row["_원본"]
        st.markdown("**이전 질문들**")
        for q in original.get("pre_queries", []):
            st.markdown(f"- {q}")
        st.markdown("**비판받은 답변**")
        st.code(original.get("llm_ans_on_last_q", ""), language=None)
        st.markdown("**사용자의 불만**")
        st.code(original.get("current_query", ""), language=None)
        st.markdown(f"**그때 검색된 문서** ({len(original.get('chunk_data', []))}개)")
        for i, chunk in enumerate(original.get("chunk_data", [])):
            st.markdown(f"`{i}` {chunk}")


if __name__ == "__main__":
    main()
