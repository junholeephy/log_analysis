"""분류 결과 탐색 대시보드.

  streamlit run dashboard.py -- --result output/conv_parsed.json \
      --dept-class output/class_dept.json --job-class output/class_job.json

에어갭 장비에서 돌아야 하므로 streamlit 하나만 추가로 필요하다. 차트는 내장
st.bar_chart 로 그린다 - plotly 를 쓰면 반입할 패키지가 늘어난다.

조직 분류 파일을 주면 집계 축을 대분류/중분류/소분류로 접었다 펼 수 있다. 팀이 수십
개일 때 소분류로만 보면 표가 읽히지 않는다 - 대분류로 접어야 "어느 본부가 문제인가"가
보이고 그다음 좁혀 들어간다. 어느 필드에 붙는 분류인지는 값을 대조해 자동 판별한다.

화면 구성은 "좁혀들어가기"다. 위에서 필터로 범위를 정하면 아래 모든 표가 같이
좁아지고, 마지막에 개별 케이스의 근거까지 펼쳐본다. 집계만 보여주고 근거를
못 보면 "왜 이 라벨이지"에서 막힌다.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

from ragdiag import taxonomy as tx

CONF_ORDER = ["high", "medium", "low"]
CONF_LABEL = {"high": "높음 (코드 검증)", "medium": "중간 (LLM+인용)",
              "low": "낮음 (사전지식 의존)"}


# ---------------------------------------------------------------------------
# 로딩
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", default="output/conv_parsed.json")
    parser.add_argument("--dept-class", help="부서 분류 체계 JSON")
    parser.add_argument("--job-class", help="직급 분류 체계 JSON")
    known, _ = parser.parse_known_args(sys.argv[1:])
    return known


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
                 "먼저 conv_parse.py 로 분류를 돌리세요.")
        st.stop()

    df = load(str(path))
    org = load_org(args.dept_class, args.job_class, flat_records(str(path)))
    if df.empty:
        st.warning("분류된 턴이 없습니다.")
        st.stop()

    st.title("대화 로그 실패 분류")
    st.caption(f"{path}  ·  {len(df)}건")

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

    cols = st.columns(5)
    cols[0].metric("분류된 턴", len(view))
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
    gaps = view[view["case"].isin(["case20"]) & (view["없던것"] != "")]
    st.subheader(f"코퍼스 보강 목록 ({len(gaps)}건)")
    st.caption("문서에 없어서 답할 수 없었던 것. 문서팀에 그대로 넘길 수 있는 목록이다.")
    if gaps.empty:
        st.info("해당 케이스가 없습니다.")
    else:
        for dept in sorted(gaps["부서"].unique()):
            block = gaps[gaps["부서"] == dept]
            with st.expander(f"{dept} · {len(block)}건", expanded=len(gaps) <= 12):
                for _, row in block.iterrows():
                    st.markdown(f"- **{row['원한것']}**  \n"
                                f"  <sub>질문: {row['질문']}</sub>",
                                unsafe_allow_html=True)

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
    meta[3].metric("LLM 호출", row["LLM호출"])

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
