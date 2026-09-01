"""테스트 전역 준비.

라벨 실값(llm_eval / llm_emotion 의 이름과 점수)은 저장소에 없다 — 사내 코드값이라
올리지 않는다 (규격 §1.1 · C3). 저장소에는 `labels.py` 의 자리표시자만 있다.

그러면 테스트가 두 갈래로 잘못 갈 수 있다.

- 자리표시자에 기대면 "질의유형 K" 같은 이름을 검증하게 되어 **아무것도 안 잰다.**
  이름이 다 같은 모양이라 표기 변형 해석이 통과해도 통과한 게 아니다.
- 로컬에 실값 파일이 있는 장비에서만 통과하는 테스트를 두면 더 나쁘다. 깨끗한
  사본에서 조용히 건너뛰거나 실패한다.

그래서 테스트는 **자기 테이블을 끼운다.** 아래 이름과 점수는 여기서 지어낸 것이고
사내 값과 무관하다. 재는 것은 값이 아니라 장치다 — 표기 변형 해석, 기대점수 계산,
필터 적용, 그리고 실값이 없을 때 조용히 0건이 나오지 않게 막는 것.
"""

import pytest

from ragdiag.labels import Label

# 실제 테이블이 가진 **성질**만 흉내낸다:
#   - 이름에 공백이 있다 (필터는 붙여쓰고 로그는 띄어쓴다)
#   - 점수가 0~100 을 다 쓴다 (필터가 낮은 점수를 고른다)
#   - 오타 별칭이 있다 (eval 시스템이 문서의 오타를 그대로 뱉는다)
#   - 그룹으로 묶인다
TEST_QUERY = {
    label.letter: label for label in [
        Label("A", "질의 알파", 80, "묶음 하나"),
        Label("B", "질의 브라보", 50, "묶음 하나"),
        Label("C", "질의 찰리", 45, "묶음 하나"),
        Label("D", "질의 델타", 60, "묶음 하나", aliases=("질의 뎔타",)),
        Label("E", "질의 에코", 60, "묶음 하나"),
        Label("F", "질의 폭스", 45, "묶음 둘"),
        Label("G", "질의 골프", 45, "묶음 둘"),
        Label("H", "질의 호텔", 40, "묶음 둘"),
        Label("I", "질의 인디아", 65, "묶음 둘"),
        Label("J", "질의 줄리엣", 40, "묶음 둘"),
        Label("K", "질의 킬로", 25, "묶음 둘"),
        Label("L", "질의 리마", 0, "묶음 셋"),
        Label("M", "질의 마이크", 100, "묶음 셋"),
        Label("N", "질의 노벰버", 75, "묶음 셋"),
        Label("O", "질의 오스카", 70, "묶음 셋"),
        Label("P", "질의 파파", 50, "묶음 셋"),
        Label("Q", "질의 퀘벡", 40, "묶음 셋"),
        Label("R", "질의 로미오", 50, "묶음 셋"),
    ]
}

TEST_EMOTION = {
    label.letter: label for label in [
        Label("A", "감정 알파", 100.0),
        Label("B", "감정 브라보", 87.5),
        Label("C", "감정 찰리", 75.0),
        Label("D", "감정 델타", 62.5),
        Label("E", "감정 에코", 50.0),
        Label("F", "감정 폭스", 37.5),
        Label("G", "감정 골프", 25.0),
        Label("H", "감정 호텔", 12.5),
        Label("I", "감정 인디아", 0.0),
    ]
}

# 가장 낮은 것과 가장 높은 것. 필터가 무엇을 고르는지 재는 테스트가 쓴다.
WORST_EMOTION = TEST_EMOTION["I"]
WORST_QUERY = TEST_QUERY["L"]


@pytest.fixture(autouse=True)
def _install_test_labels():
    """모든 테스트가 같은 테이블 위에서 돈다.

    autouse 인 이유: 합성 데이터 생성기(fixtures/synth.py)도 이 테이블에서 이름을
    끌어오므로, 일부 테스트만 끼우면 생성된 데이터와 검증이 어긋난다.
    """
    from ragdiag import labels

    saved_query = dict(labels.QUERY_LABELS)
    saved_emotion = dict(labels.EMOTION_LABELS)
    labels.install(query=TEST_QUERY, emotion=TEST_EMOTION)
    try:
        yield
    finally:
        labels.install(query=saved_query, emotion=saved_emotion)


@pytest.fixture
def placeholder_labels():
    """자리표시자 상태를 되돌린다. 실값이 없을 때의 동작을 재는 테스트가 쓴다."""
    from ragdiag import labels

    saved_query = dict(labels.QUERY_LABELS)
    saved_emotion = dict(labels.EMOTION_LABELS)
    labels.install(query=dict(labels._PLACEHOLDER_QUERY),
                   emotion=dict(labels._PLACEHOLDER_EMOTION))
    try:
        yield
    finally:
        labels.install(query=saved_query, emotion=saved_emotion)
