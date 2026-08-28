# -*- coding: utf-8 -*-
"""시각화용 유사 로그 생성기.

대시보드가 쓸모 있으려면 데이터에 **읽어낼 신호**가 들어 있어야 한다. 무작위로
만들면 모든 부서가 비슷한 분포로 나오고, 그러면 화면이 예뻐도 아무것도 말하지 않는다.

그래서 부서마다 실패 성향을 다르게 심었다:

  해외영업팀   검색 실패가 몰린다 - 해외 규정 문서가 코퍼스에 부족한 상황
  IT인프라팀   형식·코드 요청 불이행이 많다
  재무팀       계산 오류와 근거 미활용
  인사팀       비교적 고르다 (대조군)
  생산기술팀   맥락 상실과 복합 질문 미완

각 템플릿은 파이프라인이 어떤 case 로 판정할지를 의도하고 만들었다. 골든셋·회귀셋에서
실제로 그 case 가 나온 패턴을 재사용했으므로 대체로 의도대로 나오지만, 판정은 LLM 이
하므로 **정답표가 아니다.** 분포를 보기 위한 입력일 뿐이다.
"""

import json
import random

# ---------------------------------------------------------------------------
# 문서 조각
# ---------------------------------------------------------------------------

DOMESTIC = [
    "국내 출장 식비는 1일 3만원을 상한으로 한다.",
    "국내 출장 숙박비는 1박 8만원을 상한으로 한다.",
    "출장비는 출장 종료 후 5영업일 이내에 정산한다.",
]
OVERSEA_THIN = [
    "해외 출장비 정산 절차: 출장 종료 후 5영업일 이내에 정산서를 제출한다.",
    "숙박비는 실비 정산을 원칙으로 하며 영수증을 첨부한다.",
    "출장 신청은 출발 7일 전까지 팀장 승인을 받는다.",
]
LEAVE = [
    "연차유급휴가 신청은 사전에 그룹웨어를 통해 제출한다.",
    "연차 사용 시 팀장의 승인을 받아야 한다.",
    "연차는 반차 단위로도 사용할 수 있다.",
]
MAIL = [
    "메일 용량 초과 시 오래된 메일을 정리하거나 보관함으로 이동한다.",
    "첨부파일 용량이 큰 메일부터 삭제하는 것을 권장한다.",
    "추가 용량이 필요한 경우 인프라 포털에서 증설을 요청한다.",
]
HEALTH = [
    "임직원 건강검진은 매년 1회 실시하며, 만 35세 이상은 종합검진 대상으로 한다.",
    "만 35세 미만은 일반검진을 실시한다.",
]
CARD = [
    "법인카드 발급 신청은 그룹웨어 전자결재에서 진행한다.",
    "법인카드 사용 내역은 매월 말일 기준으로 자동 집계된다.",
]

# ---------------------------------------------------------------------------
# 템플릿 — (질문, 문서, 답변, 불만, 의도한 case)
# ---------------------------------------------------------------------------

TEMPLATES = {
    # 문서에 답이 없다 (near-miss / distractor)
    "retrieval_fail": [
        ("미주 지역 해외 출장의 1일 숙박비 상한 금액이 얼마인가요?", OVERSEA_THIN,
         "해외 출장 숙박비는 실비 정산이 원칙이며, 영수증을 첨부해 5영업일 이내에 제출하시면 됩니다.",
         "정산 절차 말고 미주 지역 1일 상한 금액을 물었는데요."),
        ("유럽 출장 시 비자 수수료를 회사가 지원하나요?", OVERSEA_THIN,
         "해외 출장 관련 비용은 규정에 따라 처리됩니다.",
         "비자 수수료가 지원 대상인지 근거를 보여주세요."),
        ("법인카드 한도 상향은 누가 승인하나요?", CARD,
         "법인카드 관련 신청은 그룹웨어 전자결재에서 진행하시면 됩니다.",
         "누가 승인하는지를 물었는데 결재 시스템 얘기만 하시네요."),
        ("연차 이월이 가능한 예외 조건이 무엇인가요?", LEAVE,
         "연차는 신청 후 팀장 승인을 받아 사용하시면 됩니다.",
         "이월 조건을 물었습니다. 신청 방법이 아니라요."),
    ],
    # 문서에 답이 있는데 답변이 안 씀
    "generation_fail": [
        ("국내 출장 식비 상한이 얼마인가요?", DOMESTIC,
         "출장 식비는 회사 규정에 따라 지급되며, 자세한 금액은 부서나 직급에 따라 다를 수 있습니다.",
         "부서별로 다른 게 아니라 규정상 금액이 있을 텐데요. 정확한 금액을 알려주세요."),
        ("만 몇 세부터 종합검진 대상인가요?", HEALTH,
         "건강검진 관련 사항은 인사팀에 직접 문의해 주시기 바랍니다.",
         "문의하라는 말 말고 규정에 뭐라고 되어 있는지 알려주세요."),
        ("국내 출장 숙박비 상한은 얼마인가요?", DOMESTIC,
         "숙박비는 실비로 처리되며 상황에 따라 달라질 수 있습니다.",
         "규정에 정해진 금액이 있지 않나요?"),
    ],
    # 포맷 요구 불이행
    "format": [
        ("메일 용량 초과 시 해결 방법을 번호를 매겨 단계별로 알려주세요.", MAIL,
         "오래된 메일을 정리하시고 첨부파일이 큰 것부터 삭제하시면 되며, "
         "필요하면 인프라 포털에서 용량 증설을 요청하실 수도 있습니다.",
         "1, 2, 3 번호를 붙여서 단계별로 달라고 했는데요."),
        ("국내 출장비 항목별 상한을 표로 정리해 주세요.", DOMESTIC,
         "식비는 1일 3만원이고 숙박비는 1박 8만원입니다. 정산은 5영업일 이내입니다.",
         "표로 정리해 달라고 했습니다."),
    ],
    # 길이 요구 불이행
    "length": [
        ("연차 신청 방법을 세 줄 이내로 알려주세요.", LEAVE,
         "연차유급휴가를 신청하시려면 먼저 그룹웨어에 접속하셔야 하며, 인사 메뉴에서 "
         "휴가 신청을 선택하시고, 사용하실 기간을 지정하신 다음, 사유를 기재하시고, "
         "팀장님의 승인을 받으셔야 하며, 승인이 완료되면 알림이 발송되고, 필요한 경우 "
         "신청 내역을 정정하실 수도 있습니다.",
         "세 줄 이내로 해달라고 했는데 너무 깁니다."),
    ],
    # 말투·어조
    "tone": [
        ("전자결재 반려 사유는 어디서 확인하나요?",
         ["전자결재 반려 시 반려 사유가 문서 상세 화면에 표시된다."],
         "전자결재 목록에서 해당 문서를 클릭하시면 반려 사유가 표시됩니다.",
         "말투가 너무 딱딱한데요. 그리고 영어 용어 좀 그만 쓰세요."),
    ],
    # 맞지만 실행할 수 없는 수준
    "not_actionable": [
        ("퇴직연금은 어떻게 운용하나요?",
         ["당사는 확정기여형(DC) 퇴직연금 제도를 운영한다.",
          "가입자는 운용지시를 통해 적립금을 관리할 수 있다."],
         "당사는 확정기여형(DC) 제도를 운영하며, 가입자는 운용지시를 통해 적립금을 관리할 수 있습니다.",
         "그래서 제가 뭘 어떻게 해야 하는 건지 모르겠어요."),
    ],
    # 복합 질문 일부만
    "multi_intent": [
        ("국내 출장 식비와 숙박비 상한을 각각 알려주세요.", DOMESTIC,
         "국내 출장 식비는 1일 3만원입니다.",
         "숙박비는요? 둘 다 물어봤는데요."),
    ],
    # 계산 오류
    "calc": [
        ("5일 출장이면 식비 총액이 얼마인가요? 규정 말고 계산만 해주세요.", [],
         "5일 × 30000 = 120000 원입니다.",
         "계산이 틀렸는데요. 15만원 아닌가요?"),
    ],
    # 코드
    "code": [
        ("부서별 출장비 합계를 구하는 SQL을 작성해 주세요.", [],
         "```sql\nSELECT dept, SUM(amount) FROM travel GROUP BY\n```",
         "이 쿼리를 실행하면 문법 오류가 납니다."),
    ],
    # 맥락 상실
    "context_lost": [
        ("국내 기준으로만 알려주세요. 식비는 얼마인가요?", DOMESTIC,
         "해외 출장 식비는 미주 지역 기준 1일 80달러입니다.",
         "국내 기준이라고 말씀드렸는데요."),
    ],
    # rag_data 가 빈 리스트인 턴. 서비스가 "검색 없이 답할 수 있다"고 판단했는데
    # 실제로는 사내 문서가 있어야 답할 수 있는 질문이었다. 검색을 해서 빗나간
    # case20 과 고칠 곳이 다르다 - 이쪽은 검색을 탈지 말지 정하는 로직이다.
    "no_retrieval": [
        ("우리 회사 육아휴직 분할 사용 횟수 제한이 어떻게 되나요?", [],
         "육아휴직은 법정 기준에 따라 분할 사용이 가능하며, 자세한 사항은 사규를 확인해 주세요.",
         "법 말고 우리 회사 규정을 물었는데요."),
        ("퇴직금 중간정산 신청 요건이 무엇인가요?", [],
         "퇴직금 중간정산은 법령에서 정한 사유에 해당할 때 가능합니다.",
         "사내 절차와 필요 서류를 알려달라고 했는데 원론적인 얘기뿐이네요."),
        ("사내 동호회 지원금 한도가 얼마인가요?", [],
         "동호회 운영 지원은 회사 정책에 따라 이루어집니다.",
         "금액을 물었는데 답이 없네요."),
    ],
    # 모델 자원을 확보하지 못해 서비스가 안내 문구를 대신 내보낸 턴.
    # 답변이 검색 결과와 무관하다는 점이 핵심이다 - 문서는 멀쩡히 붙어 있는데
    # 모델이 그걸 볼 기회조차 없었다. 이 구분이 안 되면 case22(문서는 있는데
    # 안 씀)로 잘못 세고, 프롬프트를 고치러 간다.
    "service_error": [
        ("연차 이월이 가능한 예외 조건이 무엇인가요?", LEAVE,
         "서비스에 문제가 있거나, 사용자 분들이 많아서 서버에 부하가 걸리고 있어요. "
         "잠시 후 다시 시도해 주세요.",
         "답이 왜 안 나와요?"),
        ("국내 출장 식비 상한이 얼마인가요?", DOMESTIC,
         "서비스에 문제가 있거나, 사용자 분들이 많아서 서버에 부하가 걸리고 있어요.",
         "또 이 화면이네요."),
        ("건강검진 대상 연령 기준을 알려주세요.", HEALTH,
         "서비스에 문제가 있거나, 사용자 분들이 많아서 서버에 부하가 걸리고 있어요. "
         "잠시 후 다시 시도해 주세요.",
         "몇 번째 이러는지 모르겠어요."),
    ],
}

# 부서마다 실패 성향을 다르게 심는다. 이 편중이 대시보드가 읽어낼 신호다.
DEPT_PROFILE = {
    "해외영업팀": {"retrieval_fail": 6, "generation_fail": 1, "format": 1, "tone": 1,
              "no_retrieval": 1},
    "IT인프라팀": {"format": 3, "code": 2, "length": 1, "generation_fail": 1, "tone": 1,
              "service_error": 2},
    "재무팀": {"calc": 2, "generation_fail": 3, "retrieval_fail": 1, "multi_intent": 1,
            "service_error": 3},
    "인사팀": {"generation_fail": 2, "retrieval_fail": 2, "length": 1,
             "not_actionable": 1, "format": 1, "no_retrieval": 3},
    "생산기술팀": {"context_lost": 2, "multi_intent": 2, "retrieval_fail": 1,
               "not_actionable": 1},
}

GRADES = ["사원", "대리", "과장", "차장", "Staff Engineer"]
JOBS = {"해외영업팀": "해외영업", "IT인프라팀": "인프라운영", "재무팀": "자금",
        "인사팀": "인사운영", "생산기술팀": "공정기술"}

# 후속 질문의 성격 라벨. 불만이므로 낮은 점수대에 몰린다.
FOLLOWUP = [("K", "명확화 요구", 25), ("L", "명시적 부정 피드백", 0),
            ("F", "조건 변경", 45), ("C", "근거/출처 요구", 45),
            ("Q", "단순 반복/확인", 40), ("H", "형식 변경", 40)]
EMOTION = [("I", "매우 부정", 0.0), ("H", "부정", 12.5), ("G", "약간 부정", 25.0)]


def build(seed: int = 20260828) -> dict:
    rng = random.Random(seed)
    users, conv_no = [], 0

    for dept, profile in DEPT_PROFILE.items():
        plan = [kind for kind, count in profile.items() for _ in range(count)]
        rng.shuffle(plan)
        # 부서당 3~4명에게 나눠 담는다
        per_user = max(1, len(plan) // rng.randint(3, 4))
        chunks_of_plan = [plan[i:i + per_user] for i in range(0, len(plan), per_user)]

        for index, kinds in enumerate(chunks_of_plan):
            conversations = []
            for kind in kinds:
                conv_no += 1
                question, docs, answer, complaint = rng.choice(TEMPLATES[kind])
                # 앞에 잡담 턴을 한둘 붙여 대화 길이를 다양하게 만든다
                lead = rng.choice([0, 0, 1, 2])
                turns, turn_no = [], 0
                for i in range(lead):
                    turn_no += 1
                    turns.append(_turn(
                        turn_no, f"{dept} 관련해서 문의드릴 게 있습니다.",
                        "네, 말씀해 주세요.", [], rng, followup=turn_no > 1, mild=True))
                turn_no += 1
                turns.append(_turn(turn_no, question, answer, docs, rng,
                                   followup=turn_no > 1, mild=True))
                turn_no += 1
                turns.append(_turn(turn_no, complaint, "(아직 답변 없음)", [], rng,
                                   followup=True, mild=False))
                conversations.append(
                    {"conversation_id": f"C-{conv_no:04d}", "turns": turns})

            users.append({
                "user_id": f"EMP-{dept[:2]}{index:02d}",
                "db_login_id": "",
                "job_grade": rng.choice(GRADES),
                "db_dept_name": dept,
                "db_job_name": JOBS[dept],
                "db_position_name": rng.choice(["팀원", "파트장"]),
                "conversations": conversations,
            })

    total = sum(len(c["turns"]) for u in users for c in u["conversations"])
    return {"metadata": {"generated_at": "2026-08-28T09:00:00",
                         "total_users": len(users), "total_turns": total},
            "users": users}


def _turn(no, question, answer, docs, rng, followup, mild):
    """mild=True 면 중립 라벨, False 면 불만 라벨을 붙인다."""
    if not followup:
        label = emotion = None
    elif mild:
        label, emotion = ("E", "단순 연속 질문", 60), ("E", "중립", 50.0)
    else:
        label, emotion = rng.choice(FOLLOWUP), rng.choice(EMOTION)

    day = rng.randint(1, 28)
    return {
        "turn": no,
        "timestamp": f"2026-0{rng.randint(3, 8)}-{day:02d} 1{no % 9}:{rng.randint(10, 59)}:00.000",
        "prev_question": None if no == 1 else "(이전 질문)",
        "retrieved_data": json.dumps(docs, ensure_ascii=False),
        "llm_response": answer,
        "user_question": question,
        "trace_matched": "True",
        "llm_eval_result": None if label is None else label[1],
        "llm_eval_score": None if label is None else float(label[2]),
        "llm_eval_score_top1": None if label is None else label[2],
        "llm_alternatives": [] if label is None
                            else [{"label": label[0], "name": label[1], "probability": 0.93},
                                  {"label": "B", "name": "맥락 추가", "probability": 0.07}],
        "llm_emotion_result": None if emotion is None else emotion[1],
        "llm_emotion_score": None if emotion is None else emotion[2],
        "llm_emotion_score_top1": None if emotion is None else emotion[2],
        "llm_emotion_alternatives": [] if emotion is None
                                    else [{"label": emotion[0], "name": emotion[1],
                                           "probability": 0.95}],
    }
