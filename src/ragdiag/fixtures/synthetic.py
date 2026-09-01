# -*- coding: utf-8 -*-
"""합성 검증 데이터.

**이건 정확도 측정용이 아니다.** 파이프라인 배선 검증과 알려진 함정에 대한
회귀 테스트용이다. 실제 정확도는 실데이터 골든셋에서 재야 한다. 이유:

1. 데이터와 판정 프롬프트를 같은 사람이 만들면 편향을 공유한다. 여기서 나온
   일치율은 실전보다 후하다.
2. 합성 문서는 지어낸 업무 규정이라 판정자가 사전지식으로 알 리가 없다.
   leakage_probe가 그 틈을 일부 메우지만 완전히는 못 메운다.

함정 유형:
  near_miss           주제는 일치, 정확한 답은 부재 -> 과대 관대함을 잡는다
  partial             요구 일부만 문서에 존재
  distractor          그럴듯한 오답 문서가 섞여 있음
  generation_failure  문서엔 답이 있는데 답변이 무시 -> 오분류 방지 장치를 잡는다
  leakage_probe       상식으로 답 가능한 질문 + 그 답이 없는 문서 -> 사전지식 오염
  context_dependent   대명사 의존 질문 -> 쿼리 재작성 부재
  format_complaint    내용은 맞고 형식이 불만 -> 무관 케이스 배제

expect가 여러 개인 케이스는 정답이 하나로 확정되지 않는 경우다. 억지로 하나를
고르게 만드는 것보다 허용 집합을 두는 게 정직하다.

context_dependent는 케이스마다 마지막 질문에 **대상 명사가 있는지**로 정한다.
"주 며칠까지 가능한가요?"처럼 무엇에 대한 질문인지 문장에 없으면 true다.
초기에 이걸 케이스별로 따지지 않고 기본값 false로 둔 탓에 라벨 5개가 틀렸었다.

**주의: 이 셋은 판정 실패를 보고 프롬프트를 여러 차례 고치는 데 사용됐다.**
따라서 여기서 나오는 일치율은 독립적인 정확도 측정치가 아니라 "알려진 회귀가 없다"는
뜻일 뿐이다. 실제 정확도는 손대지 않은 실데이터로 재야 한다.
"""

USERS = {
    "u1": dict(user_id="EMP10021", db_login_id="kim.oversea", job_grade="과장",
               db_dept_name="해외영업팀", db_job_name="해외영업", db_position_name="파트장"),
    "u2": dict(user_id="EMP20044", db_login_id="lee.hr", job_grade="대리",
               db_dept_name="인사팀", db_job_name="인사운영", db_position_name="팀원"),
    "u3": dict(user_id="EMP30112", db_login_id="park.infra", job_grade="차장",
               db_dept_name="IT인프라팀", db_job_name="인프라운영", db_position_name="파트장"),
    "u4": dict(user_id="EMP40077", db_login_id="choi.fin", job_grade="사원",
               db_dept_name="재무팀", db_job_name="자금", db_position_name="팀원"),
}

CASES = [
    # ---------- u1 해외영업팀: rag_insufficient가 몰리도록 의도 설계 ----------
    dict(
        user="u1", conv="C-1001", turn=3, trap="near_miss", expect=["rag_insufficient"],
        pre_queries=[
            "해외 출장 갈 때 숙박비는 어떻게 정산해요?",
            "미주 지역 숙박비 1일 상한이 얼마인가요?",
        ],
        answer="해외 출장 숙박비는 실비 정산이 원칙이며, 영수증을 첨부해 출장 종료 후 "
               "5영업일 이내에 제출하시면 됩니다.",
        complaint="정산 절차 말고 미주 지역 1일 상한 금액이 얼마냐고 물었는데요.",
        chunks=[
            "해외 출장비 정산 절차: 출장 종료 후 5영업일 이내에 출장정산서와 영수증을 첨부하여 제출한다.",
            "숙박비는 실비 정산을 원칙으로 하며, 영수증이 없는 경우 정액 지급으로 대체할 수 있다.",
            "출장 신청은 출발 7일 전까지 팀장 승인을 받아야 한다.",
        ],
    ),
    dict(
        user="u1", conv="C-1001", turn=9, trap="partial", expect=["rag_partial"],
        pre_queries=[
            "해외 출장 숙박비 상한이 지역별로 다른가요?",
            "미주랑 유럽 각각 1일 숙박비 상한 알려주세요.",
        ],
        answer="해외 출장 숙박비 상한은 지역 등급에 따라 차등 적용됩니다.",
        complaint="차등 적용된다는 건 알겠고, 미주랑 유럽 숙박비가 각각 얼마인지 알려주세요.",
        chunks=[
            "미주 지역 해외 출장 시 1일 숙박비 상한은 250달러이다.",
            "해외 출장 숙박비 상한은 지역 등급(A/B/C)에 따라 차등 적용한다.",
            "숙박비 초과분은 사전 승인 시에만 인정한다.",
        ],
    ),
    dict(
        user="u1", conv="C-1002", turn=4, trap="distractor", expect=["rag_insufficient"],
        pre_queries=[
            "중국 출장 시 비자 발급 지원되나요?",
            "비자 수수료도 회사에서 지원해주나요?",
        ],
        answer="네, 해외 출장 시 비자 관련 비용은 회사에서 지원합니다.",
        complaint="그 근거가 어디에 있죠? 총무팀에 물어보니 아니라던데요.",
        chunks=[
            "해외 출장 시 항공료와 숙박비는 회사가 부담한다.",
            "여권 발급 비용은 개인 부담을 원칙으로 한다.",
            "출장 중 현지 교통비는 실비 정산 대상이다.",
        ],
    ),
    dict(
        user="u1", conv="C-1003", turn=7, trap="context_dependent",
        expect=["rag_insufficient"], context_dependent=True,
        pre_queries=[
            "해외 지사 파견 제도가 어떻게 되나요?",
            "파견 기간은 보통 얼마나 되나요?",
            "그거 연장도 가능한가요?",
        ],
        answer="연장 관련해서는 별도 규정을 확인해 주시기 바랍니다.",
        complaint="파견 기간 연장 얘기였는데 엉뚱한 답이 나왔네요.",
        chunks=[
            "해외 지사 파견 대상자는 부서장 추천 후 인사위원회에서 선정한다.",
            "파견자에게는 주거 지원비와 자녀 교육비를 지원한다.",
            "파견 전 사전 교육을 이수해야 한다.",
        ],
    ),

    # ---------- u2 인사팀 ----------
    dict(
        user="u2", conv="C-2001", turn=5, trap="partial", expect=["rag_partial"],
        pre_queries=["연차 이월 규정 알려주세요.", "미사용 연차를 다음 해로 넘길 수 있나요?"],
        answer="미사용 연차는 원칙적으로 소멸됩니다.",
        complaint="이월 가능한 경우가 있다고 들었는데, 그 조건이 정확히 뭔지 알려주세요.",
        chunks=[
            "미사용 연차는 발생일로부터 1년이 경과하면 소멸하는 것을 원칙으로 한다.",
            "다만 회사의 귀책사유로 연차를 사용하지 못한 경우 예외를 둘 수 있다.",
            "연차 사용 촉진 제도를 운영하는 경우 그 절차에 따른다.",
        ],
    ),
    dict(
        user="u2", conv="C-2001", turn=12, trap="distractor", expect=["rag_insufficient"],
        # "주 며칠까지 가능한가요?" - 무엇이 주 며칠인지 문장에 없다
        context_dependent=True,
        pre_queries=["재택근무 신청 조건이 어떻게 되나요?", "주 며칠까지 가능한가요?"],
        answer="재택근무는 주 2회까지 가능합니다.",
        complaint="주 2회라는 근거가 뭐예요? 저희 팀은 주 3회 하던데요.",
        chunks=[
            "유연근무제는 시차출퇴근제, 선택근무제, 재택근무제로 구성된다.",
            "시차출퇴근은 주 5일 모두 신청 가능하다.",
            "재택근무 신청은 직전 주 금요일까지 팀장에게 제출한다.",
        ],
    ),
    dict(
        user="u2", conv="C-2002", turn=2, trap="generation_failure",
        expect=["rag_sufficient_generation_failed"],
        pre_queries=["건강검진 대상자 기준이 어떻게 되나요?", "만 몇 세부터 종합검진 대상인가요?"],
        answer="건강검진 관련 사항은 인사팀에 직접 문의해 주시기 바랍니다.",
        complaint="문의하라는 말 말고, 규정에 뭐라고 되어 있는지 알려주세요.",
        chunks=[
            "임직원 건강검진은 매년 1회 실시하며, 만 35세 이상은 종합검진 대상으로 한다.",
            "만 35세 미만은 일반검진을 실시한다.",
            "검진 미수검자는 별도 안내에 따라 추가 일정을 배정한다.",
        ],
    ),
    dict(
        user="u2", conv="C-2002", turn=8, trap="leakage_probe", expect=["rag_insufficient"],
        pre_queries=["연차는 법적으로 언제부터 발생하나요?", "1년 미만 근로자도 연차가 생기나요?"],
        answer="근로기준법상 1년 미만 근로자는 1개월 개근 시 1일의 유급휴가가 발생합니다.",
        complaint="그게 우리 회사 사규에도 그렇게 되어 있나요? 업무 규정 근거를 보여주세요.",
        chunks=[
            "연차유급휴가 신청은 사전에 그룹웨어를 통해 제출한다.",
            "연차 사용 시 팀장의 승인을 받아야 한다.",
            "연차는 반차 단위로도 사용할 수 있다.",
        ],
    ),
    dict(
        user="u2", conv="C-2003", turn=6, trap="context_dependent",
        expect=["rag_insufficient"], context_dependent=True,
        pre_queries=["교육비 지원 제도 알려주세요.", "어학 과정도 포함되나요?", "그건 연간 한도가 얼마예요?"],
        answer="교육비는 연간 한도 내에서 지원됩니다.",
        complaint="어학 과정 연간 한도 금액이 얼마냐고 물었어요.",
        chunks=[
            "임직원 교육비 지원은 직무 관련 과정을 대상으로 한다.",
            "교육 수료 후 증빙을 제출해야 지원금이 지급된다.",
            "교육비 지원은 사전 신청을 원칙으로 한다.",
        ],
    ),
    dict(
        user="u2", conv="C-2003", turn=11, trap="format_complaint", expect=["out_of_scope"],
        pre_queries=["복지포인트 사용처 알려주세요."],
        answer="복지포인트는 운영 환경 복지몰, 제휴 가맹점, 도서 구입, 문화 활동 등에 사용할 수 있습니다.",
        complaint="답변이 너무 짧아요. 좀 더 친절하게 풀어서 설명해주면 안 되나요?",
        chunks=[
            "복지포인트는 운영 환경 복지몰과 제휴 가맹점에서 사용할 수 있다.",
            "도서 구입 및 문화 활동에도 사용 가능하다.",
            "복지포인트는 매년 1월에 일괄 부여되며 당해 연도 내에 소진해야 한다.",
        ],
    ),
    dict(
        user="u2", conv="C-2004", turn=4, trap="ambiguous_depth",
        # "우리 회사는 어떤 방식인가요?" - 무엇의 방식인지 문장에 없다
        context_dependent=True,
        expect=["rag_sufficient_other", "rag_partial"],
        pre_queries=["퇴직연금 DC형과 DB형 차이가 뭔가요?", "우리 회사는 어떤 방식인가요?"],
        answer="당사는 확정기여형(DC) 퇴직연금을 운영하고 있으며, 회사가 매년 연간 임금총액의 "
               "12분의 1 이상을 부담금으로 납입합니다.",
        complaint="그건 알겠는데, 그래서 제가 뭘 어떻게 해야 하는 건지 모르겠어요.",
        chunks=[
            "당사는 확정기여형(DC) 퇴직연금 제도를 운영한다.",
            "회사는 매년 연간 임금총액의 12분의 1 이상을 부담금으로 납입한다.",
            "가입자는 운용지시를 통해 적립금을 관리할 수 있다.",
        ],
    ),

    # ---------- u3 IT인프라팀 ----------
    dict(
        user="u3", conv="C-3001", turn=3, trap="near_miss", expect=["rag_insufficient"],
        pre_queries=["운영 환경 VPN 접속이 안 되는데요.", "MFA 기기를 바꿨는데 VPN 인증이 계속 실패해요."],
        answer="VPN 접속 문제는 대부분 네트워크 설정 때문입니다. 클라이언트를 재설치해 보세요.",
        complaint="MFA 기기 변경했을 때 재등록 절차를 알려달라니까요.",
        chunks=[
            "VPN 클라이언트 설치 및 초기 설정 안내: 운영 환경 포털에서 설치 파일을 내려받아 실행한다.",
            "VPN 접속 오류 시 네트워크 어댑터를 재설정하고 클라이언트를 재시작한다.",
            "MFA는 전사 필수 정책이며 모든 운영 환경 시스템 접근에 적용된다.",
        ],
    ),
    dict(
        user="u3", conv="C-3001", turn=10, trap="partial", expect=["rag_partial"],
        # "신청하면 얼마나 쓸 수 있어요?" - 무엇을 신청/사용하는지 문장에 없다
        context_dependent=True,
        pre_queries=["운영 환경 GPU 서버 사용 신청 어떻게 하나요?", "신청하면 얼마나 쓸 수 있어요?"],
        answer="GPU 서버는 신청 후 승인을 받아 사용하실 수 있습니다.",
        complaint="사용 가능 시간이랑 동시 점유 제한이 어떻게 되는지 구체적으로 알려주세요.",
        chunks=[
            "GPU 서버 사용은 사전 신청 및 팀장 승인을 필요로 한다.",
            "1회 신청 시 최대 72시간까지 점유할 수 있다.",
            "신청은 운영 환경 인프라 포털에서 진행한다.",
        ],
    ),
    dict(
        user="u3", conv="C-3002", turn=5, trap="distractor", expect=["rag_insufficient"],
        pre_queries=["운영 환경 계정 비밀번호 정책이 어떻게 되나요?", "비밀번호 변경 주기가 몇 개월인가요?"],
        answer="비밀번호는 90일마다 변경하셔야 합니다.",
        complaint="90일 맞아요? 최근에 정책이 바뀌었다고 들었는데 근거를 보여주세요.",
        chunks=[
            "계정 비밀번호는 영문 대소문자, 숫자, 특수문자를 포함해 12자 이상으로 설정한다.",
            "퇴직자 계정은 퇴직일로부터 30일 이내에 비활성화한다.",
            "관리자 계정은 일반 계정과 분리하여 운영한다.",
        ],
    ),
    dict(
        user="u3", conv="C-3002", turn=14, trap="generation_failure",
        expect=["rag_sufficient_generation_failed"],
        pre_queries=["노트북 교체 주기가 어떻게 되나요?"],
        answer="노트북 교체는 상황에 따라 다르며, 일반적으로 고장이 발생하면 교체가 가능합니다.",
        complaint="고장 안 나면요? 정해진 주기가 있을 텐데요.",
        chunks=[
            "업무용 노트북의 표준 교체 주기는 지급일로부터 4년으로 한다.",
            "교체 주기 이전이라도 수리비가 잔존가치를 초과하면 교체할 수 있다.",
            "반납된 장비는 데이터 완전 삭제 후 폐기한다.",
        ],
    ),
    dict(
        user="u3", conv="C-3003", turn=2, trap="leakage_probe", expect=["rag_insufficient"],
        pre_queries=["HTTPS 기본 포트가 몇 번인가요?", "운영 환경 서버 방화벽에서 열려 있는 포트요."],
        answer="HTTPS는 443 포트를 사용합니다.",
        complaint="일반론 말고, 운영 환경 방화벽에서 실제로 허용된 포트 목록을 알려주세요.",
        chunks=[
            "방화벽 정책 변경은 정보보안팀 승인 후 적용한다.",
            "포트 개방 요청은 인프라 포털에서 접수한다.",
            "미사용 포트는 분기별 점검을 통해 차단한다.",
        ],
    ),
    dict(
        user="u3", conv="C-3003", turn=9, trap="format_complaint", expect=["out_of_scope"],
        # "단계별로 정리해줄 수 있나요?" - 무엇을 정리하는지 문장에 없다
        context_dependent=True,
        pre_queries=["운영 환경 메일 용량 초과 시 해결 방법 알려주세요.", "단계별로 정리해줄 수 있나요?"],
        answer="운영 환경 메일 시스템의 용량이 초과되는 경우에는 여러 가지 방법을 고려해 볼 수 있는데, "
               "우선 오래된 메일을 정리하는 것이 일반적이며, 첨부파일이 큰 메일부터 확인하시는 것이 "
               "효율적이고, 보관함으로 이동하는 방법도 있으며, 필요하다면 인프라 포털에서 용량 증설을 "
               "요청하실 수도 있습니다.",
        complaint="단계별로 번호 매겨서 정리해달라니까 또 줄글로 쓰셨네요.",
        chunks=[
            "메일 용량 초과 시 오래된 메일을 정리하거나 보관함으로 이동한다.",
            "첨부파일 용량이 큰 메일부터 삭제하는 것을 권장한다.",
            "추가 용량이 필요한 경우 인프라 포털에서 증설을 요청한다.",
        ],
    ),

    # ---------- u4 재무팀 ----------
    dict(
        user="u4", conv="C-4001", turn=4, trap="near_miss", expect=["rag_insufficient"],
        pre_queries=["법인카드 한도 조정 신청은 어디서 하나요?", "한도 상향은 누가 승인하나요?"],
        answer="법인카드 관련 신청은 그룹웨어 전자결재에서 진행하시면 됩니다.",
        complaint="누가 승인하는지를 물었는데 결재 시스템 얘기만 하네요.",
        chunks=[
            "법인카드 발급 신청은 그룹웨어 전자결재 > 총무 > 법인카드 신청 메뉴에서 진행한다.",
            "법인카드 사용 내역은 매월 말일 기준으로 자동 집계된다.",
            "분실 시 즉시 카드사에 신고하고 총무팀에 통보한다.",
        ],
    ),
    dict(
        user="u4", conv="C-4001", turn=11, trap="partial", expect=["rag_partial"],
        pre_queries=["경조사비 지원 기준이 어떻게 되나요?", "본인 결혼이랑 부모상 각각 얼마인가요?"],
        answer="경조사비는 사규에 따라 지원됩니다.",
        complaint="금액을 물어봤는데요. 본인 결혼이랑 부모상 각각 얼마죠?",
        chunks=[
            "경조사비 지원 기준: 본인 결혼 100만원, 자녀 결혼 50만원.",
            "경조 휴가는 별도 규정에 따른다.",
            "경조사비 신청은 증빙서류 첨부 후 총무팀에 제출한다.",
        ],
    ),
    dict(
        user="u4", conv="C-4002", turn=3, trap="generation_failure",
        # "국내 출장 기준으로요." - 무엇의 기준인지 문장에 없다
        context_dependent=True,
        expect=["rag_sufficient_generation_failed"],
        pre_queries=["출장 중 식비 상한이 얼마인가요?", "국내 출장 기준으로요."],
        answer="출장 식비는 회사 규정에 따라 지급되며, 자세한 금액은 부서별로 다를 수 있습니다.",
        complaint="일반적으로 말고 정확한 금액이요.",
        chunks=[
            "국내 출장 식비는 1일 3만원을 상한으로 한다.",
            "국내 출장 숙박비는 1박 8만원을 상한으로 한다.",
            "출장비는 출장 종료 후 정산한다.",
        ],
    ),
    dict(
        user="u4", conv="C-4002", turn=7, trap="leakage_probe", expect=["rag_insufficient"],
        pre_queries=["부가가치세율이 몇 퍼센트인가요?", "우리 회사 세금계산서 발행 시 적용 세율이요."],
        answer="부가가치세율은 10%입니다.",
        complaint="업무 문서 기준으로 확인해주세요. 영세율 대상도 있을 텐데요.",
        chunks=[
            "세금계산서는 거래일이 속하는 달의 다음 달 10일까지 발행한다.",
            "세금계산서 발행은 재무팀 승인 후 진행한다.",
            "전자세금계산서 발행 시스템은 그룹웨어와 연동되어 있다.",
        ],
    ),
    dict(
        user="u4", conv="C-4003", turn=6, trap="context_dependent",
        expect=["rag_insufficient"], context_dependent=True,
        pre_queries=["법인차량 운행일지를 꼭 써야 하나요?", "안 쓰면 어떻게 되나요?",
                     "그럼 그건 언제까지 제출해야 해요?"],
        answer="관련 서류는 기한 내에 제출하시면 됩니다.",
        complaint="운행일지 제출 기한을 물었는데 그냥 기한 내라니요.",
        chunks=[
            "법인차량은 업무 목적으로만 사용해야 한다.",
            "운행일지 미작성 시 사적 사용으로 간주될 수 있다.",
            "법인차량 배정은 총무팀에서 관리한다.",
        ],
    ),
    dict(
        user="u4", conv="C-4003", turn=13, trap="format_complaint", expect=["out_of_scope"],
        pre_queries=["전자결재 반려 사유 확인하는 법 알려주세요."],
        answer="전자결재 목록에서 해당 문서를 클릭하면 반려 사유를 확인할 수 있습니다.",
        complaint="말투가 너무 딱딱한데요. 그리고 영어 용어 좀 그만 쓰세요.",
        chunks=[
            "전자결재 반려 시 반려 사유가 문서 상세 화면에 표시된다.",
            "반려된 문서는 수정 후 재상신할 수 있다.",
            "결재선 변경은 기안자만 가능하다.",
        ],
    ),
]


def build(rag_as_string: bool = True) -> tuple[dict, dict]:
    """(입력 JSON, 기대 라벨 맵)을 만든다. 실데이터와 같은 중첩 구조로 낸다.

    rag_as_string=True가 기본인 이유: 실데이터의 rag_data는 청크를 \n\n으로
    이어붙인 통문자열이다. 픽스처가 배열로 나오면 실전에서만 도는 경로(청크 경계
    복원)가 회귀 테스트에서 빠진다.
    """
    from ragdiag.load import mask

    by_user: dict[str, dict] = {}
    expected: dict[str, dict] = {}

    for case in CASES:
        u = case["user"]
        node = by_user.setdefault(u, {**USERS[u], "conversations": {}})
        conv = node["conversations"].setdefault(case["conv"], [])
        conv.append({
            "turn": case["turn"],
            "pre_queries": case["pre_queries"],
            "llm_ans_on_last_q": case["answer"],
            "current_query": case["complaint"],
            "rag_data": "\n\n".join(case["chunks"]) if rag_as_string else case["chunks"],
        })
        case_id = f"{mask(USERS[u]['user_id'])}:{case['conv']}:{case['turn']}"
        expected[case_id] = {
            "trap": case["trap"],
            "expect": case["expect"],
            "context_dependent": case.get("context_dependent", False),
        }

    results = []
    for u, node in by_user.items():
        convs = [
            {"conversation_id": cid, "turns": sorted(turns, key=lambda t: t["turn"])}
            for cid, turns in node.pop("conversations").items()
        ]
        results.append({**node, "conversations": convs})

    return {"analysis_results": results}, expected


# ---------------------------------------------------------------------------
# 구 라벨(6개) -> 신 taxonomy case 매핑
#
# 이 셋은 case20/case22 판별 전용으로 만들어졌고 실제 LLM 으로 23/23 검증된 유일한
# 케이스 집합이다. 새 파이프라인이 같은 결과를 내는지 확인하는 데 쓴다.
#
# 1:1 이 아니다. out_of_scope 하나가 형식·길이·언어로 쪼개졌고, rag_partial 은
# case20 에 흡수됐다. 쪼개진 쪽은 케이스마다 expect_case 로 직접 지정한다.
# ---------------------------------------------------------------------------

LEGACY_TO_CASE = {
    "rag_insufficient": {"case20"},
    "rag_partial": {"case20"},                       # 신 체계에서는 둘 다 case20
    "rag_sufficient_generation_failed": {"case22"},
    "rag_sufficient_other": {"case13"},              # 의도와 다른 답변
    "unclassified": {"unclassified", "out_of_taxonomy"},
}

# out_of_scope 였던 3건과, 구 기준에서도 정답이 하나로 확정되지 않던 1건.
EXPECT_CASE = {
    # 구 기준 expect 가 ["rag_sufficient_other", "rag_partial"] 이었다.
    # rag_partial 은 신 체계에서 case20 이므로 그쪽도 정답이다.
    "C-2004:4": {"case13", "case20"},
    "C-2003:11": {"case11"},                 # "답변이 너무 짧아요" -> 길이
    "C-3003:9": {"case12"},                  # "번호 매겨서" -> 포맷
    # "말투가 딱딱하고 영어 용어가 많다". v1 에서는 갈 곳이 없어 미분류로 떨어졌고,
    # 그것이 v2 에 case16(말투·어조)를 추가한 이유다. 이제 제 자리로 간다.
    "C-4003:13": {"case16"},
}


def expected_cases(conv_id: str, turn: int, legacy_label: str) -> set:
    key = f"{conv_id}:{turn}"
    if key in EXPECT_CASE:
        return EXPECT_CASE[key]
    return LEGACY_TO_CASE.get(legacy_label, {legacy_label})
