"""백엔드 테스트. LLM 호출 없이 도는 부분만.

CLI 경로에는 서버측 스키마 강제가 없어서 JSON 추출과 검증이 직접 방어선이 된다.
"""

import pytest
from pydantic import BaseModel, ValidationError

from ragdiag.backends import Usage, extract_json
from ragdiag.prompts import output_contract
from ragdiag.schema import GroundingCheck, NeedAnalysis, SufficiencyJudgment


def test_extracts_plain_json():
    assert extract_json('{"a": 1}') == '{"a": 1}'


def test_strips_markdown_fence():
    # CLI는 시스템 프롬프트로 막아도 코드펜스를 붙이는 경우가 있다.
    assert extract_json('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_ignores_prose_before_and_after():
    assert extract_json('결과입니다:\n{"a": 1}\n이상입니다.') == '{"a": 1}'


def test_braces_inside_strings_do_not_break_parsing():
    # 인용문에 중괄호가 들어올 수 있다. rfind('}')로는 못 잡는다.
    raw = '{"quote": "규정 {제3조} 참고", "n": 1} 뒤에 붙은 산문 }'
    assert extract_json(raw) == '{"quote": "규정 {제3조} 참고", "n": 1}'


def test_nested_objects_are_kept_whole():
    assert extract_json('{"a": {"b": {"c": 1}}}') == '{"a": {"b": {"c": 1}}}'


def test_escaped_quote_inside_string():
    raw = r'{"quote": "그는 \"맞다\"고 했다"}'
    assert extract_json(raw) == raw


@pytest.mark.parametrize("bad", ["없음", "[1,2,3]", '{"a": 1'])
def test_malformed_input_raises(bad):
    with pytest.raises(ValueError):
        extract_json(bad)


@pytest.mark.parametrize("model", [NeedAnalysis, SufficiencyJudgment, GroundingCheck])
def test_contract_lists_every_field_in_declaration_order(model):
    # 필드 순서에 설계가 담겨 있다. reasoning이 먼저여야 결론이 근거의 결과가 된다.
    contract = output_contract(model)
    positions = [contract.index(name) for name in model.model_fields]
    assert positions == sorted(positions)
    assert all(name in contract for name in model.model_fields)


def test_contract_spells_out_enum_values():
    contract = output_contract(NeedAnalysis)
    for value in ["content_gap", "wrong_content", "format_or_style", "other"]:
        assert f'"{value}"' in contract


def test_contract_expands_nested_array_items():
    # evidence가 배열이라는 것만으론 부족하다. 원소 구조까지 알려줘야 한다.
    contract = output_contract(SufficiencyJudgment)
    assert "chunk_index" in contract and "quote" in contract


def test_contract_forbids_extra_text():
    assert "JSON 객체 하나만" in output_contract(GroundingCheck)


def test_schema_validation_rejects_bad_enum():
    with pytest.raises(ValidationError):
        NeedAnalysis.model_validate_json(
            '{"reasoning":"r","resolved_question":"q","unmet_need":"n",'
            '"complaint_type":"엉뚱한값","context_dependent":false}'
        )


def test_usage_accumulates():
    total = Usage()
    total.add(Usage(100, 20, 0.05))
    total.add(Usage(200, 30, 0.05))
    assert (total.input_tokens, total.output_tokens) == (300, 50)
    assert total.cost_usd == pytest.approx(0.10)


def test_strip_reasoning_takes_text_after_the_last_close_tag():
    from ragdiag.backends import strip_reasoning

    assert strip_reasoning("<think>생각</think>답") == "답"
    assert strip_reasoning("<thinking>생각</thinking>답") == "답"
    assert strip_reasoning("여는 태그 없이 생각만</think>답") == "답"
    assert strip_reasoning("추론 없음") == "추론 없음"


def test_strip_reasoning_uses_the_last_tag_not_the_first():
    from ragdiag.backends import strip_reasoning

    # 반복되거나 중첩된 블록에서도 마지막 뒤를 취해야 한다.
    assert strip_reasoning("<think>a</think>중간<think>b</think>진짜답") == "진짜답"
