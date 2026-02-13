from heimdex_media_pipelines.ocr.pii import detect_pii, redact_pii


def test_detect_kr_rrn_pattern():
    out = detect_pii("주민등록번호 900101-1234567")
    assert any(x[0] == "kr_rrn" for x in out)


def test_detect_credit_card_pattern():
    out = detect_pii("결제카드 4111111111111111")
    assert any(x[0] == "credit_card" for x in out)


def test_detect_kr_phone_pattern_with_dashes():
    out = detect_pii("연락처 010-1234-5678")
    assert any(x[0] == "kr_phone" for x in out)


def test_detect_kr_phone_pattern_with_dots():
    out = detect_pii("연락처 010.1234.5678")
    assert any(x[0] == "kr_phone" for x in out)


def test_detect_kr_phone_pattern_with_spaces():
    out = detect_pii("연락처 010 1234 5678")
    assert any(x[0] == "kr_phone" for x in out)


def test_detect_email_pattern():
    out = detect_pii("메일 test.user+tag@example.co.kr")
    assert any(x[0] == "email" for x in out)


def test_detect_multiple_pii_in_one_string():
    out = detect_pii("010-1234-5678 / 900101-1234567 / a@b.com")
    kinds = {x[0] for x in out}
    assert {"kr_phone", "kr_rrn", "email"}.issubset(kinds)


def test_detect_pii_returns_empty_for_clean_text():
    assert detect_pii("안녕하세요 상품 소개입니다") == []


def test_redact_pii_replaces_kr_phone():
    assert redact_pii("010-1234-5678") == "[KR_PHONE]"


def test_redact_pii_replaces_kr_rrn():
    assert redact_pii("900101-1234567") == "[KR_RRN]"


def test_redact_pii_replaces_credit_card():
    assert redact_pii("4111111111111111") == "[CREDIT_CARD]"


def test_redact_pii_replaces_email():
    assert redact_pii("person@example.com") == "[EMAIL]"


def test_redact_pii_multiple_categories():
    text = "010-1234-5678 / person@example.com / 900101-1234567"
    out = redact_pii(text)
    assert "[KR_PHONE]" in out
    assert "[EMAIL]" in out
    assert "[KR_RRN]" in out


def test_redact_pii_clean_text_unchanged():
    text = "개인정보 없는 텍스트"
    assert redact_pii(text) == text
