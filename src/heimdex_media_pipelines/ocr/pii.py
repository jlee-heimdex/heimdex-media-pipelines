from __future__ import annotations

import re

KR_RRN = re.compile(r"\b\d{6}[-–]\d{7}\b")
CREDIT_CARD = re.compile(r"\b(?:4\d{12}(?:\d{3})?|5[1-5]\d{14}|3[47]\d{13}|6(?:011|5\d{2})\d{12})\b")
KR_PHONE = re.compile(
    r"(?<!\d)"
    # Country code "+82" (optionally followed by ")" / "-" / "." / space and a
    # dropped leading 0) OR a domestic leading "0".
    r"(?:\+82[\s)\-–.]*0?|0)"
    # Area / mobile prefix: 2 (Seoul), mobile 1[016789], or 3X-6X landlines.
    r"(?:2|1[016789]|[3-6]\d)"
    # Separator: dash (incl. en-dash), dot, space — or none.
    r"[-–.\s]?"
    r"\d{3,4}"
    r"[-–.\s]?"
    r"\d{4}"
    r"(?!\d)"
)
EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

# Korean vehicle license plates.
#   - Old format: 2 digits + 1 Hangul + 4 digits      (e.g. "12가3456")
#   - New format: 3 digits + 1 Hangul + 4 digits      (e.g. "123가4567")
#   - Optional regional prefix (Hangul, 2 chars)      (e.g. "서울12가3456")
# OCR often inserts stray spaces between the groups, so allow optional
# whitespace between every element. The middle Hangul character is the
# "용도기호" (usage code), always a single syllable from the set defined
# by the Ministry of Land, but we accept any Hangul syllable to stay robust
# against OCR confusions.
KR_LICENSE_PLATE = re.compile(
    r"(?:[\uac00-\ud7a3]{2}\s*)?"   # optional region (e.g. 서울, 경기)
    r"\d{2,3}\s*"                    # 2 or 3 digit head
    r"[\uac00-\ud7a3]\s*"            # usage-code Hangul syllable
    r"\d{4}"                         # 4 digit tail
)

PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("kr_rrn", KR_RRN),
    ("credit_card", CREDIT_CARD),
    ("kr_phone", KR_PHONE),
    ("email", EMAIL),
    ("kr_license_plate", KR_LICENSE_PLATE),
]


def detect_pii(text: str) -> list[tuple[str, str, int, int]]:
    findings: list[tuple[str, str, int, int]] = []
    for category, pattern in PII_PATTERNS:
        for match in pattern.finditer(text):
            findings.append((category, match.group(0), match.start(), match.end()))
    findings.sort(key=lambda x: x[2])
    return findings


def redact_pii(text: str) -> str:
    redacted = text
    replacements = {
        "kr_rrn": "[KR_RRN]",
        "credit_card": "[CREDIT_CARD]",
        "kr_phone": "[KR_PHONE]",
        "email": "[EMAIL]",
        "kr_license_plate": "[KR_LICENSE_PLATE]",
    }
    for category, pattern in PII_PATTERNS:
        redacted = pattern.sub(replacements[category], redacted)
    return redacted
