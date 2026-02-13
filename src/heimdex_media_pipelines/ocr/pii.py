from __future__ import annotations

import re

KR_RRN = re.compile(r"\b\d{6}[-–]\d{7}\b")
CREDIT_CARD = re.compile(r"\b(?:4\d{12}(?:\d{3})?|5[1-5]\d{14}|3[47]\d{13}|6(?:011|5\d{2})\d{12})\b")
KR_PHONE = re.compile(r"\b01[016789][-–.\s]?\d{3,4}[-–.\s]?\d{4}\b")
EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("kr_rrn", KR_RRN),
    ("credit_card", CREDIT_CARD),
    ("kr_phone", KR_PHONE),
    ("email", EMAIL),
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
    }
    for category, pattern in PII_PATTERNS:
        redacted = pattern.sub(replacements[category], redacted)
    return redacted
