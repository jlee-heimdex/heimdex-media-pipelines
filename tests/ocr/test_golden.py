"""Golden dataset tests for the OCR pipeline.

These tests require:
- paddleocr + paddlepaddle installed (optional OCR deps)
- Test media clips at ~/Projects/heimdex/heimdex-test-media/ocr/
- Not running in a minimal CI image (needs full image support)

Run with:
    pytest -m ocr_golden tests/ocr/test_golden.py -v

All clips are shared internally and must NOT be committed to version control.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------

_MEDIA_DIR = Path.home() / "Projects" / "heimdex" / "heimdex-test-media" / "ocr"

_paddleocr_available = True
try:
    import importlib

    importlib.import_module("paddleocr")
except ImportError:
    _paddleocr_available = False

_SKIP_NO_PADDLEOCR = pytest.mark.skipif(
    not _paddleocr_available,
    reason="paddleocr is not installed (install with: pip install paddleocr>=2.8.0 paddlepaddle>=2.6.1)",
)

_SKIP_NO_MEDIA = pytest.mark.skipif(
    not _MEDIA_DIR.is_dir(),
    reason=f"OCR test media directory not found: {_MEDIA_DIR}",
)

_SKIP_MINIMAL_CI = pytest.mark.skipif(
    os.environ.get("CI_MINIMAL", "").lower() in ("1", "true", "yes"),
    reason="Skipping OCR golden tests on minimal CI image",
)

# Marks applied to media-dependent test classes (not module-level,
# because TestSanitizationVerification and TestLevenshteinInline don't
# need paddleocr or media).
_GOLDEN_MARKS = [
    pytest.mark.ocr_golden,
    _SKIP_NO_PADDLEOCR,
    _SKIP_NO_MEDIA,
    _SKIP_MINIMAL_CI,
]


# ---------------------------------------------------------------------------
# Inline Levenshtein distance (pure Python — no external deps)
# ---------------------------------------------------------------------------


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute the Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Insertions, deletions, substitutions
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (0 if c1 == c2 else 1)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


def _fuzzy_similarity(s1: str, s2: str) -> float:
    """Return similarity ratio between 0.0 and 1.0."""
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    return 1.0 - (_levenshtein_distance(s1, s2) / max_len)


def _fuzzy_contains(haystack: str, needle: str, threshold: float = 0.7) -> bool:
    """Check if needle appears in haystack with fuzzy matching.

    Slides a window of len(needle) ± 20% across the haystack and checks
    if any window has similarity >= threshold.
    """
    if not needle:
        return True
    if not haystack:
        return False

    # Exact substring first (fast path)
    if needle.lower() in haystack.lower():
        return True

    needle_lower = needle.lower()
    haystack_lower = haystack.lower()
    needle_len = len(needle_lower)

    # Window sizes: needle_len ± 20%
    min_window = max(1, int(needle_len * 0.8))
    max_window = int(needle_len * 1.2) + 1

    for window_size in range(min_window, min(max_window, len(haystack_lower) + 1)):
        for start in range(len(haystack_lower) - window_size + 1):
            window = haystack_lower[start : start + window_size]
            if _fuzzy_similarity(window, needle_lower) >= threshold:
                return True

    return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clip_path(name: str) -> Path:
    """Return path to a test clip, skipping if it doesn't exist."""
    path = _MEDIA_DIR / name
    if not path.exists():
        pytest.skip(f"Test clip not found: {path}")
    return path


def _run_pipeline_on_clip(
    clip_name: str,
    tmp_path: Path,
    *,
    lang: str = "korean",
    redact_pii: bool = False,
) -> dict[str, Any]:
    """Run the OCR pipeline on a single clip and return the result dict.

    Creates a minimal scenes.json for the clip and runs run_ocr_pipeline().
    """
    from heimdex_media_pipelines.ocr.pipeline import run_ocr_pipeline

    clip = _clip_path(clip_name)

    # Create keyframe dir with clip as keyframe
    keyframe_dir = tmp_path / "frames"
    keyframe_dir.mkdir(exist_ok=True)

    # Link or copy the clip as a keyframe image
    scene_id = "test_scene_0"
    keyframe_path = keyframe_dir / f"{scene_id}.jpg"
    if not keyframe_path.exists():
        # If clip is already a jpg/png image, symlink it
        os.symlink(clip, keyframe_path)

    # Create a minimal scenes result file
    scenes_path = tmp_path / "scenes.json"
    scenes_data = {
        "video_id": f"golden_{clip_name}",
        "scenes": [
            {
                "scene_id": scene_id,
                "keyframe_timestamp_ms": 0,
            }
        ],
    }
    scenes_path.write_text(json.dumps(scenes_data), encoding="utf-8")

    out_path = tmp_path / "ocr_result.json"

    result = run_ocr_pipeline(
        scenes_result_path=str(scenes_path),
        keyframe_dir=str(keyframe_dir),
        out_path=str(out_path),
        lang=lang,
        redact_pii_flag=redact_pii,
    )

    return result.model_dump()


# ---------------------------------------------------------------------------
# Golden Tests
# ---------------------------------------------------------------------------


@pytest.mark.ocr_golden
@_SKIP_NO_PADDLEOCR
@_SKIP_NO_MEDIA
@_SKIP_MINIMAL_CI
class TestPriceOverlay:
    """Test OCR on a price overlay image (e.g. ₩39,900 PRODUCT X)."""

    CLIP_NAME = "price_overlay.jpg"
    EXPECTED_SUBSTRINGS = ["₩", "39,900"]

    def test_detects_price_text(self, tmp_path):
        result = _run_pipeline_on_clip(self.CLIP_NAME, tmp_path)
        scenes = result["scenes"]
        assert len(scenes) == 1

        ocr_text = scenes[0].get("ocr_text_raw", "")
        for expected in self.EXPECTED_SUBSTRINGS:
            assert _fuzzy_contains(ocr_text, expected, threshold=0.6), (
                f"Expected fuzzy match for '{expected}' in OCR text: '{ocr_text}'"
            )

    def test_char_count_nonzero(self, tmp_path):
        result = _run_pipeline_on_clip(self.CLIP_NAME, tmp_path)
        scenes = result["scenes"]
        # At least some characters detected
        ocr_text = scenes[0].get("ocr_text_raw", "")
        assert len(ocr_text) > 0, "Expected non-empty OCR text for price overlay"

    def test_performance_under_limit(self, tmp_path):
        t0 = time.perf_counter()
        result = _run_pipeline_on_clip(self.CLIP_NAME, tmp_path)
        elapsed = time.perf_counter() - t0

        assert elapsed < 10.0, f"OCR pipeline took {elapsed:.1f}s (limit: 10s)"

        perf = result.get("meta", {}).get("perf", {})
        if perf:
            assert perf.get("frames_processed", 0) >= 1


@pytest.mark.ocr_golden
@_SKIP_NO_PADDLEOCR
@_SKIP_NO_MEDIA
@_SKIP_MINIMAL_CI
class TestBilingualOverlay:
    """Test OCR on a bilingual (Korean + English) overlay image."""

    CLIP_NAME = "bilingual_overlay.jpg"
    EXPECTED_KOREAN = "상품"
    EXPECTED_ENGLISH = "PRODUCT"

    def test_detects_korean_text(self, tmp_path):
        result = _run_pipeline_on_clip(self.CLIP_NAME, tmp_path)
        ocr_text = result["scenes"][0].get("ocr_text_raw", "")
        assert _fuzzy_contains(ocr_text, self.EXPECTED_KOREAN, threshold=0.6), (
            f"Expected Korean text '{self.EXPECTED_KOREAN}' in OCR: '{ocr_text}'"
        )

    def test_detects_english_text(self, tmp_path):
        result = _run_pipeline_on_clip(self.CLIP_NAME, tmp_path)
        ocr_text = result["scenes"][0].get("ocr_text_raw", "")
        assert _fuzzy_contains(ocr_text, self.EXPECTED_ENGLISH, threshold=0.6), (
            f"Expected English text '{self.EXPECTED_ENGLISH}' in OCR: '{ocr_text}'"
        )


@pytest.mark.ocr_golden
@_SKIP_NO_PADDLEOCR
@_SKIP_NO_MEDIA
@_SKIP_MINIMAL_CI
class TestNoOverlay:
    """Test OCR on an image with no text overlay — should return empty."""

    CLIP_NAME = "no_overlay.jpg"

    def test_returns_empty_or_minimal_text(self, tmp_path):
        result = _run_pipeline_on_clip(self.CLIP_NAME, tmp_path)
        ocr_text = result["scenes"][0].get("ocr_text_raw", "")
        # No overlay should produce empty or very short text (noise)
        assert len(ocr_text) < 10, (
            f"Expected empty/minimal OCR for no-overlay image, got {len(ocr_text)} chars: '{ocr_text}'"
        )

    def test_performance_under_limit(self, tmp_path):
        t0 = time.perf_counter()
        _run_pipeline_on_clip(self.CLIP_NAME, tmp_path)
        elapsed = time.perf_counter() - t0
        assert elapsed < 10.0, f"OCR pipeline took {elapsed:.1f}s (limit: 10s)"


@pytest.mark.ocr_golden
@_SKIP_NO_PADDLEOCR
@_SKIP_NO_MEDIA
@_SKIP_MINIMAL_CI
class TestAdversarialOverlay:
    """Test OCR sanitization on adversarial text overlays.

    Verifies that the pipeline correctly sanitizes:
    - HTML/script injection
    - Bidi override characters
    - Raw HTML entities
    """

    CLIP_NAME = "adversarial_overlay.jpg"

    def test_no_raw_script_tags(self, tmp_path):
        result = _run_pipeline_on_clip(self.CLIP_NAME, tmp_path)
        ocr_text = result["scenes"][0].get("ocr_text_raw", "")
        assert "<script>" not in ocr_text.lower(), (
            f"Raw <script> tag found in sanitized OCR output: '{ocr_text}'"
        )
        assert "</script>" not in ocr_text.lower(), (
            f"Raw </script> tag found in sanitized OCR output: '{ocr_text}'"
        )

    def test_no_bidi_override_characters(self, tmp_path):
        result = _run_pipeline_on_clip(self.CLIP_NAME, tmp_path)
        ocr_text = result["scenes"][0].get("ocr_text_raw", "")
        bidi_chars = set("\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069")
        found = [c for c in ocr_text if c in bidi_chars]
        assert len(found) == 0, (
            f"Bidi override characters found in OCR output: {[hex(ord(c)) for c in found]}"
        )

    def test_no_raw_html_tags(self, tmp_path):
        import re

        result = _run_pipeline_on_clip(self.CLIP_NAME, tmp_path)
        ocr_text = result["scenes"][0].get("ocr_text_raw", "")
        html_tags = re.findall(r"<[a-zA-Z][^>]*>", ocr_text)
        assert len(html_tags) == 0, (
            f"Raw HTML tags found in sanitized OCR output: {html_tags}"
        )


# ---------------------------------------------------------------------------
# Sanitization unit tests (don't need media)
# ---------------------------------------------------------------------------

# These run unconditionally (no media/paddleocr needed) since they test
# the sanitize_ocr_text function directly.


@pytest.mark.ocr_golden
class TestSanitizationVerification:
    """Verify sanitization properties of the OCR pipeline text processing."""

    def test_script_injection_sanitized(self):
        from heimdex_media_pipelines.ocr.pipeline import sanitize_ocr_text

        text = '<script>alert("xss")</script>₩39,900'
        result = sanitize_ocr_text(text)
        assert "<script>" not in result
        assert "alert" in result  # text content preserved
        assert "₩39,900" in result or "39,900" in result

    def test_bidi_chars_stripped(self):
        from heimdex_media_pipelines.ocr.pipeline import sanitize_ocr_text

        text = "\u202eEvil\u2066 text\u2069"
        result = sanitize_ocr_text(text)
        assert "\u202e" not in result
        assert "\u2066" not in result
        assert "\u2069" not in result
        assert "Evil" in result

    def test_html_entities_escaped(self):
        from heimdex_media_pipelines.ocr.pipeline import sanitize_ocr_text

        text = "Price A & B"
        result = sanitize_ocr_text(text)
        assert "&amp;" in result
        assert "Price A" in result

    def test_nested_html_stripped(self):
        from heimdex_media_pipelines.ocr.pipeline import sanitize_ocr_text

        text = '<div onclick="hack()">₩39,900</div>'
        result = sanitize_ocr_text(text)
        assert "<div" not in result
        assert "onclick" not in result
        assert "39,900" in result


# ---------------------------------------------------------------------------
# Levenshtein self-tests (validate our inline implementation)
# ---------------------------------------------------------------------------


@pytest.mark.ocr_golden
class TestLevenshteinInline:
    """Validate the inline Levenshtein implementation."""

    def test_identical_strings(self):
        assert _levenshtein_distance("hello", "hello") == 0

    def test_empty_strings(self):
        assert _levenshtein_distance("", "") == 0

    def test_one_empty(self):
        assert _levenshtein_distance("abc", "") == 3
        assert _levenshtein_distance("", "abc") == 3

    def test_single_substitution(self):
        assert _levenshtein_distance("cat", "bat") == 1

    def test_single_insertion(self):
        assert _levenshtein_distance("cat", "cats") == 1

    def test_single_deletion(self):
        assert _levenshtein_distance("cats", "cat") == 1

    def test_complex(self):
        assert _levenshtein_distance("kitten", "sitting") == 3

    def test_similarity_identical(self):
        assert _fuzzy_similarity("hello", "hello") == 1.0

    def test_similarity_completely_different(self):
        sim = _fuzzy_similarity("abc", "xyz")
        assert sim < 0.5

    def test_fuzzy_contains_exact(self):
        assert _fuzzy_contains("hello world", "world") is True

    def test_fuzzy_contains_missing(self):
        assert _fuzzy_contains("hello world", "zzzzz") is False

    def test_fuzzy_contains_near_match(self):
        # "wrold" is close to "world" (1 transposition)
        assert _fuzzy_contains("hello wrold", "world", threshold=0.6) is True
