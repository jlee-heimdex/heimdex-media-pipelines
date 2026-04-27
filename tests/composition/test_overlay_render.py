"""Tests for bake_overlay_png — PIL overlay renderer.

Golden fixtures live in tests/composition/fixtures/bakes/*.png.
On first run (or with --regen-goldens), they are generated from the current
implementation and committed as truth. Subsequent runs compare against them.

Regenerate goldens:
    pytest tests/composition/test_overlay_render.py --regen-goldens -v

SSIM tolerance: 0.99 (lowered to 0.98 if PIL version sensitivity is detected).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# Optional SSIM import — fall back to byte-equality if scikit-image missing
# ---------------------------------------------------------------------------

try:
    from skimage.metrics import structural_similarity as _ssim_fn
    import numpy as _np

    def _images_match(a: Image.Image, b: Image.Image, threshold: float = 0.99) -> tuple[bool, float]:
        arr_a = _np.array(a.convert("RGBA"))
        arr_b = _np.array(b.convert("RGBA"))
        if arr_a.shape != arr_b.shape:
            return False, 0.0
        score = _ssim_fn(arr_a, arr_b, channel_axis=-1, data_range=255)
        return score >= threshold, score

    _COMPARE_MODE = "ssim"

except ImportError:
    def _images_match(a: Image.Image, b: Image.Image, threshold: float = 0.99) -> tuple[bool, float]:  # type: ignore[misc]
        if a.size != b.size:
            return False, 0.0
        return a.tobytes() == b.tobytes(), 1.0 if a.tobytes() == b.tobytes() else 0.0

    _COMPARE_MODE = "byte-equal"


# ---------------------------------------------------------------------------
# Fixtures directory + conftest hook
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "bakes"
FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

# Minimal stub font directory — tests create/use a real font if available,
# otherwise fall back to Pillow's built-in default font via a temp dir trick.
_FONT_DIR = os.environ.get("HEIMDEX_TEST_FONT_DIR", str(FIXTURES_DIR / "fonts"))


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "regen_goldens: regenerate golden PNG fixtures")


def _golden_path(name: str) -> Path:
    return FIXTURES_DIR / f"{name}.png"


def _assert_golden(img: Image.Image, name: str, threshold: float = 0.99, regen: bool = False) -> None:
    """Compare img against golden PNG, or save it if regen=True or golden missing."""
    path = _golden_path(name)
    if regen or not path.exists():
        img.save(str(path))
        pytest.skip(f"Golden '{name}.png' {'regenerated' if path.exists() else 'created'}. Re-run without --regen-goldens.")
    golden = Image.open(str(path))
    matched, score = _images_match(img, golden, threshold)
    assert matched, (
        f"Golden mismatch for '{name}' (mode={_COMPARE_MODE}, score={score:.4f}, "
        f"threshold={threshold}). Re-run with --regen-goldens to update."
    )


# ---------------------------------------------------------------------------
# Pytest hook to support --regen-goldens flag
# ---------------------------------------------------------------------------

def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--regen-goldens",
        action="store_true",
        default=False,
        help="Regenerate golden PNG fixtures from current implementation.",
    )


@pytest.fixture
def regen(request: pytest.FixtureRequest) -> bool:
    return request.config.getoption("--regen-goldens", default=False)


# ---------------------------------------------------------------------------
# Font setup — create a stub fonts dir with minimal fonts for CI
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def font_dir(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Return a fonts directory.

    Resolution order:
      1. ``HEIMDEX_TEST_FONT_DIR`` env var (CI / Docker).
      2. Sibling worker repo (real Pretendard + NotoSansKR — preferred for
         golden fidelity since this is what production renders against).
      3. Fall back to a tempdir of system fonts copied + renamed
         (CI on Linux without the worker repo checked out).

    Uses ``shutil.copyfile`` (not ``copy2``) — ``copy2`` triggers ``os.chflags``
    on macOS which fails with EPERM on ``tmp_path`` destinations.
    """
    env_dir = os.environ.get("HEIMDEX_TEST_FONT_DIR")
    if env_dir and os.path.isdir(env_dir):
        return env_dir

    # Prefer sibling worker repo's bundled Pretendard/Noto fonts — same files
    # the render worker uses in production, so goldens reflect real output.
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.normpath(os.path.join(here, "..", "..", ".."))
    sibling_fonts = os.path.normpath(
        os.path.join(
            repo_root,
            "dev-heimdex-for-livecommerce",
            "services",
            "shorts-render-worker",
            "fonts",
        )
    )
    if os.path.isdir(sibling_fonts) and os.path.exists(
        os.path.join(sibling_fonts, "Pretendard-Regular.ttf")
    ):
        return sibling_fonts

    # CI fallback: copy a system font under the expected names.
    candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    found_font = next((p for p in candidates if os.path.exists(p)), None)

    tmp_dir = tmp_path_factory.mktemp("fonts")
    if found_font is None:
        return str(tmp_dir)

    import shutil
    for name in (
        "Pretendard-Regular.ttf",
        "Pretendard-Bold.ttf",
        "NotoSansKR-Regular.ttf",
        "NotoSansKR-Bold.ttf",
    ):
        shutil.copyfile(found_font, str(tmp_dir / name))

    return str(tmp_dir)


# ---------------------------------------------------------------------------
# Spec factories
# ---------------------------------------------------------------------------

def _text_spec(**kwargs) -> dict:
    base = {
        "kind": "text",
        "id": "t1",
        "start_ms": 0,
        "end_ms": 1000,
        "text": "Hello",
        "font_family": "Pretendard",
        "font_size_px": 36,
        "font_weight": 400,
        "italic": False,
        "underline": False,
        "font_color": "#FFFFFF",
        "text_align": "center",
        "line_height": 1.3,
        "letter_spacing": 0.0,
        "transform": {"x": 0.5, "y": 0.5, "rotation_deg": 0.0},
        "effects": {"opacity": 1.0},
    }
    base.update(kwargs)
    return base


def _bg_spec(**kwargs) -> dict:
    base = {
        "kind": "background",
        "id": "bg1",
        "start_ms": 0,
        "end_ms": 1000,
        "fill_color": "#000000",
        "transform": {"x": 0.5, "y": 0.5, "rotation_deg": 0.0, "width_px": 200, "height_px": 100},
        "effects": {"opacity": 1.0},
    }
    base.update(kwargs)
    return base


def _parse_overlay(data: dict):
    """Parse overlay spec dict using Pydantic model."""
    from heimdex_media_contracts.composition.overlays import TextOverlaySpec, BackgroundOverlaySpec
    if data["kind"] == "text":
        return TextOverlaySpec.model_validate(data)
    return BackgroundOverlaySpec.model_validate(data)


# ---------------------------------------------------------------------------
# Import under test
# ---------------------------------------------------------------------------

from heimdex_media_pipelines.composition.overlay_render import bake_overlay_png  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bake(spec_dict: dict, font_dir: str) -> Image.Image:
    ov = _parse_overlay(spec_dict)
    img = bake_overlay_png(ov, canvas_width=1080, canvas_height=1920, font_dir=font_dir)
    assert img.mode == "RGBA"
    assert img.width > 0
    assert img.height > 0
    return img


# ---------------------------------------------------------------------------
# Tests: text_plain
# ---------------------------------------------------------------------------

class TestTextPlain:
    def test_returns_rgba(self, font_dir: str) -> None:
        img = _bake(_text_spec(), font_dir)
        assert img.mode == "RGBA"

    def test_has_non_transparent_pixels(self, font_dir: str) -> None:
        img = _bake(_text_spec(), font_dir)
        pixels = list(img.getdata())
        opaque = [p for p in pixels if p[3] > 0]
        assert len(opaque) > 0, "Expected some opaque pixels for text"

    def test_golden(self, font_dir: str, regen: bool) -> None:
        img = _bake(_text_spec(), font_dir)
        _assert_golden(img, "text_plain", regen=regen)


# ---------------------------------------------------------------------------
# Tests: text_italic
# ---------------------------------------------------------------------------

class TestTextItalic:
    def test_golden(self, font_dir: str, regen: bool) -> None:
        img = _bake(_text_spec(italic=True), font_dir)
        assert img.mode == "RGBA"
        _assert_golden(img, "text_italic", regen=regen)


# ---------------------------------------------------------------------------
# Tests: text_underline
# ---------------------------------------------------------------------------

class TestTextUnderline:
    def test_golden(self, font_dir: str, regen: bool) -> None:
        img = _bake(_text_spec(underline=True), font_dir)
        _assert_golden(img, "text_underline", regen=regen)


# ---------------------------------------------------------------------------
# Tests: text_italic_underline
# ---------------------------------------------------------------------------

class TestTextItalicUnderline:
    def test_golden(self, font_dir: str, regen: bool) -> None:
        img = _bake(_text_spec(italic=True, underline=True), font_dir)
        _assert_golden(img, "text_italic_underline", regen=regen)


# ---------------------------------------------------------------------------
# Tests: text_stroked
# ---------------------------------------------------------------------------

class TestTextStroked:
    def test_golden(self, font_dir: str, regen: bool) -> None:
        spec = _text_spec(
            effects={
                "opacity": 1.0,
                "stroke": {"color": "#000000", "width_px": 2},
            }
        )
        img = _bake(spec, font_dir)
        _assert_golden(img, "text_stroked", regen=regen)


# ---------------------------------------------------------------------------
# Tests: text_shadowed_hard
# ---------------------------------------------------------------------------

class TestTextShadowedHard:
    def test_golden(self, font_dir: str, regen: bool) -> None:
        spec = _text_spec(
            effects={
                "opacity": 1.0,
                "shadow": {"color": "#000000", "offset_x": 4, "offset_y": 4, "blur_px": 0, "spread_px": 0},
            }
        )
        img = _bake(spec, font_dir)
        _assert_golden(img, "text_shadowed_hard", regen=regen)


# ---------------------------------------------------------------------------
# Tests: text_shadowed_soft
# ---------------------------------------------------------------------------

class TestTextShadowedSoft:
    def test_golden(self, font_dir: str, regen: bool) -> None:
        spec = _text_spec(
            effects={
                "opacity": 1.0,
                "shadow": {"color": "#000000", "offset_x": 0, "offset_y": 0, "blur_px": 8, "spread_px": 2},
            }
        )
        img = _bake(spec, font_dir)
        _assert_golden(img, "text_shadowed_soft", regen=regen)


# ---------------------------------------------------------------------------
# Tests: text_rotated
# ---------------------------------------------------------------------------

class TestTextRotated:
    def test_golden(self, font_dir: str, regen: bool) -> None:
        spec = _text_spec(
            transform={"x": 0.5, "y": 0.5, "rotation_deg": 30.0}
        )
        img = _bake(spec, font_dir)
        _assert_golden(img, "text_rotated", regen=regen)

    def test_rotation_expands_canvas(self, font_dir: str) -> None:
        plain = _bake(_text_spec(), font_dir)
        rotated = _bake(_text_spec(transform={"x": 0.5, "y": 0.5, "rotation_deg": 45.0}), font_dir)
        # 45° rotation of a non-square image expands dimensions
        assert rotated.width != plain.width or rotated.height != plain.height


# ---------------------------------------------------------------------------
# Tests: text_full
# ---------------------------------------------------------------------------

class TestTextFull:
    def test_golden(self, font_dir: str, regen: bool) -> None:
        spec = _text_spec(
            italic=True,
            underline=True,
            letter_spacing=2.0,
            effects={
                "opacity": 0.5,
                "stroke": {"color": "#000000", "width_px": 1},
                "shadow": {"color": "#333333", "offset_x": 3, "offset_y": 3, "blur_px": 6, "spread_px": 1},
            },
            transform={"x": 0.5, "y": 0.5, "rotation_deg": 15.0},
        )
        img = _bake(spec, font_dir)
        _assert_golden(img, "text_full", regen=regen)


# ---------------------------------------------------------------------------
# Tests: text_korean
# ---------------------------------------------------------------------------

class TestTextKorean:
    def test_golden(self, font_dir: str, regen: bool) -> None:
        spec = _text_spec(
            text="라이브 특가",
            font_family="Noto Sans KR",
            italic=True,
        )
        img = _bake(spec, font_dir)
        _assert_golden(img, "text_korean", regen=regen)


# ---------------------------------------------------------------------------
# Tests: text_with_highlight
# ---------------------------------------------------------------------------

class TestTextWithHighlight:
    def test_golden(self, font_dir: str, regen: bool) -> None:
        spec = _text_spec(
            highlight_color="#FF0000",
            highlight_padding_px=8,
            highlight_opacity=0.8,
        )
        img = _bake(spec, font_dir)
        _assert_golden(img, "text_with_highlight", regen=regen)

    def test_highlight_pixels_present(self, font_dir: str) -> None:
        """Highlight box must produce non-transparent pixels at padding region."""
        spec = _text_spec(
            highlight_color="#FF0000",
            highlight_padding_px=8,
            highlight_opacity=1.0,
        )
        img = _bake(spec, font_dir)
        pixels = list(img.getdata())
        red_pixels = [p for p in pixels if p[0] > 200 and p[3] > 0]
        assert len(red_pixels) > 0, "Expected red highlight pixels"


# ---------------------------------------------------------------------------
# Tests: bg_plain
# ---------------------------------------------------------------------------

class TestBgPlain:
    def test_dimensions(self, font_dir: str) -> None:
        img = _bake(_bg_spec(), font_dir)
        # Image has extra padding; content area should be at least 200x100
        assert img.width >= 200
        assert img.height >= 100

    def test_golden(self, font_dir: str, regen: bool) -> None:
        img = _bake(_bg_spec(), font_dir)
        _assert_golden(img, "bg_plain", regen=regen)


# ---------------------------------------------------------------------------
# Tests: bg_rotated_shadow
# ---------------------------------------------------------------------------

class TestBgRotatedShadow:
    def test_golden(self, font_dir: str, regen: bool) -> None:
        spec = _bg_spec(
            transform={
                "x": 0.5, "y": 0.5, "rotation_deg": 20.0,
                "width_px": 200, "height_px": 100,
            },
            effects={
                "opacity": 1.0,
                "shadow": {"color": "#000000", "offset_x": 5, "offset_y": 5, "blur_px": 10, "spread_px": 3},
            },
        )
        img = _bake(spec, font_dir)
        _assert_golden(img, "bg_rotated_shadow", regen=regen)


# ---------------------------------------------------------------------------
# Tests: opacity application
# ---------------------------------------------------------------------------

class TestOpacity:
    def test_half_opacity_reduces_alpha(self, font_dir: str) -> None:
        full = _bake(_text_spec(effects={"opacity": 1.0}), font_dir)
        half = _bake(_text_spec(effects={"opacity": 0.5}), font_dir)
        # Max alpha of half-opacity image should be ~50% of full
        full_max_alpha = max(p[3] for p in full.getdata())
        half_max_alpha = max(p[3] for p in half.getdata())
        assert half_max_alpha < full_max_alpha * 0.7, (
            f"Expected half-opacity to significantly reduce max alpha "
            f"(full={full_max_alpha}, half={half_max_alpha})"
        )

    def test_zero_opacity_fully_transparent(self, font_dir: str) -> None:
        img = _bake(_text_spec(effects={"opacity": 0.0}), font_dir)
        all_transparent = all(p[3] == 0 for p in img.getdata())
        assert all_transparent, "opacity=0.0 should produce fully transparent image"


# ---------------------------------------------------------------------------
# Tests: letter_spacing
# ---------------------------------------------------------------------------

class TestLetterSpacing:
    def test_wider_with_positive_spacing(self, font_dir: str) -> None:
        normal = _bake(_text_spec(letter_spacing=0.0), font_dir)
        spaced = _bake(_text_spec(letter_spacing=8.0), font_dir)
        assert spaced.width >= normal.width, "Positive letter_spacing should widen text"


# ---------------------------------------------------------------------------
# Tests: multiline
# ---------------------------------------------------------------------------

class TestMultiline:
    def test_taller_than_single_line(self, font_dir: str) -> None:
        single = _bake(_text_spec(text="Hello"), font_dir)
        multi = _bake(_text_spec(text="Hello\nWorld"), font_dir)
        assert multi.height > single.height, "Two lines should be taller than one"


# ---------------------------------------------------------------------------
# Tests: background spec validation
# ---------------------------------------------------------------------------

class TestBackgroundRequiresDimensions:
    def test_missing_dims_raises(self) -> None:
        from pydantic import ValidationError
        from heimdex_media_contracts.composition.overlays import BackgroundOverlaySpec
        with pytest.raises(ValidationError):
            BackgroundOverlaySpec(
                id="bg1", start_ms=0, end_ms=1000,
                fill_color="#FF0000",
                transform={"x": 0.5, "y": 0.5, "rotation_deg": 0.0},
                # Missing width_px / height_px
            )
