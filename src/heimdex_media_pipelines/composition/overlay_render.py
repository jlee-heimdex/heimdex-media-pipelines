"""PIL bake renderer for V2 overlay specs.

Takes an OverlaySpec (TextOverlaySpec or BackgroundOverlaySpec) and returns a
transparent RGBA PIL.Image with ALL effects baked in:

  - italic (real italic font or faux-italic skew if not available)
  - underline (drawn line per row)
  - letter_spacing (glyph-by-glyph advance)
  - line_height (per-line y advance)
  - highlight box (text-fitted padded rect behind text)
  - fill color (background overlays)
  - stroke (via PIL stroke_width / outline)
  - shadow (spread via dilation, blur via GaussianBlur, offset)
  - opacity (alpha multiply of final composed image)
  - rotation (expand=True so caller just positions with overlay= filter)

The returned image has its OWN dimensions (not canvas size). The caller
positions it on the canvas via ffmpeg overlay= using overlay.transform.x/y.
Caller does NOT need to apply any further transforms.

Regenerate golden fixtures via:
    pytest tests/composition/test_overlay_render.py --regen-goldens
"""

from __future__ import annotations

import os
from typing import Union

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from heimdex_media_contracts.composition.filters import (
    FontNotFoundError,
    _FONT_EXTENSIONS,
    _FONT_FILE_BASES,
)
from heimdex_media_contracts.composition.overlays import (
    BackgroundOverlaySpec,
    EffectsSpec,
    ShadowSpec,
    StrokeSpec,
    TextOverlaySpec,
)

# ---------------------------------------------------------------------------
# Public type alias (matches contracts discriminated union)
# ---------------------------------------------------------------------------

OverlaySpec = Union[TextOverlaySpec, BackgroundOverlaySpec]

# Padding added around the rendered content before rotation, so shadow
# and stroke don't get clipped at the image boundary.
_RENDER_PADDING = 8


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def bake_overlay_png(
    overlay: OverlaySpec,
    canvas_width: int,
    canvas_height: int,
    font_dir: str,
) -> Image.Image:
    """Render an overlay to a transparent RGBA PIL.Image with all effects baked in.

    The returned image has its OWN dimensions (not canvas size) — caller positions
    it on the canvas via ffmpeg overlay= using overlay.transform.x/y.

    For text overlays: image size auto-computed from text + font + padding.
    For background overlays: image size = transform.width_px x height_px.

    All effects (italic, underline, stroke, shadow with blur/spread, opacity,
    rotation) are baked into the image. Caller does NOT need to apply any
    further transforms — just position with overlay= filter.
    """
    if overlay.kind == "text":
        return _bake_text(overlay, font_dir)
    else:
        return _bake_background(overlay)


# ---------------------------------------------------------------------------
# Text overlay bake
# ---------------------------------------------------------------------------

def _bake_text(ov: TextOverlaySpec, font_dir: str) -> Image.Image:
    stroke: StrokeSpec | None = ov.effects.stroke
    shadow: ShadowSpec | None = ov.effects.shadow

    # Extra padding needed by shadow / stroke so content is never clipped
    shadow_pad = _shadow_extra_padding(shadow)
    stroke_pad = stroke.width_px if stroke else 0
    extra = max(shadow_pad, stroke_pad) + _RENDER_PADDING

    # --- Load font ---
    italic_path = _resolve_italic_font_path(ov.font_family, ov.font_weight, font_dir)
    use_faux_italic = italic_path is None and ov.italic
    font_path = (
        italic_path
        if italic_path is not None
        else _resolve_font_path(ov.font_family, ov.font_weight, font_dir)
    )
    font = ImageFont.truetype(font_path, ov.font_size_px)

    # --- Lay out text rows ---
    rows = ov.text.split("\n") if ov.text else [""]
    row_metrics = [_measure_row(font, row, ov.letter_spacing) for row in rows]

    text_w = max(w for w, _ in row_metrics) if row_metrics else 0
    line_advance = int(ov.font_size_px * ov.line_height)
    text_h = line_advance * len(rows)

    # Highlight box dimensions (behind text)
    hp = ov.highlight_padding_px
    inner_w = text_w
    inner_h = text_h
    total_w = inner_w + 2 * hp + 2 * extra
    total_h = inner_h + 2 * hp + 2 * extra

    img = Image.new("RGBA", (total_w, total_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    text_origin_x = extra + hp
    text_origin_y = extra + hp

    # --- Highlight box ---
    if ov.has_highlight and ov.highlight_color is not None:
        hc = _hex_to_rgba(ov.highlight_color)
        alpha = int(hc[3] * ov.highlight_opacity) if len(hc) == 4 else int(255 * ov.highlight_opacity)
        box_color = (hc[0], hc[1], hc[2], alpha)
        draw.rectangle(
            [
                text_origin_x - hp,
                text_origin_y - hp,
                text_origin_x + inner_w + hp,
                text_origin_y + inner_h + hp,
            ],
            fill=box_color,
        )

    # --- Shadow layer (rendered separately, pasted behind) ---
    if shadow is not None:
        shadow_layer = _render_text_shadow(
            rows=rows,
            row_metrics=row_metrics,
            font=font,
            letter_spacing=ov.letter_spacing,
            line_advance=line_advance,
            font_color=ov.font_color,
            stroke=stroke,
            shadow=shadow,
            total_w=total_w,
            total_h=total_h,
            text_origin_x=text_origin_x,
            text_origin_y=text_origin_y,
        )
        img = Image.alpha_composite(img, shadow_layer)
        draw = ImageDraw.Draw(img)

    # --- Draw each text row ---
    stroke_width = stroke.width_px if stroke else 0
    stroke_fill = _hex_to_rgba(stroke.color) if stroke else None
    font_rgba = _hex_to_rgba(ov.font_color)

    for row_idx, (row, (row_w, row_h)) in enumerate(zip(rows, row_metrics)):
        row_y = text_origin_y + row_idx * line_advance
        # Alignment x offset
        if ov.text_align == "center":
            row_x = text_origin_x + (inner_w - row_w) // 2
        elif ov.text_align == "right":
            row_x = text_origin_x + inner_w - row_w
        else:
            row_x = text_origin_x

        if ov.letter_spacing == 0.0:
            # Fast path: draw the whole string at once
            draw.text(
                (row_x, row_y),
                row,
                font=font,
                fill=font_rgba,
                stroke_width=stroke_width,
                stroke_fill=stroke_fill,
            )
        else:
            _draw_spaced_text(draw, row_x, row_y, row, font, font_rgba, ov.letter_spacing, stroke_width, stroke_fill)

        # Underline
        if ov.underline:
            ascent, descent = font.getmetrics()
            underline_y = row_y + ascent + 1
            draw.line(
                [(row_x, underline_y), (row_x + row_w, underline_y)],
                fill=font_rgba,
                width=max(1, ov.font_size_px // 20),
            )

    # --- Faux italic skew ---
    if use_faux_italic:
        img = _apply_faux_italic(img)

    # --- Rotation ---
    if ov.transform.rotation_deg != 0.0:
        img = img.rotate(
            -ov.transform.rotation_deg,
            expand=True,
            resample=Image.BICUBIC,
        )

    # --- Global opacity ---
    if ov.effects.opacity < 1.0:
        img = _apply_opacity(img, ov.effects.opacity)

    return img


# ---------------------------------------------------------------------------
# Background overlay bake
# ---------------------------------------------------------------------------

def _bake_background(ov: BackgroundOverlaySpec) -> Image.Image:
    stroke: StrokeSpec | None = ov.effects.stroke
    shadow: ShadowSpec | None = ov.effects.shadow

    w_px = ov.transform.width_px  # guaranteed non-None by model_validator
    h_px = ov.transform.height_px

    shadow_pad = _shadow_extra_padding(shadow)
    stroke_pad = stroke.width_px if stroke else 0
    extra = max(shadow_pad, stroke_pad) + _RENDER_PADDING

    total_w = w_px + 2 * extra
    total_h = h_px + 2 * extra

    img = Image.new("RGBA", (total_w, total_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    rect = [extra, extra, extra + w_px, extra + h_px]

    # --- Shadow ---
    if shadow is not None:
        shadow_layer = _render_rect_shadow(
            rect=rect,
            shadow=shadow,
            total_w=total_w,
            total_h=total_h,
        )
        img = Image.alpha_composite(img, shadow_layer)
        draw = ImageDraw.Draw(img)

    # --- Fill ---
    fill_rgba = _hex_to_rgba(ov.fill_color)
    stroke_rgba = _hex_to_rgba(stroke.color) if stroke else None
    stroke_width = stroke.width_px if stroke else 0

    draw.rectangle(
        rect,
        fill=fill_rgba,
        outline=stroke_rgba,
        width=stroke_width if stroke else 0,
    )

    # --- Rotation ---
    if ov.transform.rotation_deg != 0.0:
        img = img.rotate(
            -ov.transform.rotation_deg,
            expand=True,
            resample=Image.BICUBIC,
        )

    # --- Global opacity ---
    if ov.effects.opacity < 1.0:
        img = _apply_opacity(img, ov.effects.opacity)

    return img


# ---------------------------------------------------------------------------
# Shadow helpers
# ---------------------------------------------------------------------------

def _render_text_shadow(
    *,
    rows: list[str],
    row_metrics: list[tuple[int, int]],
    font: ImageFont.FreeTypeFont,
    letter_spacing: float,
    line_advance: int,
    font_color: str,
    stroke: StrokeSpec | None,
    shadow: ShadowSpec,
    total_w: int,
    total_h: int,
    text_origin_x: int,
    text_origin_y: int,
) -> Image.Image:
    """Render a blurred/spread/offset shadow for text."""
    # Draw text into an alpha-only mask at the same positions
    mask = Image.new("L", (total_w, total_h), 0)
    mask_draw = ImageDraw.Draw(mask)
    font_rgba = _hex_to_rgba(font_color)
    stroke_width = stroke.width_px if stroke else 0

    for row_idx, (row, (row_w, _)) in enumerate(zip(rows, row_metrics)):
        row_y = text_origin_y + row_idx * line_advance
        row_x = text_origin_x
        if letter_spacing == 0.0:
            mask_draw.text(
                (row_x, row_y),
                row,
                font=font,
                fill=255,
                stroke_width=stroke_width,
                stroke_fill=255 if stroke_width else None,
            )
        else:
            _draw_spaced_text(
                mask_draw, row_x, row_y, row, font, 255,
                letter_spacing, stroke_width, 255 if stroke_width else None,
            )

    return _finalize_shadow(mask, shadow, total_w, total_h)


def _render_rect_shadow(
    *,
    rect: list[int],
    shadow: ShadowSpec,
    total_w: int,
    total_h: int,
) -> Image.Image:
    """Render a blurred/spread/offset shadow for a rectangle."""
    mask = Image.new("L", (total_w, total_h), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rectangle(rect, fill=255)
    return _finalize_shadow(mask, shadow, total_w, total_h)


def _finalize_shadow(
    mask: Image.Image,
    shadow: ShadowSpec,
    total_w: int,
    total_h: int,
) -> Image.Image:
    """Apply spread (dilation), blur, color, and offset to a shadow mask."""
    # Spread via MaxFilter (cheap dilation approximation)
    if shadow.spread_px > 0:
        size = shadow.spread_px * 2 + 1
        mask = mask.filter(ImageFilter.MaxFilter(size))

    # Blur
    if shadow.blur_px > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=shadow.blur_px))

    # Tint
    shadow_color = _hex_to_rgba(shadow.color)
    shadow_layer = Image.new("RGBA", (total_w, total_h), (0, 0, 0, 0))
    colored = Image.new("RGBA", (total_w, total_h), shadow_color[:3] + (255,))
    shadow_layer.paste(colored, mask=mask)

    # Offset: shift the shadow layer
    if shadow.offset_x != 0 or shadow.offset_y != 0:
        shifted = Image.new("RGBA", (total_w, total_h), (0, 0, 0, 0))
        shifted.paste(shadow_layer, (shadow.offset_x, shadow.offset_y))
        shadow_layer = shifted

    return shadow_layer


def _shadow_extra_padding(shadow: ShadowSpec | None) -> int:
    """Compute extra pixels needed to prevent shadow from clipping."""
    if shadow is None:
        return 0
    return abs(shadow.offset_x) + abs(shadow.offset_y) + shadow.blur_px + shadow.spread_px


# ---------------------------------------------------------------------------
# Font resolution (italic-aware extension of contracts' _resolve_font_path)
# ---------------------------------------------------------------------------

_ITALIC_FONT_NAMES: dict[str, str] = {
    "Pretendard": "Pretendard-Italic",
    "Noto Sans KR": "NotoSansKR-Italic",
}


def _resolve_font_path(family: str, weight: int, font_dir: str) -> str:
    """Resolve non-italic font (raises FontNotFoundError if not found)."""
    font_dir = font_dir.rstrip("/")
    weight_suffix = "Bold" if weight >= 600 else "Regular"
    family_bases = _FONT_FILE_BASES.get(family)
    if family_bases is None:
        raise FontNotFoundError(
            f"Unsupported font_family={family!r}"
        )
    base = family_bases[weight_suffix]
    for ext in _FONT_EXTENSIONS:
        candidate = f"{font_dir}/{base}{ext}"
        if os.path.exists(candidate):
            return candidate
    raise FontNotFoundError(
        f"No font file found for family={family!r} weight={weight} in {font_dir!r}"
    )


def _resolve_italic_font_path(family: str, weight: int, font_dir: str) -> str | None:
    """Try to find a real italic font file. Returns None if not found (triggers faux italic)."""
    italic_base = _ITALIC_FONT_NAMES.get(family)
    if italic_base is None:
        return None
    font_dir = font_dir.rstrip("/")
    for ext in _FONT_EXTENSIONS:
        candidate = f"{font_dir}/{italic_base}{ext}"
        if os.path.exists(candidate):
            return candidate
    return None


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _measure_row(font: ImageFont.FreeTypeFont, text: str, letter_spacing: float) -> tuple[int, int]:
    """Return (width, height) for a single text row given font + letter_spacing."""
    if not text:
        bbox = font.getbbox(" ")
        return 0, (bbox[3] - bbox[1]) if bbox else 0
    if letter_spacing == 0.0:
        bbox = font.getbbox(text)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    # Sum individual glyph widths + spacing
    total_w = 0
    max_h = 0
    for i, ch in enumerate(text):
        bbox = font.getbbox(ch)
        glyph_w = bbox[2] - bbox[0]
        glyph_h = bbox[3] - bbox[1]
        total_w += glyph_w
        if i < len(text) - 1:
            total_w += int(letter_spacing)
        max_h = max(max_h, glyph_h)
    return total_w, max_h


def _draw_spaced_text(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: tuple | int,
    letter_spacing: float,
    stroke_width: int = 0,
    stroke_fill: tuple | int | None = None,
) -> None:
    """Draw text glyph-by-glyph with custom letter spacing."""
    cursor_x = x
    for i, ch in enumerate(text):
        draw.text(
            (cursor_x, y),
            ch,
            font=font,
            fill=fill,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
        )
        bbox = font.getbbox(ch)
        glyph_w = bbox[2] - bbox[0]
        cursor_x += glyph_w
        if i < len(text) - 1:
            cursor_x += int(letter_spacing)


def _apply_faux_italic(img: Image.Image, skew_factor: float = 0.25) -> Image.Image:
    """Apply a simple horizontal shear (faux italic) to the image."""
    w, h = img.size
    # Affine transform: x' = x + skew * (h - y), y' = y  (right-leaning slant)
    # PIL affine coefficients: (a, b, c, d, e, f) such that
    #   input_x = a * out_x + b * out_y + c
    #   input_y = d * out_x + e * out_y + f
    a, b, c = 1.0, -skew_factor, skew_factor * h
    d, e, f = 0.0, 1.0, 0.0
    new_w = int(w + skew_factor * h)
    result = Image.new("RGBA", (new_w, h), (0, 0, 0, 0))
    result.paste(img.transform(
        (new_w, h),
        Image.AFFINE,
        (a, b, c, d, e, f),
        resample=Image.BICUBIC,
    ), (0, 0))
    return result


def _apply_opacity(img: Image.Image, opacity: float) -> Image.Image:
    """Multiply the alpha channel of an RGBA image by opacity [0, 1]."""
    r, g, b, a = img.split()
    a = a.point(lambda px: int(px * opacity))
    return Image.merge("RGBA", (r, g, b, a))


def _hex_to_rgba(hex_color: str) -> tuple[int, int, int, int]:
    """Convert #RRGGBB or #RRGGBBAA hex string to (R, G, B, A) tuple."""
    h = hex_color.lstrip("#")
    if len(h) == 6:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return r, g, b, 255
    elif len(h) == 8:
        r, g, b, a = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), int(h[6:8], 16)
        return r, g, b, a
    raise ValueError(f"Invalid hex color: {hex_color!r}")
