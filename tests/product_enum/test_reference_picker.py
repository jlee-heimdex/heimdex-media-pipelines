"""Reference picker unit tests.

Composite scoring — covers ordering invariants. Synthetic PIL images
keep the test fast.
"""

from __future__ import annotations

import pytest
from PIL import Image

from heimdex_media_pipelines.product_enum.config import EnumerationConfig
from heimdex_media_pipelines.product_enum.reference_picker import (
    pick_reference_frame,
    reference_quality_score,
)
from heimdex_media_pipelines.product_enum.vlm_client import (
    EnumerationDetection,
)


def _det(bbox: tuple[int, int, int, int], conf: float = 0.9) -> EnumerationDetection:
    return EnumerationDetection(
        keyframe_scene_id="s",
        keyframe_frame_idx=0,
        label="x",
        bbox_xywh=bbox,
        confidence=conf,
    )


def _gray_crop(w: int = 100, h: int = 100, level: int = 128) -> Image.Image:
    return Image.new("L", (w, h), level)


def test_prominence_is_bbox_over_frame():
    config = EnumerationConfig()
    det = _det((0, 0, 200, 100))  # 200x100 bbox
    breakdown = reference_quality_score(
        det,
        crop=_gray_crop(200, 100),
        frame_size=(1000, 1000),
        cluster_size=1,
        config=config,
    )
    # 200*100 / 1000*1000 = 0.02
    assert abs(breakdown["prominence"] - 0.02) < 1e-9


def test_centeredness_max_at_frame_center():
    config = EnumerationConfig()
    # 100x100 bbox centered on a 1000x1000 frame → bbox center at (500,500)
    det = _det((450, 450, 100, 100))
    breakdown = reference_quality_score(
        det,
        crop=_gray_crop(),
        frame_size=(1000, 1000),
        cluster_size=1,
        config=config,
    )
    assert breakdown["centeredness"] >= 0.99


def test_centeredness_min_at_corner():
    config = EnumerationConfig()
    det = _det((0, 0, 50, 50))  # bbox center at (25, 25) — far from (500,500)
    breakdown = reference_quality_score(
        det,
        crop=_gray_crop(50, 50),
        frame_size=(1000, 1000),
        cluster_size=1,
        config=config,
    )
    assert breakdown["centeredness"] < 0.4


def test_temporal_stability_saturates():
    config = EnumerationConfig()
    det = _det((0, 0, 100, 100))
    # cluster_size=1 → low stability; cluster_size=10 → 1.0 (saturated)
    low = reference_quality_score(
        det, crop=_gray_crop(), frame_size=(1000, 1000),
        cluster_size=1, config=config,
    )
    high = reference_quality_score(
        det, crop=_gray_crop(), frame_size=(1000, 1000),
        cluster_size=10, config=config,
    )
    assert low["temporal_stability"] < high["temporal_stability"]
    assert high["temporal_stability"] == 1.0


def test_invalid_frame_size_raises():
    config = EnumerationConfig()
    with pytest.raises(ValueError, match="invalid frame_size"):
        reference_quality_score(
            _det((0, 0, 10, 10)),
            crop=_gray_crop(),
            frame_size=(0, 1000),
            cluster_size=1,
            config=config,
        )


def test_picker_orders_on_composite_when_prominence_dominates():
    """Use IDENTICAL crops + frame sizes so sharpness, centeredness,
    and temporal_stability all tie. The only signal left is
    prominence — verifying the picker honors the composite.

    (Earlier draft of this test used differently-sized uniform gray
    crops to "cancel" sharpness, but PIL's FIND_EDGES filter has
    boundary artefacts that make sharpness scale with the
    surface-to-area ratio of the crop. Using identical crops sidesteps
    the trap.)
    """
    config = EnumerationConfig()
    big = _det((400, 400, 200, 200))      # prominence 0.04
    small = _det((480, 480, 60, 60))      # prominence 0.0036
    # Identical crops so sharpness ties exactly.
    crop = _gray_crop(100, 100)
    crops = [crop, crop]
    sizes = [(1000, 1000), (1000, 1000)]
    best_idx, breakdown = pick_reference_frame(
        [big, small], crops, sizes,
        cluster_size=2, config=config,
    )
    assert best_idx == 0
    assert breakdown["prominence"] > 0.03


def test_picker_returns_breakdown_for_winner():
    """The picker's second return value should match the winning
    detection's breakdown — used by the worker for logs / debugging."""
    config = EnumerationConfig()
    crop = _gray_crop()
    sizes = [(1000, 1000), (1000, 1000)]
    high_conf = _det((400, 400, 300, 300), conf=0.95)
    low_conf = _det((400, 400, 100, 100), conf=0.65)
    best_idx, breakdown = pick_reference_frame(
        [high_conf, low_conf], [crop, crop], sizes,
        cluster_size=2, config=config,
    )
    # high_conf has 0.09 prominence vs 0.01 → wins.
    assert best_idx == 0
    # Returned breakdown corresponds to the WINNING detection's score
    # (its prominence) — not the loser's.
    assert breakdown["prominence"] >= 0.08


def test_length_mismatch_raises():
    config = EnumerationConfig()
    with pytest.raises(ValueError, match="length mismatch"):
        pick_reference_frame(
            [_det((0, 0, 10, 10))], [], [],
            cluster_size=1, config=config,
        )
