"""Noise-filter unit tests — every rejection reason should fire on the
matching breakdown."""

from __future__ import annotations

from heimdex_media_pipelines.product_enum.clusterer import ProductCluster
from heimdex_media_pipelines.product_enum.config import EnumerationConfig
from heimdex_media_pipelines.product_enum.noise_filter import apply_noise_filter
from heimdex_media_pipelines.product_enum.vlm_client import EnumerationDetection


def _det(conf: float = 0.9) -> EnumerationDetection:
    return EnumerationDetection(
        keyframe_scene_id="s",
        keyframe_frame_idx=0,
        label="x",
        bbox_xywh=(0, 0, 100, 100),
        confidence=conf,
    )


def test_passes_when_all_floors_met():
    cfg = EnumerationConfig()
    cluster = ProductCluster(
        detections=[_det(0.9), _det(0.9)],
        embeddings=[[0.0], [0.0]],
        centroid=[0.0],
    )
    breakdown = {"prominence": 0.05}
    assert apply_noise_filter(
        cluster=cluster, best_detection_index=0,
        best_breakdown=breakdown, config=cfg,
    ) is None


def test_single_keyframe_rejected_first():
    cfg = EnumerationConfig()
    cluster = ProductCluster(
        detections=[_det(0.9)],
        embeddings=[[0.0]],
        centroid=[0.0],
    )
    # Even with bad confidence + low prominence, single_keyframe wins
    # (cheapest check, runs first per docstring contract).
    breakdown = {"prominence": 0.001}
    assert apply_noise_filter(
        cluster=cluster, best_detection_index=0,
        best_breakdown=breakdown, config=cfg,
    ) == "single_keyframe"


def test_low_confidence_rejected():
    cfg = EnumerationConfig()
    cluster = ProductCluster(
        detections=[_det(0.5), _det(0.5)],
        embeddings=[[0.0], [0.0]],
        centroid=[0.0],
    )
    breakdown = {"prominence": 0.05}
    assert apply_noise_filter(
        cluster=cluster, best_detection_index=0,
        best_breakdown=breakdown, config=cfg,
    ) == "low_confidence"


def test_low_prominence_rejected():
    cfg = EnumerationConfig()
    cluster = ProductCluster(
        detections=[_det(0.9), _det(0.9)],
        embeddings=[[0.0], [0.0]],
        centroid=[0.0],
    )
    breakdown = {"prominence": 0.01}  # below 3% floor
    assert apply_noise_filter(
        cluster=cluster, best_detection_index=0,
        best_breakdown=breakdown, config=cfg,
    ) == "low_prominence"


def test_thresholds_overridable():
    """Verify the floors are read from EnumerationConfig — not
    hardcoded — so the worker can tighten them per-org."""
    cfg = EnumerationConfig(
        min_supporting_keyframes=1,
        min_prominence_pct=0.001,
        min_enumeration_confidence=0.0,
    )
    cluster = ProductCluster(
        detections=[_det(0.1)],
        embeddings=[[0.0]],
        centroid=[0.0],
    )
    breakdown = {"prominence": 0.002}
    # All relaxed → should pass.
    assert apply_noise_filter(
        cluster=cluster, best_detection_index=0,
        best_breakdown=breakdown, config=cfg,
    ) is None
