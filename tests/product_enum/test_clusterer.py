"""Cosine clustering unit tests.

Pure-function — no GPU, no network. Embeddings are hand-rolled
deterministic vectors so the threshold behavior is testable without
SigLIP2 in the loop.
"""

from __future__ import annotations

import math

import pytest

from heimdex_media_pipelines.product_enum.clusterer import (
    ProductCluster,
    cluster_detections,
)
from heimdex_media_pipelines.product_enum.vlm_client import (
    EnumerationDetection,
)


def _det(scene_id: str, label: str = "x") -> EnumerationDetection:
    return EnumerationDetection(
        keyframe_scene_id=scene_id,
        keyframe_frame_idx=0,
        label=label,
        bbox_xywh=(10, 10, 100, 100),
        confidence=0.9,
    )


def _normalize(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n > 0 else v


def test_empty_input_returns_empty():
    assert cluster_detections([], [], cosine_threshold=0.85) == []


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        cluster_detections(
            [_det("s1")], [_normalize([1.0, 0.0]), _normalize([0.0, 1.0])],
            cosine_threshold=0.85,
        )


def test_single_detection_forms_one_cluster():
    clusters = cluster_detections(
        [_det("s1")], [_normalize([1.0, 0.0, 0.0])], cosine_threshold=0.85,
    )
    assert len(clusters) == 1
    assert len(clusters[0].detections) == 1


def test_distinct_vectors_form_separate_clusters():
    # Two near-orthogonal embeddings → cosine ≈ 0 → separate clusters.
    clusters = cluster_detections(
        [_det("s1"), _det("s2")],
        [_normalize([1.0, 0.0]), _normalize([0.0, 1.0])],
        cosine_threshold=0.85,
    )
    assert len(clusters) == 2


def test_similar_vectors_merge():
    # Two embeddings within 5° of each other (cos ≈ 0.996) → same cluster.
    a = _normalize([1.0, 0.0, 0.0])
    b = _normalize([math.cos(math.radians(5)), math.sin(math.radians(5)), 0.0])
    clusters = cluster_detections(
        [_det("s1"), _det("s2")], [a, b], cosine_threshold=0.85,
    )
    assert len(clusters) == 1
    assert len(clusters[0].detections) == 2


def test_threshold_at_boundary():
    # Exactly at threshold — boundary case. Implementation uses ``>=``
    # so equal-to-threshold MERGES (broader recall is the safer
    # default for early-stage tuning).
    a = _normalize([1.0, 0.0])
    b = _normalize([0.85, math.sqrt(1 - 0.85 * 0.85)])  # cos == 0.85
    clusters = cluster_detections(
        [_det("s1"), _det("s2")], [a, b], cosine_threshold=0.85,
    )
    assert len(clusters) == 1


def test_centroid_pulls_clusters_together():
    # Three detections: A, A', B.
    # A and A' are similar (cos ~ 0.99). B is similar to A' but not A.
    # The centroid of (A, A') should sit between them, so B's
    # comparison against the centroid (not just the first member) is
    # what determines membership.
    a = _normalize([1.0, 0.0])
    a_prime = _normalize([math.cos(math.radians(3)), math.sin(math.radians(3))])
    b = _normalize([math.cos(math.radians(20)), math.sin(math.radians(20))])
    # Pick threshold = 0.94 — A vs B is below, but A's centroid w/
    # A_prime sits at ~1.5° so centroid-to-B is ~18.5° (cos ~ 0.948).
    # Should merge B in.
    clusters = cluster_detections(
        [_det("s1"), _det("s2"), _det("s3")],
        [a, a_prime, b],
        cosine_threshold=0.94,
    )
    assert len(clusters) == 1
    assert len(clusters[0].detections) == 3


def test_same_frame_detections_never_merge_even_when_visually_similar():
    """Adversarial-review regression (Codex 2026-04-29, finding #3).

    Two detections from the SAME ``(scene_id, frame_idx)`` are
    mutually exclusive products by construction — the VLM looked at
    one frame and chose to emit two records. Even when their
    embeddings are nearly identical (e.g. two packaging variants of
    the same SKU line), the clusterer must keep them separate.
    Merging would silently drop one product from the user-facing
    gallery.

    The cosine here is ~0.998 — well above the 0.85 threshold — so
    without the same-frame guard this would collapse to one cluster.
    """
    a = _normalize([1.0, 0.0, 0.0])
    b = _normalize([math.cos(math.radians(3)), math.sin(math.radians(3)), 0.0])
    same_frame_a = EnumerationDetection(
        keyframe_scene_id="s1", keyframe_frame_idx=42,
        label="serum-A", bbox_xywh=(100, 100, 200, 300), confidence=0.9,
    )
    same_frame_b = EnumerationDetection(
        keyframe_scene_id="s1", keyframe_frame_idx=42,
        label="serum-B", bbox_xywh=(450, 100, 200, 300), confidence=0.9,
    )
    clusters = cluster_detections(
        [same_frame_a, same_frame_b], [a, b], cosine_threshold=0.85,
    )
    assert len(clusters) == 2, (
        "same-frame disjointness violated: two detections from "
        "(scene_id=s1, frame_idx=42) must NOT merge regardless of "
        "embedding similarity"
    )


def test_same_frame_guard_does_not_block_cross_frame_merge():
    """Sibling regression: the same-frame guard must be scoped to
    ``(scene_id, frame_idx)`` only — detections in different frames
    with similar embeddings should still merge as before.
    """
    a = _normalize([1.0, 0.0, 0.0])
    b = _normalize([math.cos(math.radians(3)), math.sin(math.radians(3)), 0.0])
    cross_frame_a = EnumerationDetection(
        keyframe_scene_id="s1", keyframe_frame_idx=10,
        label="serum", bbox_xywh=(0, 0, 100, 100), confidence=0.9,
    )
    cross_frame_b = EnumerationDetection(
        keyframe_scene_id="s2", keyframe_frame_idx=20,
        label="serum", bbox_xywh=(0, 0, 100, 100), confidence=0.9,
    )
    clusters = cluster_detections(
        [cross_frame_a, cross_frame_b], [a, b], cosine_threshold=0.85,
    )
    assert len(clusters) == 1


def test_same_frame_detection_skipped_for_existing_cluster_but_can_join_another():
    """When a candidate cluster is disqualified by the same-frame
    guard, the new detection should still be allowed to join a
    DIFFERENT cluster that doesn't share its frame.
    """
    # Two detections in scene s1/frame 1, similar to each other.
    a_s1 = _normalize([1.0, 0.0])
    b_s1 = _normalize([math.cos(math.radians(2)), math.sin(math.radians(2))])
    # A third detection in scene s2/frame 5, also similar.
    c_s2 = _normalize([math.cos(math.radians(4)), math.sin(math.radians(4))])
    det_a = EnumerationDetection("s1", 1, "p", (0, 0, 50, 50), 0.9)
    det_b = EnumerationDetection("s1", 1, "p", (60, 0, 50, 50), 0.9)
    det_c = EnumerationDetection("s2", 5, "p", (0, 0, 50, 50), 0.9)
    # Order: A (cluster 0), B (different frame from A: NEW cluster 1),
    # C (different frame from both: should join the closest cluster
    # that doesn't share its frame).
    clusters = cluster_detections(
        [det_a, det_b, det_c], [a_s1, b_s1, c_s2], cosine_threshold=0.85,
    )
    # A and B must remain separate (same frame).
    # C is in s2/5 — not blocked by either, so should merge with the
    # closest (cluster 0, since A's centroid is closest to C).
    assert len(clusters) == 2
    cluster_sizes = sorted(len(c.detections) for c in clusters)
    assert cluster_sizes == [1, 2]


def test_cluster_centroid_is_normalized():
    a = _normalize([1.0, 0.0])
    b = _normalize([math.cos(math.radians(5)), math.sin(math.radians(5))])
    clusters = cluster_detections(
        [_det("s1"), _det("s2")], [a, b], cosine_threshold=0.85,
    )
    centroid = clusters[0].centroid
    norm = math.sqrt(sum(x * x for x in centroid))
    assert abs(norm - 1.0) < 1e-6
