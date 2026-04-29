"""Reject clusters that are likely noise.

Plan §6 + §11 noise rules, applied AFTER reference picking so we
filter on the best-of-cluster representative. Returns a list of
``(cluster, reason)`` tuples; ``reason`` is ``None`` for accepted
clusters.

Reasons mirror :class:`heimdex_media_contracts.product.RejectedReason`
so the worker can persist them straight to ``rejected_reason`` on
the API callback.
"""

from __future__ import annotations

from heimdex_media_pipelines.product_enum.clusterer import ProductCluster
from heimdex_media_pipelines.product_enum.config import EnumerationConfig


def apply_noise_filter(
    *,
    cluster: ProductCluster,
    best_detection_index: int,
    best_breakdown: dict[str, float],
    config: EnumerationConfig,
) -> str | None:
    """Return ``None`` if the cluster passes; otherwise the rejection
    reason.

    Order matters — we report the first failing rule, so worker logs
    of "rejected_reason" stay actionable. ``single_keyframe`` is
    checked first because it's the cheapest dedupe (no need to look
    at scores if the cluster has only one supporting frame).
    """
    if len(cluster.detections) < config.min_supporting_keyframes:
        return "single_keyframe"

    best = cluster.detections[best_detection_index]
    if best.confidence < config.min_enumeration_confidence:
        return "low_confidence"

    if best_breakdown.get("prominence", 0.0) < config.min_prominence_pct:
        return "low_prominence"

    return None
