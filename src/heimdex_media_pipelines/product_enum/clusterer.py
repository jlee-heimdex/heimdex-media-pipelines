"""Greedy cosine clustering over SigLIP2 embeddings.

Single-link clustering: each new detection's embedding is compared to
the running centroid of every existing cluster. Above
``cluster_cosine_threshold`` it joins the closest cluster (centroid
updated as a running mean of L2-normalized vectors, then re-normalized
so future cosines stay valid). Below all thresholds it forms a new
cluster.

This is intentionally simple. Greedy single-link suits the small N
(≤ 60 keyframes × ≤ ~5 detections each = ≤ 300 vectors) we expect per
video; we re-evaluate only if recall drops on the staging goldens.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from heimdex_media_pipelines.product_enum.vlm_client import EnumerationDetection


@dataclass
class ProductCluster:
    """A set of detections that the clusterer judged to be the same
    product. ``centroid`` is the running L2-normalized mean used for
    similarity comparisons; ``embeddings`` mirrors ``detections``
    one-to-one for the reference picker.
    """

    detections: list[EnumerationDetection] = field(default_factory=list)
    embeddings: list[list[float]] = field(default_factory=list)
    centroid: list[float] = field(default_factory=list)


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length, presumed-normalized vecs.

    Tolerates non-normalized inputs but is faster on normalized ones
    (the SigLIP2 embedder already L2-normalizes). Returns 0 if either
    vector has zero norm.
    """
    if len(a) != len(b):
        raise ValueError(f"vector dim mismatch: {len(a)} vs {len(b)}")
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def _normalized_mean(vectors: list[list[float]]) -> list[float]:
    """L2-normalized arithmetic mean of normalized vectors. Used for
    centroid maintenance; keeps subsequent cosine computations valid
    without re-normalizing on each comparison.
    """
    if not vectors:
        return []
    dim = len(vectors[0])
    acc = [0.0] * dim
    for v in vectors:
        for i, x in enumerate(v):
            acc[i] += x
    n = len(vectors)
    for i in range(dim):
        acc[i] /= n
    norm = math.sqrt(sum(x * x for x in acc))
    if norm <= 0:
        return acc
    return [x / norm for x in acc]


def cluster_detections(
    detections: list[EnumerationDetection],
    embeddings: list[list[float]],
    *,
    cosine_threshold: float,
) -> list[ProductCluster]:
    """Group detections into clusters by cosine similarity.

    Detections must come pre-paired with their SigLIP2 embeddings —
    the worker computes embeddings via
    :mod:`heimdex_media_pipelines.siglip2.embed` between VLM and
    cluster steps.

    **Invariant: same-frame disjointness.** Two detections in the same
    ``(scene_id, frame_idx)`` are MUTUALLY EXCLUSIVE products by
    construction — the VLM saw both at once and explicitly chose to
    emit two records. Even when their crops are visually similar
    (e.g. two near-identical packaging variants of the same SKU line),
    we must NOT merge them, because the persisted catalog row would
    silently lose one product from the user-facing gallery.

    The clusterer enforces this by skipping any candidate cluster
    whose existing members already cover the new detection's
    ``(scene_id, frame_idx)``. Such a detection forms its own
    cluster regardless of cosine similarity. Without this guard a
    fragile assumption creeps in ("the VLM never emits dupes per
    frame") that quietly breaks correctness when violated.
    """
    if len(detections) != len(embeddings):
        raise ValueError(
            f"detections / embeddings length mismatch: "
            f"{len(detections)} vs {len(embeddings)}"
        )
    clusters: list[ProductCluster] = []
    for det, emb in zip(detections, embeddings):
        best_idx = -1
        best_sim = cosine_threshold
        det_key = (det.keyframe_scene_id, det.keyframe_frame_idx)
        for idx, cluster in enumerate(clusters):
            # Same-frame disjointness guard. Cluster membership is
            # disqualified up front when ANY existing detection in
            # the candidate cluster shares this detection's
            # (scene_id, frame_idx). Cheap: ~O(cluster_size) per
            # candidate, dominated by the cosine call anyway.
            if any(
                (m.keyframe_scene_id, m.keyframe_frame_idx) == det_key
                for m in cluster.detections
            ):
                continue
            sim = _cosine(emb, cluster.centroid)
            if sim >= best_sim:
                best_sim = sim
                best_idx = idx
        if best_idx >= 0:
            clusters[best_idx].detections.append(det)
            clusters[best_idx].embeddings.append(emb)
            clusters[best_idx].centroid = _normalized_mean(
                clusters[best_idx].embeddings
            )
        else:
            clusters.append(
                ProductCluster(
                    detections=[det],
                    embeddings=[emb],
                    centroid=list(emb),
                )
            )
    return clusters
