"""Pick the best canonical reference frame for a cluster.

Composite quality score per detection — the highest-scoring detection
in a cluster becomes the canonical reference (its crop is uploaded to
S3, its bbox is persisted on the catalog row, SAM2 anchors here in
Phase 3 tracking).

Components (plan §6 — calibrated on staging goldens):
* ``prominence``        bbox area / frame area, in [0, 1]
* ``sharpness``         normalized Laplacian variance, clamped [0, 1]
* ``centeredness``      1 − (distance from frame center / frame
                        diagonal-half), in [0, 1]
* ``occlusion_penalty`` placeholder for face/hand overlap; 0 in v1
                        (caller may override per-detection)
* ``temporal_stability`` cluster size, normalized by an arbitrary
                         saturating cap (5 supporting keyframes ≈
                         maximally stable)

Each component is in [0, 1]; the final score is a weighted sum (the
occlusion term is subtracted via its weight). Scores are NOT bounded
to [0, 1] — they're only used for ordering within a cluster.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from heimdex_media_pipelines.product_enum.config import EnumerationConfig
from heimdex_media_pipelines.product_enum.vlm_client import EnumerationDetection

if TYPE_CHECKING:  # pragma: no cover
    from PIL import Image


# Cluster size at which temporal_stability saturates to 1.0. Empirical
# — Korean live-commerce hosts typically show a product 3-7 times
# across a stream, so 5 is a sensible saturating cap.
_TEMPORAL_STABILITY_SAT = 5


def reference_quality_score(
    detection: EnumerationDetection,
    *,
    crop: "Image.Image",
    frame_size: tuple[int, int],
    cluster_size: int,
    occlusion_overlap: float = 0.0,
    config: EnumerationConfig,
) -> dict[str, float]:
    """Compute the composite + components for one detection.

    ``frame_size`` is the source keyframe ``(width, height)`` — the
    detection's bbox is in pixel coordinates relative to this frame.
    ``occlusion_overlap`` is in [0, 1] (fraction of bbox covered by
    face/hand bboxes); v1 callers pass 0.

    Returns a dict with the composite score under ``"composite"`` and
    per-component scores so the worker can log breakdowns for
    debugging without re-computing.
    """
    fw, fh = frame_size
    if fw <= 0 or fh <= 0:
        raise ValueError(f"invalid frame_size: {frame_size}")

    x, y, w, h = detection.bbox_xywh
    prominence = max(0.0, min(1.0, (w * h) / (fw * fh)))

    sharpness = _crop_sharpness(crop)
    centeredness = _bbox_centeredness((x, y, w, h), fw, fh)
    occlusion_penalty = max(0.0, min(1.0, occlusion_overlap))
    temporal_stability = min(1.0, cluster_size / _TEMPORAL_STABILITY_SAT)

    composite = (
        config.weight_prominence * prominence
        + config.weight_sharpness * sharpness
        + config.weight_centeredness * centeredness
        - config.weight_occlusion_penalty * occlusion_penalty
        + config.weight_temporal_stability * temporal_stability
    )
    return {
        "composite": composite,
        "prominence": prominence,
        "sharpness": sharpness,
        "centeredness": centeredness,
        "occlusion_penalty": occlusion_penalty,
        "temporal_stability": temporal_stability,
    }


def pick_reference_frame(
    detections: list[EnumerationDetection],
    crops: list["Image.Image"],
    frame_sizes: list[tuple[int, int]],
    *,
    cluster_size: int,
    occlusion_overlaps: list[float] | None = None,
    config: EnumerationConfig,
) -> tuple[int, dict[str, float]]:
    """Return ``(best_index, breakdown)`` from a cluster's detections.

    The caller is the pipeline orchestrator, which has already
    matched detections, crops, and frame sizes one-to-one (the
    keyframe ``Image`` came from the worker's S3 download; the
    ``frame_size`` is its dimensions; the ``bbox`` is relative to it).
    """
    if not (len(detections) == len(crops) == len(frame_sizes)):
        raise ValueError(
            f"length mismatch: detections={len(detections)} "
            f"crops={len(crops)} frame_sizes={len(frame_sizes)}"
        )
    overlaps = occlusion_overlaps or [0.0] * len(detections)
    best_idx = 0
    best: dict[str, float] = {}
    best_score = float("-inf")
    for i, det in enumerate(detections):
        scored = reference_quality_score(
            det,
            crop=crops[i],
            frame_size=frame_sizes[i],
            cluster_size=cluster_size,
            occlusion_overlap=overlaps[i],
            config=config,
        )
        if scored["composite"] > best_score:
            best_score = scored["composite"]
            best_idx = i
            best = scored
    return best_idx, best


# ---------- low-level helpers ----------

def _crop_sharpness(crop: "Image.Image") -> float:
    """Laplacian variance of the crop, normalized into [0, 1].

    Higher = sharper. Saturates at variance == 500.0 — empirically
    Korean live-commerce keyframes hit ~50-300 for sharp shots, ~5-20
    for motion-blurred. The saturation point is loose; we only need
    monotonicity for ordering within a cluster.
    """
    try:
        import numpy as np
        from PIL import ImageFilter
    except Exception:  # pragma: no cover — pipeline always has these
        return 0.5

    grey = crop.convert("L")
    laplacian = grey.filter(ImageFilter.FIND_EDGES)
    arr = np.asarray(laplacian, dtype=np.float32)
    var = float(arr.var())
    return max(0.0, min(1.0, var / 500.0))


def _bbox_centeredness(
    bbox_xywh: tuple[int, int, int, int],
    frame_w: int,
    frame_h: int,
) -> float:
    """1 − (distance from bbox center to frame center) / (half-diagonal).

    Linear; close-to-center → 1, corner → ~0.
    """
    x, y, w, h = bbox_xywh
    cx = x + w / 2.0
    cy = y + h / 2.0
    fcx = frame_w / 2.0
    fcy = frame_h / 2.0
    half_diag = ((frame_w / 2.0) ** 2 + (frame_h / 2.0) ** 2) ** 0.5
    if half_diag <= 0:
        return 0.0
    dist = ((cx - fcx) ** 2 + (cy - fcy) ** 2) ** 0.5
    return max(0.0, 1.0 - dist / half_diag)
