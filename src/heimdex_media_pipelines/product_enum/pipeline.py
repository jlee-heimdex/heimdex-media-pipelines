"""Enumeration pipeline orchestrator — the only function the worker
should call.

Inputs are PIL images already downloaded from S3 by the worker (the
pipeline is pure: no I/O). Outputs are :class:`CanonicalProduct`
records that the worker turns into ``ProductCatalogEntry`` rows on
the API callback.

Flow (matches plan §6.1):

    1. Subsample keyframes (cap at ``config.max_keyframes``).
    2. Batch keyframes for the VLM call (``config.vlm_batch_size``
       per call).
    3. Call ``vlm_client.detect_products`` per batch → flatten into
       a single list of detections.
    4. Crop each detection from its source keyframe, run SigLIP2 on
       the crop → 768-dim L2-normalized embedding.
    5. Cluster detections by SigLIP2 cosine similarity.
    6. For each cluster, pick the canonical reference frame using the
       composite quality score.
    7. Apply the noise filter; mark rejected clusters with their
       reason instead of dropping (the worker persists them so we
       can tune thresholds later without re-running enumeration).
    8. Return all clusters as :class:`CanonicalProduct` records.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from heimdex_media_pipelines.product_enum.clusterer import (
    ProductCluster,
    cluster_detections,
)
from heimdex_media_pipelines.product_enum.config import EnumerationConfig
from heimdex_media_pipelines.product_enum.noise_filter import apply_noise_filter
from heimdex_media_pipelines.product_enum.reference_picker import (
    pick_reference_frame,
)
from heimdex_media_pipelines.product_enum.vlm_client import (
    EnumerationDetection,
    VlmClient,
    VlmDetectionBatch,
)

if TYPE_CHECKING:  # pragma: no cover
    from PIL import Image

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SceneKeyframe:
    """Worker → pipeline input row.

    The worker resolves ``scene_id`` and ``frame_idx`` from the API
    scene metadata, downloads the keyframe from S3, and constructs
    this dataclass. The pipeline does not know how the image was
    obtained.
    """

    scene_id: str
    frame_idx: int
    image: "Image.Image"


@dataclass(frozen=True)
class CanonicalProduct:
    """Pipeline output — one per cluster.

    The worker maps this 1:1 to a
    ``heimdex_media_contracts.product.ProductCatalogEntry`` for the
    API callback. ``canonical_crop`` is a PIL Image the worker
    uploads to S3 to populate ``canonical_crop_s3_key`` on the row.

    Rejected clusters carry ``rejected_reason`` (matching
    :class:`RejectedReason` in contracts) and are still posted to the
    API so threshold tuning can re-classify without re-running
    enumeration.
    """

    canonical_scene_id: str
    canonical_frame_idx: int
    canonical_bbox_xywh: tuple[int, int, int, int]
    canonical_crop: "Image.Image"
    llm_label: str
    siglip2_embedding: list[float]
    enumeration_confidence: float
    prominence_score: float
    cluster_size: int
    rejected_reason: str | None


# ---------- protocol for the SigLIP2 dependency ----------
#
# Defined inline (rather than imported from siglip2/) so this module
# can be tested with a stub embedder that does not load real weights.
# The production wiring in
# ``services/product-enumerate-worker/src/main.py`` passes the real
# ``embed_pil_image_batch`` function bound to a loaded singleton.

class _Embedder:  # pragma: no cover — protocol only, never instantiated
    def __call__(self, images: list["Image.Image"]) -> list[list[float]]:
        ...


def enumerate_products(
    *,
    keyframes: list[SceneKeyframe],
    vlm_client: VlmClient,
    embedder: _Embedder,
    system_prompt: str,
    user_prompt_template: str,
    config: EnumerationConfig,
) -> tuple[list[CanonicalProduct], float]:
    """Run the full enumeration pass.

    Returns ``(products, total_cost_usd)`` — the cost is the sum of
    every batched VLM call so the worker can roll it up into the
    next ``/internal/products/{job_id}/heartbeat`` payload.

    ``user_prompt_template`` may include ``{num_keyframes}`` which
    will be substituted per batch (matches
    ``heimdex_media_contracts.product.EnumerationPrompt.USER_TEMPLATE``).
    """
    if not keyframes:
        return [], 0.0

    # 1. Subsample keyframes (deterministic — keep evenly-spaced).
    if len(keyframes) > config.max_keyframes:
        stride = len(keyframes) / config.max_keyframes
        sampled = [
            keyframes[int(i * stride)] for i in range(config.max_keyframes)
        ]
    else:
        sampled = list(keyframes)

    # 2. Batch + 3. Call VLM.
    all_detections: list[EnumerationDetection] = []
    detection_to_keyframe_idx: list[int] = []  # index into ``sampled``
    total_cost = 0.0
    for start in range(0, len(sampled), config.vlm_batch_size):
        batch = sampled[start:start + config.vlm_batch_size]
        batch_input = [(kf.scene_id, kf.frame_idx, kf.image) for kf in batch]
        user_prompt = user_prompt_template.format(num_keyframes=len(batch))
        result: VlmDetectionBatch = vlm_client.detect_products(
            keyframes=batch_input,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        total_cost += result.cost_usd
        # Map each detection back to its source keyframe so we can
        # crop correctly later.
        scene_to_idx = {
            (kf.scene_id, kf.frame_idx): start + i
            for i, kf in enumerate(batch)
        }
        for det in result.detections:
            all_detections.append(det)
            kf_idx = scene_to_idx.get((det.keyframe_scene_id, det.keyframe_frame_idx))
            if kf_idx is None:
                # VLM hallucinated a scene_id not in this batch —
                # log + drop. (The contracts validation would also
                # catch this on the wire side.)
                logger.warning(
                    "vlm_hallucinated_scene_id",
                    extra={
                        "scene_id": det.keyframe_scene_id,
                        "frame_idx": det.keyframe_frame_idx,
                    },
                )
                detection_to_keyframe_idx.append(-1)
            else:
                detection_to_keyframe_idx.append(kf_idx)

    # Drop hallucinated detections cleanly so downstream lengths match.
    valid_pairs = [
        (det, idx) for det, idx in zip(all_detections, detection_to_keyframe_idx)
        if idx >= 0
    ]
    if not valid_pairs:
        return [], total_cost
    all_detections = [d for d, _ in valid_pairs]
    detection_to_keyframe_idx = [i for _, i in valid_pairs]

    # 4. Crop + embed.
    crops: list["Image.Image"] = []
    frame_sizes: list[tuple[int, int]] = []
    for det, kf_idx in zip(all_detections, detection_to_keyframe_idx):
        source = sampled[kf_idx].image
        crop = _safe_crop(source, det.bbox_xywh)
        crops.append(crop)
        frame_sizes.append(source.size)
    embeddings = embedder(crops)
    if len(embeddings) != len(all_detections):
        raise RuntimeError(
            f"embedder length mismatch: {len(embeddings)} vs {len(all_detections)}"
        )

    # 5. Cluster.
    clusters = cluster_detections(
        all_detections, embeddings,
        cosine_threshold=config.cluster_cosine_threshold,
    )

    # 6 + 7 + 8. Pick + filter + emit.
    products: list[CanonicalProduct] = []
    for cluster in clusters:
        cluster_indices = [
            i for i, det in enumerate(all_detections)
            if det in cluster.detections
        ]
        # Build the per-cluster crop / frame-size lists in the same
        # order as cluster.detections.
        cluster_crops = [crops[i] for i in cluster_indices]
        cluster_frame_sizes = [frame_sizes[i] for i in cluster_indices]

        best_idx, breakdown = pick_reference_frame(
            cluster.detections,
            cluster_crops,
            cluster_frame_sizes,
            cluster_size=len(cluster.detections),
            config=config,
        )
        rejected = apply_noise_filter(
            cluster=cluster,
            best_detection_index=best_idx,
            best_breakdown=breakdown,
            config=config,
        )
        best_det = cluster.detections[best_idx]
        products.append(
            CanonicalProduct(
                canonical_scene_id=best_det.keyframe_scene_id,
                canonical_frame_idx=best_det.keyframe_frame_idx,
                canonical_bbox_xywh=best_det.bbox_xywh,
                canonical_crop=cluster_crops[best_idx],
                llm_label=best_det.label,
                siglip2_embedding=list(cluster.centroid),
                enumeration_confidence=best_det.confidence,
                prominence_score=breakdown.get("prominence", 0.0),
                cluster_size=len(cluster.detections),
                rejected_reason=rejected,
            )
        )

    return products, total_cost


def _safe_crop(
    image: "Image.Image",
    bbox_xywh: tuple[int, int, int, int],
) -> "Image.Image":
    """PIL crop with bbox clamped to the image bounds.

    Defends against off-by-one errors in the VLM's bbox output —
    cropping with negative or out-of-bounds coordinates raises
    silently (PIL returns a black-padded image), which would corrupt
    the SigLIP2 embedding without warning.
    """
    w, h = image.size
    x, y, bw, bh = bbox_xywh
    left = max(0, min(x, w))
    top = max(0, min(y, h))
    right = max(left + 1, min(x + bw, w))
    bottom = max(top + 1, min(y + bh, h))
    return image.crop((left, top, right, bottom))
