"""Product enumeration library for shorts-auto product mode v2.

Lazy first-stage of the two-stage pipeline: given a list of scene
keyframes from a Korean live-commerce video, return a deduplicated
catalog of distinct products with canonical reference frames + 768-dim
SigLIP2 embeddings.

This module is **pure** — no S3, no HTTP, no DB. Workers (in
``services/product-enumerate-worker/``) handle I/O and call
:func:`enumerate_products` with already-downloaded PIL images.

See ``dev-heimdex-for-livecommerce/.claude/plans/shorts-auto-product-v2.md``
§6 for the full design and locked decisions.

Public surface:
* :class:`SceneKeyframe` — one keyframe + its scene id, used as the
  pipeline's input row
* :class:`EnumerationDetection` — one VLM-detected product crop
  before clustering
* :class:`ProductCluster` — a set of detections collapsed to a single
  canonical entry
* :class:`CanonicalProduct` — final output: one product per cluster
  with quality metadata + SigLIP2 embedding
* :class:`EnumerationConfig` — all knobs (thresholds, caps)
* :func:`enumerate_products` — orchestrator; the only function the
  worker calls
* :class:`VlmClient` — Protocol; production impl in ``vlm_client.py``,
  test impl can be a stub
"""

from heimdex_media_pipelines.product_enum.config import (
    DEFAULT_ENUMERATION_VERSION,
    EnumerationConfig,
)
from heimdex_media_pipelines.product_enum.clusterer import (
    ProductCluster,
    cluster_detections,
)
from heimdex_media_pipelines.product_enum.noise_filter import (
    apply_noise_filter,
)
from heimdex_media_pipelines.product_enum.pipeline import (
    CanonicalProduct,
    SceneKeyframe,
    enumerate_products,
)
from heimdex_media_pipelines.product_enum.reference_picker import (
    pick_reference_frame,
    reference_quality_score,
)
from heimdex_media_pipelines.product_enum.vlm_client import (
    EnumerationDetection,
    VlmClient,
    VlmDetectionBatch,
)

__all__ = [
    "CanonicalProduct",
    "DEFAULT_ENUMERATION_VERSION",
    "EnumerationConfig",
    "EnumerationDetection",
    "ProductCluster",
    "SceneKeyframe",
    "VlmClient",
    "VlmDetectionBatch",
    "apply_noise_filter",
    "cluster_detections",
    "enumerate_products",
    "pick_reference_frame",
    "reference_quality_score",
]
