"""PII blur pipeline: face + OWLv2 text-guided detection + mosaic blur.

Produces a blurred copy of a video plus a structured detection manifest
that downstream search indexing can consume. GPU recommended (OWLv2 is
prohibitively slow on CPU); degrades gracefully.

Public surface:
    BlurConfig       — all knobs (do_faces, do_owl, stride, thresholds, ...)
    BlurPipeline     — frame-loop orchestrator, returns BlurResult
    BlurResult       — summary + detection list (serializable to manifest JSON)
    DetectionRecord  — one detected + blurred region
    DEFAULT_OWL_QUERIES — built-in text queries grouped by category
"""

from heimdex_media_pipelines.blur.config import BlurConfig, DetectionRecord, BlurResult
from heimdex_media_pipelines.blur.masks import (
    BlurProgressEvent,
    CategoryMaskAggregator,
    CategoryMaskWriter,
    ProgressCallback,
    ProgressThrottler,
)
from heimdex_media_pipelines.blur.pipeline import BlurPipeline
from heimdex_media_pipelines.blur.primitives import (
    apply_mosaic_blur,
    apply_mosaic_blur_norm,
)
from heimdex_media_pipelines.blur.queries import (
    DEFAULT_OWL_QUERIES,
    DIRECT_BLUR_CATEGORIES,
    label_to_category,
)

__all__ = [
    "BlurConfig",
    "BlurPipeline",
    "BlurProgressEvent",
    "BlurResult",
    "CategoryMaskAggregator",
    "CategoryMaskWriter",
    "DetectionRecord",
    "DEFAULT_OWL_QUERIES",
    "DIRECT_BLUR_CATEGORIES",
    "ProgressCallback",
    "ProgressThrottler",
    "label_to_category",
    "apply_mosaic_blur",
    "apply_mosaic_blur_norm",
]
