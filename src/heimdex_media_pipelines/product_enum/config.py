"""Knobs and version constants for product enumeration.

All thresholds default to the values in the v2 plan §6. Callers
override per-deployment via the worker's env vars (the worker
constructs an :class:`EnumerationConfig` from those).
"""

from __future__ import annotations

from dataclasses import dataclass


# Pipeline version stamped on every produced :class:`CanonicalProduct`.
# Bumping this version invalidates cached catalog entries via the
# version-bump banner in the API. Keep in sync with:
#   * services/api/app/config.py::auto_shorts_product_v2_enumeration_version
#   * heimdex_media_contracts.product_enum (no — version travels in the
#     worker callback payload, not in contracts)
DEFAULT_ENUMERATION_VERSION = "v1.0"


@dataclass(frozen=True)
class EnumerationConfig:
    """All thresholds for the enumeration pass.

    Defaults match plan §6 + §11. Mutable per-call so workers can
    tighten/loosen on a per-org basis (future feature) without
    redeployment.
    """

    # Worker subsamples to at most this many keyframes before LLM —
    # caps cost on long livestreams.
    max_keyframes: int = 60
    # Keyframes per VLM call — the prompt fits ~10 image_url entries
    # cleanly in gpt-4o-mini's structured output mode.
    vlm_batch_size: int = 10

    # SigLIP2 cosine threshold for collapsing detections into the same
    # product cluster. Calibration on staging goldens may push this
    # up/down — start conservative (high recall, may over-merge),
    # tighten if precision suffers.
    cluster_cosine_threshold: float = 0.85

    # Noise-filter floors applied per cluster before the cluster is
    # promoted to a canonical entry.
    min_supporting_keyframes: int = 2          # single-keyframe rejection
    min_prominence_pct: float = 0.03           # bbox / frame area
    min_enumeration_confidence: float = 0.6    # VLM confidence on best ref

    # Reference picker composite weights. Sum to 1.0 by convention but
    # not enforced — relative weights are what matter.
    weight_prominence: float = 0.35
    weight_sharpness: float = 0.25
    weight_centeredness: float = 0.10
    weight_occlusion_penalty: float = 0.15     # subtracted via -occlusion
    weight_temporal_stability: float = 0.15

    enumeration_version: str = DEFAULT_ENUMERATION_VERSION
