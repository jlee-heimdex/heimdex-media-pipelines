"""Configuration, detection records, and result aggregation for the blur pipeline.

These are plain dataclasses rather than pydantic models so the library stays
contract-free at import time. The worker maps between these and the
``heimdex_media_contracts.blur`` pydantic models at the system boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Iterable


# Categories that can appear in a manifest. "face" is added by the face
# detector path, the rest by OWLv2. Kept narrow so consumers can switch
# on a closed set.
ALLOWED_CATEGORIES: frozenset[str] = frozenset({
    "face",
    "license_plate",
    "logo",
    "card_object",
    "object",  # fallback for custom queries
})


@dataclass
class BlurConfig:
    """All knobs the blur pipeline exposes.

    Defaults match what the senior prototype shipped. ``categories``
    intentionally excludes ``logo`` — enabling logo blur is a deliberate
    tenant opt-in because it will blur products in livecommerce footage.
    """

    # What to detect
    do_faces: bool = True
    do_owl: bool = True
    categories: tuple[str, ...] = ("face", "license_plate", "card_object")

    # Face detector
    face_detector: str = "scrfd"  # "scrfd" or "haar"
    min_face_confidence: float = 0.5
    face_shrink_px: int = 4

    # OWLv2
    owl_model: str = "google/owlv2-base-patch16-ensemble"
    owl_stride: int = 5
    owl_score_threshold: float = 0.35
    owl_shrink_px: int = 5
    custom_owl_queries: tuple[str, ...] | None = None

    # Blur strength
    mosaic_cells: int = 100
    feather: int = 3

    # Compute
    use_gpu: bool = True

    def __post_init__(self) -> None:
        unknown = set(self.categories) - ALLOWED_CATEGORIES
        if unknown:
            raise ValueError(
                f"Unknown blur categories: {sorted(unknown)}; "
                f"allowed={sorted(ALLOWED_CATEGORIES)}"
            )
        if self.owl_stride < 1:
            raise ValueError(f"owl_stride must be >= 1, got {self.owl_stride}")
        if not (0.0 <= self.owl_score_threshold <= 1.0):
            raise ValueError(
                f"owl_score_threshold must be in [0,1], got {self.owl_score_threshold}"
            )
        if not (0.0 <= self.min_face_confidence <= 1.0):
            raise ValueError(
                f"min_face_confidence must be in [0,1], got {self.min_face_confidence}"
            )

    @property
    def blur_faces(self) -> bool:
        return self.do_faces and "face" in self.categories

    @property
    def owl_categories(self) -> tuple[str, ...]:
        return tuple(c for c in self.categories if c != "face")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DetectionRecord:
    """One blurred region in the output video.

    ``t_ms`` is the inclusive start timestamp of the source frame. For
    detections reused via the OWL stride cache, ``from_cache`` is True so
    callers can downweight them during any downstream post-processing.
    """

    frame_idx: int
    t_ms: int
    category: str
    label: str
    confidence: float
    bbox_norm: tuple[float, float, float, float]
    from_cache: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_idx": self.frame_idx,
            "t_ms": self.t_ms,
            "category": self.category,
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "bbox_norm": [round(v, 5) for v in self.bbox_norm],
            "from_cache": self.from_cache,
        }


@dataclass
class BlurResult:
    """Aggregated output of one ``BlurPipeline.process_video`` call."""

    input_path: str
    output_path: str
    fps: float
    width: int
    height: int
    frame_count: int
    total_ms: float
    owl_infer_ms: float
    owl_infer_frames: int
    detections: list[DetectionRecord] = field(default_factory=list)
    config: BlurConfig | None = None

    def summary(self) -> dict[str, int]:
        """Category → count of detections (unique frames blurred)."""
        counts: dict[str, int] = {}
        for d in self.detections:
            counts[d.category] = counts.get(d.category, 0) + 1
        return counts

    def to_manifest(self) -> dict[str, Any]:
        """Serialize to the on-disk manifest JSON shape.

        The worker uploads this verbatim to S3; downstream search
        indexing consumes it without re-reading the video.
        """
        return {
            "schema_version": "1",
            "input_path": self.input_path,
            "output_path": self.output_path,
            "video": {
                "fps": round(self.fps, 3),
                "width": self.width,
                "height": self.height,
                "frame_count": self.frame_count,
            },
            "timing": {
                "total_ms": round(self.total_ms, 1),
                "owl_infer_ms": round(self.owl_infer_ms, 1),
                "owl_infer_frames": self.owl_infer_frames,
                "avg_fps": round(
                    self.frame_count * 1000.0 / self.total_ms, 2
                ) if self.total_ms > 0 else 0.0,
            },
            "config": self.config.to_dict() if self.config else None,
            "summary": self.summary(),
            "detections": [d.to_dict() for d in self.detections],
        }


def iter_category_counts(detections: Iterable[DetectionRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for d in detections:
        counts[d.category] = counts.get(d.category, 0) + 1
    return counts


__all__ = [
    "ALLOWED_CATEGORIES",
    "BlurConfig",
    "BlurResult",
    "DetectionRecord",
    "iter_category_counts",
]
