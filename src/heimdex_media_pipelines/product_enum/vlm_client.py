"""VLM (gpt-4o-mini vision) client protocol + production impl.

The :class:`VlmClient` protocol is what the pipeline depends on;
concrete impls handle the OpenAI HTTP call, retry, and JSON-mode
response parsing. Tests inject a :class:`StubVlmClient` (lives in
``tests/product_enum/conftest.py``) so unit tests don't need network.

Output shape mirrors
``heimdex_media_contracts.product.EnumerationDetection`` exactly so
the worker can hand the list straight to the API callback without a
re-shape — but we redefine here as a plain dataclass to keep
heimdex_media_pipelines free of pydantic at import time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol


if TYPE_CHECKING:  # pragma: no cover
    from PIL import Image


@dataclass(frozen=True)
class EnumerationDetection:
    """One product detection produced by the VLM, before clustering.

    Mirrors ``heimdex_media_contracts.product.EnumerationDetection``.
    Multiple detections may merge into one cluster after SigLIP2
    similarity matching.
    """

    keyframe_scene_id: str
    keyframe_frame_idx: int
    label: str
    bbox_xywh: tuple[int, int, int, int]
    confidence: float


@dataclass(frozen=True)
class VlmDetectionBatch:
    """Result for one batched VLM call: detections + the reasoning
    cost (USD) we consumed. The worker accumulates ``cost_usd`` and
    posts it back to the API on each heartbeat for the per-org cap.
    """

    detections: list[EnumerationDetection]
    cost_usd: float
    # Optional opaque metadata for debugging — the prompt version
    # used, the model latency, etc. Don't rely on schema stability.
    debug: dict[str, object] = field(default_factory=dict)


class VlmClient(Protocol):
    """Protocol the pipeline depends on. The production impl wraps
    OpenAI's client + handles JSON-mode + retry + cost accounting.
    Tests provide a deterministic stub.
    """

    def detect_products(
        self,
        *,
        keyframes: list[tuple[str, int, "Image.Image"]],
        system_prompt: str,
        user_prompt: str,
    ) -> VlmDetectionBatch:
        """Run a single batched VLM call.

        ``keyframes`` is a list of ``(scene_id, frame_idx, image)``
        triples — the keyframe id pair is echoed back on each
        detection so the pipeline knows which crop produced it.
        """
        ...
