"""OWLv2 open-vocabulary object detector wrapper.

Thin adapter around HuggingFace ``Owlv2ForObjectDetection`` that returns
detections in normalized ``[x1, y1, x2, y2]`` form for easy mosaic-blur
compositing. Import of transformers/torch is lazy so the rest of the
blur package remains importable on machines without the heavy ML stack
(needed for unit tests and the pipeline library doctor command).

A lightweight ``Detector`` protocol is defined so ``BlurPipeline`` can
swap in a canned/mock detector during tests without pulling in
transformers.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Protocol

import cv2

logger = logging.getLogger(__name__)


class Detector(Protocol):
    """Minimal contract for anything that can answer text-guided detection."""

    def detect(
        self,
        frame_bgr: Any,
        queries: list[str],
    ) -> list[dict[str, Any]]:
        ...


class OWLv2Detector:
    """HuggingFace OWLv2 wrapper.

    Loads ``google/owlv2-base-patch16-ensemble`` (or any OWLv2-compatible
    checkpoint) once at construction and exposes ``detect()`` for the
    per-frame loop.

    Device selection defers to the standard pipeline helper
    ``heimdex_media_pipelines.device`` if no explicit device is passed.
    """

    def __init__(
        self,
        model_id: str = "google/owlv2-base-patch16-ensemble",
        device: str | None = None,
        score_threshold: float = 0.35,
    ) -> None:
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        self._score_threshold = score_threshold
        self._model_id = model_id

        logger.info("owlv2_loading", model=model_id, device=device)
        t0 = time.time()
        self._processor = Owlv2Processor.from_pretrained(model_id)
        self._model = Owlv2ForObjectDetection.from_pretrained(model_id).to(device)
        self._model.eval()
        logger.info("owlv2_loaded", elapsed_s=round(time.time() - t0, 2))

    @property
    def device(self) -> str:
        return self._device

    @property
    def model_id(self) -> str:
        return self._model_id

    def detect(
        self,
        frame_bgr: Any,
        queries: list[str],
    ) -> list[dict[str, Any]]:
        """Run text-guided detection on one BGR frame.

        Returns a list of ``{"bbox": [x1,y1,x2,y2] (normalized 0..1),
        "label": str, "confidence": float}`` dicts.
        """
        import torch
        from PIL import Image

        if not queries:
            return []

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        h, w = frame_bgr.shape[:2]

        inputs = self._processor(
            text=[queries],
            images=image,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        target_sizes = torch.tensor([(h, w)], device=self._device)
        results = self._processor.image_processor.post_process_object_detection(
            outputs=outputs,
            threshold=self._score_threshold,
            target_sizes=target_sizes,
        )[0]

        detections: list[dict[str, Any]] = []
        for box, score, label_idx in zip(
            results["boxes"], results["scores"], results["labels"]
        ):
            x1, y1, x2, y2 = box.cpu().tolist()
            label_index = int(label_idx.cpu())
            if 0 <= label_index < len(queries):
                label_text = queries[label_index]
            else:
                label_text = f"class_{label_index}"

            detections.append({
                "bbox": [
                    max(0.0, min(1.0, x1 / w)),
                    max(0.0, min(1.0, y1 / h)),
                    max(0.0, min(1.0, x2 / w)),
                    max(0.0, min(1.0, y2 / h)),
                ],
                "label": label_text.strip(),
                "confidence": float(score.cpu()),
            })

        return detections


__all__ = ["Detector", "OWLv2Detector"]
