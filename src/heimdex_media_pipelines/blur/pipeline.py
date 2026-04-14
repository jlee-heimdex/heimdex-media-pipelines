"""Frame-by-frame blur pipeline orchestrator.

Reads an input video, runs face detection (SCRFD/Haar) on every frame
and OWLv2 text-guided detection every N frames (stride cache in
between), blurs all matched regions, and writes the result to an output
video. Returns a ``BlurResult`` with a structured detection list that
can be serialized to a manifest JSON.

The frame loop is kept deliberately close to the senior engineer's
prototype so downstream behavior matches the reference outputs he has
already reviewed.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import cv2

from heimdex_media_pipelines.blur.config import (
    BlurConfig,
    BlurResult,
    DetectionRecord,
)
from heimdex_media_pipelines.blur.owlv2 import Detector
from heimdex_media_pipelines.blur.primitives import (
    apply_mosaic_blur,
    apply_mosaic_blur_norm,
)
from heimdex_media_pipelines.blur.queries import (
    DEFAULT_OWL_QUERIES,
    label_to_category,
)

logger = logging.getLogger(__name__)


def _build_owl_queries(
    config: BlurConfig,
) -> tuple[list[str], dict[str, list[str]]]:
    """Return ``(flat_queries, query_map)`` for the given config.

    Precedence:
      1. ``config.custom_owl_queries`` overrides everything (all treated
         as the ``"object"`` fallback category).
      2. Otherwise use ``DEFAULT_OWL_QUERIES`` intersected with
         ``config.owl_categories``.
    """
    if config.custom_owl_queries:
        flat = [q.strip() for q in config.custom_owl_queries if q and q.strip()]
        return flat, {"object": list(flat)}

    query_map: dict[str, list[str]] = {}
    flat: list[str] = []
    for cat in config.owl_categories:
        qs = DEFAULT_OWL_QUERIES.get(cat)
        if not qs:
            continue
        query_map[cat] = list(qs)
        flat.extend(qs)
    return flat, query_map


class BlurPipeline:
    """Orchestrates face + OWLv2 blur over a video.

    Construct once, reuse for many videos. The heavy models (OWLv2,
    SCRFD) are loaded lazily on first ``process_video`` call unless
    ``warm_up`` is called explicitly — worker processes should warm up
    at boot so per-message latency is predictable.
    """

    def __init__(
        self,
        config: BlurConfig | None = None,
        *,
        detector: Detector | None = None,
    ) -> None:
        self._config = config or BlurConfig()
        self._owl: Detector | None = detector
        self._face_detector_name: str | None = None
        self._face_cascade: Any = None
        self._face_scrfd_app: Any = None
        self._face_ready = False

    @property
    def config(self) -> BlurConfig:
        return self._config

    # ----- lifecycle -----

    def warm_up(self) -> None:
        """Pre-load face + OWLv2 models. Safe to call multiple times."""
        if self._config.blur_faces:
            self._ensure_face_detector()
        if self._config.do_owl:
            self._ensure_owl()

    def _ensure_face_detector(self) -> None:
        if self._face_ready:
            return
        from heimdex_media_pipelines.faces.detect import _init_detector

        try:
            name, cascade, scrfd_app = _init_detector(
                self._config.face_detector,
                scrfd_det_size=640,
                scrfd_ctx_id=0 if self._config.use_gpu else -1,
                scrfd_det_thresh=self._config.min_face_confidence,
            )
        except Exception as exc:
            logger.warning(
                "face_detector_init_failed_fallback_haar",
                detector=self._config.face_detector,
                error=str(exc),
            )
            name, cascade, scrfd_app = _init_detector(
                "haar", scrfd_det_size=640, scrfd_ctx_id=-1,
            )
        self._face_detector_name = name
        self._face_cascade = cascade
        self._face_scrfd_app = scrfd_app
        self._face_ready = True

    def _ensure_owl(self) -> None:
        if self._owl is not None:
            return
        from heimdex_media_pipelines.blur.owlv2 import OWLv2Detector

        device = None if self._config.use_gpu else "cpu"
        self._owl = OWLv2Detector(
            model_id=self._config.owl_model,
            device=device,
            score_threshold=self._config.owl_score_threshold,
        )

    # ----- main loop -----

    def process_video(
        self,
        in_path: str | Path,
        out_path: str | Path,
    ) -> BlurResult:
        from heimdex_media_pipelines.faces.detect import _detect_on_frame

        in_path = Path(in_path)
        out_path = Path(out_path)
        if not in_path.exists():
            raise FileNotFoundError(f"Input video not found: {in_path}")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(in_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {in_path}")
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            writer = self._open_writer(out_path, fps, width, height)
            try:
                result = self._run_loop(
                    cap=cap,
                    writer=writer,
                    fps=fps,
                    width=width,
                    height=height,
                    detect_on_frame=_detect_on_frame,
                )
            finally:
                writer.release()
        finally:
            cap.release()

        result.input_path = str(in_path)
        result.output_path = str(out_path)
        result.fps = fps
        result.width = width
        result.height = height
        return result

    # ----- internals -----

    def _open_writer(
        self,
        out_path: Path,
        fps: float,
        width: int,
        height: int,
    ) -> cv2.VideoWriter:
        for codec in ("mp4v", "avc1", "x264"):
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
            if writer.isOpened():
                return writer
            writer.release()
        raise RuntimeError(f"Could not open writer for: {out_path}")

    def _run_loop(
        self,
        *,
        cap: cv2.VideoCapture,
        writer: cv2.VideoWriter,
        fps: float,
        width: int,
        height: int,
        detect_on_frame: Any,
    ) -> BlurResult:
        cfg = self._config

        do_faces = cfg.blur_faces
        if do_faces:
            self._ensure_face_detector()

        do_owl = cfg.do_owl and cfg.owl_categories
        if do_owl:
            self._ensure_owl()
        flat_queries, query_map = _build_owl_queries(cfg)
        if not flat_queries:
            do_owl = False

        cached_owl_dets: list[dict[str, Any]] = []
        detections: list[DetectionRecord] = []

        frame_idx = 0
        owl_total_ms = 0.0
        owl_infer_count = 0
        t_start = time.time()

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            t_ms = int(round(frame_idx * 1000.0 / fps))

            # ---- face pass (every frame) ----
            face_bboxes: list[dict[str, Any]] = []
            if do_faces:
                try:
                    raw = detect_on_frame(
                        frame,
                        self._face_detector_name,
                        20,  # min_size
                        self._face_cascade,
                        self._face_scrfd_app,
                        cfg.min_face_confidence,
                    )
                except Exception as exc:
                    logger.warning("face_detect_failed", frame=frame_idx, error=str(exc))
                    raw = []
                face_bboxes = [
                    b for b in raw
                    if float(b.get("det_conf", 0.0)) >= cfg.min_face_confidence
                ]
                for b in face_bboxes:
                    detections.append(DetectionRecord(
                        frame_idx=frame_idx,
                        t_ms=t_ms,
                        category="face",
                        label="face",
                        confidence=float(b.get("det_conf", 0.0)),
                        bbox_norm=(
                            b["x"] / width,
                            b["y"] / height,
                            (b["x"] + b["w"]) / width,
                            (b["y"] + b["h"]) / height,
                        ),
                        from_cache=False,
                    ))

            # ---- OWL pass (every Nth frame; reuse in between) ----
            # Matches the senior engineer's validated behavior: the
            # manifest records only FRESH detections (one entry per OWL
            # inference call). Cached-frame blurs still happen — they
            # just don't inflate the manifest. A future tracking layer
            # can re-emit dense per-frame records if search indexing
            # needs them.
            if do_owl and self._owl is not None:
                if frame_idx % max(cfg.owl_stride, 1) == 0:
                    t_owl = time.time()
                    try:
                        raw_dets = self._owl.detect(frame, flat_queries)
                    except Exception as exc:
                        logger.warning("owl_detect_failed", frame=frame_idx, error=str(exc))
                        raw_dets = []
                    cached_owl_dets = [
                        d for d in raw_dets
                        if d["confidence"] >= cfg.owl_score_threshold
                    ]
                    owl_total_ms += (time.time() - t_owl) * 1000.0
                    owl_infer_count += 1
                    for det in cached_owl_dets:
                        cat = label_to_category(det["label"], query_map)
                        detections.append(DetectionRecord(
                            frame_idx=frame_idx,
                            t_ms=t_ms,
                            category=cat,
                            label=det["label"],
                            confidence=float(det["confidence"]),
                            bbox_norm=tuple(det["bbox"]),  # type: ignore[arg-type]
                            from_cache=False,
                        ))
                owl_dets = cached_owl_dets
            else:
                owl_dets = []

            # ---- apply blurs ----
            if do_faces:
                for b in face_bboxes:
                    apply_mosaic_blur(
                        frame,
                        b["x"] + cfg.face_shrink_px,
                        b["y"] + cfg.face_shrink_px,
                        b["w"] - cfg.face_shrink_px * 2,
                        b["h"] - cfg.face_shrink_px * 2,
                        mosaic_cells=cfg.mosaic_cells,
                        feather=cfg.feather,
                    )
            if owl_dets:
                sx = cfg.owl_shrink_px / width
                sy = cfg.owl_shrink_px / height
                for det in owl_dets:
                    bbox = det["bbox"]
                    shrunk = [
                        min(bbox[0] + sx, bbox[2]),
                        min(bbox[1] + sy, bbox[3]),
                        max(bbox[2] - sx, bbox[0]),
                        max(bbox[3] - sy, bbox[1]),
                    ]
                    apply_mosaic_blur_norm(
                        frame, shrunk,
                        mosaic_cells=cfg.mosaic_cells,
                        feather=cfg.feather,
                    )

            writer.write(frame)
            frame_idx += 1

        total_ms = (time.time() - t_start) * 1000.0
        return BlurResult(
            input_path="",  # populated by caller
            output_path="",
            fps=fps,
            width=width,
            height=height,
            frame_count=frame_idx,
            total_ms=total_ms,
            owl_infer_ms=owl_total_ms,
            owl_infer_frames=owl_infer_count,
            detections=detections,
            config=cfg,
        )


__all__ = ["BlurPipeline"]
