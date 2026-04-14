"""End-to-end smoke test of BlurPipeline with real I/O but no ML.

Uses a synthetic video (generated in-test) and a canned detector stub so
the test runs in CI without GPU, without transformers, without
insightface. Verifies:

  * the pipeline reads the input, writes an output mp4 with the same
    frame count and dimensions
  * canned detections are blurred in the output
  * the stride cache reuses detections between inference frames
  * the manifest summary matches what was injected
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from heimdex_media_pipelines.blur import BlurConfig, BlurPipeline


class _StubDetector:
    """Returns a single deterministic detection for every call."""

    def __init__(self) -> None:
        self.calls = 0

    def detect(self, frame_bgr: Any, queries: list[str]) -> list[dict[str, Any]]:
        self.calls += 1
        return [
            {
                "bbox": [0.25, 0.25, 0.75, 0.75],
                "label": "korean license plate",
                "confidence": 0.9,
            }
        ]


def test_pipeline_processes_synthetic_video(synthetic_video: Path, tmp_path: Path):
    out_path = tmp_path / "out.mp4"
    config = BlurConfig(
        do_faces=False,            # skip face path — no insightface in unit tests
        do_owl=True,
        categories=("license_plate",),
        owl_stride=5,
        owl_score_threshold=0.3,
    )
    detector = _StubDetector()
    pipeline = BlurPipeline(config, detector=detector)
    result = pipeline.process_video(synthetic_video, out_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 0

    # Output must have the same frame count + dimensions as input.
    cap = cv2.VideoCapture(str(out_path))
    try:
        assert cap.isOpened()
        out_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        out_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()

    assert out_frames == result.frame_count
    assert (out_w, out_h) == (result.width, result.height)

    # 15 frames with stride 5 → detector runs on frames 0, 5, 10  → 3 calls.
    # The manifest records one DetectionRecord per *fresh* inference
    # (matches the senior engineer's validated behavior); cached-frame
    # blurs still happen but don't inflate the manifest.
    assert detector.calls == 3
    assert len(result.detections) == 3
    assert all(not d.from_cache for d in result.detections)
    assert all(d.category == "license_plate" for d in result.detections)
    # Fresh detections land on exactly the stride frames 0, 5, 10.
    assert [d.frame_idx for d in result.detections] == [0, 5, 10]

    summary = result.summary()
    assert summary == {"license_plate": 3}

    manifest = result.to_manifest()
    assert manifest["schema_version"] == "2"
    assert manifest["video"]["frame_count"] == 15
    assert manifest["summary"] == {"license_plate": 3}
    assert len(manifest["detections"]) == 3
    # v2 manifests include a placeholder mask_s3_keys field that the
    # worker populates after uploading each BlurResult.mask_paths entry.
    assert manifest["mask_s3_keys"] is None
    # No masks requested — mask_paths stays empty.
    assert result.mask_paths == {}


def test_pipeline_no_detector_no_owl(synthetic_video: Path, tmp_path: Path):
    """With do_owl=False and no face detector, the pipeline is a pure
    copy-through."""
    out_path = tmp_path / "copy.mp4"
    config = BlurConfig(do_faces=False, do_owl=False, categories=("face",))
    pipeline = BlurPipeline(config)
    result = pipeline.process_video(synthetic_video, out_path)

    assert out_path.exists()
    assert result.frame_count == 15
    assert result.detections == []
    assert result.owl_infer_frames == 0


def test_pipeline_missing_input(tmp_path: Path):
    pipeline = BlurPipeline(BlurConfig(do_faces=False, do_owl=False))
    with pytest.raises(FileNotFoundError):
        pipeline.process_video(tmp_path / "nope.mp4", tmp_path / "out.mp4")


def test_pipeline_below_threshold_filtered(synthetic_video: Path, tmp_path: Path):
    """Low-confidence detections below the threshold get dropped."""
    class _LowConfDetector:
        def detect(self, frame, queries):
            return [
                {"bbox": [0.1, 0.1, 0.2, 0.2], "label": "credit card", "confidence": 0.1},
            ]

    config = BlurConfig(
        do_faces=False, do_owl=True, categories=("card_object",),
        owl_stride=5, owl_score_threshold=0.5,
    )
    pipeline = BlurPipeline(config, detector=_LowConfDetector())
    result = pipeline.process_video(synthetic_video, tmp_path / "out.mp4")
    # Detector was called but nothing passed threshold → no detections recorded.
    assert result.detections == []
