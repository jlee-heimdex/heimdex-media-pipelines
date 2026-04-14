"""Unit tests for :mod:`heimdex_media_pipelines.blur.masks`.

Scope:

* ``ProgressThrottler`` — debounces by time AND pct, always flushes on
  phase change, swallows callback exceptions.
* ``CategoryMaskWriter`` — opens/closes an ffmpeg subprocess, rejects
  frames of the wrong dtype/shape, surfaces ffmpeg stderr on failure.
* ``CategoryMaskAggregator`` — end-to-end: paint N frames with known
  bboxes, decode the resulting FFV1 MKVs, verify every frame has white
  pixels exactly where bboxes were supplied.
* ``BlurPipeline`` integration — running the pipeline with
  ``emit_masks=True`` populates ``BlurResult.mask_paths`` and produces
  one mask video per category with the right frame count.

Every test skips cleanly if ``ffmpeg`` is not on PATH — these tests
must pass on any developer's laptop without special setup.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from heimdex_media_pipelines.blur import (
    BlurConfig,
    BlurPipeline,
    BlurProgressEvent,
    CategoryMaskAggregator,
    CategoryMaskWriter,
    ProgressThrottler,
)


requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="ffmpeg not installed — skipping mask emission tests",
)


# ---------- ProgressThrottler ----------


def test_throttler_emits_first_event_immediately():
    seen: list[BlurProgressEvent] = []
    t = ProgressThrottler(seen.append, min_interval_s=0.0, min_pct_delta=0.0)
    t.emit("detecting", 5.0)
    assert len(seen) == 1
    assert seen[0].phase == "detecting"
    assert seen[0].progress_pct == 5.0


def test_throttler_suppresses_small_deltas_in_same_phase():
    seen: list[BlurProgressEvent] = []
    t = ProgressThrottler(seen.append, min_interval_s=0.0, min_pct_delta=5.0)
    t.emit("detecting", 10.0)   # flushes (phase change from None)
    t.emit("detecting", 11.0)   # below 5pt threshold → suppressed
    t.emit("detecting", 15.0)   # exactly 5pt delta → flushes
    assert [e.progress_pct for e in seen] == [10.0, 15.0]


def test_throttler_always_flushes_on_phase_change():
    seen: list[BlurProgressEvent] = []
    t = ProgressThrottler(
        seen.append, min_interval_s=3600.0, min_pct_delta=1000.0,
    )
    t.emit("initializing", 0.0)     # flushes (first)
    t.emit("initializing", 0.1)     # suppressed
    t.emit("detecting", 0.2)        # flushes (phase change)
    t.emit("finalizing", 100.0)     # flushes (phase change)
    assert [e.phase for e in seen] == ["initializing", "detecting", "finalizing"]


def test_throttler_time_gate_blocks_rapid_emits():
    """Both the pct delta AND the time interval must be satisfied to
    fire. A tight loop hammering emit() with large pct jumps still
    gets throttled down to one event per ``min_interval_s``.
    """
    seen: list[BlurProgressEvent] = []
    t = ProgressThrottler(seen.append, min_interval_s=0.1, min_pct_delta=1.0)
    t.emit("detecting", 0.0)        # fires (phase change from None)
    t.emit("detecting", 50.0)       # pct gate open, time gate closed → suppressed
    t.emit("detecting", 80.0)       # same — no time has passed
    time.sleep(0.12)
    t.emit("detecting", 90.0)       # both gates open → fires
    assert [e.progress_pct for e in seen] == [0.0, 90.0]


def test_throttler_swallows_callback_exceptions():
    """A broken callback must never crash the pipeline."""

    def boom(_evt: BlurProgressEvent) -> None:
        raise RuntimeError("network down")

    t = ProgressThrottler(boom, min_interval_s=0.0, min_pct_delta=0.0)
    t.emit("detecting", 10.0)  # must not raise
    t.emit("encoding", 50.0)


def test_throttler_no_callback_is_noop():
    t = ProgressThrottler(None)
    t.emit("detecting", 10.0)  # must not raise


# ---------- CategoryMaskWriter ----------


@requires_ffmpeg
def test_mask_writer_rejects_wrong_dtype(tmp_path: Path):
    w = CategoryMaskWriter(
        category="face",
        output_path=tmp_path / "face.mkv",
        width=32, height=24, fps=10.0,
        ffmpeg_binary=shutil.which("ffmpeg") or "ffmpeg",
    )
    w.open()
    try:
        with pytest.raises(ValueError):
            w.write_frame(np.zeros((24, 32), dtype=np.float32))
    finally:
        # Close cleanly even though we never wrote a frame — the
        # subprocess needs to be reaped.
        try:
            w.close()
        except Exception:
            pass


@requires_ffmpeg
def test_mask_writer_rejects_wrong_shape(tmp_path: Path):
    w = CategoryMaskWriter(
        category="face",
        output_path=tmp_path / "face.mkv",
        width=32, height=24, fps=10.0,
        ffmpeg_binary=shutil.which("ffmpeg") or "ffmpeg",
    )
    w.open()
    try:
        with pytest.raises(ValueError):
            w.write_frame(np.zeros((10, 10), dtype=np.uint8))
    finally:
        try:
            w.close()
        except Exception:
            pass


# ---------- CategoryMaskAggregator end-to-end ----------


def _decode_mask_frames(path: Path) -> list[np.ndarray]:
    """Read every frame of an FFV1 mask MKV into uint8 grayscale."""
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened(), f"failed to open mask video {path}"
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            # cv2 decodes as BGR even for gray sources; collapse.
            if frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame)
    finally:
        cap.release()
    return frames


@requires_ffmpeg
def test_aggregator_writes_expected_white_regions(tmp_path: Path):
    width, height, fps = 64, 48, 10.0
    categories = ("face", "license_plate")
    plan = [
        # frame_idx → {category: [(x1,y1,x2,y2), ...]}
        {"face": [(4, 4, 20, 20)]},                       # frame 0
        {},                                               # frame 1 (blank)
        {"license_plate": [(30, 10, 60, 30)]},            # frame 2
        {                                                 # frame 3 (both)
            "face": [(0, 0, 10, 10)],
            "license_plate": [(40, 30, 60, 48)],
        },
    ]

    with CategoryMaskAggregator(
        mask_dir=tmp_path,
        width=width, height=height, fps=fps,
        categories=categories,
    ) as agg:
        for regions in plan:
            agg.paint(regions)  # type: ignore[arg-type]

    paths = agg.mask_paths
    assert set(paths.keys()) == {"face", "license_plate"}
    for category, path in paths.items():
        assert path.exists(), f"{category} mask missing"
        assert path.suffix == ".mkv"
        assert path.stat().st_size > 0

    face_frames = _decode_mask_frames(paths["face"])
    plate_frames = _decode_mask_frames(paths["license_plate"])
    assert len(face_frames) == len(plan)
    assert len(plate_frames) == len(plan)

    # Frame 0: face rect filled, plate blank.
    assert face_frames[0][10, 10] > 200
    assert plate_frames[0].max() == 0

    # Frame 1: both blank.
    assert face_frames[1].max() == 0
    assert plate_frames[1].max() == 0

    # Frame 2: plate rect filled, face blank.
    assert face_frames[2].max() == 0
    assert plate_frames[2][20, 45] > 200

    # Frame 3: both filled at the planted locations.
    assert face_frames[3][5, 5] > 200
    assert plate_frames[3][40, 50] > 200
    # And empty at the *other* category's region (keeps layers disjoint).
    assert face_frames[3][40, 50] == 0
    assert plate_frames[3][5, 5] == 0


@requires_ffmpeg
def test_aggregator_dedupes_categories(tmp_path: Path):
    with CategoryMaskAggregator(
        mask_dir=tmp_path,
        width=16, height=16, fps=5.0,
        categories=("face", "face", "license_plate"),
    ) as agg:
        agg.paint({"face": [(0, 0, 4, 4)]})
    # Duplicate category collapsed to a single writer + single path.
    assert set(agg.categories) == {"face", "license_plate"}
    assert len(agg.mask_paths) == 2


@requires_ffmpeg
def test_aggregator_requires_ffmpeg(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "heimdex_media_pipelines.blur.masks.shutil.which",
        lambda *_a, **_kw: None,
    )
    with pytest.raises(RuntimeError, match="ffmpeg"):
        with CategoryMaskAggregator(
            mask_dir=tmp_path, width=8, height=8, fps=1.0, categories=("face",),
        ):
            pass


@requires_ffmpeg
def test_aggregator_rejects_paint_after_close(tmp_path: Path):
    with CategoryMaskAggregator(
        mask_dir=tmp_path, width=8, height=8, fps=1.0, categories=("face",),
    ) as agg:
        agg.paint({"face": [(0, 0, 4, 4)]})
    with pytest.raises(RuntimeError):
        agg.paint({"face": [(0, 0, 4, 4)]})


def test_aggregator_rejects_empty_category_list(tmp_path: Path):
    with pytest.raises(ValueError):
        CategoryMaskAggregator(
            mask_dir=tmp_path, width=8, height=8, fps=1.0, categories=(),
        )


# ---------- pipeline integration (emit_masks=True) ----------


class _StaticPlateDetector:
    """Returns a single deterministic detection for every call."""

    def detect(self, frame_bgr: Any, queries: list[str]) -> list[dict[str, Any]]:
        return [
            {
                "bbox": [0.25, 0.25, 0.75, 0.75],
                "label": "korean license plate",
                "confidence": 0.9,
            }
        ]


@requires_ffmpeg
def test_pipeline_emits_masks_when_enabled(synthetic_video: Path, tmp_path: Path):
    mask_dir = tmp_path / "masks"
    events: list[BlurProgressEvent] = []

    config = BlurConfig(
        do_faces=False,
        do_owl=True,
        categories=("license_plate",),
        owl_stride=5,
        owl_score_threshold=0.3,
        emit_masks=True,
        mask_dir=mask_dir,
        progress_callback=events.append,
    )
    pipeline = BlurPipeline(config, detector=_StaticPlateDetector())
    result = pipeline.process_video(synthetic_video, tmp_path / "out.mp4")

    # Mask artifact contract.
    assert "license_plate" in result.mask_paths
    mask_path = result.mask_paths["license_plate"]
    assert mask_path.exists()
    assert mask_path.suffix == ".mkv"
    assert mask_path.parent == mask_dir

    # Frame count parity — mask must be frame-locked to source.
    decoded = _decode_mask_frames(mask_path)
    assert len(decoded) == result.frame_count

    # OWL fires on stride frames (0, 5, 10). Each fresh detection also
    # leaves the detection cached for the next stride, so every frame
    # in the loop has some detection. The mask must be non-empty on
    # every stride frame at least.
    for frame_idx in (0, 5, 10):
        frame = decoded[frame_idx]
        assert frame.max() > 200, f"mask frame {frame_idx} should be non-empty"

    # Progress callback got real events, including the phase transitions.
    phases_seen = {e.phase for e in events}
    assert "initializing" in phases_seen
    assert "detecting" in phases_seen
    assert "finalizing" in phases_seen
    # Last event reaches the finalizing ceiling.
    assert events[-1].phase == "finalizing"
    assert events[-1].progress_pct >= 95.0


@requires_ffmpeg
def test_pipeline_emit_masks_requires_mask_dir():
    with pytest.raises(ValueError, match="mask_dir"):
        BlurConfig(
            do_faces=False, do_owl=True,
            categories=("license_plate",),
            emit_masks=True,
        )
