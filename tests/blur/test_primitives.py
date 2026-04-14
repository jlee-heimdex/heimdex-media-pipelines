"""Blur primitives — verify the ROI is actually modified and edges respected."""

from __future__ import annotations

import numpy as np

from heimdex_media_pipelines.blur.primitives import (
    apply_mosaic_blur,
    apply_mosaic_blur_norm,
)


def test_apply_mosaic_blur_modifies_roi():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:] = 255  # white
    # Inject a high-frequency checkerboard inside the ROI so the blur has
    # something to smooth.
    for y in range(20, 60):
        for x in range(20, 60):
            if (x + y) % 2 == 0:
                frame[y, x] = (0, 0, 0)

    roi_before = frame[20:60, 20:60].copy()
    apply_mosaic_blur(frame, 20, 20, 40, 40)
    roi_after = frame[20:60, 20:60]

    assert not np.array_equal(roi_before, roi_after)
    # Areas outside the ROI must be untouched.
    assert np.array_equal(frame[:20, :], np.full((20, 100, 3), 255, dtype=np.uint8))
    assert np.array_equal(frame[60:, :], np.full((40, 100, 3), 255, dtype=np.uint8))


def test_apply_mosaic_blur_handles_edge_crop():
    frame = np.full((100, 100, 3), 200, dtype=np.uint8)
    apply_mosaic_blur(frame, -10, -10, 30, 30)  # partial ROI, clipped to (0,0,20,20)
    assert frame[5, 5].tolist() != [0, 0, 0]  # no crash, frame still valid


def test_apply_mosaic_blur_noop_on_empty_bbox():
    frame = np.full((50, 50, 3), 123, dtype=np.uint8)
    before = frame.copy()
    apply_mosaic_blur(frame, 10, 10, 0, 0)
    apply_mosaic_blur(frame, 60, 60, 5, 5)  # fully outside
    assert np.array_equal(frame, before)


def test_apply_mosaic_blur_norm_coordinates():
    frame = np.full((100, 200, 3), 255, dtype=np.uint8)
    # Hard vertical edge inside the target region: left half white, right half black.
    frame[30:70, 100:140] = (0, 0, 0)

    apply_mosaic_blur_norm(frame, [0.3, 0.3, 0.7, 0.7])
    # Corners must be untouched.
    assert frame[0, 0].tolist() == [255, 255, 255]
    assert frame[99, 199].tolist() == [255, 255, 255]
    # The edge at column 100 must have been smoothed — it's no longer
    # a pure 255 → 0 step.
    left_px = int(frame[50, 95, 0])
    right_px = int(frame[50, 105, 0])
    assert left_px < 255
    assert right_px > 0
