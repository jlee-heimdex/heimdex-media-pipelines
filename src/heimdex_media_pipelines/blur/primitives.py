"""Region blur primitives.

Provides a visually uniform Gaussian blur over rectangular ROIs. The
kernel size scales with ROI dimensions so a small license-plate crop and
a large face crop both look equally obscured to a human reviewer.

These helpers are deliberately framework-agnostic — no torch, no
transformers — so they can run inside any worker that already has
OpenCV + NumPy.
"""

from __future__ import annotations

import math
from typing import Sequence

import cv2
import numpy as np


def apply_mosaic_blur(
    frame: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    *,
    mosaic_cells: int = 12,
    feather: int = 6,
) -> None:
    """Blur the rectangular ROI ``(x, y, w, h)`` in-place.

    ``mosaic_cells`` and ``feather`` are accepted for API compatibility
    with the senior's prototype but a single Gaussian pass has proven
    sufficient in practice — the kernel scales with ROI size so the
    perceived strength is uniform across different bbox sizes.
    """
    del mosaic_cells, feather  # reserved for future pixelation mode

    H, W = frame.shape[:2]
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(W, int(x + w))
    y2 = min(H, int(y + h))
    if x2 <= x1 or y2 <= y1:
        return

    roi = frame[y1:y2, x1:x2]
    rh, rw = roi.shape[:2]

    # Ensure odd, floor 31 — below that the blur is barely perceptible.
    k = max(31, int(max(rw, rh) * 0.4) | 1)
    frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)


def apply_mosaic_blur_norm(
    frame: np.ndarray,
    bbox_norm: Sequence[float],
    *,
    mosaic_cells: int = 12,
    feather: int = 6,
    pad: float = 0.0,
) -> None:
    """Same as :func:`apply_mosaic_blur` for a normalized ``[x1, y1, x2, y2]``
    bbox in ``[0, 1]`` coordinates. ``pad`` expands each side by that
    fraction of the bbox dimension.
    """
    H, W = frame.shape[:2]
    x_min, y_min, x_max, y_max = bbox_norm
    if pad > 0:
        bw = x_max - x_min
        bh = y_max - y_min
        x_min -= bw * pad
        x_max += bw * pad
        y_min -= bh * pad
        y_max += bh * pad
    x1 = int(math.floor(max(0.0, x_min) * W))
    y1 = int(math.floor(max(0.0, y_min) * H))
    x2 = int(math.ceil(min(1.0, x_max) * W))
    y2 = int(math.ceil(min(1.0, y_max) * H))
    apply_mosaic_blur(
        frame, x1, y1, x2 - x1, y2 - y1,
        mosaic_cells=mosaic_cells, feather=feather,
    )


__all__ = ["apply_mosaic_blur", "apply_mosaic_blur_norm"]
