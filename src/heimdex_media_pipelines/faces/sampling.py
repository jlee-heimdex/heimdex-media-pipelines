"""Face detection frame sampling.

Pure timestamp math is delegated to ``heimdex_media_contracts.faces.sampling``.
The video-duration probing helper (``_video_duration_s``) lives here because it
requires ``cv2``, which is NOT allowed in the contracts package.

Public API:
    sample_timestamps(video_path, fps, scene_boundaries_s, boundary_window_s)
"""

import os
from typing import Iterable, List, Optional

import cv2

from heimdex_media_contracts.faces.sampling import (  # noqa: F401
    _dedupe_sorted,
    sample_timestamps as _sample_timestamps_pure,
)


def _video_duration_s(video_path: str) -> float:
    """Probe video duration using OpenCV.  Requires ``cv2``."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    if fps <= 0.0 or frame_count <= 0.0:
        return 0.0
    return float(frame_count) / float(fps)


def sample_timestamps(
    video_path: str,
    fps: float = 1.0,
    scene_boundaries_s: Optional[Iterable[float]] = None,
    boundary_window_s: float = 0.5,
) -> List[float]:
    """Return a list of timestamps (seconds) sampled at the desired fps.

    If scene boundaries are provided, add extra samples around each boundary.

    This is a thin wrapper that probes the video duration with ``cv2`` and then
    delegates the pure math to ``heimdex_media_contracts.faces.sampling``.
    """
    if fps <= 0:
        raise ValueError("fps must be > 0")
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    duration_s = _video_duration_s(video_path)
    if not duration_s or duration_s <= 0:
        return []

    return _sample_timestamps_pure(
        duration_s=duration_s,
        fps=fps,
        scene_boundaries_s=scene_boundaries_s,
        boundary_window_s=boundary_window_s,
    )
