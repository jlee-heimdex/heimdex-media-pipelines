"""Shared fixtures for blur tests."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest


def _make_synthetic_video(path: Path, *, n_frames: int = 15, fps: float = 5.0,
                          width: int = 160, height: int = 120) -> Path:
    """Write a deterministic synthetic mp4 — no ML models touched."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        pytest.skip("mp4v codec unavailable on this platform")
    try:
        for i in range(n_frames):
            frame = np.full((height, width, 3), (i * 10) % 256, dtype=np.uint8)
            cv2.rectangle(frame, (30, 30), (90, 90), (0, 255, 0), -1)
            cv2.putText(
                frame, f"{i}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
            )
            writer.write(frame)
    finally:
        writer.release()
    return path


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    return _make_synthetic_video(tmp_path / "in.mp4")
