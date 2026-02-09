"""Keyframe extraction using ffmpeg seek + single frame capture."""

from __future__ import annotations

import os
import subprocess
from typing import List, Optional

from heimdex_media_contracts.scenes.schemas import SceneBoundary


def extract_keyframe(
    video_path: str,
    timestamp_ms: int,
    out_path: str,
    ffmpeg_bin: Optional[str] = None,
) -> str:
    """Extract a single JPEG frame at the given timestamp.

    Args:
        video_path: Path to video file.
        timestamp_ms: Seek position in milliseconds.
        out_path: Output JPEG file path.
        ffmpeg_bin: Override ffmpeg binary path.

    Returns:
        The ``out_path`` on success.

    Raises:
        RuntimeError: If ffmpeg fails or output file is empty/missing.
    """
    ffmpeg = ffmpeg_bin or "ffmpeg"
    ts_s = timestamp_ms / 1000.0

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    cmd = [
        ffmpeg, "-y",
        "-ss", f"{ts_s:.3f}",
        "-i", video_path,
        "-frames:v", "1",
        "-q:v", "2",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg keyframe extraction failed at {timestamp_ms}ms: "
            f"{result.stderr.strip()}"
        )

    if not os.path.isfile(out_path) or os.path.getsize(out_path) == 0:
        raise RuntimeError(
            f"Keyframe output missing or empty: {out_path}"
        )

    return out_path


def extract_all_keyframes(
    video_path: str,
    scenes: List[SceneBoundary],
    out_dir: str,
    ffmpeg_bin: Optional[str] = None,
) -> List[str]:
    """Extract one keyframe per scene, writing JPEGs to ``out_dir``.

    Each output file is named ``{scene_id}.jpg``.  The ``keyframe_path``
    attribute on each SceneBoundary is updated in-place.

    Returns:
        List of output file paths in scene order.
    """
    os.makedirs(out_dir, exist_ok=True)
    paths: List[str] = []

    for scene in scenes:
        out_path = os.path.join(out_dir, f"{scene.scene_id}.jpg")
        extract_keyframe(
            video_path,
            scene.keyframe_timestamp_ms,
            out_path,
            ffmpeg_bin=ffmpeg_bin,
        )
        scene.keyframe_path = out_path
        paths.append(out_path)

    return paths
