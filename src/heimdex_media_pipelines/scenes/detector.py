"""Scene boundary detection using ffmpeg's scenecut filter.

No new pip dependencies — only requires ffmpeg on PATH.
"""

from __future__ import annotations

import re
import subprocess
from typing import List, Optional

from heimdex_media_contracts.scenes.schemas import SceneBoundary


_SHOWINFO_RE = re.compile(r"pts_time:([\d.]+)")

_DEFAULT_THRESHOLD = 0.3


def _probe_duration_ms(video_path: str) -> int:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr.strip()}")
    return int(float(result.stdout.strip()) * 1000)


def detect_scenes(
    video_path: str,
    video_id: str,
    threshold: float = _DEFAULT_THRESHOLD,
    min_scene_duration_ms: int = 500,
    ffmpeg_bin: Optional[str] = None,
) -> List[SceneBoundary]:
    """Detect scene boundaries using ffmpeg's select filter with scene score.

    Args:
        video_path: Path to video file.
        video_id: Identifier used to build scene_id values.
        threshold: Scene change sensitivity (0.0 = every frame, 1.0 = never).
        min_scene_duration_ms: Minimum scene duration; shorter scenes are merged
            into the previous scene.
        ffmpeg_bin: Override ffmpeg binary path.

    Returns:
        Sorted list of ``SceneBoundary`` objects covering the full video duration.
    """
    ffmpeg = ffmpeg_bin or "ffmpeg"

    total_duration_ms = _probe_duration_ms(video_path)
    if total_duration_ms <= 0:
        return []

    cmd = [
        ffmpeg, "-i", video_path,
        "-filter:v", f"select='gt(scene,{threshold})',showinfo",
        "-f", "null", "-",
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=600,
    )

    cut_times_ms: List[int] = []
    for line in result.stderr.splitlines():
        if "showinfo" not in line:
            continue
        match = _SHOWINFO_RE.search(line)
        if match:
            ts_ms = int(float(match.group(1)) * 1000)
            if ts_ms > 0:
                cut_times_ms.append(ts_ms)

    cut_times_ms = sorted(set(cut_times_ms))

    boundaries: List[int] = [0]
    for ts in cut_times_ms:
        if ts - boundaries[-1] >= min_scene_duration_ms:
            boundaries.append(ts)
    boundaries.append(total_duration_ms)

    scenes: List[SceneBoundary] = []
    for i in range(len(boundaries) - 1):
        start_ms = boundaries[i]
        end_ms = boundaries[i + 1]
        if end_ms <= start_ms:
            continue
        keyframe_ts = start_ms + (end_ms - start_ms) // 2
        scene_id = f"{video_id}_scene_{i:03d}"
        scenes.append(SceneBoundary(
            scene_id=scene_id,
            index=i,
            start_ms=start_ms,
            end_ms=end_ms,
            keyframe_timestamp_ms=keyframe_ts,
        ))

    return scenes
