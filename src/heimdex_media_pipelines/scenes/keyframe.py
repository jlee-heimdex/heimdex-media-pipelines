"""Keyframe extraction using ffmpeg seek + single frame capture."""

from __future__ import annotations

import logging
import os
import subprocess
import time
from typing import List, Optional

from heimdex_media_contracts.scenes.schemas import SceneBoundary

logger = logging.getLogger(__name__)

_BATCH_KEYFRAMES_ENV = "HEIMDEX_BATCH_KEYFRAMES"


def _batch_keyframes_enabled() -> bool:
    """Return True when the batch-keyframe feature flag is active."""
    return os.getenv(_BATCH_KEYFRAMES_ENV, "").lower() in ("true", "1")


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
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
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


def _extract_batch(
    video_path: str,
    scenes: List[SceneBoundary],
    out_dir: str,
    ffmpeg_bin: Optional[str] = None,
) -> List[str]:
    """Extract all keyframes in a single ffmpeg invocation.

    Builds a ``select`` filter expression that matches exact timestamps,
    writes numbered temporary files, then renames them to ``{scene_id}.jpg``.

    Raises:
        RuntimeError: If ffmpeg fails or any expected output file is missing.
    """
    ffmpeg = ffmpeg_bin or "ffmpeg"
    os.makedirs(out_dir, exist_ok=True)

    # Sort by timestamp for deterministic frame-to-scene mapping.
    sorted_scenes = sorted(scenes, key=lambda s: s.keyframe_timestamp_ms)

    select_parts: List[str] = []
    for scene in sorted_scenes:
        ts_s = scene.keyframe_timestamp_ms / 1000.0
        select_parts.append(f"eq(t\\,{ts_s:.3f})")
    select_expr = "+".join(select_parts)

    tmp_pattern = os.path.join(out_dir, "_batch_%04d.jpg")
    cmd = [
        ffmpeg, "-y",
        "-i", video_path,
        "-vf", f"select='{select_expr}'",
        "-vsync", "0",
        "-q:v", "2",
        tmp_pattern,
    ]

    timeout = max(120 * len(scenes), 120)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(
            f"Batch keyframe extraction failed: {result.stderr.strip()}"
        )

    paths: List[str] = []
    for i, scene in enumerate(sorted_scenes):
        tmp_path = os.path.join(out_dir, f"_batch_{i + 1:04d}.jpg")
        final_path = os.path.join(out_dir, f"{scene.scene_id}.jpg")
        if not os.path.isfile(tmp_path) or os.path.getsize(tmp_path) == 0:
            raise RuntimeError(
                f"Batch keyframe missing or empty for scene "
                f"{scene.scene_id}: expected {tmp_path}"
            )
        os.replace(tmp_path, final_path)
        scene.keyframe_path = final_path
        paths.append(final_path)

    return paths


def _cleanup_batch_temps(out_dir: str) -> None:
    """Remove leftover ``_batch_*.jpg`` temp files after a failed batch run."""
    try:
        for entry in os.listdir(out_dir):
            if entry.startswith("_batch_") and entry.endswith(".jpg"):
                os.remove(os.path.join(out_dir, entry))
    except OSError:
        pass


def extract_all_keyframes(
    video_path: str,
    scenes: List[SceneBoundary],
    out_dir: str,
    ffmpeg_bin: Optional[str] = None,
) -> List[str]:
    """Extract one keyframe per scene, writing JPEGs to ``out_dir``.

    Each output file is named ``{scene_id}.jpg``.  The ``keyframe_path``
    attribute on each SceneBoundary is updated in-place.

    When ``HEIMDEX_BATCH_KEYFRAMES=true``, uses a single ffmpeg process
    for all frames.  Falls back to sequential extraction on any failure.

    Returns:
        List of output file paths in scene order.
    """
    if not scenes:
        return []

    t0 = time.monotonic()
    mode = "sequential"

    # --- batch path (feature-flagged) ---
    if _batch_keyframes_enabled() and len(scenes) >= 2:
        try:
            batch_paths = _extract_batch(
                video_path, scenes, out_dir, ffmpeg_bin=ffmpeg_bin,
            )
            mode = "batch"
            elapsed = time.monotonic() - t0
            logger.info(
                "keyframe_extraction mode=%s scenes=%d elapsed_s=%.3f",
                mode, len(scenes), elapsed,
            )
            # Return paths in original scene order (batch sorts internally).
            path_by_id = {
                os.path.basename(p).removesuffix(".jpg"): p
                for p in batch_paths
            }
            return [path_by_id[s.scene_id] for s in scenes]
        except Exception:
            logger.warning(
                "Batch keyframe extraction failed, falling back to sequential",
                exc_info=True,
            )
            _cleanup_batch_temps(out_dir)
            # Reset keyframe_path so sequential path starts clean.
            for scene in scenes:
                scene.keyframe_path = None
            t0 = time.monotonic()

    # --- sequential path (default / fallback) ---
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

    elapsed = time.monotonic() - t0
    logger.info(
        "keyframe_extraction mode=%s scenes=%d elapsed_s=%.3f",
        mode, len(scenes), elapsed,
    )
    return paths
