"""Orchestrator for multi-signal scene splitting.

Combines visual cut detection (ffmpeg) with speech-aware signals
(pauses, speaker turns) through the pure combiner in contracts.

This is the recommended entry point when speech data is available.
Falls back to visual-only detection when it isn't.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional

from heimdex_media_contracts.scenes.combiner import combine_signals
from heimdex_media_contracts.scenes.presets import resolve_config
from heimdex_media_contracts.scenes.schemas import SceneBoundary
from heimdex_media_contracts.scenes.splitting import SplitConfig

from heimdex_media_pipelines.scenes.detector import _probe_duration_ms, detect_scenes
from heimdex_media_pipelines.scenes.signals import (
    extract_speech_pauses,
    extract_speaker_turns,
)


def split_scenes(
    video_path: str,
    video_id: str,
    config: SplitConfig | None = None,
    preset: str | None = None,
    overrides: dict | None = None,
    speech_result_path: str | None = None,
    speech_segments: list[dict] | None = None,
    ffmpeg_bin: str | None = None,
) -> List[SceneBoundary]:
    """Detect scene boundaries using visual cuts + speech signals.

    This wraps :func:`detect_scenes` for visual cut extraction, then
    enriches the result with speech-aware splitting via the contracts
    combiner when speech data is available.

    Config resolution order:
      1. Explicit ``config`` parameter (highest priority).
      2. ``preset`` + ``overrides`` resolved via :func:`resolve_config`.
      3. Default ``SplitConfig()`` (lowest priority).

    Speech data can be provided as:
      - ``speech_result_path``: path to a JSON file with a ``"segments"`` key.
      - ``speech_segments``: pre-loaded list of segment dicts.
      If both are given, ``speech_segments`` takes precedence.

    Falls back to visual-only detection (identical to :func:`detect_scenes`)
    when no speech data is available or ``speech_split_enabled`` is False.

    Args:
        video_path: Path to video file.
        video_id: Identifier for scene_id construction.
        config: Explicit splitting config (overrides preset).
        preset: Preset name (e.g. "default", "fine", "coarse").
        overrides: Per-field overrides on top of the preset.
        speech_result_path: Path to STT result JSON file.
        speech_segments: Pre-loaded speech segments (list of dicts).
        ffmpeg_bin: Override ffmpeg binary path.

    Returns:
        Sorted list of :class:`SceneBoundary` objects.
    """
    cfg = config or resolve_config(preset, overrides)

    # --- Load speech data if needed ---
    segments = speech_segments
    if segments is None and speech_result_path:
        segments = _load_speech_segments(speech_result_path)

    has_speech = bool(segments) and cfg.speech_split_enabled

    if not has_speech:
        # Fast path: visual-only, identical to legacy detect_scenes()
        return detect_scenes(
            video_path=video_path,
            video_id=video_id,
            threshold=cfg.visual_threshold,
            min_scene_duration_ms=cfg.min_scene_duration_ms,
            max_scene_duration_ms=cfg.max_scene_duration_ms,
            ffmpeg_bin=ffmpeg_bin,
        )

    # --- Multi-signal path ---
    total_duration_ms = _probe_duration_ms(video_path)
    if total_duration_ms <= 0:
        return []

    # Step 1: extract visual cuts (raw timestamps, not SceneBoundary objects)
    visual_cuts_ms = _extract_visual_cuts(
        video_path, cfg.visual_threshold, ffmpeg_bin,
    )

    # Step 2: extract speech signals
    speech_pauses = extract_speech_pauses(
        segments, min_gap_ms=cfg.speech_pause_min_gap_ms,
    )
    speaker_turns = extract_speaker_turns(segments)

    # Step 3: combine all signals via pure combiner
    boundaries = combine_signals(
        visual_cuts_ms=visual_cuts_ms,
        speech_pauses=speech_pauses,
        speaker_turns=speaker_turns,
        total_duration_ms=total_duration_ms,
        config=cfg,
    )

    # Step 4: build SceneBoundary objects
    return _boundaries_to_scenes(boundaries, video_id, total_duration_ms)


def _load_speech_segments(path: str) -> list[dict]:
    """Load speech segments from a JSON file."""
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        data = json.load(f)
    return data.get("segments", [])


def _extract_visual_cuts(
    video_path: str,
    threshold: float,
    ffmpeg_bin: str | None,
) -> list[int]:
    """Run ffmpeg scene detection and return raw cut timestamps in ms.

    Reuses the existing detect_scenes logic but extracts only the
    cut timestamps without building SceneBoundary objects.
    """
    import re
    import subprocess

    ffmpeg = ffmpeg_bin or "ffmpeg"
    cmd = [
        ffmpeg, "-i", video_path,
        "-filter:v", f"select='gt(scene,{threshold})',showinfo",
        "-f", "null", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    showinfo_re = re.compile(r"pts_time:([\d.]+)")
    cuts: list[int] = []
    for line in result.stderr.splitlines():
        if "showinfo" not in line:
            continue
        match = showinfo_re.search(line)
        if match:
            ts_ms = int(float(match.group(1)) * 1000)
            if ts_ms > 0:
                cuts.append(ts_ms)

    return sorted(set(cuts))


def _boundaries_to_scenes(
    sorted_boundaries: list[int],
    video_id: str,
    total_duration_ms: int,
) -> List[SceneBoundary]:
    """Convert sorted boundary timestamps to SceneBoundary objects."""
    scenes: List[SceneBoundary] = []

    for i in range(len(sorted_boundaries) - 1):
        start_ms = sorted_boundaries[i]
        end_ms = sorted_boundaries[i + 1]
        if end_ms <= start_ms:
            continue

        keyframe_ts = start_ms + (end_ms - start_ms) // 2
        eof_margin_ms = max(total_duration_ms - 500, 0)
        keyframe_ts = max(start_ms, min(keyframe_ts, eof_margin_ms))

        scene_id = f"{video_id}_scene_{i:03d}"
        scenes.append(SceneBoundary(
            scene_id=scene_id,
            index=i,
            start_ms=start_ms,
            end_ms=end_ms,
            keyframe_timestamp_ms=keyframe_ts,
        ))

    return scenes
