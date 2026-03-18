"""ffmpeg subprocess execution for composition rendering.

Takes a CompositionSpec from heimdex-media-contracts, resolves media files,
builds the filter graph (using contracts), and runs the final ffmpeg encode.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass

from heimdex_media_contracts.composition import CompositionSpec, build_filter_graph

logger = logging.getLogger(__name__)


@dataclass
class RenderResult:
    output_path: str
    duration_ms: int
    size_bytes: int
    render_time_ms: int


def extract_clip(
    input_path: str,
    output_path: str,
    start_ms: int,
    end_ms: int,
) -> None:
    """Stream copy clip segment (near-instant, no re-encode).

    Uses -ss before -i for fast seek, -c copy for no re-encode.
    """
    start_s = start_ms / 1000.0
    duration_s = (end_ms - start_ms) / 1000.0

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_s),
        "-t", str(duration_s),
        "-i", input_path,
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(
            f"extract_clip failed (exit {result.returncode}): {result.stderr.strip()}"
        )


def render_composition(
    spec: CompositionSpec,
    media_paths: dict[str, str],
    output_path: str,
    font_dir: str,
    use_gpu: bool = False,
) -> RenderResult:
    """Render a composition spec to a final MP4.

    Steps:
    1. extract_clip() for each scene_clip
    2. build_filter_graph() from heimdex-media-contracts
    3. Run final ffmpeg encode via subprocess.run()
    """
    t0 = time.monotonic()

    with tempfile.TemporaryDirectory(prefix="heimdex_comp_") as tmp_dir:
        # 1. Extract clips
        clip_paths: list[str] = []
        for i, clip in enumerate(spec.scene_clips):
            src = media_paths[clip.video_id]
            clip_path = os.path.join(tmp_dir, f"clip_{i:03d}.mp4")
            extract_clip(src, clip_path, clip.start_ms, clip.end_ms)
            clip_paths.append(clip_path)

        # 2. Build filter graph
        filter_graph = build_filter_graph(
            clips=spec.scene_clips,
            subtitles=spec.subtitles,
            output=spec.output,
            font_dir=font_dir,
        )

        # 3. Build ffmpeg command
        cmd: list[str] = ["ffmpeg", "-y"]

        # Add extracted clips as inputs
        for cp in clip_paths:
            cmd.extend(["-i", cp])

        cmd.extend(["-filter_complex", filter_graph])

        # Determine final output labels
        has_subtitles = len(spec.subtitles) > 0
        video_label = "final" if has_subtitles else f"canvas{len(spec.scene_clips)}"
        cmd.extend(["-map", f"[{video_label}]", "-map", "[aout]"])

        # Encoding preset
        if use_gpu:
            cmd.extend(["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23"])
        else:
            cmd.extend(["-c:v", "libx264", "-preset", "medium", "-crf", "23"])

        # Audio + faststart
        cmd.extend(["-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart"])

        cmd.append(output_path)

        logger.info(
            "composition_render_started",
            extra={
                "num_clips": len(spec.scene_clips),
                "num_subtitles": len(spec.subtitles),
                "output": output_path,
                "use_gpu": use_gpu,
            },
        )

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg render failed (exit {result.returncode}): {result.stderr.strip()}"
            )

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    size_bytes = os.path.getsize(output_path)

    logger.info(
        "composition_render_complete",
        extra={
            "output": output_path,
            "size_bytes": size_bytes,
            "render_time_ms": elapsed_ms,
        },
    )

    return RenderResult(
        output_path=output_path,
        duration_ms=spec.total_duration_ms,
        size_bytes=size_bytes,
        render_time_ms=elapsed_ms,
    )
