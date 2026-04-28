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

from heimdex_media_contracts.composition import (
    CompositionSpec,
    build_filter_graph,
    build_overlay_filter_chain,
)

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
    """Extract clip segment with frame-accurate seeking.

    Uses -ss before -i for fast seek, then re-encodes to ensure
    accurate start point (stream copy can only cut at keyframes,
    causing frozen frames at non-keyframe seek positions).
    """
    start_s = start_ms / 1000.0
    duration_s = (end_ms - start_ms) / 1000.0

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_s),
        "-i", input_path,
        "-t", str(duration_s),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac", "-b:a", "128k",
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
    2. Bake each overlay (text or background) to a transparent RGBA PNG via
       heimdex_media_pipelines.composition.overlay_render.bake_overlay_png.
       Effects (italic, underline, rotation, opacity, stroke, shadow w/
       blur+spread) are baked in; ffmpeg only positions and gates timing.
    3. build_filter_graph() from heimdex-media-contracts (clips + legacy
       subtitles via drawtext) + build_overlay_filter_chain() (PNG overlays).
    4. Run final ffmpeg encode via subprocess.run().

    Stream layout for V2 overlays:
      - Clip inputs: indices 0..N-1
      - Overlay PNG inputs: indices N..N+M-1 (each via -loop 1 so the
        single frame is available for the entire enable= window)
      - Last video label routes through canvas{N} → [final] → [vout],
        skipping intermediate labels when subtitles or overlays are absent.
    """
    t0 = time.monotonic()

    overlays = sorted(spec.overlays, key=lambda o: o.layer_index)

    with tempfile.TemporaryDirectory(prefix="heimdex_comp_") as tmp_dir:
        # 1. Extract clips
        clip_paths: list[str] = []
        for i, clip in enumerate(spec.scene_clips):
            src = media_paths[clip.video_id]
            clip_path = os.path.join(tmp_dir, f"clip_{i:03d}.mp4")
            extract_clip(src, clip_path, clip.start_ms, clip.end_ms)
            clip_paths.append(clip_path)

        # 2. Bake overlays — only imports PIL when there's something to bake
        # so consumers without the [composition] extra still render legacy
        # compositions (subtitles only) without an ImportError.
        overlay_png_paths: list[str] = []
        if overlays:
            from heimdex_media_pipelines.composition.overlay_render import (
                bake_overlay_png,
            )

            for i, ov in enumerate(overlays):
                png_path = os.path.join(tmp_dir, f"overlay_{i:03d}.png")
                img = bake_overlay_png(
                    ov,
                    canvas_width=spec.output.width,
                    canvas_height=spec.output.height,
                    font_dir=font_dir,
                )
                img.save(png_path, format="PNG")
                overlay_png_paths.append(png_path)

        # 3. Build filter graph (clips + drawtext subtitles)
        filter_graph = build_filter_graph(
            clips=spec.scene_clips,
            subtitles=spec.subtitles,
            output=spec.output,
            font_dir=font_dir,
        )

        # 3b. Append overlay filter chain (V2 PNG overlays)
        if overlays:
            n_clips = len(spec.scene_clips)
            has_subtitles = len(spec.subtitles) > 0
            overlay_label_in = "final" if has_subtitles else f"canvas{n_clips}"
            overlay_chain = build_overlay_filter_chain(
                overlays=overlays,
                overlay_input_indices=list(
                    range(n_clips, n_clips + len(overlays))
                ),
                label_in=overlay_label_in,
                final_label="vout",
            )
            filter_graph = filter_graph + ";\n" + ";\n".join(overlay_chain)

        # 4. Build ffmpeg command
        cmd: list[str] = ["ffmpeg", "-y"]

        # Add extracted clips as inputs
        for cp in clip_paths:
            cmd.extend(["-i", cp])

        # Add overlay PNGs as looped image inputs. -loop 1 turns the single
        # PNG into a continuous video stream so the overlay= filter has
        # frames available throughout its enable= window.
        for png in overlay_png_paths:
            cmd.extend(["-loop", "1", "-i", png])

        cmd.extend(["-filter_complex", filter_graph])

        # Determine final video label — overlays are last in the chain when
        # present, then drawtext subtitles, then the last clip canvas.
        if overlays:
            video_label = "vout"
        elif len(spec.subtitles) > 0:
            video_label = "final"
        else:
            video_label = f"canvas{len(spec.scene_clips)}"
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
