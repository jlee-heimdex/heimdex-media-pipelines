"""CLI commands for scene detection pipeline."""

from __future__ import annotations

import json
import os
import time
from typing import Optional

import typer

import heimdex_media_pipelines as _pkg

app = typer.Typer(name="scenes", help="Scene detection, keyframe extraction, and assembly commands.")


def _write_result(data: dict, out: str) -> None:
    abs_out = os.path.abspath(out)
    os.makedirs(os.path.dirname(abs_out) or ".", exist_ok=True)
    tmp_path = abs_out + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, abs_out)


@app.command()
def detect(
    video: str = typer.Option(..., help="Path to input video file"),
    video_id: str = typer.Option(..., help="Video identifier for scene_id construction"),
    threshold: float = typer.Option(0.3, help="Scene change threshold (0.0-1.0)"),
    out: str = typer.Option(..., help="Output JSON file path"),
) -> None:
    """Detect scene boundaries only (no transcript assembly)."""
    from heimdex_media_pipelines.scenes.detector import detect_scenes

    t0 = time.time()
    scenes = detect_scenes(video, video_id, threshold=threshold)
    elapsed = time.time() - t0

    result = {
        "schema_version": "1.0",
        "pipeline_version": _pkg.__version__,
        "model_version": "ffmpeg_scenecut",
        "video_path": video,
        "video_id": video_id,
        "num_scenes": len(scenes),
        "scenes": [s.model_dump() for s in scenes],
        "timing_s": round(elapsed, 3),
    }
    _write_result(result, out)
    typer.echo(f"Wrote {out} ({len(scenes)} scenes in {elapsed:.1f}s)")


@app.command()
def assemble(
    video: str = typer.Option(..., help="Path to input video file"),
    video_id: str = typer.Option(..., help="Video identifier"),
    boundaries_json: str = typer.Option(..., help="Path to scene boundaries JSON"),
    speech_result: Optional[str] = typer.Option(None, help="Path to speech result JSON"),
    out: str = typer.Option(..., help="Output JSON file path"),
) -> None:
    """Assemble full scene documents from boundaries + speech result."""
    from heimdex_media_contracts.scenes.schemas import SceneBoundary
    from heimdex_media_pipelines.scenes.assembler import assemble_scenes
    from heimdex_media_pipelines.scenes.detector import _probe_duration_ms

    with open(boundaries_json) as f:
        data = json.load(f)

    boundaries = [SceneBoundary(**s) for s in data.get("scenes", data.get("boundaries", []))]
    total_duration_ms = _probe_duration_ms(video)

    t0 = time.time()
    result = assemble_scenes(
        video_path=video,
        video_id=video_id,
        scene_boundaries=boundaries,
        speech_result_path=speech_result,
        pipeline_version=_pkg.__version__,
        total_duration_ms=total_duration_ms,
        processing_time_s=0.0,
    )
    result.processing_time_s = round(time.time() - t0, 3)

    _write_result(result.model_dump(), out)
    typer.echo(f"Wrote {out} ({len(result.scenes)} scenes)")


@app.command(name="pipeline")
def run_pipeline(
    video: str = typer.Option(..., help="Path to input video file"),
    video_id: str = typer.Option(..., help="Video identifier"),
    speech_result: Optional[str] = typer.Option(None, "--speech-result", help="Path to speech result JSON"),
    threshold: float = typer.Option(0.3, help="Scene change threshold (0.0-1.0)"),
    keyframe_dir: Optional[str] = typer.Option(None, help="Directory for keyframe JPEGs"),
    out: str = typer.Option(..., help="Output JSON file path"),
) -> None:
    """Full pipeline: detect scenes → extract keyframes → assemble documents."""
    from heimdex_media_pipelines.scenes.assembler import assemble_scenes
    from heimdex_media_pipelines.scenes.detector import (
        _probe_duration_ms,
        detect_scenes,
    )
    from heimdex_media_pipelines.scenes.keyframe import extract_all_keyframes

    t0 = time.time()

    scenes = detect_scenes(video, video_id, threshold=threshold)
    total_duration_ms = _probe_duration_ms(video)

    if keyframe_dir and scenes:
        extract_all_keyframes(video, scenes, keyframe_dir)

    result = assemble_scenes(
        video_path=video,
        video_id=video_id,
        scene_boundaries=scenes,
        speech_result_path=speech_result,
        pipeline_version=_pkg.__version__,
        model_version="ffmpeg_scenecut",
        total_duration_ms=total_duration_ms,
        processing_time_s=round(time.time() - t0, 3),
    )

    _write_result(result.model_dump(), out)
    typer.echo(
        f"Wrote {out} ({len(result.scenes)} scenes, "
        f"status={result.status}, {result.processing_time_s:.1f}s)"
    )
