"""CLI commands for composition rendering pipeline."""

from __future__ import annotations

import json
import os
import time
from typing import Optional

import typer

import heimdex_media_pipelines as _pkg

app = typer.Typer(name="composition", help="Composition rendering commands.")


def _write_result(data: dict, out: str) -> None:
    abs_out = os.path.abspath(out)
    os.makedirs(os.path.dirname(abs_out) or ".", exist_ok=True)
    tmp_path = abs_out + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, abs_out)


@app.command()
def render(
    spec: str = typer.Option(..., help="Path to CompositionSpec JSON file"),
    media_dir: str = typer.Option(..., help="Directory containing source media files"),
    output: str = typer.Option(..., help="Output MP4 file path"),
    font_dir: str = typer.Option("/fonts", help="Directory containing font files"),
    use_gpu: bool = typer.Option(False, "--use-gpu", help="Use GPU encoding (h264_nvenc)"),
    result_json: Optional[str] = typer.Option(None, "--result-json", help="Write result metadata to JSON"),
) -> None:
    """Render a composition spec to a final MP4."""
    from heimdex_media_contracts.composition import CompositionSpec
    from heimdex_media_pipelines.composition.render import render_composition

    with open(spec) as f:
        spec_data = json.load(f)

    composition = CompositionSpec(**spec_data)

    # Build media_paths from media_dir: {video_id: path}
    media_paths: dict[str, str] = {}
    for clip in composition.scene_clips:
        if clip.video_id not in media_paths:
            # Look for video file in media_dir by video_id
            for ext in (".mp4", ".mov", ".mkv", ".webm", ".avi"):
                candidate = os.path.join(media_dir, f"{clip.video_id}{ext}")
                if os.path.isfile(candidate):
                    media_paths[clip.video_id] = candidate
                    break
            else:
                raise typer.BadParameter(
                    f"Media file not found for video_id={clip.video_id} in {media_dir}"
                )

    t0 = time.time()
    result = render_composition(
        spec=composition,
        media_paths=media_paths,
        output_path=output,
        font_dir=font_dir,
        use_gpu=use_gpu,
    )
    elapsed = time.time() - t0

    typer.echo(
        f"Rendered {output} "
        f"({result.duration_ms}ms, {result.size_bytes} bytes, {elapsed:.1f}s)"
    )

    if result_json:
        result_data = {
            "schema_version": "1.0",
            "pipeline_version": _pkg.__version__,
            "output_path": result.output_path,
            "duration_ms": result.duration_ms,
            "size_bytes": result.size_bytes,
            "render_time_ms": result.render_time_ms,
            "timing_s": round(elapsed, 3),
        }
        _write_result(result_data, result_json)
        typer.echo(f"Wrote {result_json}")
