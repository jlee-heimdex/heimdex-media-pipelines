"""CLI for the blur pipeline.

    python -m heimdex_media_pipelines blur process \\
        --video /path/in.mp4 --out /path/out.mp4 --manifest /path/manifest.json

Mirrors the style of ``faces/cli.py`` and ``ocr/cli.py``: Typer app,
JSON-writable outputs, no side imports at module load time.
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

import typer

import heimdex_media_pipelines as _pkg

app = typer.Typer(name="blur", help="PII blur pipeline (faces + OWLv2).")


def _write_json(data: dict, out: str) -> None:
    abs_out = os.path.abspath(out)
    os.makedirs(os.path.dirname(abs_out) or ".", exist_ok=True)
    tmp_path = abs_out + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, abs_out)


@app.command()
def process(
    video: str = typer.Option(..., "--video", help="Path to input video"),
    out: str = typer.Option(..., "--out", help="Path to write blurred video"),
    manifest: Optional[str] = typer.Option(
        None, "--manifest", help="Optional path to write detection manifest JSON"
    ),
    face_detector: str = typer.Option(
        "scrfd", "--face-detector",
        help="Face detector backend: 'scrfd' (GPU preferred, insightface required) "
             "or 'haar' (CPU fallback, no extra deps). Matches run_blur_owl.py.",
    ),
    no_faces: bool = typer.Option(False, "--no-faces", help="Disable face blur"),
    no_owl: bool = typer.Option(False, "--no-owl", help="Disable OWLv2 blur"),
    owl_stride: int = typer.Option(5, "--owl-stride", help="Run OWLv2 every Nth frame"),
    score_threshold: float = typer.Option(
        0.35, "--score-threshold", help="OWLv2 confidence threshold"
    ),
    owl_model: str = typer.Option(
        "google/owlv2-base-patch16-ensemble", "--owl-model",
        help="HuggingFace OWLv2 model id",
    ),
    categories: str = typer.Option(
        "face,license_plate,card_object",
        "--categories",
        help="Comma-separated category list. Allowed: face, license_plate, card_object, logo",
    ),
    custom_queries: Optional[str] = typer.Option(
        None, "--custom-queries", help="Comma-separated OWL queries; overrides categories"
    ),
    no_gpu: bool = typer.Option(False, "--no-gpu", help="Force CPU mode"),
    mosaic_cells: int = typer.Option(100, "--mosaic-cells"),
    feather: int = typer.Option(3, "--feather"),
    min_face_confidence: float = typer.Option(0.5, "--min-face-confidence"),
) -> None:
    """Blur faces + OWLv2-detected PII regions in a video."""
    from heimdex_media_pipelines.blur import BlurConfig, BlurPipeline

    cats = tuple(c.strip() for c in categories.split(",") if c.strip())
    custom = (
        tuple(q.strip() for q in custom_queries.split(",") if q.strip())
        if custom_queries
        else None
    )

    if face_detector not in ("scrfd", "haar"):
        raise typer.BadParameter("--face-detector must be 'scrfd' or 'haar'")

    config = BlurConfig(
        do_faces=not no_faces,
        do_owl=not no_owl,
        categories=cats,
        face_detector=face_detector,
        owl_model=owl_model,
        owl_stride=owl_stride,
        owl_score_threshold=score_threshold,
        custom_owl_queries=custom,
        mosaic_cells=mosaic_cells,
        feather=feather,
        min_face_confidence=min_face_confidence,
        use_gpu=not no_gpu,
    )

    pipeline = BlurPipeline(config)
    t0 = time.time()
    result = pipeline.process_video(video, out)
    elapsed = time.time() - t0

    typer.echo(
        f"Wrote {out}  frames={result.frame_count}  "
        f"elapsed={elapsed:.1f}s  "
        f"avg_fps={result.frame_count / max(elapsed, 0.001):.1f}  "
        f"detections={len(result.detections)}"
    )

    if manifest:
        data = {
            "schema_version": "1",
            "pipeline_version": _pkg.__version__,
            **result.to_manifest(),
        }
        _write_json(data, manifest)
        typer.echo(f"Wrote {manifest}")


__all__ = ["app"]
