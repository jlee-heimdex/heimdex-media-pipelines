"""CLI commands for face pipelines."""

import json
import os
import time
from typing import Optional

import typer

import heimdex_media_pipelines as _pkg

app = typer.Typer(name="faces", help="Face detection, embedding, and registration commands.")


def _write_result(data: dict, out: str) -> None:
    abs_out = os.path.abspath(out)
    os.makedirs(os.path.dirname(abs_out), exist_ok=True)
    tmp_path = abs_out + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=True)
    os.replace(tmp_path, abs_out)


@app.command()
def detect(
    video: str = typer.Option(..., help="Path to input video file"),
    fps: float = typer.Option(1.0, help="Sampling rate (frames per second)"),
    min_size: int = typer.Option(40, help="Minimum face size in pixels"),
    detector: str = typer.Option("scrfd", help="Detector backend: 'scrfd' or 'haar'"),
    det_size: int = typer.Option(640, help="SCRFD detection input size"),
    ctx_id: int = typer.Option(-1, help="GPU context ID (-1 for CPU)"),
    out: str = typer.Option(..., help="Output JSON file path"),
) -> None:
    """Run face detection at sampled timestamps."""
    from heimdex_media_pipelines.faces.sampling import sample_timestamps
    from heimdex_media_pipelines.faces.detect import detect_faces

    t0 = time.time()
    timestamps = sample_timestamps(video, fps=fps)
    detections = detect_faces(
        video,
        timestamps,
        min_size=min_size,
        detector=detector,
        scrfd_det_size=det_size,
        scrfd_ctx_id=ctx_id,
    )
    elapsed = time.time() - t0

    result = {
        "schema_version": "1.0",
        "pipeline_version": _pkg.__version__,
        "model_version": detector,
        "video_path": video,
        "fps": fps,
        "num_timestamps": len(timestamps),
        "num_frames_with_faces": sum(1 for d in detections if d.get("bboxes")),
        "total_faces": sum(len(d.get("bboxes", [])) for d in detections),
        "detections": detections,
        "timing_s": round(elapsed, 3),
    }
    _write_result(result, out)
    typer.echo(f"Wrote {out} ({result['total_faces']} faces in {elapsed:.1f}s)")


@app.command()
def embed(
    video: str = typer.Option(..., help="Path to input video file"),
    detections: str = typer.Option(..., help="Path to detections JSONL file"),
    q_min: Optional[float] = typer.Option(None, help="Minimum quality threshold"),
    align: bool = typer.Option(False, help="Use face alignment for embeddings"),
    det_size: int = typer.Option(640, help="SCRFD detection input size"),
    ctx_id: int = typer.Option(-1, help="GPU context ID (-1 for CPU)"),
    out: str = typer.Option(..., help="Output JSON file path"),
) -> None:
    """Extract face embeddings from detection results."""
    from heimdex_media_pipelines.faces.embed import extract_embeddings

    t0 = time.time()
    embeddings = extract_embeddings(
        video,
        detections,
        q_min=q_min,
        align=align,
        det_size=det_size,
        ctx_id=ctx_id,
    )
    elapsed = time.time() - t0

    result = {
        "schema_version": "1.0",
        "pipeline_version": _pkg.__version__,
        "model_version": "buffalo_l",
        "video_path": video,
        "detections_path": detections,
        "num_embeddings": len(embeddings),
        "embeddings": embeddings,
        "timing_s": round(elapsed, 3),
    }
    _write_result(result, out)
    typer.echo(f"Wrote {out} ({len(embeddings)} embeddings in {elapsed:.1f}s)")


@app.command()
def register(
    identity_id: str = typer.Option(..., help="Identity ID"),
    ref_dir: Optional[str] = typer.Option(None, help="Directory with reference images"),
    ref_images: Optional[list[str]] = typer.Option(None, help="Individual reference image paths"),
    det_size: int = typer.Option(640, help="SCRFD detection input size"),
    ctx_id: int = typer.Option(-1, help="GPU context ID (-1 for CPU)"),
    exemplars_k: int = typer.Option(5, help="Number of exemplar embeddings to keep"),
    out: str = typer.Option(..., help="Output JSON file path"),
) -> None:
    """Build face identity template from reference images."""
    from heimdex_media_pipelines.faces.register import build_identity_template

    t0 = time.time()
    images = ref_images or []
    template_path = build_identity_template(
        identity_id=identity_id,
        ref_images=images,
        ref_dir=ref_dir,
        out_path=out,
        det_size=det_size,
        ctx_id=ctx_id,
        exemplars_k=exemplars_k,
    )
    elapsed = time.time() - t0

    typer.echo(f"Wrote {template_path} in {elapsed:.1f}s")
