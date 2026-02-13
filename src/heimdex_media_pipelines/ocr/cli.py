from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import typer

app = typer.Typer(name="ocr", help="OCR text extraction pipeline commands.")


def _write_result(data: Mapping[str, Any], out: str) -> None:
    abs_out = os.path.abspath(out)
    os.makedirs(os.path.dirname(abs_out) or ".", exist_ok=True)
    tmp_path = abs_out + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(dict(data), f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, abs_out)


@app.command()
def pipeline(
    scenes_result: str = typer.Option(..., "--scenes-result", help="Path to scenes result JSON"),
    keyframe_dir: str = typer.Option(..., "--keyframe-dir", help="Directory containing scene keyframes"),
    out: str = typer.Option(..., "--out", help="Output JSON file path"),
    lang: str = typer.Option("korean", "--lang", help="PaddleOCR language"),
    use_gpu: bool = typer.Option(False, "--use-gpu", help="Use GPU inference"),
    redact_pii: bool = typer.Option(False, "--redact-pii", help="Redact detected PII in OCR text"),
) -> None:
    from heimdex_media_pipelines.ocr.pipeline import run_ocr_pipeline

    result = run_ocr_pipeline(
        scenes_result_path=scenes_result,
        keyframe_dir=keyframe_dir,
        out_path=out,
        lang=lang,
        use_gpu=use_gpu,
        redact_pii_flag=redact_pii,
    )
    _write_result(result.model_dump(), out)
    typer.echo(
        f"Wrote {out} ({len(result.scenes)} scenes, "
        f"frames={result.total_frames_processed}, status={result.status})"
    )


@app.command()
def extract(
    image: str = typer.Option(..., "--image", help="Path to input image file"),
    out: str = typer.Option(..., "--out", help="Output JSON file path"),
    lang: str = typer.Option("korean", "--lang", help="PaddleOCR language"),
    use_gpu: bool = typer.Option(False, "--use-gpu", help="Use GPU inference"),
) -> None:
    from heimdex_media_pipelines.ocr.engine import create_ocr_engine

    engine = create_ocr_engine(lang=lang, use_gpu=use_gpu)
    blocks = engine.detect(Path(image))
    result = {
        "image": image,
        "num_blocks": len(blocks),
        "blocks": [b.model_dump() for b in blocks],
    }
    _write_result(result, out)
    typer.echo(f"Wrote {out} ({len(blocks)} blocks)")
