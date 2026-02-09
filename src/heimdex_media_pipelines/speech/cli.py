"""CLI commands for speech pipelines."""

import json
import os
import time
from typing import Optional

import typer

import heimdex_media_pipelines as _pkg

app = typer.Typer(name="speech", help="Speech-to-text and speech segment pipeline commands.")


def _write_result(data: dict, out: str) -> None:
    abs_out = os.path.abspath(out)
    os.makedirs(os.path.dirname(abs_out), exist_ok=True)
    tmp_path = abs_out + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, abs_out)


@app.command()
def transcribe(
    video: str = typer.Option(..., help="Path to input video file"),
    model: str = typer.Option("base", help="Whisper model name"),
    language: Optional[str] = typer.Option(None, help="Language code (e.g. 'ko', 'en')"),
    device: str = typer.Option("auto", help="Device: 'auto', 'cpu', or 'cuda'"),
    out: str = typer.Option(..., help="Output JSON file path"),
) -> None:
    """Transcribe video audio using Whisper."""
    from heimdex_media_pipelines.speech.stt import STTProcessor

    t0 = time.time()
    processor = STTProcessor(model_name=model, language=language, device=device)
    segments = processor.process(video)
    elapsed = time.time() - t0

    result = {
        "schema_version": "1.0",
        "pipeline_version": _pkg.__version__,
        "model_version": model,
        "video_path": video,
        "num_segments": len(segments),
        "total_duration_s": max((s.end_s for s in segments), default=0.0),
        "segments": [s.to_dict() for s in segments],
        "timing_s": round(elapsed, 3),
    }
    _write_result(result, out)
    typer.echo(f"Wrote {out} ({len(segments)} segments in {elapsed:.1f}s)")


@app.command()
def pipeline(
    video: str = typer.Option(..., help="Path to input video file"),
    model: str = typer.Option("base", help="Whisper model name"),
    language: Optional[str] = typer.Option(None, help="Language code (e.g. 'ko', 'en')"),
    save_transcript: bool = typer.Option(True, help="Save intermediate transcript.json"),
    artifacts_dir: Optional[str] = typer.Option(None, help="Artifacts output directory"),
    out: str = typer.Option(..., help="Output JSON file path"),
) -> None:
    """Run full speech pipeline: STT -> tagging -> ranking."""
    from heimdex_media_pipelines.speech.pipeline import SpeechSegmentsPipeline

    t0 = time.time()
    pipe = SpeechSegmentsPipeline(whisper_model=model, language=language)
    pipeline_result = pipe.run(
        video_path=video,
        save_transcript=save_transcript,
        artifacts_dir=artifacts_dir,
    )
    elapsed = time.time() - t0

    result = {
        "schema_version": "1.0",
        "pipeline_version": _pkg.__version__,
        "model_version": model,
        **pipeline_result.to_dict(),
        "timing_s": round(elapsed, 3),
    }
    _write_result(result, out)
    typer.echo(
        f"Wrote {out} ({len(pipeline_result.segments)} segments, "
        f"status={pipeline_result.status}, {elapsed:.1f}s)"
    )
