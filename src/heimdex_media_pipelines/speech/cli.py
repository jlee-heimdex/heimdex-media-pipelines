"""CLI commands for speech pipelines."""

import json
import os
import time
from typing import Any, Mapping, Optional

import typer

import heimdex_media_pipelines as _pkg

app = typer.Typer(name="speech", help="Speech-to-text and speech segment pipeline commands.")


def _write_result(data: Mapping[str, Any], out: str) -> None:
    abs_out = os.path.abspath(out)
    os.makedirs(os.path.dirname(abs_out), exist_ok=True)
    tmp_path = abs_out + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, abs_out)


def _get_perf_meta(processor: Any) -> dict[str, Any]:
    perf = getattr(processor, "_last_perf", None)
    if perf is None:
        return {}
    return perf.to_dict()


@app.command()
def transcribe(
    video: str = typer.Option(..., help="Path to input video file"),
    model: str = typer.Option("base", help="Whisper model name"),
    language: Optional[str] = typer.Option(None, help="Language code (e.g. 'ko', 'en')"),
    device: str = typer.Option("auto", help="Device: 'auto', 'cpu', or 'cuda'"),
    backend: str = typer.Option("auto", help="STT backend: 'auto', 'local', 'api', 'whisper', or 'faster-whisper'"),
    compute_type: str = typer.Option("auto", help="Compute type for faster-whisper: 'auto', 'int8', 'float16', 'float32'"),
    beam_size: int = typer.Option(1, help="Beam size for decoding (1=greedy, 5=quality)"),
    out: str = typer.Option(..., help="Output JSON file path"),
) -> None:
    """Transcribe video audio using Whisper."""
    from heimdex_media_pipelines.speech.stt import create_stt_processor

    t0 = time.time()
    processor = create_stt_processor(
        backend=backend,
        model_name=model,
        language=language,
        device=device,
        api_key=os.getenv("OPENAI_API_KEY"),
        compute_type=compute_type,
        beam_size=beam_size,
    )
    segments = processor.process(video)
    elapsed = time.time() - t0

    result: dict[str, Any] = {
        "schema_version": "1.0",
        "pipeline_version": _pkg.__version__,
        "model_version": model,
        "video_path": video,
        "num_segments": len(segments),
        "total_duration_s": max((s.end_s for s in segments), default=0.0),
        "segments": [s.to_dict() for s in segments],
        "timing_s": round(elapsed, 3),
    }
    perf_meta = _get_perf_meta(processor)
    if perf_meta:
        result["meta"] = {"perf": perf_meta}
    _write_result(result, out)
    typer.echo(f"Wrote {out} ({len(segments)} segments in {elapsed:.1f}s)")


@app.command()
def pipeline(
    video: str = typer.Option(..., help="Path to input video file"),
    model: str = typer.Option("base", help="Whisper model name"),
    language: Optional[str] = typer.Option(None, help="Language code (e.g. 'ko', 'en')"),
    backend: str = typer.Option("auto", help="STT backend: 'auto', 'local', 'api', 'whisper', or 'faster-whisper'"),
    compute_type: str = typer.Option("auto", help="Compute type for faster-whisper: 'auto', 'int8', 'float16', 'float32'"),
    beam_size: int = typer.Option(1, help="Beam size for decoding (1=greedy, 5=quality)"),
    save_transcript: bool = typer.Option(True, help="Save intermediate transcript.json"),
    artifacts_dir: Optional[str] = typer.Option(None, help="Artifacts output directory"),
    out: str = typer.Option(..., help="Output JSON file path"),
) -> None:
    """Run full speech pipeline: STT -> tagging -> ranking."""
    from heimdex_media_pipelines.speech.pipeline import SpeechSegmentsPipeline

    t0 = time.time()
    pipe = SpeechSegmentsPipeline(
        whisper_model=model,
        language=language,
        backend=backend,
        api_key=os.getenv("OPENAI_API_KEY"),
        compute_type=compute_type,
        beam_size=beam_size,
    )
    pipeline_result = pipe.run(
        video_path=video,
        save_transcript=save_transcript,
        artifacts_dir=artifacts_dir,
    )
    elapsed = time.time() - t0

    result: dict[str, Any] = {
        "schema_version": "1.0",
        "pipeline_version": _pkg.__version__,
        "model_version": model,
        **pipeline_result.to_dict(),
        "timing_s": round(elapsed, 3),
    }
    perf_meta = _get_perf_meta(pipe.stt)
    if perf_meta:
        result["meta"] = {"perf": perf_meta}
    _write_result(result, out)
    typer.echo(
        f"Wrote {out} ({len(pipeline_result.segments)} segments, "
        f"status={pipeline_result.status}, {elapsed:.1f}s)"
    )
