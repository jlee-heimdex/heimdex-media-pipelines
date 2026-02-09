"""Unified CLI entrypoint for heimdex-media-pipelines.

Usage:
    python -m heimdex_media_pipelines doctor --json --out doctor.json
    python -m heimdex_media_pipelines faces detect --video ... --out ...
    python -m heimdex_media_pipelines speech transcribe --video ... --out ...
"""

import json
import os
import shutil
import sys
from typing import Optional

import typer

import heimdex_media_pipelines as _pkg
from heimdex_media_pipelines.faces.cli import app as faces_app
from heimdex_media_pipelines.scenes.cli import app as scenes_app
from heimdex_media_pipelines.speech.cli import app as speech_app

app = typer.Typer(
    name="heimdex-pipelines",
    help="Heimdex media processing pipelines CLI.",
)
app.add_typer(faces_app, name="faces")
app.add_typer(scenes_app, name="scenes")
app.add_typer(speech_app, name="speech")


def _check_importable(module_name: str) -> dict:
    """Try importing a module and return status dict."""
    try:
        mod = __import__(module_name)
        version = getattr(mod, "__version__", getattr(mod, "VERSION", "unknown"))
        return {"available": True, "version": str(version)}
    except ImportError as e:
        return {"available": False, "error": str(e)}


def _check_executable(name: str) -> dict:
    """Check if an executable is on PATH."""
    path = shutil.which(name)
    if path:
        return {"available": True, "path": path}
    return {"available": False, "error": f"{name} not found on PATH"}


@app.command()
def doctor(
    as_json: bool = typer.Option(False, "--json", help="Output as JSON"),
    out: Optional[str] = typer.Option(None, help="Write result to file path"),
) -> None:
    """Check system dependencies for all pipelines."""
    checks = {
        "package_version": _pkg.__version__,
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "dependencies": {
            "cv2": _check_importable("cv2"),
            "insightface": _check_importable("insightface"),
            "onnxruntime": _check_importable("onnxruntime"),
            "whisper": _check_importable("whisper"),
            "torch": _check_importable("torch"),
            "typer": _check_importable("typer"),
            "pydantic": _check_importable("pydantic"),
        },
        "executables": {
            "ffmpeg": _check_executable("ffmpeg"),
            "ffprobe": _check_executable("ffprobe"),
        },
    }

    # GPU check
    try:
        import torch
        checks["gpu"] = {
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
    except ImportError:
        checks["gpu"] = {"cuda_available": False, "error": "torch not installed"}

    deps = checks["dependencies"]
    exes = checks["executables"]

    has_ffmpeg = exes.get("ffmpeg", {}).get("available", False)
    checks["pipelines"] = {
        "speech": deps.get("whisper", {}).get("available", False) and has_ffmpeg,
        "faces": deps.get("cv2", {}).get("available", False) and deps.get("insightface", {}).get("available", False),
        "scenes": has_ffmpeg,
    }

    all_deps = {**deps, **exes}
    available_count = sum(1 for v in all_deps.values() if v.get("available"))
    total_count = len(all_deps)
    checks["summary"] = {
        "available": available_count,
        "total": total_count,
        "all_ok": available_count == total_count,
    }

    if as_json or out:
        text = json.dumps(checks, indent=2, ensure_ascii=False)
        if out:
            abs_out = os.path.abspath(out)
            os.makedirs(os.path.dirname(abs_out), exist_ok=True)
            tmp_path = abs_out + ".tmp"
            with open(tmp_path, "w") as f:
                f.write(text)
            os.replace(tmp_path, abs_out)
            typer.echo(f"Wrote {out}")
        else:
            typer.echo(text)
    else:
        typer.echo(f"heimdex-media-pipelines v{_pkg.__version__}")
        typer.echo(f"Python {sys.version}")
        typer.echo()
        for category_name, category in [("Dependencies", checks["dependencies"]), ("Executables", checks["executables"])]:
            typer.echo(f"  {category_name}:")
            for name, info in category.items():
                status = "OK" if info.get("available") else "MISSING"
                detail = info.get("version", info.get("path", info.get("error", "")))
                typer.echo(f"    {name:15s} [{status:7s}] {detail}")
        typer.echo()
        gpu_info = checks.get("gpu", {})
        cuda = "yes" if gpu_info.get("cuda_available") else "no"
        typer.echo(f"  GPU: cuda={cuda}")
        typer.echo()
        summary = checks["summary"]
        typer.echo(f"  {summary['available']}/{summary['total']} dependencies available")
        if not summary["all_ok"]:
            raise typer.Exit(code=1)


# Support `python -m heimdex_media_pipelines`
def main() -> None:
    app()


if __name__ == "__main__":
    main()
