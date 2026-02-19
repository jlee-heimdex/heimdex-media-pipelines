"""Smoke tests for CLI module.

These tests verify the CLI can be imported and the typer app is wired correctly.
They do NOT require GPU, ffmpeg, whisper, or insightface.
"""

import importlib
import subprocess
import sys

import pytest


def test_package_importable():
    """Package root imports without error."""
    mod = importlib.import_module("heimdex_media_pipelines")
    assert hasattr(mod, "__version__")
    assert mod.__version__ == "0.5.0"


def test_cli_module_importable():
    """CLI module imports without error."""
    mod = importlib.import_module("heimdex_media_pipelines.cli")
    assert hasattr(mod, "app")
    assert hasattr(mod, "doctor")


def test_faces_cli_importable():
    """Faces CLI module imports without error."""
    mod = importlib.import_module("heimdex_media_pipelines.faces.cli")
    assert hasattr(mod, "app")


def test_speech_cli_importable():
    """Speech CLI module imports without error."""
    mod = importlib.import_module("heimdex_media_pipelines.speech.cli")
    assert hasattr(mod, "app")


def test_ocr_cli_importable():
    mod = importlib.import_module("heimdex_media_pipelines.ocr.cli")
    assert hasattr(mod, "app")


def test_main_module_runnable():
    """python -m heimdex_media_pipelines --help works."""
    result = subprocess.run(
        [sys.executable, "-m", "heimdex_media_pipelines", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "heimdex" in result.stdout.lower() or "pipeline" in result.stdout.lower()


def test_doctor_command_help():
    """Doctor command responds to --help."""
    result = subprocess.run(
        [sys.executable, "-m", "heimdex_media_pipelines", "doctor", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "doctor" in result.stdout.lower() or "check" in result.stdout.lower()


def test_faces_detect_help():
    """Faces detect command responds to --help."""
    result = subprocess.run(
        [sys.executable, "-m", "heimdex_media_pipelines", "faces", "detect", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "--video" in result.stdout
    assert "--out" in result.stdout


def test_faces_embed_help():
    """Faces embed command responds to --help."""
    result = subprocess.run(
        [sys.executable, "-m", "heimdex_media_pipelines", "faces", "embed", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "--video" in result.stdout
    assert "--out" in result.stdout


def test_faces_register_help():
    """Faces register command responds to --help."""
    result = subprocess.run(
        [sys.executable, "-m", "heimdex_media_pipelines", "faces", "register", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "--identity-id" in result.stdout
    assert "--out" in result.stdout


def test_speech_transcribe_help():
    """Speech transcribe command responds to --help."""
    result = subprocess.run(
        [sys.executable, "-m", "heimdex_media_pipelines", "speech", "transcribe", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "--video" in result.stdout
    assert "--out" in result.stdout


def test_speech_pipeline_help():
    """Speech pipeline command responds to --help."""
    result = subprocess.run(
        [sys.executable, "-m", "heimdex_media_pipelines", "speech", "pipeline", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "--video" in result.stdout
    assert "--out" in result.stdout


def test_ocr_pipeline_help():
    result = subprocess.run(
        [sys.executable, "-m", "heimdex_media_pipelines", "ocr", "pipeline", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "--scenes-result" in result.stdout


def test_ocr_extract_help():
    result = subprocess.run(
        [sys.executable, "-m", "heimdex_media_pipelines", "ocr", "extract", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "--image" in result.stdout
