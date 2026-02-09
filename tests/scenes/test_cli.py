"""CLI smoke tests for scenes subcommand."""

import importlib
import subprocess
import sys


def test_scenes_cli_importable():
    mod = importlib.import_module("heimdex_media_pipelines.scenes.cli")
    assert hasattr(mod, "app")


def test_scenes_detect_help():
    result = subprocess.run(
        [sys.executable, "-m", "heimdex_media_pipelines", "scenes", "detect", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "--video" in result.stdout
    assert "--out" in result.stdout


def test_scenes_assemble_help():
    result = subprocess.run(
        [sys.executable, "-m", "heimdex_media_pipelines", "scenes", "assemble", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "--video" in result.stdout
    assert "--speech-result" in result.stdout or "--boundaries-json" in result.stdout


def test_scenes_pipeline_help():
    result = subprocess.run(
        [sys.executable, "-m", "heimdex_media_pipelines", "scenes", "pipeline", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "--video" in result.stdout
    assert "--out" in result.stdout
    assert "--speech-result" in result.stdout
