from __future__ import annotations

import subprocess
import sys


def test_scenes_pipeline_help_shows_ocr_flag():
    result = subprocess.run(
        [sys.executable, "-m", "heimdex_media_pipelines", "scenes", "pipeline", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "--ocr" in result.stdout


def test_scenes_pipeline_help_shows_ocr_lang_flag():
    result = subprocess.run(
        [sys.executable, "-m", "heimdex_media_pipelines", "scenes", "pipeline", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "--ocr-lang" in result.stdout


def test_scenes_pipeline_help_shows_redact_pii_flag():
    result = subprocess.run(
        [sys.executable, "-m", "heimdex_media_pipelines", "scenes", "pipeline", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "--redact-pii" in result.stdout


def test_scenes_pipeline_help_shows_no_ocr_flag():
    result = subprocess.run(
        [sys.executable, "-m", "heimdex_media_pipelines", "scenes", "pipeline", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "--no-ocr" in result.stdout
