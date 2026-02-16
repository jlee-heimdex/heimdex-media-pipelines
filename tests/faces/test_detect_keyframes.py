import json

import cv2
import numpy as np
from typer.testing import CliRunner

from heimdex_media_pipelines.faces.cli import app
from heimdex_media_pipelines.faces.detect import detect_faces_from_images


def test_detect_faces_from_images_with_synthetic_image(tmp_path):
    image_path = tmp_path / "frame.jpg"
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image)

    results = detect_faces_from_images([str(image_path)], detector="haar")

    assert len(results) == 1
    assert results[0]["image_path"] == str(image_path)
    assert "bboxes" in results[0]
    assert isinstance(results[0]["bboxes"], list)


def test_cli_detect_keyframes_mode_writes_expected_json(tmp_path):
    keyframes_dir = tmp_path / "thumbs"
    keyframes_dir.mkdir()
    first = keyframes_dir / "002.jpg"
    second = keyframes_dir / "010.jpg"
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    assert cv2.imwrite(str(first), image)
    assert cv2.imwrite(str(second), image)

    out_path = tmp_path / "detections.json"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "detect",
            "--keyframes-dir",
            str(keyframes_dir),
            "--detector",
            "haar",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(out_path.read_text())
    assert payload["source"] == "keyframes"
    assert payload["num_timestamps"] == 2
    assert "video_path" not in payload
    assert "fps" not in payload
    assert payload["detections"][0]["image_path"] == str(first)
    assert payload["detections"][1]["image_path"] == str(second)


def test_cli_detect_rejects_video_and_keyframes_together(tmp_path):
    keyframes_dir = tmp_path / "thumbs"
    keyframes_dir.mkdir()
    out_path = tmp_path / "detections.json"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "detect",
            "--video",
            "input.mp4",
            "--keyframes-dir",
            str(keyframes_dir),
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code != 0
    assert "exactly one of --video or --keyframes-dir" in result.output


def test_cli_detect_requires_video_or_keyframes(tmp_path):
    out_path = tmp_path / "detections.json"
    runner = CliRunner()
    result = runner.invoke(app, ["detect", "--out", str(out_path)])

    assert result.exit_code != 0
    assert "exactly one of --video or --keyframes-dir" in result.output
