"""Tests for keyframe extraction.

Mocks subprocess to avoid requiring ffmpeg.
"""

import os
from unittest.mock import MagicMock, call, patch

import pytest

from heimdex_media_contracts.scenes.schemas import SceneBoundary
from heimdex_media_pipelines.scenes.keyframe import (
    extract_all_keyframes,
    extract_keyframe,
)


def _make_boundary(scene_id: str, index: int, start_ms: int, end_ms: int) -> SceneBoundary:
    return SceneBoundary(
        scene_id=scene_id,
        index=index,
        start_ms=start_ms,
        end_ms=end_ms,
        keyframe_timestamp_ms=(start_ms + end_ms) // 2,
    )


class TestExtractKeyframe:
    @patch("heimdex_media_pipelines.scenes.keyframe.subprocess.run")
    def test_success_returns_path(self, mock_run, tmp_path):
        out_path = str(tmp_path / "frame.jpg")

        def side_effect(cmd, **kwargs):
            with open(out_path, "wb") as f:
                f.write(b"\xff\xd8\xff\xe0JFIF")
            m = MagicMock()
            m.returncode = 0
            return m

        mock_run.side_effect = side_effect

        result = extract_keyframe("/tmp/test.mp4", 2500, out_path)
        assert result == out_path
        assert os.path.isfile(out_path)

    @patch("heimdex_media_pipelines.scenes.keyframe.subprocess.run")
    def test_ffmpeg_failure_raises(self, mock_run, tmp_path):
        m = MagicMock()
        m.returncode = 1
        m.stderr = "Error opening input"
        mock_run.return_value = m

        with pytest.raises(RuntimeError, match="ffmpeg keyframe extraction failed"):
            extract_keyframe("/tmp/bad.mp4", 0, str(tmp_path / "frame.jpg"))

    @patch("heimdex_media_pipelines.scenes.keyframe.subprocess.run")
    def test_empty_output_raises(self, mock_run, tmp_path):
        out_path = str(tmp_path / "empty.jpg")

        def side_effect(cmd, **kwargs):
            with open(out_path, "wb") as f:
                pass
            m = MagicMock()
            m.returncode = 0
            return m

        mock_run.side_effect = side_effect

        with pytest.raises(RuntimeError, match="missing or empty"):
            extract_keyframe("/tmp/test.mp4", 0, out_path)

    @patch("heimdex_media_pipelines.scenes.keyframe.subprocess.run")
    def test_creates_parent_directory(self, mock_run, tmp_path):
        nested_path = str(tmp_path / "a" / "b" / "frame.jpg")

        def side_effect(cmd, **kwargs):
            with open(nested_path, "wb") as f:
                f.write(b"\xff\xd8")
            m = MagicMock()
            m.returncode = 0
            return m

        mock_run.side_effect = side_effect

        result = extract_keyframe("/tmp/test.mp4", 1000, nested_path)
        assert os.path.isfile(result)


class TestExtractAllKeyframes:
    @patch("heimdex_media_pipelines.scenes.keyframe.subprocess.run")
    def test_extracts_one_per_scene(self, mock_run, tmp_path):
        scenes = [
            _make_boundary("v_scene_000", 0, 0, 5000),
            _make_boundary("v_scene_001", 1, 5000, 10000),
        ]
        out_dir = str(tmp_path / "keyframes")

        def side_effect(cmd, **kwargs):
            out_path = cmd[-1]
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            with open(out_path, "wb") as f:
                f.write(b"\xff\xd8\xff")
            m = MagicMock()
            m.returncode = 0
            return m

        mock_run.side_effect = side_effect

        paths = extract_all_keyframes("/tmp/test.mp4", scenes, out_dir)
        assert len(paths) == 2
        assert all(os.path.isfile(p) for p in paths)
        assert scenes[0].keyframe_path is not None
        assert scenes[1].keyframe_path is not None
