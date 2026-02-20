"""Tests for keyframe extraction.

Mocks subprocess to avoid requiring ffmpeg.
"""

import os
from unittest.mock import MagicMock, call, patch

import pytest

from heimdex_media_contracts.scenes.schemas import SceneBoundary
from heimdex_media_pipelines.scenes.keyframe import (
    _batch_keyframes_enabled,
    _cleanup_batch_temps,
    _extract_batch,
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

    def test_empty_scenes_returns_empty(self):
        assert extract_all_keyframes("/tmp/test.mp4", [], "/tmp/out") == []


class TestExtractAllKeyframesResilience:
    @patch("heimdex_media_pipelines.scenes.keyframe.subprocess.run")
    def test_skips_failed_frame_continues_rest(self, mock_run, tmp_path):
        scenes = [
            _make_boundary("v_scene_000", 0, 0, 5000),
            _make_boundary("v_scene_001", 1, 5000, 10000),
            _make_boundary("v_scene_002", 2, 10000, 15000),
        ]
        out_dir = str(tmp_path / "keyframes")
        call_count = 0

        def side_effect(cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                m = MagicMock()
                m.returncode = 1
                m.stderr = "Could not open encoder before EOF"
                return m
            out_path = cmd[-1]
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            with open(out_path, "wb") as f:
                f.write(b"\xff\xd8\xff")
            return MagicMock(returncode=0)

        mock_run.side_effect = side_effect

        paths = extract_all_keyframes("/tmp/test.mp4", scenes, out_dir)
        assert len(paths) == 2
        assert scenes[0].keyframe_path is not None
        assert scenes[1].keyframe_path is None
        assert scenes[2].keyframe_path is not None


class TestBatchKeyframesFlag:
    def test_default_false(self):
        with patch.dict(os.environ, {}, clear=True):
            assert _batch_keyframes_enabled() is False

    @pytest.mark.parametrize("value", ["true", "True", "TRUE", "1"])
    def test_truthy_values(self, value):
        with patch.dict(os.environ, {"HEIMDEX_BATCH_KEYFRAMES": value}):
            assert _batch_keyframes_enabled() is True

    @pytest.mark.parametrize("value", ["false", "0", "", "yes", "on"])
    def test_falsy_values(self, value):
        with patch.dict(os.environ, {"HEIMDEX_BATCH_KEYFRAMES": value}):
            assert _batch_keyframes_enabled() is False


class TestExtractBatch:
    @patch("heimdex_media_pipelines.scenes.keyframe.subprocess.run")
    def test_builds_correct_ffmpeg_command(self, mock_run, tmp_path):
        scenes = [
            _make_boundary("v_scene_001", 1, 5000, 10000),
            _make_boundary("v_scene_000", 0, 0, 5000),
        ]
        out_dir = str(tmp_path / "kf")

        def side_effect(cmd, **kwargs):
            os.makedirs(out_dir, exist_ok=True)
            for i in range(1, 3):
                path = os.path.join(out_dir, f"_batch_{i:04d}.jpg")
                with open(path, "wb") as f:
                    f.write(b"\xff\xd8\xff\xe0JFIF")
            return MagicMock(returncode=0)

        mock_run.side_effect = side_effect

        paths = _extract_batch("/tmp/v.mp4", scenes, out_dir)

        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "ffmpeg"
        assert "-vf" in cmd
        vf_idx = cmd.index("-vf")
        vf_arg = cmd[vf_idx + 1]
        assert "select=" in vf_arg
        assert "eq(t\\," in vf_arg
        assert "-vsync" in cmd
        assert "0" in cmd[cmd.index("-vsync") + 1]

        assert len(paths) == 2
        assert all(os.path.isfile(p) for p in paths)
        assert all(p.endswith(".jpg") for p in paths)
        assert not any("_batch_" in os.path.basename(p) for p in paths)

    @patch("heimdex_media_pipelines.scenes.keyframe.subprocess.run")
    def test_sorts_by_timestamp(self, mock_run, tmp_path):
        scenes = [
            _make_boundary("v_scene_001", 1, 8000, 10000),
            _make_boundary("v_scene_000", 0, 0, 2000),
        ]
        out_dir = str(tmp_path / "kf")

        def side_effect(cmd, **kwargs):
            os.makedirs(out_dir, exist_ok=True)
            for i in range(1, 3):
                with open(os.path.join(out_dir, f"_batch_{i:04d}.jpg"), "wb") as f:
                    f.write(b"\xff\xd8")
            return MagicMock(returncode=0)

        mock_run.side_effect = side_effect
        paths = _extract_batch("/tmp/v.mp4", scenes, out_dir)

        assert os.path.basename(paths[0]) == "v_scene_000.jpg"
        assert os.path.basename(paths[1]) == "v_scene_001.jpg"

    @patch("heimdex_media_pipelines.scenes.keyframe.subprocess.run")
    def test_ffmpeg_failure_raises(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=1, stderr="decode error")
        scenes = [
            _make_boundary("v_scene_000", 0, 0, 1000),
            _make_boundary("v_scene_001", 1, 1000, 2000),
        ]

        with pytest.raises(RuntimeError, match="Batch keyframe extraction failed"):
            _extract_batch("/tmp/v.mp4", scenes, str(tmp_path / "kf"))

    @patch("heimdex_media_pipelines.scenes.keyframe.subprocess.run")
    def test_missing_output_file_raises(self, mock_run, tmp_path):
        out_dir = str(tmp_path / "kf")

        def side_effect(cmd, **kwargs):
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "_batch_0001.jpg"), "wb") as f:
                f.write(b"\xff\xd8")
            return MagicMock(returncode=0)

        mock_run.side_effect = side_effect
        scenes = [
            _make_boundary("v_scene_000", 0, 0, 1000),
            _make_boundary("v_scene_001", 1, 1000, 2000),
        ]

        with pytest.raises(RuntimeError, match="Batch keyframe missing"):
            _extract_batch("/tmp/v.mp4", scenes, out_dir)


class TestBatchFallback:
    @patch("heimdex_media_pipelines.scenes.keyframe.subprocess.run")
    @patch("heimdex_media_pipelines.scenes.keyframe._batch_keyframes_enabled", return_value=True)
    def test_falls_back_to_sequential_on_batch_failure(self, _mock_flag, mock_run, tmp_path):
        scenes = [
            _make_boundary("v_scene_000", 0, 0, 2000),
            _make_boundary("v_scene_001", 1, 2000, 4000),
        ]
        out_dir = str(tmp_path / "kf")
        call_count = 0

        def side_effect(cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return MagicMock(returncode=1, stderr="batch fail")
            out_path = cmd[-1]
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            with open(out_path, "wb") as f:
                f.write(b"\xff\xd8\xff")
            return MagicMock(returncode=0)

        mock_run.side_effect = side_effect

        paths = extract_all_keyframes("/tmp/v.mp4", scenes, out_dir)

        assert len(paths) == 2
        assert all(os.path.isfile(p) for p in paths)
        assert mock_run.call_count == 3
        assert scenes[0].keyframe_path is not None
        assert scenes[1].keyframe_path is not None

    @patch("heimdex_media_pipelines.scenes.keyframe.subprocess.run")
    @patch("heimdex_media_pipelines.scenes.keyframe._batch_keyframes_enabled", return_value=True)
    def test_cleans_up_temp_files_on_failure(self, _mock_flag, mock_run, tmp_path):
        out_dir = str(tmp_path / "kf")
        os.makedirs(out_dir)
        for i in range(1, 3):
            with open(os.path.join(out_dir, f"_batch_{i:04d}.jpg"), "wb") as f:
                f.write(b"\xff\xd8")

        call_count = 0

        def side_effect(cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return MagicMock(returncode=1, stderr="fail")
            out_path = cmd[-1]
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            with open(out_path, "wb") as f:
                f.write(b"\xff\xd8\xff")
            return MagicMock(returncode=0)

        mock_run.side_effect = side_effect
        scenes = [
            _make_boundary("v_scene_000", 0, 0, 2000),
            _make_boundary("v_scene_001", 1, 2000, 4000),
        ]

        extract_all_keyframes("/tmp/v.mp4", scenes, out_dir)

        remaining = [f for f in os.listdir(out_dir) if f.startswith("_batch_")]
        assert remaining == []

    @patch("heimdex_media_pipelines.scenes.keyframe.subprocess.run")
    @patch("heimdex_media_pipelines.scenes.keyframe._batch_keyframes_enabled", return_value=True)
    def test_resets_keyframe_path_before_sequential(self, _mock_flag, mock_run, tmp_path):
        scenes = [
            _make_boundary("v_scene_000", 0, 0, 2000),
            _make_boundary("v_scene_001", 1, 2000, 4000),
        ]
        scenes[0].keyframe_path = "/stale/path.jpg"
        out_dir = str(tmp_path / "kf")
        call_count = 0

        def side_effect(cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return MagicMock(returncode=1, stderr="fail")
            out_path = cmd[-1]
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            with open(out_path, "wb") as f:
                f.write(b"\xff\xd8\xff")
            return MagicMock(returncode=0)

        mock_run.side_effect = side_effect

        extract_all_keyframes("/tmp/v.mp4", scenes, out_dir)

        assert scenes[0].keyframe_path != "/stale/path.jpg"
        assert scenes[0].keyframe_path.endswith("v_scene_000.jpg")


class TestBatchPathSafety:
    @patch("heimdex_media_pipelines.scenes.keyframe.subprocess.run")
    def test_output_paths_stay_within_out_dir(self, mock_run, tmp_path):
        scenes = [
            _make_boundary("v_scene_000", 0, 0, 2000),
            _make_boundary("v_scene_001", 1, 2000, 4000),
        ]
        out_dir = str(tmp_path / "kf")

        def side_effect(cmd, **kwargs):
            os.makedirs(out_dir, exist_ok=True)
            for i in range(1, 3):
                with open(os.path.join(out_dir, f"_batch_{i:04d}.jpg"), "wb") as f:
                    f.write(b"\xff\xd8")
            return MagicMock(returncode=0)

        mock_run.side_effect = side_effect

        paths = _extract_batch("/tmp/v.mp4", scenes, out_dir)

        for p in paths:
            resolved = os.path.realpath(p)
            assert resolved.startswith(os.path.realpath(out_dir))


class TestCleanupBatchTemps:
    def test_removes_batch_files(self, tmp_path):
        for i in range(1, 4):
            (tmp_path / f"_batch_{i:04d}.jpg").write_bytes(b"\xff")
        (tmp_path / "keep.jpg").write_bytes(b"\xff")

        _cleanup_batch_temps(str(tmp_path))

        remaining = sorted(os.listdir(str(tmp_path)))
        assert remaining == ["keep.jpg"]

    def test_no_error_on_missing_dir(self):
        _cleanup_batch_temps("/nonexistent/path/that/does/not/exist")
