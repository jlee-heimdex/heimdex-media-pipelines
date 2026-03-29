"""Tests for the multi-signal scene splitter orchestrator.

These tests mock ffmpeg calls to avoid requiring actual video files.
"""

from unittest.mock import patch

import pytest

from heimdex_media_contracts.scenes.splitting import SplitConfig
from heimdex_media_pipelines.scenes.splitter import (
    _boundaries_to_scenes,
    _load_speech_segments,
    split_scenes,
)


def _mock_detect_scenes(video_path, video_id, **kwargs):
    """Return a minimal SceneBoundary list for a 60s video with no cuts."""
    from heimdex_media_contracts.scenes.schemas import SceneBoundary
    return [
        SceneBoundary(
            scene_id=f"{video_id}_scene_000",
            index=0, start_ms=0, end_ms=60_000,
            keyframe_timestamp_ms=30_000,
        ),
    ]


def _mock_probe_duration(video_path):
    return 60_000


def _mock_extract_visual_cuts(video_path, threshold, ffmpeg_bin):
    return []  # no visual cuts (talking-head content)


# ---------------------------------------------------------------------------
# _boundaries_to_scenes
# ---------------------------------------------------------------------------

class TestBoundariesToScenes:
    def test_basic(self):
        scenes = _boundaries_to_scenes([0, 25_000, 60_000], "vid1", 60_000)
        assert len(scenes) == 2
        assert scenes[0].scene_id == "vid1_scene_000"
        assert scenes[0].start_ms == 0
        assert scenes[0].end_ms == 25_000
        assert scenes[1].scene_id == "vid1_scene_001"
        assert scenes[1].start_ms == 25_000
        assert scenes[1].end_ms == 60_000

    def test_keyframe_at_midpoint(self):
        scenes = _boundaries_to_scenes([0, 10_000], "v", 10_000)
        assert scenes[0].keyframe_timestamp_ms == 5_000

    def test_keyframe_clamped_near_eof(self):
        scenes = _boundaries_to_scenes([0, 1_000], "v", 1_000)
        # midpoint=500, eof_margin=500, so clamped to 500
        assert scenes[0].keyframe_timestamp_ms == 500

    def test_empty_boundaries(self):
        scenes = _boundaries_to_scenes([], "v", 0)
        assert scenes == []

    def test_single_boundary(self):
        scenes = _boundaries_to_scenes([0], "v", 0)
        assert scenes == []


# ---------------------------------------------------------------------------
# _load_speech_segments
# ---------------------------------------------------------------------------

class TestLoadSpeechSegments:
    def test_nonexistent_file(self):
        result = _load_speech_segments("/nonexistent/path.json")
        assert result == []

    def test_valid_file(self, tmp_path):
        import json
        data = {"segments": [{"start": 0.0, "end": 5.0, "text": "hello"}]}
        path = tmp_path / "speech.json"
        path.write_text(json.dumps(data))
        result = _load_speech_segments(str(path))
        assert len(result) == 1
        assert result[0]["text"] == "hello"

    def test_file_without_segments_key(self, tmp_path):
        import json
        path = tmp_path / "speech.json"
        path.write_text(json.dumps({"other_key": []}))
        result = _load_speech_segments(str(path))
        assert result == []


# ---------------------------------------------------------------------------
# split_scenes (integration with mocks)
# ---------------------------------------------------------------------------

class TestSplitScenes:
    @patch("heimdex_media_pipelines.scenes.splitter.detect_scenes", side_effect=_mock_detect_scenes)
    def test_no_speech_falls_back_to_detect_scenes(self, mock_detect):
        """Without speech data, split_scenes delegates to detect_scenes."""
        result = split_scenes("video.mp4", "vid1", speech_segments=None)
        mock_detect.assert_called_once()
        assert len(result) == 1
        assert result[0].scene_id == "vid1_scene_000"

    @patch("heimdex_media_pipelines.scenes.splitter.detect_scenes", side_effect=_mock_detect_scenes)
    def test_speech_disabled_falls_back(self, mock_detect):
        """With speech_split_enabled=False, always uses detect_scenes."""
        segments = [{"start": 0, "end": 5, "text": "a"}, {"start": 6, "end": 10, "text": "b"}]
        config = SplitConfig(speech_split_enabled=False)
        result = split_scenes("video.mp4", "vid1", config=config, speech_segments=segments)
        mock_detect.assert_called_once()

    @patch("heimdex_media_pipelines.scenes.splitter._extract_visual_cuts", side_effect=_mock_extract_visual_cuts)
    @patch("heimdex_media_pipelines.scenes.splitter._probe_duration_ms", side_effect=_mock_probe_duration)
    def test_speech_aware_splitting(self, mock_probe, mock_cuts):
        """With speech data and no visual cuts, splits at speech boundaries."""
        segments = [
            {"start": 0.0, "end": 10.0, "text": "hello", "speaker_id": "S0"},
            {"start": 10.5, "end": 20.0, "text": "world", "speaker_id": "S0"},
            {"start": 25.0, "end": 35.0, "text": "foo", "speaker_id": "S1"},
            {"start": 36.0, "end": 50.0, "text": "bar", "speaker_id": "S1"},
            {"start": 51.0, "end": 59.0, "text": "baz", "speaker_id": "S0"},
        ]
        config = SplitConfig(
            target_scene_duration_ms=20_000,
            max_scene_duration_ms=45_000,
        )
        result = split_scenes(
            "video.mp4", "vid1",
            config=config,
            speech_segments=segments,
        )
        # Should have more than 1 scene (the speaker turn at ~25s and ~51s are candidates)
        assert len(result) >= 2
        # All scenes should cover the full 60s duration
        assert result[0].start_ms == 0
        assert result[-1].end_ms == 60_000

    @patch("heimdex_media_pipelines.scenes.splitter._extract_visual_cuts", side_effect=_mock_extract_visual_cuts)
    @patch("heimdex_media_pipelines.scenes.splitter._probe_duration_ms", side_effect=_mock_probe_duration)
    def test_preset_resolution(self, mock_probe, mock_cuts):
        """Preset name is resolved correctly."""
        segments = [
            {"start": 0, "end": 10, "text": "a", "speaker_id": "S0"},
            {"start": 15, "end": 25, "text": "b", "speaker_id": "S1"},
            {"start": 30, "end": 40, "text": "c", "speaker_id": "S0"},
            {"start": 45, "end": 55, "text": "d", "speaker_id": "S1"},
        ]
        result_fine = split_scenes(
            "video.mp4", "vid1", preset="fine", speech_segments=segments,
        )
        result_coarse = split_scenes(
            "video.mp4", "vid1", preset="coarse", speech_segments=segments,
        )
        # Fine preset should produce more scenes than coarse
        assert len(result_fine) >= len(result_coarse)

    @patch("heimdex_media_pipelines.scenes.splitter._extract_visual_cuts", side_effect=_mock_extract_visual_cuts)
    @patch("heimdex_media_pipelines.scenes.splitter._probe_duration_ms", return_value=0)
    def test_zero_duration_returns_empty(self, mock_probe, mock_cuts):
        result = split_scenes(
            "video.mp4", "vid1",
            speech_segments=[{"start": 0, "end": 5, "text": "a"}],
        )
        assert result == []

    @patch("heimdex_media_pipelines.scenes.splitter.detect_scenes", side_effect=_mock_detect_scenes)
    def test_visual_only_preset(self, mock_detect):
        """visual_only preset uses detect_scenes even with speech data."""
        segments = [{"start": 0, "end": 5, "text": "a"}, {"start": 6, "end": 10, "text": "b"}]
        result = split_scenes(
            "video.mp4", "vid1", preset="visual_only", speech_segments=segments,
        )
        mock_detect.assert_called_once()

    def test_invalid_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            split_scenes("video.mp4", "vid1", preset="nonexistent")

    @patch("heimdex_media_pipelines.scenes.splitter.detect_scenes", side_effect=_mock_detect_scenes)
    def test_speech_from_file(self, mock_detect, tmp_path):
        """speech_result_path loads segments from file."""
        import json
        data = {"segments": [{"start": 0, "end": 5, "text": "hello"}]}
        path = tmp_path / "speech.json"
        path.write_text(json.dumps(data))

        # With speech_split_enabled=False, still uses detect_scenes
        # (but verifies the path loading doesn't crash)
        config = SplitConfig(speech_split_enabled=False)
        result = split_scenes(
            "video.mp4", "vid1",
            config=config,
            speech_result_path=str(path),
        )
        assert len(result) >= 1
