"""Tests for scene boundary detection.

These tests mock subprocess calls to avoid requiring ffmpeg in CI.
"""

from unittest.mock import MagicMock, patch

import pytest

from heimdex_media_pipelines.scenes.detector import detect_scenes, _probe_duration_ms


MOCK_FFPROBE_STDOUT = "30.500000\n"

MOCK_FFMPEG_STDERR_3_CUTS = """\
[Parsed_showinfo_1 @ 0x1234] n:   0 pts:   5000 pts_time:5.000000 fmt:yuv420p
[Parsed_showinfo_1 @ 0x1234] n:   1 pts:  12000 pts_time:12.000000 fmt:yuv420p
[Parsed_showinfo_1 @ 0x1234] n:   2 pts:  22000 pts_time:22.000000 fmt:yuv420p
"""

MOCK_FFMPEG_STDERR_NO_CUTS = """\
[info] No scene changes detected
"""


def _mock_run_factory(ffprobe_stdout: str, ffmpeg_stderr: str):
    def mock_run(cmd, **kwargs):
        m = MagicMock()
        m.returncode = 0
        if cmd[0] == "ffprobe" or (isinstance(cmd, list) and "ffprobe" in cmd[0]):
            m.stdout = ffprobe_stdout
            m.stderr = ""
        else:
            m.stdout = ""
            m.stderr = ffmpeg_stderr
        return m
    return mock_run


class TestDetectScenes:
    @patch("heimdex_media_pipelines.scenes.detector.subprocess.run")
    def test_detects_three_scenes_from_three_cuts(self, mock_run):
        mock_run.side_effect = _mock_run_factory(MOCK_FFPROBE_STDOUT, MOCK_FFMPEG_STDERR_3_CUTS)

        scenes = detect_scenes("/tmp/test.mp4", "vid001", threshold=0.3)

        assert len(scenes) == 4
        assert scenes[0].start_ms == 0
        assert scenes[0].end_ms == 5000
        assert scenes[-1].end_ms == 30500
        for s in scenes:
            assert s.scene_id.startswith("vid001_scene_")

    @patch("heimdex_media_pipelines.scenes.detector.subprocess.run")
    def test_no_cuts_returns_single_scene(self, mock_run):
        mock_run.side_effect = _mock_run_factory(MOCK_FFPROBE_STDOUT, MOCK_FFMPEG_STDERR_NO_CUTS)

        scenes = detect_scenes("/tmp/test.mp4", "vid001")

        assert len(scenes) == 1
        assert scenes[0].start_ms == 0
        assert scenes[0].end_ms == 30500

    @patch("heimdex_media_pipelines.scenes.detector.subprocess.run")
    def test_output_sorted_by_start_ms(self, mock_run):
        mock_run.side_effect = _mock_run_factory(MOCK_FFPROBE_STDOUT, MOCK_FFMPEG_STDERR_3_CUTS)

        scenes = detect_scenes("/tmp/test.mp4", "vid001")

        for i in range(1, len(scenes)):
            assert scenes[i].start_ms >= scenes[i - 1].start_ms

    @patch("heimdex_media_pipelines.scenes.detector.subprocess.run")
    def test_scene_id_format(self, mock_run):
        mock_run.side_effect = _mock_run_factory(MOCK_FFPROBE_STDOUT, MOCK_FFMPEG_STDERR_3_CUTS)

        scenes = detect_scenes("/tmp/test.mp4", "my_video")

        for i, s in enumerate(scenes):
            assert s.scene_id == f"my_video_scene_{i:03d}"
            assert s.index == i

    @patch("heimdex_media_pipelines.scenes.detector.subprocess.run")
    def test_threshold_zero_many_scenes(self, mock_run):
        cuts = "\n".join(
            f"[Parsed_showinfo_1 @ 0x1] n: {i} pts: {i*1000} pts_time:{i}.000000 fmt:yuv420p"
            for i in range(1, 30)
        )
        mock_run.side_effect = _mock_run_factory(MOCK_FFPROBE_STDOUT, cuts)

        scenes = detect_scenes("/tmp/test.mp4", "vid", threshold=0.0)

        assert len(scenes) >= 2

    @patch("heimdex_media_pipelines.scenes.detector.subprocess.run")
    def test_threshold_one_single_scene(self, mock_run):
        mock_run.side_effect = _mock_run_factory(MOCK_FFPROBE_STDOUT, MOCK_FFMPEG_STDERR_NO_CUTS)

        scenes = detect_scenes("/tmp/test.mp4", "vid", threshold=1.0)

        assert len(scenes) == 1


class TestEofClamp:
    @patch("heimdex_media_pipelines.scenes.detector.subprocess.run")
    def test_last_scene_keyframe_clamped_away_from_eof(self, mock_run):
        stderr = (
            "[Parsed_showinfo_1 @ 0x1] n:0 pts:4500 pts_time:4.500000 fmt:yuv420p\n"
        )
        mock_run.side_effect = _mock_run_factory("5.000\n", stderr)

        scenes = detect_scenes("/tmp/test.mp4", "vid")
        last = scenes[-1]
        assert last.start_ms == 4500
        assert last.end_ms == 5000
        unclamped_midpoint = 4500 + (5000 - 4500) // 2
        assert last.keyframe_timestamp_ms < unclamped_midpoint
        assert last.keyframe_timestamp_ms <= 5000 - 500

    @patch("heimdex_media_pipelines.scenes.detector.subprocess.run")
    def test_short_video_keyframe_stays_at_start(self, mock_run):
        mock_run.side_effect = _mock_run_factory("0.400\n", "")

        scenes = detect_scenes("/tmp/test.mp4", "vid")
        assert len(scenes) == 1
        assert scenes[0].keyframe_timestamp_ms >= 0

    @patch("heimdex_media_pipelines.scenes.detector.subprocess.run")
    def test_normal_scene_unaffected_by_clamp(self, mock_run):
        mock_run.side_effect = _mock_run_factory("30.0\n", "")

        scenes = detect_scenes("/tmp/test.mp4", "vid")
        assert scenes[0].keyframe_timestamp_ms == 15000


class TestProbeDuration:
    @patch("heimdex_media_pipelines.scenes.detector.subprocess.run")
    def test_parses_duration(self, mock_run):
        m = MagicMock()
        m.returncode = 0
        m.stdout = "45.123456\n"
        mock_run.return_value = m

        assert _probe_duration_ms("/tmp/test.mp4") == 45123

    @patch("heimdex_media_pipelines.scenes.detector.subprocess.run")
    def test_ffprobe_failure_raises(self, mock_run):
        m = MagicMock()
        m.returncode = 1
        m.stderr = "No such file"
        mock_run.return_value = m

        with pytest.raises(RuntimeError, match="ffprobe failed"):
            _probe_duration_ms("/tmp/missing.mp4")
