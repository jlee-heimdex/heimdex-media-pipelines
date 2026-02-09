"""Tests for scene assembly logic."""

import json
import os

from heimdex_media_contracts.scenes.schemas import SceneBoundary, SceneDetectionResult
from heimdex_media_pipelines.scenes.assembler import assemble_scenes


def _boundary(scene_id: str, index: int, start_ms: int, end_ms: int) -> SceneBoundary:
    return SceneBoundary(
        scene_id=scene_id,
        index=index,
        start_ms=start_ms,
        end_ms=end_ms,
        keyframe_timestamp_ms=(start_ms + end_ms) // 2,
    )


def _write_speech_result(path: str, segments: list[dict]) -> None:
    with open(path, "w") as f:
        json.dump({
            "schema_version": "1.0",
            "pipeline_version": "0.1.0",
            "model_version": "base",
            "segments": segments,
        }, f)


class TestAssembleScenes:
    def test_three_scenes_five_segments(self, tmp_path):
        speech_path = str(tmp_path / "speech.json")
        _write_speech_result(speech_path, [
            {"start": 1.0, "end": 2.0, "text": "hello"},
            {"start": 3.0, "end": 4.0, "text": "world"},
            {"start": 6.0, "end": 7.0, "text": "second scene"},
            {"start": 11.0, "end": 12.0, "text": "third"},
            {"start": 13.0, "end": 14.0, "text": "scene"},
        ])

        boundaries = [
            _boundary("v_scene_000", 0, 0, 5000),
            _boundary("v_scene_001", 1, 5000, 10000),
            _boundary("v_scene_002", 2, 10000, 15000),
        ]

        result = assemble_scenes(
            video_path="/tmp/test.mp4",
            video_id="v",
            scene_boundaries=boundaries,
            speech_result_path=speech_path,
            total_duration_ms=15000,
        )

        assert isinstance(result, SceneDetectionResult)
        assert len(result.scenes) == 3
        assert result.scenes[0].transcript_raw == "hello world"
        assert result.scenes[0].speech_segment_count == 2
        assert result.scenes[1].transcript_raw == "second scene"
        assert result.scenes[1].speech_segment_count == 1
        assert result.scenes[2].transcript_raw == "third scene"
        assert result.scenes[2].speech_segment_count == 2

    def test_no_speech_result(self):
        boundaries = [_boundary("v_scene_000", 0, 0, 5000)]

        result = assemble_scenes(
            video_path="/tmp/test.mp4",
            video_id="v",
            scene_boundaries=boundaries,
        )

        assert len(result.scenes) == 1
        assert result.scenes[0].transcript_raw == ""
        assert result.scenes[0].speech_segment_count == 0

    def test_empty_segments_in_speech_result(self, tmp_path):
        speech_path = str(tmp_path / "speech.json")
        _write_speech_result(speech_path, [])

        boundaries = [_boundary("v_scene_000", 0, 0, 5000)]

        result = assemble_scenes(
            video_path="/tmp/test.mp4",
            video_id="v",
            scene_boundaries=boundaries,
            speech_result_path=speech_path,
        )

        assert result.scenes[0].transcript_raw == ""
        assert result.scenes[0].speech_segment_count == 0

    def test_output_has_three_field_contract(self, tmp_path):
        speech_path = str(tmp_path / "speech.json")
        _write_speech_result(speech_path, [])

        result = assemble_scenes(
            video_path="/tmp/test.mp4",
            video_id="v",
            scene_boundaries=[_boundary("v_scene_000", 0, 0, 5000)],
            speech_result_path=speech_path,
        )

        assert result.schema_version == "1.0"
        assert result.pipeline_version != ""
        assert result.model_version != ""

    def test_transcript_norm_is_lowercased(self, tmp_path):
        speech_path = str(tmp_path / "speech.json")
        _write_speech_result(speech_path, [
            {"start": 1.0, "end": 2.0, "text": "Hello WORLD"},
        ])

        result = assemble_scenes(
            video_path="/tmp/test.mp4",
            video_id="v",
            scene_boundaries=[_boundary("v_scene_000", 0, 0, 5000)],
            speech_result_path=speech_path,
        )

        assert result.scenes[0].transcript_raw == "Hello WORLD"
        assert result.scenes[0].transcript_norm == "hello world"

    def test_valid_scene_detection_result_json(self, tmp_path):
        speech_path = str(tmp_path / "speech.json")
        _write_speech_result(speech_path, [
            {"start": 1.0, "end": 2.0, "text": "test"},
        ])

        result = assemble_scenes(
            video_path="/tmp/test.mp4",
            video_id="v",
            scene_boundaries=[_boundary("v_scene_000", 0, 0, 5000)],
            speech_result_path=speech_path,
            total_duration_ms=5000,
        )

        data = result.model_dump()
        assert "scenes" in data
        assert data["scenes"][0]["scene_id"] == "v_scene_000"
        assert data["total_duration_ms"] == 5000
