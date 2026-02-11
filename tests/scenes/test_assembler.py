"""Tests for scene assembly logic."""

import json
import os

from heimdex_media_contracts.scenes.schemas import SceneBoundary, SceneDetectionResult
from heimdex_media_pipelines.scenes.assembler import (
    _dicts_to_speech_segments,
    _extract_product_entities,
    assemble_scenes,
)


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

    def test_keyword_tags_populated_from_korean_speech(self, tmp_path):
        speech_path = str(tmp_path / "speech.json")
        _write_speech_result(speech_path, [
            {"start": 1.0, "end": 2.0, "text": "지금 바로 구매하세요"},
            {"start": 3.0, "end": 4.0, "text": "가격이 정말 좋아요 할인 중"},
        ])

        boundaries = [_boundary("v_scene_000", 0, 0, 5000)]
        result = assemble_scenes(
            video_path="/tmp/test.mp4",
            video_id="v",
            scene_boundaries=boundaries,
            speech_result_path=speech_path,
        )

        scene = result.scenes[0]
        assert "cta" in scene.keyword_tags
        assert "price" in scene.keyword_tags

    def test_product_tags_and_entities_from_korean_speech(self, tmp_path):
        speech_path = str(tmp_path / "speech.json")
        _write_speech_result(speech_path, [
            {"start": 1.0, "end": 2.0, "text": "이 세럼 정말 좋아요"},
            {"start": 3.0, "end": 4.0, "text": "립스틱 색상이 예뻐요"},
        ])

        boundaries = [_boundary("v_scene_000", 0, 0, 5000)]
        result = assemble_scenes(
            video_path="/tmp/test.mp4",
            video_id="v",
            scene_boundaries=boundaries,
            speech_result_path=speech_path,
        )

        scene = result.scenes[0]
        assert "skincare" in scene.product_tags
        assert "makeup" in scene.product_tags
        assert "세럼" in scene.product_entities
        assert "립스틱" in scene.product_entities

    def test_no_speech_produces_empty_tags(self):
        boundaries = [_boundary("v_scene_000", 0, 0, 5000)]
        result = assemble_scenes(
            video_path="/tmp/test.mp4",
            video_id="v",
            scene_boundaries=boundaries,
        )

        scene = result.scenes[0]
        assert scene.keyword_tags == []
        assert scene.product_tags == []
        assert scene.product_entities == []
        assert scene.people_cluster_ids == []

    def test_people_cluster_ids_always_empty(self, tmp_path):
        speech_path = str(tmp_path / "speech.json")
        _write_speech_result(speech_path, [
            {"start": 1.0, "end": 2.0, "text": "test segment"},
        ])

        boundaries = [
            _boundary("v_scene_000", 0, 0, 5000),
            _boundary("v_scene_001", 1, 5000, 10000),
        ]
        result = assemble_scenes(
            video_path="/tmp/test.mp4",
            video_id="v",
            scene_boundaries=boundaries,
            speech_result_path=speech_path,
        )

        for scene in result.scenes:
            assert scene.people_cluster_ids == []

    def test_pipeline_version_defaults_to_0_3_0(self, tmp_path):
        speech_path = str(tmp_path / "speech.json")
        _write_speech_result(speech_path, [])

        result = assemble_scenes(
            video_path="/tmp/test.mp4",
            video_id="v",
            scene_boundaries=[_boundary("v_scene_000", 0, 0, 5000)],
            speech_result_path=speech_path,
        )

        assert result.pipeline_version == "0.3.0"

    def test_start_s_end_s_key_convention(self, tmp_path):
        speech_path = str(tmp_path / "speech.json")
        _write_speech_result(speech_path, [
            {"start_s": 1.0, "end_s": 2.0, "text": "alternate keys"},
        ])

        boundaries = [_boundary("v_scene_000", 0, 0, 5000)]
        result = assemble_scenes(
            video_path="/tmp/test.mp4",
            video_id="v",
            scene_boundaries=boundaries,
            speech_result_path=speech_path,
        )

        assert result.scenes[0].transcript_raw == "alternate keys"
        assert result.scenes[0].speech_segment_count == 1

    def test_multi_scene_tag_isolation(self, tmp_path):
        speech_path = str(tmp_path / "speech.json")
        _write_speech_result(speech_path, [
            {"start": 1.0, "end": 2.0, "text": "지금 구매 세럼"},
            {"start": 6.0, "end": 7.0, "text": "샴푸 할인"},
        ])

        boundaries = [
            _boundary("v_scene_000", 0, 0, 5000),
            _boundary("v_scene_001", 1, 5000, 10000),
        ]
        result = assemble_scenes(
            video_path="/tmp/test.mp4",
            video_id="v",
            scene_boundaries=boundaries,
            speech_result_path=speech_path,
        )

        # given: scene 0 has "지금 구매 세럼" → cta + skincare
        assert "cta" in result.scenes[0].keyword_tags
        assert "skincare" in result.scenes[0].product_tags
        assert "haircare" not in result.scenes[0].product_tags

        # given: scene 1 has "샴푸 할인" → price + haircare
        assert "price" in result.scenes[1].keyword_tags
        assert "haircare" in result.scenes[1].product_tags
        assert "skincare" not in result.scenes[1].product_tags


class TestDictsToSpeechSegments:
    def test_standard_keys(self):
        segs = _dicts_to_speech_segments([
            {"start": 1.0, "end": 2.0, "text": "hello", "confidence": 0.95},
        ])
        assert len(segs) == 1
        assert segs[0].start == 1.0
        assert segs[0].end == 2.0
        assert segs[0].text == "hello"
        assert segs[0].confidence == 0.95

    def test_alternate_keys(self):
        segs = _dicts_to_speech_segments([
            {"start_s": 3.0, "end_s": 4.0, "text": "alt"},
        ])
        assert segs[0].start == 3.0
        assert segs[0].end == 4.0

    def test_empty_list(self):
        assert _dicts_to_speech_segments([]) == []

    def test_missing_text_defaults_empty(self):
        segs = _dicts_to_speech_segments([{"start": 0.0, "end": 1.0}])
        assert segs[0].text == ""


class TestExtractProductEntities:
    def test_skincare_match(self):
        from heimdex_media_contracts.speech.tagger import PRODUCT_KEYWORD_DICT
        tags, entities = _extract_product_entities("세럼이 좋아요", dict(PRODUCT_KEYWORD_DICT))
        assert "skincare" in tags
        assert "세럼" in entities

    def test_no_match_returns_empty(self):
        tags, entities = _extract_product_entities("hello world", {"skincare": ["세럼"]})
        assert tags == []
        assert entities == []

    def test_multiple_categories(self):
        tags, entities = _extract_product_entities(
            "세럼과 립스틱",
            {"skincare": ["세럼"], "makeup": ["립스틱"]},
        )
        assert tags == ["makeup", "skincare"]
        assert entities == ["립스틱", "세럼"]
