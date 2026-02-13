from __future__ import annotations

from pathlib import Path

import heimdex_media_pipelines.ocr as ocr_mod
from heimdex_media_contracts.ocr.schemas import OCRBlock
from heimdex_media_contracts.scenes.schemas import SceneDetectionResult, SceneDocument
from heimdex_media_pipelines.scenes.assembler import enrich_scenes_with_ocr


def _make_result(num_scenes: int = 3) -> SceneDetectionResult:
    scenes = [
        SceneDocument(
            scene_id=f"testvid_scene_{i:03d}",
            video_id="testvid",
            index=i,
            start_ms=i * 10_000,
            end_ms=(i + 1) * 10_000,
            keyframe_timestamp_ms=i * 10_000 + 5_000,
            transcript_raw=f"segment {i}",
            transcript_norm=f"segment {i}",
            transcript_char_count=len(f"segment {i}"),
        )
        for i in range(num_scenes)
    ]
    return SceneDetectionResult(
        pipeline_version="0.3.0",
        model_version="ffmpeg_scenecut",
        video_path="/tmp/test.mp4",
        video_id="testvid",
        total_duration_ms=num_scenes * 10_000,
        scenes=scenes,
    )


class _FakeEngine:
    def __init__(self, text: str = "할인 안내") -> None:
        self.text = text

    def detect(self, _image_path: str | Path) -> list[OCRBlock]:
        return [OCRBlock(text=self.text, confidence=0.95, bbox=[0.0, 0.0, 1.0, 1.0])]


def _patch_valid_keyframes(monkeypatch) -> None:
    monkeypatch.setattr(ocr_mod, "safe_keyframe_path", lambda scene_id, _: Path(f"/tmp/{scene_id}.jpg"))
    monkeypatch.setattr(ocr_mod, "validate_keyframe", lambda _: True)
    monkeypatch.setattr(ocr_mod, "sanitize_ocr_text", lambda text: text.strip())


def test_enrich_returns_unchanged_when_ocr_not_installed(monkeypatch):
    result = _make_result()
    before = result.model_dump()

    def _raise_import_error(*_args, **_kwargs):
        raise ImportError("paddleocr is not installed")

    monkeypatch.setattr(ocr_mod, "create_ocr_engine", _raise_import_error)
    after = enrich_scenes_with_ocr(result, "/tmp/frames")

    assert after is result
    assert after.model_dump() == before


def test_enrich_returns_unchanged_when_version_gate_fails(monkeypatch):
    result = _make_result()
    before = result.model_dump()

    def _raise_runtime_error(*_args, **_kwargs):
        raise RuntimeError("paddlepaddle>=2.6.1 required")

    monkeypatch.setattr(ocr_mod, "create_ocr_engine", _raise_runtime_error)
    after = enrich_scenes_with_ocr(result, "/tmp/frames")

    assert after is result
    assert after.model_dump() == before


def test_enrich_skips_when_scene_count_exceeds_cap(monkeypatch):
    result = _make_result(num_scenes=60)
    before = result.model_dump()
    called = {"value": False}

    def _should_not_run(*_args, **_kwargs):
        called["value"] = True
        return _FakeEngine()

    monkeypatch.setattr(ocr_mod, "create_ocr_engine", _should_not_run)
    after = enrich_scenes_with_ocr(result, "/tmp/frames", max_scenes=50)

    assert after is result
    assert after.model_dump() == before
    assert called["value"] is False


def test_enrich_populates_ocr_fields_on_valid_keyframes(monkeypatch):
    result = _make_result(num_scenes=2)
    _patch_valid_keyframes(monkeypatch)
    monkeypatch.setattr(ocr_mod, "create_ocr_engine", lambda **_kwargs: _FakeEngine(text="가격 할인"))

    after = enrich_scenes_with_ocr(result, "/tmp/frames")

    assert after is not result
    assert after.scenes[0].ocr_text_raw == "가격 할인"
    assert after.scenes[0].ocr_char_count == len("가격 할인")
    assert after.scenes[1].ocr_text_raw == "가격 할인"
    assert after.scenes[1].ocr_char_count == len("가격 할인")


def test_enrich_handles_missing_keyframes_gracefully(monkeypatch):
    result = _make_result(num_scenes=2)
    monkeypatch.setattr(ocr_mod, "create_ocr_engine", lambda **_kwargs: _FakeEngine())
    monkeypatch.setattr(ocr_mod, "safe_keyframe_path", lambda _scene_id, _dir: None)

    after = enrich_scenes_with_ocr(result, "/tmp/frames")

    assert after.scenes[0].ocr_text_raw == ""
    assert after.scenes[0].ocr_char_count == 0
    assert after.scenes[1].ocr_text_raw == ""
    assert after.scenes[1].ocr_char_count == 0


def test_enrich_handles_engine_detect_error_per_scene(monkeypatch):
    result = _make_result(num_scenes=3)
    _patch_valid_keyframes(monkeypatch)

    class _EngineWithOneFailure:
        def detect(self, image_path: str | Path) -> list[OCRBlock]:
            if "scene_000" in str(image_path):
                raise ValueError("decode failed")
            return [OCRBlock(text="세일중", confidence=0.9, bbox=[0.0, 0.0, 1.0, 1.0])]

    monkeypatch.setattr(ocr_mod, "create_ocr_engine", lambda **_kwargs: _EngineWithOneFailure())

    after = enrich_scenes_with_ocr(result, "/tmp/frames")

    assert after.scenes[0].ocr_text_raw == ""
    assert after.scenes[1].ocr_text_raw == "세일중"
    assert after.scenes[2].ocr_text_raw == "세일중"


def test_enrich_with_pii_redaction(monkeypatch):
    result = _make_result(num_scenes=1)
    _patch_valid_keyframes(monkeypatch)
    monkeypatch.setattr(ocr_mod, "create_ocr_engine", lambda **_kwargs: _FakeEngine(text="문의 user@example.com"))

    after = enrich_scenes_with_ocr(result, "/tmp/frames", redact_pii=True)

    assert "user@example.com" not in after.scenes[0].ocr_text_raw
    assert "[EMAIL]" in after.scenes[0].ocr_text_raw


def test_enrich_preserves_existing_transcript_fields(monkeypatch):
    result = _make_result(num_scenes=1)
    _patch_valid_keyframes(monkeypatch)
    monkeypatch.setattr(ocr_mod, "create_ocr_engine", lambda **_kwargs: _FakeEngine(text="한정 세일"))

    before_raw = result.scenes[0].transcript_raw
    before_norm = result.scenes[0].transcript_norm

    after = enrich_scenes_with_ocr(result, "/tmp/frames")

    assert after.scenes[0].transcript_raw == before_raw
    assert after.scenes[0].transcript_norm == before_norm


def test_enrich_result_backward_compatible_json(monkeypatch):
    result = _make_result(num_scenes=2)
    monkeypatch.setattr(ocr_mod, "create_ocr_engine", lambda **_kwargs: _FakeEngine())
    monkeypatch.setattr(ocr_mod, "safe_keyframe_path", lambda _scene_id, _dir: None)

    payload = enrich_scenes_with_ocr(result, "/tmp/frames").model_dump()

    assert "scenes" in payload
    assert payload["scenes"][0]["ocr_text_raw"] == ""
    assert payload["scenes"][0]["ocr_char_count"] == 0
    assert payload["scenes"][1]["ocr_text_raw"] == ""
    assert payload["scenes"][1]["ocr_char_count"] == 0


def test_enrich_max_scenes_boundary(monkeypatch):
    result = _make_result(num_scenes=50)
    _patch_valid_keyframes(monkeypatch)
    monkeypatch.setattr(ocr_mod, "create_ocr_engine", lambda **_kwargs: _FakeEngine(text="상품 안내"))

    after = enrich_scenes_with_ocr(result, "/tmp/frames", max_scenes=50)

    assert after.scenes[0].ocr_text_raw == "상품 안내"
    assert after.scenes[49].ocr_text_raw == "상품 안내"


def test_enrich_passes_engine_configuration(monkeypatch):
    result = _make_result(num_scenes=1)
    _patch_valid_keyframes(monkeypatch)
    received: dict[str, object] = {}

    def _capture_config(*, lang: str, use_gpu: bool):
        received["lang"] = lang
        received["use_gpu"] = use_gpu
        return _FakeEngine(text="promo")

    monkeypatch.setattr(ocr_mod, "create_ocr_engine", _capture_config)
    enrich_scenes_with_ocr(result, "/tmp/frames", lang="english", use_gpu=True)

    assert received == {"lang": "english", "use_gpu": True}


def test_enrich_leaves_scene_when_gated_text_is_empty(monkeypatch):
    result = _make_result(num_scenes=1)
    _patch_valid_keyframes(monkeypatch)
    monkeypatch.setattr(ocr_mod, "create_ocr_engine", lambda **_kwargs: _FakeEngine(text="!!"))

    after = enrich_scenes_with_ocr(result, "/tmp/frames")

    assert after.scenes[0].ocr_text_raw == ""
    assert after.scenes[0].ocr_char_count == 0
