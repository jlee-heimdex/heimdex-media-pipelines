import json
import os
import sys
import types

from heimdex_media_contracts.ocr.schemas import OCRBlock

from heimdex_media_pipelines.ocr import pipeline


def _install_fake_pil(monkeypatch, width=1920, height=1080):
    class FakeImageObj:
        def __init__(self):
            self.size = (width, height)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

    image_mod = types.SimpleNamespace(open=lambda _: FakeImageObj())
    pil_mod = types.SimpleNamespace(Image=image_mod)
    monkeypatch.setitem(sys.modules, "PIL", pil_mod)
    monkeypatch.setitem(sys.modules, "PIL.Image", image_mod)


def test_sanitize_strips_bidi_chars():
    text = "abc\u202Edef\u2067ghi"
    assert pipeline.sanitize_ocr_text(text) == "abcdefghi"


def test_sanitize_strips_script_tags():
    text = "A<script>alert(1)</script>B"
    assert pipeline.sanitize_ocr_text(text) == "Aalert(1)B"


def test_sanitize_strips_html_tags_bold_div():
    text = "<div>Hello <b>World</b></div>"
    assert pipeline.sanitize_ocr_text(text) == "Hello World"


def test_sanitize_escapes_entities():
    text = "A & B"
    assert pipeline.sanitize_ocr_text(text) == "A &amp; B"


def test_sanitize_combined_bidi_html_entities():
    text = "\u202e<div>a & b</div>"
    assert pipeline.sanitize_ocr_text(text) == "a &amp; b"


def test_sanitize_clean_text_passthrough_without_ampersand():
    assert pipeline.sanitize_ocr_text("just text") == "just text"


def test_sanitize_empty_string_returns_empty():
    assert pipeline.sanitize_ocr_text("") == ""


def test_validate_keyframe_false_for_nonexistent_file():
    assert pipeline.validate_keyframe("/tmp/definitely-missing.jpg") is False


def test_validate_keyframe_false_for_oversized_file(monkeypatch, tmp_path):
    image = tmp_path / "a.jpg"
    image.write_bytes(b"x")
    monkeypatch.setattr(os.path, "getsize", lambda _: 51 * 1024 * 1024)
    assert pipeline.validate_keyframe(image) is False


def test_validate_keyframe_false_for_too_many_pixels(monkeypatch, tmp_path):
    _install_fake_pil(monkeypatch, width=6000, height=5000)
    image = tmp_path / "a.jpg"
    image.write_bytes(b"x")
    monkeypatch.setattr(os.path, "getsize", lambda _: 10)
    assert pipeline.validate_keyframe(image) is False


def test_validate_keyframe_false_for_extreme_aspect(monkeypatch, tmp_path):
    _install_fake_pil(monkeypatch, width=2100, height=100)
    image = tmp_path / "a.jpg"
    image.write_bytes(b"x")
    monkeypatch.setattr(os.path, "getsize", lambda _: 10)
    assert pipeline.validate_keyframe(image) is False


def test_validate_keyframe_true_for_valid_image(monkeypatch, tmp_path):
    _install_fake_pil(monkeypatch, width=1920, height=1080)
    image = tmp_path / "a.jpg"
    image.write_bytes(b"x")
    monkeypatch.setattr(os.path, "getsize", lambda _: 10)
    assert pipeline.validate_keyframe(image) is True


def test_validate_keyframe_false_when_pillow_missing(monkeypatch, tmp_path):
    image = tmp_path / "a.jpg"
    image.write_bytes(b"x")
    monkeypatch.setattr(os.path, "getsize", lambda _: 10)
    monkeypatch.delitem(sys.modules, "PIL", raising=False)
    monkeypatch.delitem(sys.modules, "PIL.Image", raising=False)

    import builtins

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "PIL":
            raise ImportError("No module named PIL")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert pipeline.validate_keyframe(image) is False


def test_safe_keyframe_path_invalid_scene_id():
    assert pipeline.safe_keyframe_path("bad/id", "/tmp") is None


def test_safe_keyframe_path_path_traversal_attempt():
    assert pipeline.safe_keyframe_path("../etc/passwd_scene_0", "/tmp") is None


def test_safe_keyframe_path_valid_scene_id(tmp_path):
    out = pipeline.safe_keyframe_path("video_scene_1", tmp_path)
    assert out == (tmp_path / "video_scene_1.jpg").resolve()


def test_safe_keyframe_path_special_chars_rejected(tmp_path):
    assert pipeline.safe_keyframe_path("video_scene_1$", tmp_path) is None


def test_run_ocr_pipeline_orchestrates_and_writes_file(monkeypatch, tmp_path):
    scenes_path = tmp_path / "scenes.json"
    out_path = tmp_path / "ocr.json"
    keyframe_dir = tmp_path / "frames"
    keyframe_dir.mkdir()

    payload = {
        "video_id": "vid1",
        "scenes": [
            {"scene_id": "vid1_scene_1", "keyframe_timestamp_ms": 1200},
            {"scene_id": "vid1_scene_2", "keyframe_timestamp_ms": 2400},
        ],
    }
    scenes_path.write_text(json.dumps(payload), encoding="utf-8")

    for sid in ("vid1_scene_1", "vid1_scene_2"):
        (keyframe_dir / f"{sid}.jpg").write_bytes(b"x")

    monkeypatch.setattr(pipeline, "validate_keyframe", lambda p: True)

    class FakeEngine:
        def detect(self, image_path):
            return [
                OCRBlock(text="010-1234-5678", confidence=0.95, bbox=[0.1, 0.1, 0.4, 0.2]),
                OCRBlock(text="상품", confidence=0.2, bbox=[0.2, 0.2, 0.5, 0.3]),
            ]

    monkeypatch.setattr(pipeline, "create_ocr_engine", lambda lang, use_gpu: FakeEngine())

    result = pipeline.run_ocr_pipeline(
        scenes_result_path=str(scenes_path),
        keyframe_dir=str(keyframe_dir),
        out_path=str(out_path),
        redact_pii_flag=True,
    )

    assert result.video_id == "vid1"
    assert len(result.scenes) == 2
    assert result.total_frames_processed == 2
    assert "[KR_PHONE]" in result.scenes[0].ocr_text_raw
    assert out_path.exists()


def test_run_ocr_pipeline_skips_invalid_keyframe(monkeypatch, tmp_path):
    scenes_path = tmp_path / "scenes.json"
    out_path = tmp_path / "ocr.json"
    keyframe_dir = tmp_path / "frames"
    keyframe_dir.mkdir()
    scenes_path.write_text(
        json.dumps({"video_id": "v", "scenes": [{"scene_id": "v_scene_1", "keyframe_timestamp_ms": 1}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(pipeline, "validate_keyframe", lambda p: False)
    monkeypatch.setattr(pipeline, "create_ocr_engine", lambda lang, use_gpu: object())

    result = pipeline.run_ocr_pipeline(
        scenes_result_path=str(scenes_path),
        keyframe_dir=str(keyframe_dir),
        out_path=str(out_path),
    )
    assert result.total_frames_processed == 0
    assert len(result.scenes) == 1
    assert result.scenes[0].ocr_text_raw == ""


def test_run_ocr_pipeline_raises_import_error_when_engine_missing(monkeypatch, tmp_path):
    scenes_path = tmp_path / "scenes.json"
    out_path = tmp_path / "ocr.json"
    keyframe_dir = tmp_path / "frames"
    keyframe_dir.mkdir()
    scenes_path.write_text(json.dumps({"video_id": "v", "scenes": []}), encoding="utf-8")

    def raise_import(*args, **kwargs):
        raise ImportError("paddleocr missing")

    monkeypatch.setattr(pipeline, "create_ocr_engine", raise_import)
    try:
        pipeline.run_ocr_pipeline(
            scenes_result_path=str(scenes_path),
            keyframe_dir=str(keyframe_dir),
            out_path=str(out_path),
        )
        assert False
    except ImportError as e:
        assert "paddleocr" in str(e)
