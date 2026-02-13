import json
import logging
import sys
import types

import pytest

from heimdex_media_pipelines.ocr.engine import OCRPerfTimings, PaddleOCREngine, create_ocr_engine


def _install_fake_pil(monkeypatch, width=100, height=50):
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


def test_perf_timings_defaults_to_zero():
    perf = OCRPerfTimings()
    values = perf.to_dict()
    assert values["model_load_s"] == 0.0
    assert values["ocr_inference_s"] == 0.0
    assert values["postprocess_s"] == 0.0
    assert values["total_s"] == 0.0
    assert values["frames_processed"] == 0
    assert values["frames_skipped"] == 0
    assert values["avg_confidence"] == 0.0


def test_perf_timings_to_dict_has_all_keys():
    perf = OCRPerfTimings()
    assert set(perf.to_dict().keys()) == {
        "model_load_s",
        "ocr_inference_s",
        "postprocess_s",
        "total_s",
        "frames_processed",
        "frames_skipped",
        "avg_confidence",
    }


def test_perf_timings_log_emits_json(caplog):
    with caplog.at_level(logging.INFO, logger="heimdex_media_pipelines.ocr.engine"):
        OCRPerfTimings(total_s=1.23456, frames_processed=3).log(video_id="v1", model="paddleocr")
    msg = next(r.message for r in caplog.records if "ocr_perf" in r.message)
    payload = json.loads(msg.split("ocr_perf ", 1)[1])
    assert payload["event"] == "ocr_perf"
    assert payload["video_id"] == "v1"
    assert payload["model"] == "paddleocr"
    assert payload["total_s"] == 1.235


def test_create_ocr_engine_raises_when_paddle_too_old(monkeypatch):
    monkeypatch.setattr("heimdex_media_pipelines.ocr.engine.importlib_metadata.version", lambda name: "2.6.0")
    with pytest.raises(RuntimeError, match="paddlepaddle>=2.6.1"):
        create_ocr_engine()


def test_create_ocr_engine_succeeds_at_minimum_version(monkeypatch):
    def fake_version(name):
        if name == "paddlepaddle":
            return "2.6.1"
        if name == "paddleocr":
            return "2.8.0"
        raise AssertionError(name)

    monkeypatch.setattr("heimdex_media_pipelines.ocr.engine.importlib_metadata.version", fake_version)
    engine = create_ocr_engine(lang="korean", use_gpu=False)
    assert isinstance(engine, PaddleOCREngine)
    assert engine.lang == "korean"


def test_paddle_engine_init_stores_params():
    engine = PaddleOCREngine(lang="en", use_angle_cls=False, use_gpu=True)
    assert engine.lang == "en"
    assert engine.use_angle_cls is False
    assert engine.use_gpu is True
    assert engine._model is None


def test_detect_returns_ocr_blocks(monkeypatch, tmp_path):
    _install_fake_pil(monkeypatch, width=200, height=100)

    class FakePaddleOCR:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def ocr(self, image_path, cls=True):
            return [[[[[20, 10], [80, 10], [80, 30], [20, 30]], ("상품명", 0.95)]]]

    fake_module = types.SimpleNamespace(PaddleOCR=FakePaddleOCR)
    monkeypatch.setitem(sys.modules, "paddleocr", fake_module)

    image_path = tmp_path / "f.jpg"
    image_path.write_bytes(b"x")

    engine = PaddleOCREngine()
    blocks = engine.detect(image_path)
    assert len(blocks) == 1
    assert blocks[0].text == "상품명"
    assert blocks[0].confidence == pytest.approx(0.95)
    assert blocks[0].bbox == pytest.approx([0.1, 0.1, 0.4, 0.3])


def test_detect_returns_empty_on_error(monkeypatch):
    class BrokenModel:
        def ocr(self, image_path, cls=True):
            raise RuntimeError("boom")

    engine = PaddleOCREngine()
    engine._model = BrokenModel()
    assert engine.detect("/tmp/nope.jpg") == []


def test_detect_lazy_loads_model_once(monkeypatch, tmp_path):
    _install_fake_pil(monkeypatch)
    calls = {"init": 0}

    class FakePaddleOCR:
        def __init__(self, **kwargs):
            calls["init"] += 1

        def ocr(self, image_path, cls=True):
            return [[[[[1, 1], [2, 1], [2, 2], [1, 2]], ("x", 0.8)]]]

    monkeypatch.setitem(sys.modules, "paddleocr", types.SimpleNamespace(PaddleOCR=FakePaddleOCR))
    image_path = tmp_path / "i.jpg"
    image_path.write_bytes(b"x")

    engine = PaddleOCREngine()
    engine.detect(image_path)
    engine.detect(image_path)
    assert calls["init"] == 1


def test_create_ocr_engine_raises_when_paddleocr_missing(monkeypatch):
    from importlib import metadata as importlib_metadata

    def fake_version(name):
        if name == "paddlepaddle":
            return "2.7.0"
        raise importlib_metadata.PackageNotFoundError(name)

    monkeypatch.setattr("heimdex_media_pipelines.ocr.engine.importlib_metadata.version", fake_version)
    with pytest.raises(ImportError, match="paddleocr"):
        create_ocr_engine()
