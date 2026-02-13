import importlib


def test_ocr_module_importable():
    mod = importlib.import_module("heimdex_media_pipelines.ocr")
    assert hasattr(mod, "create_ocr_engine")


def test_ocr_cli_module_importable():
    mod = importlib.import_module("heimdex_media_pipelines.ocr.cli")
    assert hasattr(mod, "app")


def test_ocr_engine_module_importable():
    mod = importlib.import_module("heimdex_media_pipelines.ocr.engine")
    assert hasattr(mod, "PaddleOCREngine")


def test_ocr_pipeline_module_importable():
    mod = importlib.import_module("heimdex_media_pipelines.ocr.pipeline")
    assert hasattr(mod, "run_ocr_pipeline")


def test_ocr_pii_module_importable():
    mod = importlib.import_module("heimdex_media_pipelines.ocr.pii")
    assert hasattr(mod, "detect_pii")
