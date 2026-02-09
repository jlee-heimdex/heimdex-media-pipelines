"""Smoke tests for faces pipeline modules.

Tests import-ability and basic structure. Does NOT require GPU, insightface,
or actual video files.
"""

import importlib

import pytest


def test_detect_module_importable():
    """faces.detect module imports without error."""
    mod = importlib.import_module("heimdex_media_pipelines.faces.detect")
    assert hasattr(mod, "detect_faces")
    assert callable(mod.detect_faces)


def test_embed_module_importable():
    """faces.embed module imports without error."""
    mod = importlib.import_module("heimdex_media_pipelines.faces.embed")
    assert hasattr(mod, "extract_embeddings")
    assert hasattr(mod, "run_embeddings")
    assert callable(mod.extract_embeddings)
    assert callable(mod.run_embeddings)


def test_register_module_importable():
    """faces.register module imports without error."""
    mod = importlib.import_module("heimdex_media_pipelines.faces.register")
    assert hasattr(mod, "build_identity_template")
    assert callable(mod.build_identity_template)


def test_pipeline_module_importable():
    """faces.pipeline module imports without error."""
    mod = importlib.import_module("heimdex_media_pipelines.faces.pipeline")
    assert hasattr(mod, "run_pipeline")
    assert callable(mod.run_pipeline)


def test_sampling_module_importable():
    """faces.sampling module imports without error."""
    mod = importlib.import_module("heimdex_media_pipelines.faces.sampling")
    assert hasattr(mod, "sample_timestamps")
    assert callable(mod.sample_timestamps)


def test_sampling_delegates_to_contracts():
    """sampling._sample_timestamps_pure is available from contracts."""
    mod = importlib.import_module("heimdex_media_pipelines.faces.sampling")
    assert hasattr(mod, "_sample_timestamps_pure")


def test_detect_faces_missing_video():
    """detect_faces raises FileNotFoundError for missing video."""
    from heimdex_media_pipelines.faces.detect import detect_faces

    with pytest.raises(FileNotFoundError):
        detect_faces("/nonexistent/video.mp4", [1.0, 2.0])


def test_register_missing_identity():
    """build_identity_template raises ValueError for empty identity_id."""
    from heimdex_media_pipelines.faces.register import build_identity_template

    with pytest.raises(ValueError, match="identity_id"):
        build_identity_template("", [])


def test_register_no_images():
    """build_identity_template raises ValueError when no images provided."""
    from heimdex_media_pipelines.faces.register import build_identity_template

    with pytest.raises(ValueError, match="reference image"):
        build_identity_template("test-id", [])


def test_sampling_invalid_fps():
    """sample_timestamps raises ValueError for fps <= 0."""
    from heimdex_media_pipelines.faces.sampling import sample_timestamps

    with pytest.raises(ValueError, match="fps"):
        sample_timestamps("/some/video.mp4", fps=0)

    with pytest.raises(ValueError, match="fps"):
        sample_timestamps("/some/video.mp4", fps=-1)


def test_sampling_missing_video():
    """sample_timestamps raises FileNotFoundError for missing video."""
    from heimdex_media_pipelines.faces.sampling import sample_timestamps

    with pytest.raises(FileNotFoundError):
        sample_timestamps("/nonexistent/video.mp4", fps=1.0)
