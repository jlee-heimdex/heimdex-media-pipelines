"""CLI import + Typer wiring smoke test."""

from __future__ import annotations

import importlib


def test_blur_cli_module_importable():
    mod = importlib.import_module("heimdex_media_pipelines.blur.cli")
    assert hasattr(mod, "app")


def test_blur_cli_registered_on_root():
    root = importlib.import_module("heimdex_media_pipelines.cli")
    # Typer registers subcommands as CommandInfo/TyperInfo on the app.
    names = [g.name for g in root.app.registered_groups]
    assert "blur" in names


def test_blur_package_public_api():
    from heimdex_media_pipelines import blur
    for name in (
        "BlurConfig", "BlurPipeline", "BlurResult", "DetectionRecord",
        "DEFAULT_OWL_QUERIES", "apply_mosaic_blur", "apply_mosaic_blur_norm",
    ):
        assert hasattr(blur, name), name
