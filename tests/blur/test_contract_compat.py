"""Cross-repo contract compatibility tests.

This file is the one place in the pipelines repo that is allowed to
import from ``heimdex_media_contracts``. It exists to catch drift
between :class:`BlurResult`'s serialization shape and the contracts-
side :class:`BlurManifest` / :class:`BlurJobResult` pydantic models —
the exact boundary the worker validates at.

Why this lives here instead of in contracts:

* Contracts stays dependency-free (no opencv, no numpy, no runtime
  pipeline code). It can't run ``BlurPipeline.process_video``.
* Pipelines already depends on contracts (``heimdex-media-contracts
  >=0.8.0`` in pyproject.toml), so importing here is cheap.
* The shape under test is *what the pipeline actually emits*, not a
  hand-written fixture. A golden JSON fixture in contracts would
  drift as the pipeline evolves; this test re-reads live output.

Each test runs the real pipeline on the tiny synthetic fixture and
validates the output against the pydantic models. A failure means
the pipeline and contracts have diverged — bump contracts, adjust
``BlurResult.to_manifest()``, or both, depending on which side
represents the new intent.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

# Contract models — imported ONLY by this test module, not the
# pipeline library itself.
from heimdex_media_contracts.blur import (
    ALLOWED_BLUR_CATEGORIES,
    BlurDetectionSummary,
    BlurJobResult,
    BlurManifest,
)

from heimdex_media_pipelines.blur import BlurConfig, BlurPipeline
from heimdex_media_pipelines.blur.config import ALLOWED_CATEGORIES


requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="ffmpeg not installed",
)


class _StubDetector:
    """Canned OWLv2 substitute. Same shape as the real detector but
    returns a single deterministic box per call.
    """

    def detect(self, frame_bgr: Any, queries: list[str]) -> list[dict[str, Any]]:
        return [
            {
                "bbox": [0.10, 0.10, 0.40, 0.40],
                "label": "korean license plate",
                "confidence": 0.92,
            }
        ]


# ---------- BlurResult.to_manifest() ↔ BlurManifest ----------


def _manifest_with_pipeline_version(pipeline_manifest: dict[str, Any]) -> dict[str, Any]:
    """The pipeline library emits a manifest without ``pipeline_version``
    — the worker injects it at upload time. For the contract test we
    inject a stub value so ``BlurManifest.model_validate`` succeeds.
    """
    return {**pipeline_manifest, "pipeline_version": "test"}


def test_pipeline_manifest_validates_against_contracts_v2(synthetic_video: Path, tmp_path: Path):
    """The pipeline's ``to_manifest()`` output must validate as a v2
    BlurManifest after the worker injects the pipeline_version field.
    This is the exact parsing step drive-blur-worker performs before
    uploading the JSON to S3.
    """
    config = BlurConfig(
        do_faces=False,
        do_owl=True,
        categories=("license_plate",),
        owl_stride=5,
    )
    pipeline = BlurPipeline(config, detector=_StubDetector())
    result = pipeline.process_video(synthetic_video, tmp_path / "out.mp4")

    raw = _manifest_with_pipeline_version(result.to_manifest())
    model = BlurManifest.model_validate(raw)

    # Sanity — new-writes default to schema_version "2".
    assert model.schema_version == "2"
    # Pipeline emits a ``mask_s3_keys: None`` placeholder that the
    # worker overwrites post-upload. That's still valid at the
    # contract level.
    assert model.mask_s3_keys is None
    # Roundtrip the full model — any silent field drift blows up here.
    # Compare via canonical JSON rather than model equality because
    # ``config`` is an untyped dict whose nested tuples (e.g. the
    # pipeline's ``categories`` tuple) come back as lists after a
    # JSON trip — that's expected, not a regression.
    roundtrip_json = BlurManifest.model_validate_json(model.model_dump_json()).model_dump_json()
    assert roundtrip_json == model.model_dump_json()


@requires_ffmpeg
def test_pipeline_manifest_with_masks_validates(synthetic_video: Path, tmp_path: Path):
    """When emit_masks=True, the pipeline still produces a valid v2
    manifest. The mask_s3_keys field stays None at the pipeline layer
    — the worker fills it in from the local paths returned on
    BlurResult.mask_paths — but the manifest itself is emitted
    unchanged.
    """
    mask_dir = tmp_path / "masks"
    config = BlurConfig(
        do_faces=False,
        do_owl=True,
        categories=("license_plate",),
        owl_stride=5,
        emit_masks=True,
        mask_dir=mask_dir,
    )
    pipeline = BlurPipeline(config, detector=_StubDetector())
    result = pipeline.process_video(synthetic_video, tmp_path / "out.mp4")

    # The pipeline populated mask_paths; the manifest still has
    # mask_s3_keys=None (S3 upload is the worker's job).
    assert "license_plate" in result.mask_paths
    raw = _manifest_with_pipeline_version(result.to_manifest())
    model = BlurManifest.model_validate(raw)
    assert model.mask_s3_keys is None

    # Simulate the worker injecting mask_s3_keys and re-validate.
    raw_with_keys = {
        **raw,
        "mask_s3_keys": {
            "license_plate": f"blurred/vid/job-1/masks/license_plate.mkv",
        },
    }
    model2 = BlurManifest.model_validate(raw_with_keys)
    assert model2.mask_s3_keys == {"license_plate": "blurred/vid/job-1/masks/license_plate.mkv"}


def test_pipeline_summary_matches_contracts_detection_summary(synthetic_video: Path, tmp_path: Path):
    """The ``summary`` dict the pipeline emits is structurally
    compatible with :class:`BlurDetectionSummary`. Fresh categories
    added to the pipeline but not contracts — or vice versa — will
    fail the from_counts validation.
    """
    config = BlurConfig(
        do_faces=False,
        do_owl=True,
        categories=("license_plate",),
        owl_stride=5,
    )
    pipeline = BlurPipeline(config, detector=_StubDetector())
    result = pipeline.process_video(synthetic_video, tmp_path / "out.mp4")

    summary = BlurDetectionSummary.from_counts(result.summary())
    assert summary.license_plate >= 1
    # Extra=allow on the model — any surplus categories get captured
    # without blowing up, but the known ones must map correctly.
    assert summary.face == 0
    assert summary.logo == 0


# ---------- ALLOWED_CATEGORIES invariant ----------


def test_allowed_categories_match_contracts_literal():
    """The pipeline's ALLOWED_CATEGORIES must stay in lockstep with
    contracts' ALLOWED_BLUR_CATEGORIES. Any drift means the worker
    could reject valid pipeline output OR accept categories the
    pipeline can't produce.
    """
    assert ALLOWED_CATEGORIES == ALLOWED_BLUR_CATEGORIES


# ---------- BlurJobResult round-trip with pipeline detections_summary ----------


def test_blur_job_result_accepts_pipeline_summary(synthetic_video: Path, tmp_path: Path):
    """Closer to the worker's real path: it takes the pipeline's
    ``result.summary()`` and hands it to ``BlurDetectionSummary`` as
    part of a ``BlurJobResult`` callback. Verify the pipeline's
    serialized summary shape is directly injectable.
    """
    config = BlurConfig(
        do_faces=False,
        do_owl=True,
        categories=("license_plate",),
        owl_stride=5,
    )
    pipeline = BlurPipeline(config, detector=_StubDetector())
    result = pipeline.process_video(synthetic_video, tmp_path / "out.mp4")

    job_result = BlurJobResult(
        job_id=uuid4(),
        lease_token=uuid4(),
        file_id=uuid4(),
        org_id=uuid4(),
        video_id="vid-abc",
        status="done",
        blurred_s3_key="blurred/vid-abc/job-1/blurred.mp4",
        manifest_s3_key="blurred/vid-abc/job-1/manifest.json",
        mask_s3_keys={"license_plate": "blurred/vid-abc/job-1/masks/license_plate.mkv"},
        detections_summary=BlurDetectionSummary.from_counts(result.summary()),
    )
    restored = BlurJobResult.model_validate_json(job_result.model_dump_json())
    assert restored.status == "done"
    assert restored.mask_s3_keys is not None
    assert restored.detections_summary is not None
    assert restored.detections_summary.license_plate >= 1
