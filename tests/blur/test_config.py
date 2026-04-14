"""Unit tests for BlurConfig, DetectionRecord, BlurResult."""

from __future__ import annotations

import pytest

from heimdex_media_pipelines.blur import BlurConfig, DetectionRecord, BlurResult


def test_defaults_sane():
    c = BlurConfig()
    assert c.do_faces is True
    assert c.do_owl is True
    assert "face" in c.categories
    assert "license_plate" in c.categories
    assert "card_object" in c.categories
    # Logo blur MUST be off by default — livecommerce constraint.
    assert "logo" not in c.categories
    assert c.blur_faces is True
    assert c.owl_categories == ("license_plate", "card_object")


def test_rejects_unknown_category():
    with pytest.raises(ValueError, match="Unknown blur categories"):
        BlurConfig(categories=("face", "definitely-not-a-thing"))


def test_rejects_bad_stride():
    with pytest.raises(ValueError, match="owl_stride"):
        BlurConfig(owl_stride=0)


def test_rejects_bad_threshold():
    with pytest.raises(ValueError, match="score_threshold"):
        BlurConfig(owl_score_threshold=1.5)


def test_blur_faces_gated_on_category():
    c = BlurConfig(do_faces=True, categories=("license_plate",))
    assert c.blur_faces is False


def test_detection_record_serialization():
    d = DetectionRecord(
        frame_idx=3, t_ms=120, category="license_plate",
        label="korean license plate", confidence=0.87654,
        bbox_norm=(0.10001, 0.20002, 0.30003, 0.40004),
    )
    out = d.to_dict()
    assert out["frame_idx"] == 3
    assert out["category"] == "license_plate"
    assert out["confidence"] == 0.8765
    assert out["bbox_norm"] == [0.10001, 0.20002, 0.30003, 0.40004]
    assert out["from_cache"] is False


def test_blur_result_summary():
    dets = [
        DetectionRecord(0, 0, "face", "face", 0.9, (0, 0, 0.1, 0.1)),
        DetectionRecord(0, 0, "face", "face", 0.8, (0.2, 0.2, 0.3, 0.3)),
        DetectionRecord(1, 40, "license_plate", "korean license plate",
                        0.7, (0.5, 0.5, 0.6, 0.6)),
    ]
    r = BlurResult(
        input_path="/tmp/in.mp4", output_path="/tmp/out.mp4",
        fps=25.0, width=1280, height=720, frame_count=30,
        total_ms=1234.0, owl_infer_ms=900.0, owl_infer_frames=6,
        detections=dets, config=BlurConfig(),
    )
    assert r.summary() == {"face": 2, "license_plate": 1}
    manifest = r.to_manifest()
    assert manifest["schema_version"] == "1"
    assert manifest["video"]["frame_count"] == 30
    assert manifest["summary"]["face"] == 2
    assert len(manifest["detections"]) == 3
    assert manifest["timing"]["avg_fps"] > 0
