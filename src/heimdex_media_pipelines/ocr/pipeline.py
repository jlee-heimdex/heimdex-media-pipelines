from __future__ import annotations

import html
import importlib
import json
import logging
import os
import re
import time
from pathlib import Path

import heimdex_media_pipelines as _pkg
from heimdex_media_contracts.ocr.gating import concat_blocks, filter_blocks_by_confidence, gate_ocr_text
from heimdex_media_contracts.ocr.schemas import OCRFrameResult, OCRPipelineResult, OCRSceneResult

from heimdex_media_pipelines.ocr.engine import OCRPerfTimings, create_ocr_engine
from heimdex_media_pipelines.ocr.pii import redact_pii

logger = logging.getLogger(__name__)

_SCENE_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+_scene_\d+$")


def sanitize_ocr_text(text: str) -> str:
    cleaned = re.sub(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069]", "", text)
    cleaned = re.sub(r"<[^>]+>", "", cleaned)
    cleaned = html.escape(cleaned)
    return cleaned.strip()


def validate_keyframe(path: str | Path) -> bool:
    path_obj = Path(path)
    if not path_obj.exists():
        return False

    try:
        size = os.path.getsize(path_obj)
        if size > 50 * 1024 * 1024:
            logger.warning("Skipping keyframe larger than 50MB: %s", path_obj)
            return False

        try:
            image_mod = importlib.import_module("PIL.Image")
        except ImportError:
            logger.warning("Pillow is required to validate keyframe image metadata")
            return False

        with image_mod.open(path_obj) as img:
            width, height = img.size

        if width * height > 25_000_000:
            logger.warning("Skipping keyframe larger than 25M pixels: %s (%sx%s)", path_obj, width, height)
            return False

        ratio = max(width, height) / max(min(width, height), 1)
        if ratio > 20:
            logger.warning("Skipping extreme aspect-ratio keyframe: %s (%sx%s)", path_obj, width, height)
            return False

        return True
    except Exception as e:
        logger.warning("Keyframe validation failed for %s: %s", path_obj, e)
        return False


def safe_keyframe_path(scene_id: str, keyframe_dir: str | Path) -> Path | None:
    if not _SCENE_ID_RE.fullmatch(scene_id):
        return None

    base = Path(keyframe_dir).resolve()
    candidate = (Path(keyframe_dir) / f"{scene_id}.jpg").resolve()
    if not candidate.is_relative_to(base):
        return None
    return candidate


def run_ocr_pipeline(
    scenes_result_path: str,
    keyframe_dir: str,
    out_path: str,
    *,
    lang: str = "korean",
    use_gpu: bool = False,
    redact_pii_flag: bool = False,
) -> OCRPipelineResult:
    t_total = time.perf_counter()
    perf = OCRPerfTimings()

    with open(scenes_result_path, encoding="utf-8") as f:
        scenes_payload = json.load(f)

    scenes_data = scenes_payload.get("scenes", [])
    video_id = str(scenes_payload.get("video_id", ""))

    t_model = time.perf_counter()
    engine = create_ocr_engine(lang=lang, use_gpu=use_gpu)
    perf.model_load_s = round(time.perf_counter() - t_model, 3)

    scene_results: list[OCRSceneResult] = []
    confidences: list[float] = []

    for scene in scenes_data:
        scene_id = scene.get("scene_id")
        if not isinstance(scene_id, str):
            continue

        scene_result = OCRSceneResult(scene_id=scene_id)
        keyframe_path = safe_keyframe_path(scene_id, keyframe_dir)
        if keyframe_path is None or not validate_keyframe(keyframe_path):
            perf.frames_skipped += 1
            scene_results.append(scene_result)
            continue

        t_infer = time.perf_counter()
        blocks = engine.detect(keyframe_path)
        perf.ocr_inference_s += time.perf_counter() - t_infer

        t_post = time.perf_counter()
        filtered = filter_blocks_by_confidence(blocks)
        text_concat = concat_blocks(filtered)
        cleaned = sanitize_ocr_text(text_concat)
        if redact_pii_flag:
            cleaned = redact_pii(cleaned)
        gated = gate_ocr_text(cleaned)
        perf.postprocess_s += time.perf_counter() - t_post

        for block in filtered:
            confidences.append(block.confidence)

        if gated:
            scene_result.frames.append(
                OCRFrameResult(
                    frame_ts_ms=int(scene.get("keyframe_timestamp_ms", 0) or 0),
                    blocks=filtered,
                    text_concat=gated,
                    processing_time_ms=round((time.perf_counter() - t_infer) * 1000.0, 3),
                )
            )
            scene_result.ocr_text_raw = gated
            perf.frames_processed += 1
        else:
            perf.frames_skipped += 1

        scene_results.append(scene_result)

    if confidences:
        perf.avg_confidence = sum(confidences) / len(confidences)

    perf.ocr_inference_s = round(perf.ocr_inference_s, 3)
    perf.postprocess_s = round(perf.postprocess_s, 3)
    perf.total_s = round(time.perf_counter() - t_total, 3)
    perf.log(video_id=video_id, model="paddleocr")

    result = OCRPipelineResult(
        schema_version="1.0",
        pipeline_version=_pkg.__version__,
        model_version="paddleocr",
        video_id=video_id,
        scenes=scene_results,
        total_frames_processed=perf.frames_processed,
        processing_time_s=perf.total_s,
        status="success",
        meta={"perf": perf.to_dict()},
    )

    abs_out = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(abs_out) or ".", exist_ok=True)
    tmp_path = abs_out + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, abs_out)

    return result
