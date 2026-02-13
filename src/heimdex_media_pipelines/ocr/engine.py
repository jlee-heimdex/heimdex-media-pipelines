from __future__ import annotations

import json
import importlib
import logging
import re
from dataclasses import asdict, dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Protocol

from heimdex_media_contracts.ocr.schemas import OCRBlock

logger = logging.getLogger(__name__)

_MIN_PADDLE_VERSION = "2.6.1"


def _parse_version(v: str) -> tuple[int, ...]:
    parts = re.findall(r"\d+", v)
    nums = [int(x) for x in parts[:3]]
    while len(nums) < 3:
        nums.append(0)
    return tuple(nums)


class OCREngine(Protocol):
    def detect(self, image_path: str | Path) -> list[OCRBlock]: ...


@dataclass
class OCRPerfTimings:
    model_load_s: float = 0.0
    ocr_inference_s: float = 0.0
    postprocess_s: float = 0.0
    total_s: float = 0.0
    frames_processed: int = 0
    frames_skipped: int = 0
    avg_confidence: float = 0.0

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)

    def log(self, video_id: str = "", model: str = "") -> None:
        payload = {
            "event": "ocr_perf",
            "video_id": video_id,
            "model": model,
            **{k: round(v, 3) if isinstance(v, float) else v for k, v in self.to_dict().items()},
        }
        logger.info("ocr_perf %s", json.dumps(payload, ensure_ascii=False))


class PaddleOCREngine:

    def __init__(self, lang: str = "korean", use_angle_cls: bool = True, use_gpu: bool = False):
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self.use_gpu = use_gpu
        self._model: Any = None
        self._api_version: int = 0

    def _load_model(self) -> None:
        if self._model is not None:
            return
        paddle_module_name = "paddle" + "ocr"
        paddleocr_module = importlib.import_module(paddle_module_name)
        paddle_ocr_cls = getattr(paddleocr_module, "PaddleOCR")

        if hasattr(paddle_ocr_cls, "predict"):
            self._model = paddle_ocr_cls(lang=self.lang)
            self._api_version = 3
        else:
            self._model = paddle_ocr_cls(
                use_angle_cls=self.use_angle_cls,
                lang=self.lang,
                use_gpu=self.use_gpu,
                show_log=False,
            )
            self._api_version = 2

    def _detect_v3(self, image_path: str | Path, width: int, height: int) -> list[OCRBlock]:
        results = self._model.predict(str(image_path))
        blocks: list[OCRBlock] = []
        for page in results:
            texts = page.get("rec_texts", [])
            scores = page.get("rec_scores", [])
            polys = page.get("dt_polys", [])
            for text, conf, poly in zip(texts, scores, polys):
                text = str(text).strip()
                conf = float(conf)
                if not text:
                    continue
                xs = [float(p[0]) for p in poly]
                ys = [float(p[1]) for p in poly]
                x_min = max(0.0, min(1.0, min(xs) / max(width, 1)))
                y_min = max(0.0, min(1.0, min(ys) / max(height, 1)))
                x_max = max(0.0, min(1.0, max(xs) / max(width, 1)))
                y_max = max(0.0, min(1.0, max(ys) / max(height, 1)))
                blocks.append(
                    OCRBlock(text=text, confidence=conf, bbox=[x_min, y_min, x_max, y_max])
                )
        return blocks

    def _detect_v2(self, image_path: str | Path, width: int, height: int) -> list[OCRBlock]:
        result = self._model.ocr(str(image_path), cls=self.use_angle_cls)
        page = result[0] if result and isinstance(result, list) else []
        if not page:
            return []

        blocks: list[OCRBlock] = []
        for line in page:
            if not line or len(line) < 2:
                continue
            points = line[0]
            text_conf = line[1]
            if not isinstance(text_conf, (tuple, list)) or len(text_conf) < 2:
                continue
            text = str(text_conf[0]).strip()
            conf = float(text_conf[1])
            if not text:
                continue
            xs = [float(p[0]) for p in points]
            ys = [float(p[1]) for p in points]
            x_min = max(0.0, min(1.0, min(xs) / max(width, 1)))
            y_min = max(0.0, min(1.0, min(ys) / max(height, 1)))
            x_max = max(0.0, min(1.0, max(xs) / max(width, 1)))
            y_max = max(0.0, min(1.0, max(ys) / max(height, 1)))
            blocks.append(
                OCRBlock(text=text, confidence=conf, bbox=[x_min, y_min, x_max, y_max])
            )
        return blocks

    def detect(self, image_path: str | Path) -> list[OCRBlock]:
        try:
            if self._model is None:
                self._load_model()

            image_mod = importlib.import_module("PIL.Image")

            with image_mod.open(image_path) as img:
                width, height = img.size

            if self._api_version == 3:
                return self._detect_v3(image_path, width, height)
            else:
                return self._detect_v2(image_path, width, height)
        except Exception as e:
            logger.warning("OCR detection failed for %s: %s", image_path, e)
            return []


def _resolve_paddle_version() -> str:
    errors: list[str] = []
    for pkg in ("paddlepaddle", "paddlepaddle-gpu"):
        try:
            return importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError as e:
            errors.append(str(e))
    raise ImportError("paddlepaddle is not installed. Install with: pip install paddlepaddle>=2.6.1")


def create_ocr_engine(lang: str = "korean", use_gpu: bool = False) -> OCREngine:
    version_str = _resolve_paddle_version()
    if _parse_version(version_str) < _parse_version(_MIN_PADDLE_VERSION):
        raise RuntimeError(
            f"paddlepaddle>={_MIN_PADDLE_VERSION} is required for OCR. Detected version: {version_str}"
        )

    try:
        _ = importlib_metadata.version("paddleocr")
    except importlib_metadata.PackageNotFoundError as e:
        raise ImportError("paddleocr is not installed. Install with: pip install paddleocr>=2.8.0") from e

    return PaddleOCREngine(lang=lang, use_gpu=use_gpu)
