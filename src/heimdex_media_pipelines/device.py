"""Shared GPU / device detection for all pipelines.

Centralises hardware probing so that faces (ONNX), speech (CTranslate2), and
OCR (PaddlePaddle) pipelines can auto-select the best available accelerator
without duplicating detection logic.

Design principles:
  - Every function is safe to call on any platform (macOS, Windows, Linux).
  - Import errors for optional packages are caught — callers always get a
    usable fallback (CPU).
  - Results are logged at INFO so the agent and operator can see which path
    was taken.
"""

from __future__ import annotations

import logging
import platform
from typing import Any, List, Tuple, Union

logger = logging.getLogger(__name__)

# Type accepted by onnxruntime.InferenceSession / insightface FaceAnalysis
ProviderOption = Union[str, Tuple[str, dict]]


def detect_onnx_providers() -> List[ProviderOption]:
    """Return the best available ONNX Runtime execution providers.

    Suitable for passing directly to ``FaceAnalysis(providers=...)``.

    Priority order (first match wins):
      macOS Apple Silicon  -> CoreMLExecutionProvider (CPU+GPU+ANE)
      NVIDIA GPU           -> CUDAExecutionProvider
      DirectX 12 GPU       -> DmlExecutionProvider
      Fallback             -> CPUExecutionProvider
    """
    try:
        import onnxruntime as ort

        available = ort.get_available_providers()
    except ImportError:
        logger.debug("onnxruntime not installed, defaulting to CPU")
        return ["CPUExecutionProvider"]
    except Exception as exc:
        logger.debug("onnxruntime provider check failed: %s", exc)
        return ["CPUExecutionProvider"]

    system = platform.system()

    if system == "Darwin" and "CoreMLExecutionProvider" in available:
        logger.info("GPU: using CoreMLExecutionProvider (macOS)")
        return [
            (
                "CoreMLExecutionProvider",
                {
                    "MLComputeUnits": "ALL",
                },
            ),
            "CPUExecutionProvider",
        ]

    # NVIDIA CUDA — requires onnxruntime-gpu
    if "CUDAExecutionProvider" in available:
        logger.info("GPU: using CUDAExecutionProvider")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    # DirectML — AMD / Intel / any DirectX 12 GPU on Windows
    if "DmlExecutionProvider" in available:
        logger.info("GPU: using DmlExecutionProvider (DirectML)")
        return ["DmlExecutionProvider", "CPUExecutionProvider"]

    logger.info("GPU: no ONNX accelerator found, using CPU")
    return ["CPUExecutionProvider"]


def detect_whisper_device() -> Tuple[str, str]:
    """Return ``(device, compute_type)`` optimised for faster-whisper.

    faster-whisper is backed by CTranslate2 which does **not** support
    Apple Metal / MPS.  macOS always gets ``("cpu", "int8")``.

    On Windows / Linux, CUDA is detected via ``ctranslate2`` first, then
    ``torch`` as fallback.
    """
    system = platform.system()

    if system == "Darwin":
        # CTranslate2 has no Metal backend; Apple Accelerate is used for
        # CPU BLAS automatically.
        logger.info("whisper device: macOS detected, using cpu + int8")
        return "cpu", "int8"

    # Non-macOS: probe for CUDA via ctranslate2 (fast, no torch import)
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            logger.info("whisper device: CUDA detected via ctranslate2, using cuda + float16")
            return "cuda", "float16"
    except ImportError:
        logger.debug("ctranslate2 not available for CUDA probe")
    except Exception as exc:
        logger.debug("ctranslate2 CUDA check failed: %s", exc)

    # Fallback: try torch (heavier import but may be already loaded)
    try:
        import torch

        if torch.cuda.is_available():
            logger.info("whisper device: CUDA detected via torch, using cuda + float16")
            return "cuda", "float16"
    except ImportError:
        pass
    except Exception as exc:
        logger.debug("torch CUDA check failed: %s", exc)

    logger.info("whisper device: no CUDA found, using cpu + int8")
    return "cpu", "int8"


def detect_paddle_gpu() -> bool:
    """Return True when PaddlePaddle has working GPU support.

    macOS always returns False (no ``paddlepaddle-gpu`` package for macOS).
    On Windows / Linux, checks ``paddlepaddle-gpu`` CUDA support.
    """
    if platform.system() == "Darwin":
        return False

    try:
        import paddle  # type: ignore[import-untyped]

        compiled = getattr(paddle.device, "is_compiled_with_cuda", lambda: False)()
        if not compiled:
            return False
        count = paddle.device.cuda.device_count()
        if count > 0:
            logger.info("paddle GPU: %d CUDA device(s) available", count)
            return True
    except ImportError:
        logger.debug("paddle not installed, GPU unavailable")
    except Exception as exc:
        logger.debug("paddle GPU check failed: %s", exc)

    return False


def get_onnx_provider_names() -> List[str]:
    """Return flat list of available ONNX Runtime provider names (strings only).

    Useful for doctor / diagnostic output.
    """
    try:
        import onnxruntime as ort

        return list(ort.get_available_providers())
    except Exception:
        return []
