"""Lazy singleton loader for the shared SigLIP2 vision encoder.

The loader is module-global by design — workers run a single model
per process and we never want to load it twice. Tests reset the
singleton between cases via :func:`reset_singleton` or inject a fake
via :func:`set_singleton_for_testing`.

Heavy imports (``torch``, ``transformers``) are deferred to the
loader function so this module can be imported in test contexts that
mock the model entirely.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

from heimdex_media_pipelines.siglip2.config import SiglipConfig

logger = logging.getLogger(__name__)


@dataclass
class LoadedSiglip:
    """Container for the loaded SigLIP2 vision model + processor +
    device. ``model`` and ``processor`` are typed as ``Any`` so this
    module stays importable without ``transformers`` / ``torch`` in
    the import path."""

    model: Any
    processor: Any
    device: Any
    dtype: Any
    config: SiglipConfig


# Module-level singleton (None until first load). The lock guards
# the lazy init from racing under concurrent first-use.
_singleton: LoadedSiglip | None = None
_lock = threading.Lock()


def get_singleton() -> LoadedSiglip | None:
    """Return the loaded singleton, or ``None`` if not yet loaded.

    Useful in tests that need to assert "we did not unintentionally
    load the model".
    """
    return _singleton


def load(config: SiglipConfig | None = None) -> LoadedSiglip:
    """Load (or return) the shared SigLIP2 model.

    Idempotent — repeated calls return the same instance regardless
    of the ``config`` passed. The first call wins; later calls with a
    different config are logged as warnings but do not reload.

    Reload requires :func:`reset_singleton` first. This shape matches
    how production workers actually use the model: single model per
    process, lifetime tied to the worker's lifetime.
    """
    global _singleton
    if _singleton is not None:
        if config is not None and config != _singleton.config:
            logger.warning(
                "siglip2_load_called_with_different_config_ignored",
                extra={
                    "loaded_model_id": _singleton.config.model_id,
                    "requested_model_id": config.model_id,
                },
            )
        return _singleton

    cfg = config or SiglipConfig()

    with _lock:
        if _singleton is not None:
            return _singleton

        import torch  # local import keeps this module importable without torch
        from transformers import AutoImageProcessor, SiglipVisionModel

        device = _resolve_device(cfg.device)
        dtype = _resolve_dtype(cfg.dtype_str, device)

        logger.info(
            "siglip2_loading",
            extra={
                "model_id": cfg.model_id,
                "device": str(device),
                "dtype": str(dtype),
            },
        )

        processor = AutoImageProcessor.from_pretrained(cfg.model_id)
        model = SiglipVisionModel.from_pretrained(
            cfg.model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=cfg.low_cpu_mem_usage,
        ).to(device)
        model.eval()

        loaded = LoadedSiglip(
            model=model,
            processor=processor,
            device=device,
            dtype=dtype,
            config=cfg,
        )
        _singleton = loaded

        params_m = round(sum(p.numel() for p in model.parameters()) / 1e6, 1)
        logger.info(
            "siglip2_loaded",
            extra={"params_m": params_m, "device": str(device)},
        )
        return loaded


def reset_singleton() -> None:
    """Evict the cached model. Used by tests; do NOT call from
    production code. Subsequent :func:`load` calls will reload from
    scratch.
    """
    global _singleton
    with _lock:
        _singleton = None


def set_singleton_for_testing(loaded: LoadedSiglip) -> None:
    """Inject a fake :class:`LoadedSiglip` (e.g., a stub model that
    returns deterministic embeddings). Tests use this to avoid
    downloading real weights.
    """
    global _singleton
    with _lock:
        _singleton = loaded


# ---------- helpers ----------

def _resolve_device(device_str: str) -> Any:
    import torch
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _resolve_dtype(dtype_str: str, device: Any) -> Any:
    import torch
    if dtype_str == "fp16":
        return torch.float16
    if dtype_str == "fp32":
        return torch.float32
    if dtype_str == "bf16":
        return torch.bfloat16
    if dtype_str == "auto":
        # Match drive-visual-embed-worker — fp16 on cuda, bf16 on cpu.
        return torch.float16 if device.type == "cuda" else torch.bfloat16
    raise ValueError(f"unknown dtype_str: {dtype_str!r}")
