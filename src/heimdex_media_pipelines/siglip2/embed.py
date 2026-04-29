"""Embed PIL images with the shared SigLIP2 vision encoder.

Output is L2-normalized 768-dim ``list[float]`` so cosine similarity
reduces to a dot product — both the OpenSearch coarse pre-filter and
the local precise pass in
``heimdex_media_pipelines.product_track`` rely on this normalization.

Two entrypoints:
* :func:`embed_pil_image` — single image, returns a single vector.
* :func:`embed_pil_image_batch` — many images, batched to GPU memory.

Workers call these directly; tests inject a stub model via
``siglip2.set_singleton_for_testing`` and bypass the heavy paths.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from heimdex_media_pipelines.siglip2.config import EMBED_DIM
from heimdex_media_pipelines.siglip2.loader import LoadedSiglip, load

if TYPE_CHECKING:  # pragma: no cover
    from PIL import Image


def embed_pil_image(image: "Image.Image", *, loaded: LoadedSiglip | None = None) -> list[float]:
    """Encode a single PIL image into a 768-dim L2-normalized vector.

    ``loaded`` lets callers pass an already-loaded singleton (the
    common case for batched tasks where the caller wants to amortize
    the load across many calls). ``None`` triggers a one-shot
    :func:`heimdex_media_pipelines.siglip2.loader.load` which is
    idempotent.
    """
    if loaded is None:
        loaded = load()

    import torch
    import torch.nn.functional as F

    inputs = loaded.processor(images=image.convert("RGB"), return_tensors="pt")
    inputs = _to_device(inputs, loaded)

    with torch.no_grad():
        outputs = loaded.model(**inputs)

    pooled = outputs.pooler_output.float()  # [1, 768]
    normalized = F.normalize(pooled, p=2, dim=-1)
    vec = normalized.squeeze(0).cpu().tolist()
    if len(vec) != EMBED_DIM:
        # Defensive — guards against an accidental model variant swap
        # that produces the wrong dimensionality. Caller's downstream
        # storage (pgvector(768)) would otherwise fail much later
        # with a less actionable error.
        raise RuntimeError(
            f"siglip2 embedding dim mismatch: got {len(vec)}, want {EMBED_DIM}"
        )
    return vec


def embed_pil_image_batch(
    images: list["Image.Image"],
    *,
    loaded: LoadedSiglip | None = None,
    batch_size: int = 16,
) -> list[list[float]]:
    """Encode a list of PIL images into 768-dim L2-normalized vectors.

    Returned vectors are in the same order as ``images``. Internally
    chunks into mini-batches of ``batch_size`` to bound peak GPU
    memory; 16 matches drive-visual-embed-worker's default and stays
    within an A100's working set with the base/256 variant.
    """
    if not images:
        return []
    if loaded is None:
        loaded = load()

    import torch
    import torch.nn.functional as F

    out: list[list[float]] = []
    for start in range(0, len(images), batch_size):
        chunk = [img.convert("RGB") for img in images[start:start + batch_size]]
        inputs = loaded.processor(images=chunk, return_tensors="pt")
        inputs = _to_device(inputs, loaded)
        with torch.no_grad():
            outputs = loaded.model(**inputs)
        pooled = outputs.pooler_output.float()  # [B, 768]
        normalized = F.normalize(pooled, p=2, dim=-1)
        for vec in normalized.cpu().tolist():
            if len(vec) != EMBED_DIM:
                raise RuntimeError(
                    f"siglip2 embedding dim mismatch: got {len(vec)}, want {EMBED_DIM}"
                )
            out.append(vec)
    return out


def _to_device(inputs: Any, loaded: LoadedSiglip) -> Any:
    """Move processor outputs to the model's device + dtype.

    Lifted from drive-visual-embed-worker — the dtype check on each
    tensor is needed because the processor returns float32 even when
    the model is loaded in fp16/bf16. Without the cast, the matmul
    inside the model raises a dtype mismatch.
    """
    import torch
    for key in inputs:
        inputs[key] = inputs[key].to(loaded.device)
        if (
            inputs[key].dtype == torch.float32
            and loaded.model.dtype != torch.float32
        ):
            inputs[key] = inputs[key].to(loaded.model.dtype)
    return inputs
