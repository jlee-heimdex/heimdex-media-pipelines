"""Shared SigLIP2 vision encoder for Heimdex pipelines.

Variant pinned to ``google/siglip2-base-patch16-256`` (768-dim) — must
match the model deployed in ``drive-visual-embed-worker`` exactly so
embeddings persisted at ingest time stay comparable to embeddings
produced at inference time by the product workers (see
``shorts-auto-product-v2.md`` plan §6 — the OS coarse pre-filter
depends on this invariant).

Pure library: no S3, no DB, no HTTP. Workers handle I/O and call
:func:`embed_pil_image` / :func:`embed_pil_image_batch`. The model
loads lazily on first call and stays in memory for the process
lifetime (idempotent — repeated calls are no-ops).

For tests, :func:`reset_singleton` evicts the cached model; injection
of a fake model is supported via :func:`set_singleton_for_testing`.
"""

from heimdex_media_pipelines.siglip2.config import (
    DEFAULT_SIGLIP2_MODEL_ID,
    EMBED_DIM,
    SiglipConfig,
)
from heimdex_media_pipelines.siglip2.embed import (
    embed_pil_image,
    embed_pil_image_batch,
)
from heimdex_media_pipelines.siglip2.loader import (
    get_singleton,
    load,
    reset_singleton,
    set_singleton_for_testing,
)

__all__ = [
    "DEFAULT_SIGLIP2_MODEL_ID",
    "EMBED_DIM",
    "SiglipConfig",
    "embed_pil_image",
    "embed_pil_image_batch",
    "get_singleton",
    "load",
    "reset_singleton",
    "set_singleton_for_testing",
]
