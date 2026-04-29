"""Pinned SigLIP2 variant + dataclass for loader configuration.

The constants here are the **single source of truth** for the SigLIP2
shape used across Heimdex. Bumping the model variant (e.g., to Large)
is a coordinated migration:

1. Bump ``DEFAULT_SIGLIP2_MODEL_ID`` here.
2. Bump ``EMBED_DIM`` to match the new variant's output.
3. Update ``drive-visual-embed-worker`` to load the same variant.
4. Add a Postgres migration on ``product_catalog_entries.siglip2_embedding``
   (and any other consumers) to widen / re-create the vector column.
5. Re-embed all OpenSearch scene-level vectors against the new variant
   (otherwise the coarse pre-filter in
   ``heimdex_media_pipelines.product_track`` becomes meaningless).

Drift between any of those steps silently breaks cosine similarity
across the platform — guard the constant carefully.
"""

from __future__ import annotations

from dataclasses import dataclass


# Variant locked to base/256 to match the deployed
# drive-visual-embed-worker exactly. See module docstring for what
# changing this implies.
DEFAULT_SIGLIP2_MODEL_ID = "google/siglip2-base-patch16-256"

# Dimensionality of the pooled image embedding for the variant above.
# ``ProductCatalogEntry.siglip2_embedding`` and the contracts
# ``ProductCatalogEntry.siglip2_embedding`` field both rely on this.
EMBED_DIM = 768


@dataclass(frozen=True)
class SiglipConfig:
    """Loader configuration. Defaults match the drive-visual-embed-worker
    deployment so reusing the singleton across modules is safe.

    Workers can override ``device`` / ``dtype`` per their environment
    (CPU for unit tests, fp16 GPU in production).
    """

    model_id: str = DEFAULT_SIGLIP2_MODEL_ID
    # ``device`` is a string ("cuda" | "cpu" | "cuda:0" …). The loader
    # resolves it to a torch.device at runtime — keeping a string here
    # avoids a torch import in the dataclass module, which keeps this
    # importable in test contexts that mock the heavy deps.
    device: str = "auto"
    # ``dtype_str``: "auto" | "fp16" | "fp32" | "bf16". "auto" picks
    # fp16 on cuda, bf16 on cpu (matches the visual-embed-worker).
    dtype_str: str = "auto"
    # ``low_cpu_mem_usage`` — kept on by default; matches visual-embed.
    low_cpu_mem_usage: bool = True
