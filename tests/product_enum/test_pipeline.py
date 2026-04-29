"""End-to-end pipeline tests with stub VLM + stub embedder.

No SigLIP2 weights, no OpenAI calls — verifies the orchestrator wires
each stage together correctly and that hallucinated VLM outputs are
dropped cleanly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from PIL import Image

from heimdex_media_pipelines.product_enum.config import EnumerationConfig
from heimdex_media_pipelines.product_enum.pipeline import (
    SceneKeyframe,
    enumerate_products,
)
from heimdex_media_pipelines.product_enum.vlm_client import (
    EnumerationDetection,
    VlmDetectionBatch,
)


def _img(w: int = 1000, h: int = 1000, fill: int = 200) -> Image.Image:
    return Image.new("RGB", (w, h), (fill, fill, fill))


def _normalize(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n > 0 else v


@dataclass
class _StubVlmClient:
    """Returns a hand-prepared list of detections per call. Tests
    construct one with the desired output and inject it into
    ``enumerate_products``."""

    detections_per_call: list[list[EnumerationDetection]]
    cost_per_call: float = 0.005

    def __post_init__(self) -> None:
        self._call_idx = 0

    def detect_products(self, *, keyframes, system_prompt, user_prompt):
        out = self.detections_per_call[self._call_idx]
        self._call_idx += 1
        return VlmDetectionBatch(detections=list(out), cost_usd=self.cost_per_call)


def _stub_embedder(vector_per_label: dict[str, list[float]]):
    """Build an embedder that returns a deterministic vector per
    detection label. Useful for steering clustering tests."""

    def _embed(images):
        # We rely on the caller to track which crops correspond to
        # which detections. For this stub, we return a fixed vector
        # for every image — which means the LABEL of each detection
        # has to be reconstructed by the test from order. Tests that
        # need label-aware embeddings should pre-construct the order.
        # Default: cycle through the dict in insertion order.
        out = []
        labels = list(vector_per_label.keys())
        for i, _ in enumerate(images):
            out.append(list(vector_per_label[labels[i % len(labels)]]))
        return out

    return _embed


# ---------- happy path ----------

def test_two_distinct_products_produce_two_clusters():
    keyframes = [
        SceneKeyframe(scene_id="s1", frame_idx=0, image=_img()),
        SceneKeyframe(scene_id="s2", frame_idx=0, image=_img()),
        SceneKeyframe(scene_id="s3", frame_idx=0, image=_img()),
        SceneKeyframe(scene_id="s4", frame_idx=0, image=_img()),
    ]
    # 1 batched call → 4 detections, alternating labels
    detections = [
        EnumerationDetection("s1", 0, "serum",   (400, 400, 200, 200), 0.9),
        EnumerationDetection("s2", 0, "serum",   (400, 400, 200, 200), 0.85),
        EnumerationDetection("s3", 0, "cleanser",(450, 450, 100, 100), 0.9),
        EnumerationDetection("s4", 0, "cleanser",(450, 450, 100, 100), 0.85),
    ]
    vlm = _StubVlmClient(detections_per_call=[detections])
    # Orthogonal embeddings per label so clustering produces 2.
    serum = _normalize([1.0, 0.0])
    cleanser = _normalize([0.0, 1.0])
    embed_lookup = {0: serum, 1: serum, 2: cleanser, 3: cleanser}
    embedder = lambda images: [embed_lookup[i] for i in range(len(images))]

    cfg = EnumerationConfig(
        max_keyframes=10, vlm_batch_size=10,
        min_supporting_keyframes=2, min_prominence_pct=0.001,
        min_enumeration_confidence=0.6,
    )
    products, cost = enumerate_products(
        keyframes=keyframes,
        vlm_client=vlm,
        embedder=embedder,
        system_prompt="sys",
        user_prompt_template="batch of {num_keyframes}",
        config=cfg,
    )
    assert cost == 0.005
    assert len(products) == 2
    labels = sorted(p.llm_label for p in products)
    assert labels == ["cleanser", "serum"]
    for p in products:
        assert p.rejected_reason is None
        assert p.cluster_size == 2


def test_zero_keyframes_returns_empty():
    vlm = _StubVlmClient(detections_per_call=[])
    products, cost = enumerate_products(
        keyframes=[], vlm_client=vlm, embedder=lambda x: [],
        system_prompt="x", user_prompt_template="x",
        config=EnumerationConfig(),
    )
    assert products == []
    assert cost == 0.0


def test_subsamples_when_keyframes_exceed_max():
    """30 keyframes with max=5 — pipeline should pass exactly 5 to the
    VLM in chronological-ish stride. Verifies cost/recall trade-off
    is enforceable via the config."""
    keyframes = [
        SceneKeyframe(scene_id=f"s{i}", frame_idx=0, image=_img())
        for i in range(30)
    ]
    captured: list[int] = []

    class _RecorderVlm:
        def detect_products(self, *, keyframes, system_prompt, user_prompt):
            captured.append(len(keyframes))
            return VlmDetectionBatch(detections=[], cost_usd=0.001)

    cfg = EnumerationConfig(max_keyframes=5, vlm_batch_size=10)
    enumerate_products(
        keyframes=keyframes, vlm_client=_RecorderVlm(),
        embedder=lambda x: [],
        system_prompt="s", user_prompt_template="b",
        config=cfg,
    )
    assert sum(captured) == 5  # capped at max_keyframes


def test_hallucinated_scene_id_dropped():
    """VLM returns a detection referencing a scene_id not in the
    batch. Pipeline must drop it without crashing the run."""
    keyframes = [SceneKeyframe(scene_id="s1", frame_idx=0, image=_img())]
    detections = [
        EnumerationDetection("s1", 0, "real",   (100, 100, 200, 200), 0.9),
        EnumerationDetection("ghost", 0, "fake", (0, 0, 50, 50), 0.9),
    ]
    vlm = _StubVlmClient(detections_per_call=[detections])
    embed_lookup = {0: _normalize([1.0, 0.0])}  # only 1 valid
    embedder = lambda images: [embed_lookup[i] for i in range(len(images))]

    cfg = EnumerationConfig(
        min_supporting_keyframes=1, min_prominence_pct=0.001,
        min_enumeration_confidence=0.6,
    )
    products, _ = enumerate_products(
        keyframes=keyframes, vlm_client=vlm, embedder=embedder,
        system_prompt="s", user_prompt_template="b",
        config=cfg,
    )
    # Only the real detection survives → 1 cluster.
    assert len(products) == 1
    assert products[0].llm_label == "real"


def test_rejected_reason_propagated_to_output():
    """Rejected clusters should still appear in the output (so the
    worker persists them), with rejected_reason set."""
    keyframes = [
        SceneKeyframe(scene_id="s1", frame_idx=0, image=_img()),
    ]
    # Single detection → single_keyframe rejection (cluster size 1).
    detections = [
        EnumerationDetection("s1", 0, "lonely", (400, 400, 200, 200), 0.9),
    ]
    vlm = _StubVlmClient(detections_per_call=[detections])
    embedder = lambda images: [_normalize([1.0, 0.0])]
    cfg = EnumerationConfig(
        min_supporting_keyframes=2,  # require ≥ 2; 1 detection fails
        min_prominence_pct=0.001,
        min_enumeration_confidence=0.6,
    )
    products, _ = enumerate_products(
        keyframes=keyframes, vlm_client=vlm, embedder=embedder,
        system_prompt="s", user_prompt_template="b",
        config=cfg,
    )
    assert len(products) == 1
    assert products[0].rejected_reason == "single_keyframe"


def test_cost_aggregated_across_batches():
    """Multiple batched VLM calls — total cost is the sum."""
    # 20 keyframes, batch_size=10 → 2 batches.
    keyframes = [
        SceneKeyframe(scene_id=f"s{i}", frame_idx=0, image=_img())
        for i in range(20)
    ]
    vlm = _StubVlmClient(
        detections_per_call=[[], []],
        cost_per_call=0.0042,
    )
    products, cost = enumerate_products(
        keyframes=keyframes, vlm_client=vlm,
        embedder=lambda x: [],
        system_prompt="s", user_prompt_template="b",
        config=EnumerationConfig(max_keyframes=20, vlm_batch_size=10),
    )
    assert products == []
    assert abs(cost - 0.0084) < 1e-9
