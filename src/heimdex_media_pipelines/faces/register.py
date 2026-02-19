"""Face identity template registration.

Extracts face embeddings from reference images, computes a centroid embedding,
and saves an identity template JSON with exemplars and intra-class statistics.
"""

import json
import math
import os
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import cv2

try:
    from insightface.app import FaceAnalysis
except Exception:
    FaceAnalysis = None


def _load_face_app(det_size: int = 640, ctx_id: int = -1):
    if FaceAnalysis is None:
        raise RuntimeError(
            "Face embedding requires insightface. Install with: python -m pip install insightface onnxruntime"
        )
    if ctx_id >= 0:
        from heimdex_media_pipelines.device import detect_onnx_providers
        providers = detect_onnx_providers()
    else:
        providers = ["CPUExecutionProvider"]
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))
    return app


def _l2_norm(vec: List[float]) -> float:
    return math.sqrt(sum(x * x for x in vec))


def _mean_vector(vectors: List[List[float]]) -> List[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    acc = [0.0] * dim
    for vec in vectors:
        for i, val in enumerate(vec):
            acc[i] += float(val)
    n = float(len(vectors))
    return [v / n for v in acc]


def _normalize(vec: List[float]) -> List[float]:
    norm = _l2_norm(vec)
    if norm <= 0:
        return vec
    return [v / norm for v in vec]


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    denom = _l2_norm(a) * _l2_norm(b)
    if denom <= 0:
        return 0.0
    return float(sum(x * y for x, y in zip(a, b)) / denom)


def _default_artifacts_dir() -> str:
    return os.path.join(os.getcwd(), "artifacts")


def _iter_ref_images(ref_images: Iterable[str], ref_dir: Optional[str]) -> List[str]:
    paths = [p for p in ref_images if p]
    if ref_dir:
        if not os.path.isdir(ref_dir):
            raise FileNotFoundError(ref_dir)
        for name in sorted(os.listdir(ref_dir)):
            if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                paths.append(os.path.join(ref_dir, name))
    unique = []
    seen = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def _extract_embedding_from_image(app, image_path: str) -> Tuple[List[float], Dict[str, float]]:
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    faces = app.get(image)
    if not faces:
        raise RuntimeError(f"No face detected in image: {image_path}")

    best_face = max(faces, key=lambda f: float(getattr(f, "det_score", 0.0)))
    emb = getattr(best_face, "embedding", None)
    if emb is None:
        raise RuntimeError(f"Failed to extract embedding from image: {image_path}")

    x1, y1, x2, y2 = best_face.bbox.astype(int).tolist()
    meta = {
        "det_conf": float(getattr(best_face, "det_score", 0.0)),
        "bbox_x1": float(x1),
        "bbox_y1": float(y1),
        "bbox_x2": float(x2),
        "bbox_y2": float(y2),
    }
    return [float(x) for x in emb], meta


def build_identity_template(
    identity_id: str,
    ref_images: Iterable[str],
    ref_dir: Optional[str] = None,
    out_path: Optional[str] = None,
    det_size: int = 640,
    ctx_id: int = -1,
    exemplars_k: int = 5,
) -> str:
    if not identity_id:
        raise ValueError("identity_id is required")

    image_paths = _iter_ref_images(ref_images, ref_dir)
    if not image_paths:
        raise ValueError("At least one reference image is required")

    app = _load_face_app(det_size=det_size, ctx_id=ctx_id)

    embeddings: List[List[float]] = []
    sources: List[Dict[str, object]] = []
    for image_path in image_paths:
        embedding, meta = _extract_embedding_from_image(app, image_path)
        embeddings.append(embedding)
        sources.append({"path": image_path, **meta})

    centroid_raw = _mean_vector(embeddings)
    centroid = _normalize(centroid_raw)

    sims = [_cosine_similarity(centroid, emb) for emb in embeddings]
    mean_sim = sum(sims) / len(sims) if sims else 0.0
    var = sum((s - mean_sim) ** 2 for s in sims) / len(sims) if sims else 0.0
    std_sim = math.sqrt(var)

    exemplars: List[Dict[str, object]] = []
    for idx in sorted(range(len(embeddings)), key=lambda i: sims[i], reverse=True)[: max(1, exemplars_k)]:
        exemplars.append(
            {
                "image_path": sources[idx]["path"],
                "similarity": float(sims[idx]),
                "embedding": _normalize(embeddings[idx]),
            }
        )

    payload = {
        "identity_id": identity_id,
        "created_at": datetime.utcnow().isoformat(),
        "embedding_dim": len(centroid),
        "centroid": centroid,
        "exemplars": exemplars,
        "intra_class": {
            "count": len(embeddings),
            "mean_cosine": float(mean_sim),
            "std_cosine": float(std_sim),
            "min_cosine": float(min(sims)) if sims else 0.0,
            "max_cosine": float(max(sims)) if sims else 0.0,
        },
        "sources": sources,
    }

    if out_path is None:
        out_dir = os.path.join(_default_artifacts_dir(), "identities")
        out_path = os.path.join(out_dir, f"{identity_id}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    return out_path
