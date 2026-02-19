"""Face embedding extraction from video frames.

Reads detection results (JSONL), re-opens frames, matches detections to
insightface faces by IoU, and extracts ArcFace embeddings with quality scoring.
"""

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2

try:
    from insightface.app import FaceAnalysis
    from insightface.utils import face_align
except Exception:
    FaceAnalysis = None
    face_align = None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _bbox_iou(a: Dict[str, Any], b: Iterable[float]) -> float:
    ax1, ay1, aw, ah = float(a["x"]), float(a["y"]), float(a["w"]), float(a["h"])
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx1, by1, bx2, by2 = map(float, b)
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0
    a_area = max(0.0, aw) * max(0.0, ah)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = a_area + b_area - inter_area
    if denom <= 0.0:
        return 0.0
    return inter_area / denom


def _compute_blur_score(crop_bgr) -> float:
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _compute_quality(
    blur_score: float,
    area_ratio: float,
    det_conf: float,
    blur_ref: float = 100.0,
    weights: Tuple[float, float, float] = (0.4, 0.3, 0.3),
) -> float:
    blur_norm = blur_score / (blur_score + max(1.0, blur_ref))
    area_norm = _clamp(area_ratio, 0.0, 1.0)
    conf_norm = _clamp(det_conf, 0.0, 1.0)
    w_blur, w_area, w_conf = weights
    denom = w_blur + w_area + w_conf
    if denom <= 0:
        return 0.0
    return (blur_norm * w_blur + area_norm * w_area + conf_norm * w_conf) / denom


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


def _embedding_from_crop(rec_model, crop_bgr, kps=None) -> Optional[List[float]]:
    if rec_model is None or crop_bgr is None or crop_bgr.size == 0:
        return None
    try:
        if kps is not None and face_align is not None:
            aligned = face_align.norm_crop(crop_bgr, kps)
            emb = rec_model.get(aligned)
        else:
            emb = rec_model.get(crop_bgr)
        if emb is None:
            return None
        return [float(x) for x in emb]
    except Exception:
        return None


def _clip_bbox(x: int, y: int, w: int, h: int, frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
    x1 = _clamp(float(x), 0.0, float(frame_w))
    y1 = _clamp(float(y), 0.0, float(frame_h))
    x2 = _clamp(float(x + w), 0.0, float(frame_w))
    y2 = _clamp(float(y + h), 0.0, float(frame_h))
    w2 = max(0, int(x2 - x1))
    h2 = max(0, int(y2 - y1))
    return int(x1), int(y1), w2, h2


def _iter_detections(detections_path: str) -> Iterable[Dict[str, Any]]:
    with open(detections_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _load_frame_from_video(cap, ts: float):
    cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return frame


def _load_frame_from_image(image_path: str):
    if not os.path.exists(image_path):
        return None
    frame = cv2.imread(image_path)
    return frame


def extract_embeddings(
    video_path: str,
    detections_path: str,
    q_min: Optional[float] = None,
    align: bool = False,
    det_size: int = 640,
    ctx_id: int = -1,
    iou_threshold: float = 0.3,
    blur_ref: float = 100.0,
    quality_weights: Tuple[float, float, float] = (0.4, 0.3, 0.3),
) -> List[Dict[str, Any]]:
    if not os.path.exists(detections_path):
        raise FileNotFoundError(detections_path)

    app = _load_face_app(det_size=det_size, ctx_id=ctx_id)
    rec_model = app.models.get("recognition") if hasattr(app, "models") else None

    cap = None
    if os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap = None

    embeddings: List[Dict[str, Any]] = []
    try:
        for row in _iter_detections(detections_path):
            image_path = row.get("image_path")
            ts = float(row.get("ts", 0.0))

            if image_path:
                frame = _load_frame_from_image(image_path)
            elif cap is not None:
                frame = _load_frame_from_video(cap, ts)
            else:
                continue

            if frame is None:
                continue

            frame_h, frame_w = frame.shape[:2]
            faces = app.get(frame)

            for bbox in row.get("bboxes", []):
                x, y, w, h = (
                    int(bbox.get("x", 0)),
                    int(bbox.get("y", 0)),
                    int(bbox.get("w", 0)),
                    int(bbox.get("h", 0)),
                )
                det_conf = float(bbox.get("det_conf", 0.0))
                x, y, w, h = _clip_bbox(x, y, w, h, frame_w, frame_h)
                if w <= 0 or h <= 0:
                    continue

                crop = frame[y : y + h, x : x + w]
                blur_score = _compute_blur_score(crop)
                area_ratio = (w * h) / float(frame_w * frame_h) if frame_w and frame_h else 0.0
                quality = _compute_quality(
                    blur_score,
                    area_ratio,
                    det_conf,
                    blur_ref=blur_ref,
                    weights=quality_weights,
                )
                if q_min is not None and quality < q_min:
                    continue

                matched_face = None
                best_iou = 0.0
                for face in faces:
                    iou = _bbox_iou(bbox, face.bbox)
                    if iou > best_iou:
                        best_iou = iou
                        matched_face = face

                embedding: Optional[List[float]] = None
                if matched_face is not None and best_iou >= iou_threshold:
                    if align:
                        kps = getattr(matched_face, "kps", None)
                        if kps is not None:
                            embedding = _embedding_from_crop(rec_model, frame, kps=kps)
                    else:
                        emb = getattr(matched_face, "embedding", None)
                        if emb is not None:
                            embedding = [float(x) for x in emb]

                if embedding is None:
                    embedding = _embedding_from_crop(rec_model, crop)

                if embedding is None:
                    continue

                embeddings.append(
                    {
                        "ts": ts,
                        "bbox": {"x": x, "y": y, "w": w, "h": h},
                        "det_conf": det_conf,
                        "blur": blur_score,
                        "bbox_area_ratio": area_ratio,
                        "quality": quality,
                        "embedding": embedding,
                    }
                )
    finally:
        if cap is not None:
            cap.release()

    return embeddings


def _write_jsonl(records: Iterable[Dict[str, Any]], out_path: str) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
    return out_path


def run_embeddings(
    video_path: str,
    detections_path: str,
    out_dir: Optional[str] = None,
    q_min: Optional[float] = None,
    align: bool = False,
    det_size: int = 640,
    ctx_id: int = -1,
    iou_threshold: float = 0.3,
) -> Dict[str, Optional[str]]:
    if out_dir is None:
        out_dir = os.path.dirname(detections_path)
    records = extract_embeddings(
        video_path,
        detections_path,
        q_min=q_min,
        align=align,
        det_size=det_size,
        ctx_id=ctx_id,
        iou_threshold=iou_threshold,
    )
    jsonl_path = _write_jsonl(records, os.path.join(out_dir, "embeddings.jsonl"))
    return {"jsonl": jsonl_path}
