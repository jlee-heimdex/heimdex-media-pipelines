"""Face detection on video frames at given timestamps.

Uses either Haar cascades or SCRFD (insightface) for detection.
"""

import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2


def _load_cascade():
    cascade_root = getattr(cv2, "data", None)
    if cascade_root is None or not hasattr(cascade_root, "haarcascades"):
        raise RuntimeError("OpenCV haarcascades path is unavailable")
    cascade_path = os.path.join(cascade_root.haarcascades, "haarcascade_frontalface_default.xml")
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise RuntimeError(f"Failed to load cascade at {cascade_path}")
    return cascade


def _init_detector(
    detector: str,
    scrfd_det_size: int,
    scrfd_ctx_id: int,
    scrfd_det_thresh: float = 0.7,
) -> Tuple[str, Optional[cv2.CascadeClassifier], Any]:
    detector = detector.lower().strip()
    if detector not in {"haar", "scrfd"}:
        raise ValueError(f"Unsupported detector: {detector}")

    if detector == "haar":
        return detector, _load_cascade(), None

    try:
        insightface_mod = __import__("insightface", fromlist=["app"])
        face_analysis_mod = getattr(insightface_mod, "app")
        FaceAnalysis = getattr(face_analysis_mod, "FaceAnalysis")
    except Exception as exc:
        raise RuntimeError(
            "SCRFD requires insightface. Install with: python -m pip install insightface onnxruntime"
        ) from exc
    if scrfd_ctx_id >= 0:
        from heimdex_media_pipelines.device import detect_onnx_providers
        providers = detect_onnx_providers()
    else:
        providers = ["CPUExecutionProvider"]
    scrfd_app = FaceAnalysis(name="buffalo_l", providers=providers)
    scrfd_app.prepare(ctx_id=scrfd_ctx_id, det_size=(scrfd_det_size, scrfd_det_size), det_thresh=scrfd_det_thresh)
    return detector, None, scrfd_app


def _detect_on_frame(
    frame: Any,
    detector: str,
    min_size: int,
    cascade: Optional[cv2.CascadeClassifier],
    scrfd_app: Any,
    min_det_conf: float = 0.0,
) -> List[Dict[str, Any]]:
    bboxes: List[Dict[str, Any]] = []
    if detector == "haar":
        if cascade is None:
            raise RuntimeError("Haar cascade detector was not initialized")
        cascade_obj: Any = cascade
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if hasattr(cascade_obj, "detectMultiScale3"):
            faces, _, level_weights = cascade_obj.detectMultiScale3(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(min_size, min_size),
                outputRejectLevels=True,
            )
            for (x, y, w, h), weight in zip(faces, level_weights):
                if w < min_size or h < min_size:
                    continue
                conf = float(weight)
                if conf < min_det_conf:
                    continue
                bboxes.append(
                    {
                        "x": int(x),
                        "y": int(y),
                        "w": int(w),
                        "h": int(h),
                        "det_conf": conf,
                    }
                )
        else:
            faces = cascade_obj.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(min_size, min_size),
            )
            for (x, y, w, h) in faces:
                if w < min_size or h < min_size:
                    continue
                bboxes.append(
                    {
                        "x": int(x),
                        "y": int(y),
                        "w": int(w),
                        "h": int(h),
                        "det_conf": 1.0,
                    }
                )
    else:
        if scrfd_app is None:
            raise RuntimeError("SCRFD detector was not initialized")
        faces = scrfd_app.get(frame)
        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int).tolist()
            w = x2 - x1
            h = y2 - y1
            if w < min_size or h < min_size:
                continue
            conf = float(face.det_score)
            if conf < min_det_conf:
                continue
            bboxes.append(
                {
                    "x": int(x1),
                    "y": int(y1),
                    "w": int(w),
                    "h": int(h),
                    "det_conf": conf,
                }
            )
    return bboxes


def detect_faces(
    video_path: str,
    timestamps_s: Iterable[float],
    min_size: int = 40,
    detector: str = "scrfd",
    scrfd_det_size: int = 640,
    scrfd_ctx_id: int = -1,
    scrfd_det_thresh: float = 0.7,
    min_det_conf: float = 0.0,
) -> List[Dict[str, Any]]:
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    cap = cv2.VideoCapture(video_path)

    detector, cascade, scrfd_app = _init_detector(detector, scrfd_det_size, scrfd_ctx_id, scrfd_det_thresh)

    results: List[Dict[str, Any]] = []
    try:
        for ts in timestamps_s:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(ts) * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                results.append({"ts": float(ts), "bboxes": []})
                continue

            bboxes = _detect_on_frame(frame, detector, min_size, cascade, scrfd_app, min_det_conf)

            results.append({"ts": float(ts), "bboxes": bboxes})
    finally:
        cap.release()

    return results


def detect_faces_from_images(
    image_paths: List[str],
    min_size: int = 40,
    detector: str = "scrfd",
    scrfd_det_size: int = 640,
    scrfd_ctx_id: int = -1,
    scrfd_det_thresh: float = 0.7,
    min_det_conf: float = 0.0,
) -> List[Dict[str, Any]]:
    detector, cascade, scrfd_app = _init_detector(detector, scrfd_det_size, scrfd_ctx_id, scrfd_det_thresh)

    results: List[Dict[str, Any]] = []
    for image_path in image_paths:
        frame = cv2.imread(image_path)
        if frame is None:
            results.append({"image_path": image_path, "bboxes": []})
            continue

        bboxes = _detect_on_frame(frame, detector, min_size, cascade, scrfd_app, min_det_conf)
        results.append({"image_path": image_path, "bboxes": bboxes})

    return results
