"""Face sampling + detection pipeline.

Orchestrates timestamp sampling and face detection into a single run.
"""

import json
import os
from typing import Iterable, Optional

from heimdex_media_pipelines.faces.sampling import sample_timestamps
from heimdex_media_pipelines.faces.detect import detect_faces


def _default_artifacts_dir() -> str:
    return os.path.join(os.getcwd(), "artifacts")


def run_pipeline(
    video_path: str,
    identity_dir: str,
    fps: float = 1.0,
    min_size: int = 40,
    scene_boundaries_s: Optional[Iterable[float]] = None,
    detector: str = "scrfd",
    scrfd_det_size: int = 640,
    scrfd_ctx_id: int = -1,
) -> str:
    """Run face sampling + detection and save detections.jsonl.

    Returns the output path.
    """
    _ = identity_dir

    video_id = os.path.splitext(os.path.basename(video_path))[0]
    timestamps = sample_timestamps(
        video_path,
        fps=fps,
        scene_boundaries_s=scene_boundaries_s,
    )
    detections = detect_faces(
        video_path,
        timestamps,
        min_size=min_size,
        detector=detector,
        scrfd_det_size=scrfd_det_size,
        scrfd_ctx_id=scrfd_ctx_id,
    )

    out_dir = os.path.join(_default_artifacts_dir(), video_id, "faces")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "detections.jsonl")

    with open(out_path, "w") as f:
        for row in detections:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    return out_path
