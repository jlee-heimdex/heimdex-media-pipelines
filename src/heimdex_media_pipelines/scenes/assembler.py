"""Scene assembly: merge speech segments into scene documents."""

from __future__ import annotations

import json
import os
from typing import List, Optional

from heimdex_media_contracts.scenes.merge import (
    aggregate_transcript,
    assign_segments_to_scenes,
)
from heimdex_media_contracts.scenes.schemas import (
    SceneBoundary,
    SceneDetectionResult,
    SceneDocument,
)


def _load_speech_segments(speech_result_path: str) -> List[dict]:
    with open(speech_result_path) as f:
        data = json.load(f)
    return data.get("segments", [])


def assemble_scenes(
    video_path: str,
    video_id: str,
    scene_boundaries: List[SceneBoundary],
    speech_result_path: Optional[str] = None,
    pipeline_version: str = "0.2.0",
    model_version: str = "ffmpeg_scenecut",
    total_duration_ms: int = 0,
    processing_time_s: float = 0.0,
) -> SceneDetectionResult:
    """Build a ``SceneDetectionResult`` from boundaries + speech segments.

    Args:
        video_path: Path to the source video.
        video_id: Video identifier for scene_id construction.
        scene_boundaries: Detected scene boundaries.
        speech_result_path: Path to speech pipeline ``result.json``.
            If None or missing, scenes will have empty transcripts.
        pipeline_version: Package version string.
        model_version: Detection method identifier.
        total_duration_ms: Video duration in milliseconds.
        processing_time_s: Elapsed pipeline time.

    Returns:
        Complete ``SceneDetectionResult`` with scene documents.
    """
    segments: List[dict] = []
    if speech_result_path and os.path.isfile(speech_result_path):
        segments = _load_speech_segments(speech_result_path)

    assignment = assign_segments_to_scenes(scene_boundaries, segments)

    scene_docs: List[SceneDocument] = []
    for boundary in scene_boundaries:
        assigned = assignment.get(boundary.scene_id, [])

        raw = aggregate_transcript(assigned)
        norm = raw.lower()

        scene_docs.append(SceneDocument(
            scene_id=boundary.scene_id,
            video_id=video_id,
            index=boundary.index,
            start_ms=boundary.start_ms,
            end_ms=boundary.end_ms,
            keyframe_timestamp_ms=boundary.keyframe_timestamp_ms,
            transcript_raw=raw,
            transcript_norm=norm,
            transcript_char_count=len(raw),
            speech_segment_count=len(assigned),
            thumbnail_path=boundary.keyframe_path,
        ))

    return SceneDetectionResult(
        schema_version="1.0",
        pipeline_version=pipeline_version,
        model_version=model_version,
        video_path=video_path,
        video_id=video_id,
        total_duration_ms=total_duration_ms,
        scenes=scene_docs,
        processing_time_s=processing_time_s,
    )
