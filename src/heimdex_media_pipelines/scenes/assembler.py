"""Scene assembly: merge speech segments into scene documents."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence

from heimdex_media_contracts.scenes.merge import (
    aggregate_scene_tags,
    aggregate_transcript,
    assign_segments_to_scenes,
)
from heimdex_media_contracts.scenes.schemas import (
    SceneBoundary,
    SceneDetectionResult,
    SceneDocument,
)
from heimdex_media_contracts.speech.schemas import SpeechSegment, TaggedSegment
from heimdex_media_contracts.speech.tagger import (
    PRODUCT_KEYWORD_DICT,
    SpeechTagger,
)


def _load_speech_segments(speech_result_path: str) -> List[dict]:
    with open(speech_result_path) as f:
        data = json.load(f)
    return data.get("segments", [])


def _dicts_to_speech_segments(raw_segments: Sequence[dict]) -> list[SpeechSegment]:
    result: list[SpeechSegment] = []
    for seg in raw_segments:
        start = seg.get("start", seg.get("start_s", 0.0))
        end = seg.get("end", seg.get("end_s", 0.0))
        text = seg.get("text", "")
        confidence = seg.get("confidence", 1.0)
        result.append(SpeechSegment(start=start, end=end, text=text, confidence=confidence))
    return result


def _extract_product_entities(
    text: str,
    product_dict: dict[str, list[str]],
) -> tuple[list[str], list[str]]:
    text_lower = text.lower()
    tags: list[str] = []
    entities: list[str] = []

    for category, keywords in product_dict.items():
        matched = [kw for kw in keywords if kw.lower() in text_lower]
        if matched:
            tags.append(category)
            entities.extend(matched)

    return sorted(set(tags)), sorted(set(entities))


def _aggregate_product_data(
    segments: Sequence[TaggedSegment],
    product_dict: dict[str, list[str]],
) -> tuple[list[str], list[str]]:
    all_tags: set[str] = set()
    all_entities: set[str] = set()

    for seg in segments:
        tags, entities = _extract_product_entities(seg.text, product_dict)
        all_tags.update(tags)
        all_entities.update(entities)

    return sorted(all_tags), sorted(all_entities)


def assemble_scenes(
    video_path: str,
    video_id: str,
    scene_boundaries: List[SceneBoundary],
    speech_result_path: Optional[str] = None,
    pipeline_version: str = "0.3.0",
    model_version: str = "ffmpeg_scenecut",
    total_duration_ms: int = 0,
    processing_time_s: float = 0.0,
    keyword_tagger: Optional[SpeechTagger] = None,
    product_dict: Optional[Dict[str, List[str]]] = None,
) -> SceneDetectionResult:
    raw_segments: List[dict] = []
    if speech_result_path and os.path.isfile(speech_result_path):
        raw_segments = _load_speech_segments(speech_result_path)

    tagger = keyword_tagger or SpeechTagger()
    prod_dict = product_dict if product_dict is not None else dict(PRODUCT_KEYWORD_DICT)

    speech_segments = _dicts_to_speech_segments(raw_segments)
    tagged_segments = tagger.tag(speech_segments) if speech_segments else []

    assignment = assign_segments_to_scenes(scene_boundaries, tagged_segments)

    scene_docs: List[SceneDocument] = []
    for boundary in scene_boundaries:
        assigned: list[TaggedSegment] = assignment.get(boundary.scene_id, [])  # type: ignore[assignment]

        raw = aggregate_transcript(assigned)
        norm = raw.lower()
        keyword_tags = aggregate_scene_tags(assigned)
        p_tags, p_entities = _aggregate_product_data(assigned, prod_dict)

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
            keyword_tags=keyword_tags,
            product_tags=p_tags,
            product_entities=p_entities,
            # TODO: Populate from face clustering pipeline once IdentityPresence
            # output is implemented.  Currently face pipeline outputs raw
            # embeddings without stable identity/cluster assignments.
            people_cluster_ids=[],
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
