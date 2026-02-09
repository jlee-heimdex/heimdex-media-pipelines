"""Smoke tests for speech pipeline modules.

Tests import-ability and basic structure. Does NOT require GPU, whisper,
torch, ffmpeg, or actual video files.
"""

import importlib

import pytest


def test_stt_module_importable():
    """speech.stt module imports without error."""
    mod = importlib.import_module("heimdex_media_pipelines.speech.stt")
    assert hasattr(mod, "STTProcessor")
    assert hasattr(mod, "TranscriptSegment")
    assert hasattr(mod, "convert_to_speech_segments")


def test_pipeline_module_importable():
    """speech.pipeline module imports without error."""
    mod = importlib.import_module("heimdex_media_pipelines.speech.pipeline")
    assert hasattr(mod, "SpeechSegmentsPipeline")


def test_transcript_segment_dataclass():
    """TranscriptSegment creates correctly and has to_dict."""
    from heimdex_media_pipelines.speech.stt import TranscriptSegment

    seg = TranscriptSegment(start_s=1.0, end_s=2.5, text="hello world")
    assert seg.start_s == 1.0
    assert seg.end_s == 2.5
    assert seg.text == "hello world"

    d = seg.to_dict()
    assert d == {"start_s": 1.0, "end_s": 2.5, "text": "hello world"}


def test_stt_processor_init():
    """STTProcessor initializes with defaults."""
    from heimdex_media_pipelines.speech.stt import STTProcessor

    proc = STTProcessor()
    assert proc.model_name == "base"
    assert proc.language is None
    assert proc.device == "auto"
    assert proc._model is None


def test_stt_processor_unknown_model_fallback():
    """STTProcessor falls back to 'base' for unknown model names."""
    from heimdex_media_pipelines.speech.stt import STTProcessor

    proc = STTProcessor(model_name="nonexistent-model")
    assert proc.model_name == "base"


def test_stt_processor_supported_models():
    """STTProcessor has expected supported models list."""
    from heimdex_media_pipelines.speech.stt import STTProcessor

    expected = {"tiny", "base", "small", "medium", "large", "large-v2", "large-v3"}
    assert set(STTProcessor.SUPPORTED_MODELS) == expected


def test_convert_to_speech_segments():
    """convert_to_speech_segments creates SpeechSegment from TranscriptSegment."""
    from heimdex_media_pipelines.speech.stt import (
        TranscriptSegment,
        convert_to_speech_segments,
    )

    transcript_segs = [
        TranscriptSegment(start_s=0.0, end_s=1.5, text="first"),
        TranscriptSegment(start_s=1.5, end_s=3.0, text="second"),
    ]
    speech_segs = convert_to_speech_segments(transcript_segs)
    assert len(speech_segs) == 2
    assert speech_segs[0].start == 0.0
    assert speech_segs[0].end == 1.5
    assert speech_segs[0].text == "first"
    assert speech_segs[0].confidence == 1.0
    assert speech_segs[1].start == 1.5
    assert speech_segs[1].text == "second"


def test_speech_pipeline_init():
    """SpeechSegmentsPipeline initializes with default components."""
    from heimdex_media_pipelines.speech.pipeline import SpeechSegmentsPipeline

    pipe = SpeechSegmentsPipeline()
    assert pipe.stt is not None
    assert pipe.tagger is not None
    assert pipe.ranker is not None


def test_speech_pipeline_missing_video():
    """SpeechSegmentsPipeline.run returns error for missing video."""
    from heimdex_media_pipelines.speech.pipeline import SpeechSegmentsPipeline

    pipe = SpeechSegmentsPipeline()
    result = pipe.run("/nonexistent/video.mp4")
    assert result.status == "error"
    assert "not found" in result.error.lower()
