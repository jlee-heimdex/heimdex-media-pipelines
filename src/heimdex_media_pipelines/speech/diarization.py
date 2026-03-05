"""Post-transcription speaker diarization using pyannote.audio.

Runs pyannote speaker diarization on audio independently of Whisper,
then assigns speaker labels to transcript segments by maximum temporal overlap.

Requires: pyannote-audio==3.3.2, torch
Pin to 3.3.2 — v4.x has a known 6x VRAM regression on long audio.
"""

from __future__ import annotations

import gc
import importlib
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from heimdex_media_pipelines.speech.stt import TranscriptSegment

logger = logging.getLogger(__name__)


@dataclass
class SpeakerTurn:
    start_s: float
    end_s: float
    speaker_id: str


class SpeakerDiarizer:
    """pyannote.audio 3.x speaker diarization pipeline.

    Designed to run alongside faster-whisper on the same GPU.
    Both models fit in ~5-6 GB VRAM total (pyannote ~1.6 GB + whisper turbo ~3-4 GB).
    """

    def __init__(
        self,
        model_name: str = "pyannote/speaker-diarization-3.1",
        hf_token: Optional[str] = None,
        device: str = "auto",
        min_speakers: int = 1,
        max_speakers: int = 4,
        segmentation_batch_size: int = 32,
        embedding_batch_size: int = 32,
    ):
        self.model_name = model_name
        self.hf_token = hf_token
        self.device = device
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.segmentation_batch_size = segmentation_batch_size
        self.embedding_batch_size = embedding_batch_size
        self._pipeline: Any = None

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        torch = importlib.import_module("torch")
        return "cuda" if torch.cuda.is_available() else "cpu"

    def load(self) -> None:
        if self._pipeline is not None:
            return

        t0 = time.perf_counter()
        pyannote_pipeline = importlib.import_module("pyannote.audio").Pipeline
        torch = importlib.import_module("torch")

        resolved_device = self._resolve_device()

        logger.info(
            "diarization_model_loading",
            extra={"model": self.model_name, "device": resolved_device},
        )

        # Explicitly pass cache_dir to avoid relying on TORCH_HOME env var
        # at runtime.  pyannote defaults to torch.hub.get_dir() which can
        # point to an unwritable path on some container platforms (Aircloud).
        # Prefer HF_HOME (set in GPU Dockerfile to /models/huggingface)
        # where models are pre-downloaded at build time.
        cache_dir = os.environ.get("HF_HOME") or os.environ.get("TORCH_HOME")

        self._pipeline = pyannote_pipeline.from_pretrained(
            self.model_name,
            use_auth_token=self.hf_token,
            cache_dir=cache_dir,
        ).to(torch.device(resolved_device))

        if hasattr(self._pipeline, "_segmentation"):
            self._pipeline._segmentation.batch_size = self.segmentation_batch_size
        if hasattr(self._pipeline, "_embedding"):
            self._pipeline._embedding.batch_size = self.embedding_batch_size

        elapsed = time.perf_counter() - t0
        logger.info(
            "diarization_model_loaded",
            extra={"model": self.model_name, "device": resolved_device, "load_seconds": round(elapsed, 1)},
        )

    def diarize(self, audio_path: Path) -> list[SpeakerTurn]:
        self.load()

        t0 = time.perf_counter()
        logger.info("diarization_started", extra={"audio_path": str(audio_path)})

        kwargs: dict[str, Any] = {}
        if self.min_speakers > 1:
            kwargs["min_speakers"] = self.min_speakers
        if self.max_speakers > 0:
            kwargs["max_speakers"] = self.max_speakers

        output = self._pipeline(str(audio_path), **kwargs)

        turns: list[SpeakerTurn] = []
        for turn, _, speaker in output.itertracks(yield_label=True):
            turns.append(SpeakerTurn(
                start_s=round(turn.start, 3),
                end_s=round(turn.end, 3),
                speaker_id=speaker,
            ))

        elapsed = time.perf_counter() - t0
        speaker_count = len({t.speaker_id for t in turns})
        logger.info(
            "diarization_complete",
            extra={
                "audio_path": str(audio_path),
                "turn_count": len(turns),
                "speaker_count": speaker_count,
                "duration_seconds": round(elapsed, 1),
            },
        )
        return turns

    def release(self) -> None:
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            gc.collect()
            try:
                torch = importlib.import_module("torch")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            logger.info("diarization_model_released")

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None


def assign_speakers_to_segments(
    segments: list[TranscriptSegment],
    speaker_turns: list[SpeakerTurn],
) -> list[TranscriptSegment]:
    """Assign speaker labels to transcript segments by maximum temporal overlap.

    Mutates ``speaker_id`` on each segment in-place and returns the same list.
    Segments with no overlapping speaker turn keep ``speaker_id=None``.
    """
    if not speaker_turns:
        return segments

    for seg in segments:
        speaker_time: dict[str, float] = {}
        for turn in speaker_turns:
            overlap = max(0.0, min(seg.end_s, turn.end_s) - max(seg.start_s, turn.start_s))
            if overlap > 0:
                speaker_time[turn.speaker_id] = speaker_time.get(turn.speaker_id, 0.0) + overlap

        if speaker_time:
            seg.speaker_id = max(speaker_time, key=speaker_time.get)

    return segments
