from __future__ import annotations

import importlib
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from heimdex_media_pipelines.speech.stt import PerfTimings, TranscriptSegment

logger = logging.getLogger(__name__)

SUPPORTED_COMPUTE_TYPES = ("int8", "int8_float16", "int8_float32", "float16", "float32", "auto")


class FasterWhisperSTTProcessor:
    SUPPORTED_MODELS = ["tiny", "base", "small", "medium", "large-v2", "large-v3", "turbo",
                        "distil-small.en", "distil-medium.en", "distil-large-v2", "distil-large-v3"]

    def __init__(
        self,
        model_name: str = "base",
        language: str | None = None,
        device: str = "auto",
        compute_type: str = "auto",
        beam_size: int = 1,
        best_of: int = 1,
        cpu_threads: int = 0,
    ):
        self.model_name = model_name
        self.language = language
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.best_of = best_of
        self.cpu_threads = cpu_threads
        self._model: Any = None
        self._last_perf: PerfTimings | None = None

        if model_name not in self.SUPPORTED_MODELS:
            logger.warning(f"Unknown model '{model_name}', using 'base'")
            self.model_name = "base"

    def _resolve_compute_type(self) -> str:
        if self.compute_type != "auto":
            return self.compute_type
        resolved_device = self._resolve_device()
        if resolved_device == "cuda":
            return "int8_float16"
        return "int8"

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        from heimdex_media_pipelines.device import detect_whisper_device
        device, _ = detect_whisper_device()
        return device

    def _load_model(self) -> float:
        if self._model is not None:
            return 0.0

        t0 = time.perf_counter()
        try:
            faster_whisper = importlib.import_module("faster_whisper")
            WhisperModel = getattr(faster_whisper, "WhisperModel")
        except ImportError as e:
            logger.error(f"Failed to import faster_whisper: {e}")
            logger.error("Install with: pip install faster-whisper")
            raise

        resolved_device = self._resolve_device()
        resolved_compute = self._resolve_compute_type()

        logger.info(f"Loading faster-whisper model: {self.model_name} "
                     f"(device={resolved_device}, compute_type={resolved_compute})")

        try:
            self._model = WhisperModel(
                self.model_name,
                device=resolved_device,
                compute_type=resolved_compute,
                cpu_threads=self.cpu_threads,
            )
        except Exception as e:
            logger.error(f"Failed to load faster-whisper model: {e}")
            raise

        elapsed = time.perf_counter() - t0
        logger.info(f"faster-whisper model loaded in {elapsed:.1f}s "
                     f"(device={resolved_device}, compute_type={resolved_compute})")
        return round(elapsed, 3)

    def extract_audio(self, video_path: Path, output_path: Path) -> Path:
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        logger.info(f"Extracting audio from {video_path}")

        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1",
            "-y", str(output_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.debug(f"ffmpeg output: {result.stderr}")

        if not output_path.exists():
            raise FileNotFoundError(f"Audio extraction failed: {output_path} not created")

        file_size = output_path.stat().st_size
        logger.info(f"Audio extracted: {output_path} ({file_size / 1024:.1f} KB)")
        return output_path

    def transcribe(self, audio_path: Path) -> list[TranscriptSegment]:
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"Transcribing audio: {audio_path}")

        if self._model is None:
            self._load_model()
        model = self._model
        if model is None:
            raise RuntimeError("faster-whisper model is not loaded")

        kwargs: dict[str, Any] = {
            "beam_size": self.beam_size,
            "best_of": self.best_of,
            "word_timestamps": False,
            "vad_filter": True,
        }
        if self.language:
            kwargs["language"] = self.language

        try:
            segments_gen, info = model.transcribe(str(audio_path), **kwargs)

            segments: list[TranscriptSegment] = []
            for seg in segments_gen:
                text = seg.text.strip()
                if text:
                    segments.append(TranscriptSegment(
                        start_s=round(seg.start, 3),
                        end_s=round(seg.end, 3),
                        text=text,
                    ))

            logger.info(f"Transcription complete: {len(segments)} segments "
                         f"(language={info.language}, prob={info.language_probability:.2f})")
            return segments

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise

    def process(self, video_path: str | Path) -> list[TranscriptSegment]:
        video_path = Path(video_path)
        self._last_perf = None

        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            return []

        t_total = time.perf_counter()
        perf = PerfTimings()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            audio_path = temp_path / "audio.wav"

            try:
                t0 = time.perf_counter()
                self.extract_audio(video_path, audio_path)
                perf.ffmpeg_extract_s = round(time.perf_counter() - t0, 3)

                perf.model_load_s = self._load_model()

                t0 = time.perf_counter()
                segments = self.transcribe(audio_path)
                perf.transcribe_s = round(time.perf_counter() - t0, 3)

                perf.postprocess_s = 0.0
                perf.total_s = round(time.perf_counter() - t_total, 3)
                self._last_perf = perf
                perf.log(
                    video_path=str(video_path),
                    backend="faster-whisper",
                    model_name=self.model_name,
                    device=self._resolve_device(),
                )
                return segments

            except FileNotFoundError as e:
                logger.error(f"File not found: {e}")
                return []
            except subprocess.CalledProcessError as e:
                logger.error(f"FFmpeg error: {e}")
                logger.error("Make sure ffmpeg is installed and accessible")
                return []
            except Exception as e:
                logger.error(f"STT processing failed: {e}")
                return []
