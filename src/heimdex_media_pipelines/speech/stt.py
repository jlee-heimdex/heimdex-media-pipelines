from __future__ import annotations

"""STT (Speech-to-Text) module.

Extracts audio from video and transcribes using Whisper.
Output: [{start_s, end_s, text}] (sentence-level segments)
"""

import json
import importlib
import importlib.util
import logging
import os
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol

logger = logging.getLogger(__name__)


class SupportsSTTProcessor(Protocol):
    def process(self, video_path: str | Path) -> list[TranscriptSegment]: ...


@dataclass
class PerfTimings:
    """Performance timings for STT pipeline stages."""
    ffmpeg_extract_s: float = 0.0
    model_load_s: float = 0.0
    transcribe_s: float = 0.0
    postprocess_s: float = 0.0
    total_s: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    def log(self, video_path: str = "", backend: str = "", model_name: str = "", device: str = "") -> None:
        """Emit a single structured JSON log line with all perf data."""
        payload = {
            "event": "stt_perf",
            "video_path": str(video_path),
            "backend": backend,
            "model_name": model_name,
            "device": device,
            **self.to_dict(),
        }
        logger.info("stt_perf %s", json.dumps(payload, ensure_ascii=False))


@dataclass
class TranscriptSegment:
    """STT result segment (sentence-level)."""
    start_s: float
    end_s: float
    text: str

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


class STTProcessor:
    """Whisper-based speech-to-text processor."""

    SUPPORTED_MODELS = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]

    def __init__(
        self,
        model_name: str = "base",
        language: Optional[str] = None,
        device: str = "auto",
    ):
        """
        Args:
            model_name: Whisper model name (tiny, base, small, medium, large)
            language: Language code (e.g. "ko", "en"). None for auto-detect
            device: Device ("auto", "cpu", "cuda")
        """
        self.model_name = model_name
        self.language = language
        self.device = device
        self._model: Any = None

        if model_name not in self.SUPPORTED_MODELS:
            logger.warning(f"Unknown model '{model_name}', using 'base'")
            self.model_name = "base"

    def _load_model(self):
        """Load Whisper model (lazy loading)."""
        if self._model is not None:
            return

        try:
            whisper = importlib.import_module("whisper")

            logger.info(f"Loading Whisper model: {self.model_name}")

            device = self.device
            if device == "auto":
                torch = importlib.import_module("torch")

                device = "cuda" if torch.cuda.is_available() else "cpu"

            self._model = whisper.load_model(self.model_name, device=device)
            logger.info(f"Whisper model loaded on {device}")

        except ImportError as e:
            logger.error(f"Failed to import whisper: {e}")
            logger.error("Install with: pip install openai-whisper")
            raise
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise

    def extract_audio(self, video_path: Path, output_path: Path) -> Path:
        """Extract audio from video (16kHz mono WAV).

        Args:
            video_path: Input video path
            output_path: Output audio path

        Returns:
            Extracted audio file path

        Raises:
            subprocess.CalledProcessError: If ffmpeg fails
            FileNotFoundError: If video file missing
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        logger.info(f"Extracting audio from {video_path}")

        cmd = [
            "ffmpeg",
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-y",
            str(output_path),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            logger.debug(f"ffmpeg output: {result.stderr}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Audio extraction failed: {e.stderr}")
            raise

        if not output_path.exists():
            raise FileNotFoundError(f"Audio extraction failed: {output_path} not created")

        file_size = output_path.stat().st_size
        logger.info(f"Audio extracted: {output_path} ({file_size / 1024:.1f} KB)")

        return output_path

    def transcribe(self, audio_path: Path) -> list[TranscriptSegment]:
        """Transcribe audio file to text.

        Args:
            audio_path: Audio file path

        Returns:
            List of TranscriptSegment (sentence-level, with timestamps)
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        self._load_model()

        logger.info(f"Transcribing audio: {audio_path}")

        try:
            options = {
                "task": "transcribe",
                "verbose": False,
            }

            if self.language:
                options["language"] = self.language

            model = self._model
            if model is None:
                raise RuntimeError("Whisper model is not loaded")

            result = model.transcribe(str(audio_path), **options)

            segments = []
            for seg in result.get("segments", []):
                transcript_seg = TranscriptSegment(
                    start_s=round(seg["start"], 3),
                    end_s=round(seg["end"], 3),
                    text=seg["text"].strip(),
                )
                if transcript_seg.text:
                    segments.append(transcript_seg)

            logger.info(f"Transcription complete: {len(segments)} segments")

            detected_lang = result.get("language", "unknown")
            logger.info(f"Detected language: {detected_lang}")

            return segments

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise

    def process(self, video_path: str | Path) -> list[TranscriptSegment]:
        """Run full STT pipeline on a video file.

        Args:
            video_path: Video file path

        Returns:
            List of TranscriptSegment
        """
        video_path = Path(video_path)
        self._last_perf: PerfTimings | None = None

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

                t0 = time.perf_counter()
                self._load_model()
                perf.model_load_s = round(time.perf_counter() - t0, 3)

                t0 = time.perf_counter()
                segments = self.transcribe(audio_path)
                perf.transcribe_s = round(time.perf_counter() - t0, 3)

                t0 = time.perf_counter()
                # postprocess is negligible here but measured for completeness
                perf.postprocess_s = round(time.perf_counter() - t0, 3)

                perf.total_s = round(time.perf_counter() - t_total, 3)
                self._last_perf = perf
                perf.log(
                    video_path=str(video_path),
                    backend="whisper",
                    model_name=self.model_name,
                    device=self.device,
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

    @staticmethod
    def save_transcript(
        segments: list[TranscriptSegment],
        output_path: Path,
        video_path: Optional[str] = None,
    ) -> None:
        """Save transcript to JSON file.

        Args:
            segments: List of TranscriptSegment
            output_path: Output JSON file path
            video_path: Original video path (for metadata)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data: dict[str, object] = {
            "video_path": str(video_path) if video_path else None,
            "segments": [seg.to_dict() for seg in segments],
            "total_segments": len(segments),
        }

        if segments:
            data["total_duration_s"] = max(seg.end_s for seg in segments)
        else:
            data["total_duration_s"] = 0.0

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Transcript saved to {output_path}")


def convert_to_speech_segments(transcript_segments: list[TranscriptSegment]):
    """Convert TranscriptSegment to SpeechSegment.

    Args:
        transcript_segments: List of STT result segments

    Returns:
        List of SpeechSegment
    """
    from heimdex_media_contracts.speech.schemas import SpeechSegment

    return [
        SpeechSegment(
            start=seg.start_s,
            end=seg.end_s,
            text=seg.text,
            confidence=1.0,
        )
        for seg in transcript_segments
    ]


def _is_importable(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _create_api_stt_processor(language: str | None, api_key: str | None) -> SupportsSTTProcessor:
    stt_api_module = importlib.import_module("heimdex_media_pipelines.speech.stt_api")
    api_processor_cls = getattr(stt_api_module, "APISTTProcessor")
    return api_processor_cls(language=language, api_key=api_key)


def _create_faster_whisper_processor(
    model_name: str,
    language: str | None,
    device: str,
    compute_type: str,
    beam_size: int,
    best_of: int,
) -> SupportsSTTProcessor:
    stt_faster_module = importlib.import_module("heimdex_media_pipelines.speech.stt_faster")
    processor_cls = getattr(stt_faster_module, "FasterWhisperSTTProcessor")
    return processor_cls(
        model_name=model_name,
        language=language,
        device=device,
        compute_type=compute_type,
        beam_size=beam_size,
        best_of=best_of,
    )


def create_stt_processor(
    backend: str = "auto",
    model_name: str = "base",
    language: str | None = None,
    device: str = "auto",
    api_key: str | None = None,
    compute_type: str = "auto",
    beam_size: int = 1,
    best_of: int = 1,
) -> STTProcessor | SupportsSTTProcessor:
    backend = backend.lower().strip()

    env_backend = os.getenv("HEIMDEX_STT_BACKEND", "").lower().strip()
    if env_backend and backend == "auto":
        backend = env_backend

    valid_backends = {"auto", "local", "api", "faster-whisper", "whisper"}
    if backend not in valid_backends:
        raise ValueError(f"Invalid backend '{backend}'. Expected one of: {', '.join(sorted(valid_backends))}")

    has_whisper = _is_importable("whisper")
    has_torch = _is_importable("torch")
    has_faster_whisper = _is_importable("faster_whisper")
    has_openai = _is_importable("openai")
    has_api_key = bool(api_key or os.getenv("OPENAI_API_KEY"))

    if backend == "whisper" or backend == "local":
        if not has_whisper:
            raise ImportError("Local STT backend requires whisper. Install with: pip install openai-whisper")
        return STTProcessor(model_name=model_name, language=language, device=device)

    if backend == "faster-whisper":
        if not has_faster_whisper:
            raise ImportError("faster-whisper backend requires faster-whisper. Install with: pip install faster-whisper")
        return _create_faster_whisper_processor(
            model_name=model_name, language=language, device=device,
            compute_type=compute_type, beam_size=beam_size, best_of=best_of,
        )

    if backend == "api":
        if not has_openai:
            raise ImportError("API STT backend requires openai SDK. Install with: pip install openai")
        if not has_api_key:
            raise ImportError("API STT backend requires OPENAI_API_KEY environment variable")
        return _create_api_stt_processor(language=language, api_key=api_key)

    # auto mode: prefer faster-whisper → openai-whisper → API
    if has_faster_whisper:
        return _create_faster_whisper_processor(
            model_name=model_name, language=language, device=device,
            compute_type=compute_type, beam_size=beam_size, best_of=best_of,
        )

    if has_whisper and has_torch:
        return STTProcessor(model_name=model_name, language=language, device=device)

    if has_openai and has_api_key:
        return _create_api_stt_processor(language=language, api_key=api_key)

    raise ImportError(
        "No usable STT backend found. Install faster-whisper (recommended), "
        "openai-whisper+torch (local), or openai SDK + OPENAI_API_KEY (API)."
    )
