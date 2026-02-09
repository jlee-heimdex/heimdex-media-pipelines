"""STT (Speech-to-Text) module.

Extracts audio from video and transcribes using Whisper.
Output: [{start_s, end_s, text}] (sentence-level segments)
"""

import json
import logging
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TranscriptSegment:
    """STT result segment (sentence-level)."""
    start_s: float
    end_s: float
    text: str

    def to_dict(self) -> dict:
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
        self._model = None

        if model_name not in self.SUPPORTED_MODELS:
            logger.warning(f"Unknown model '{model_name}', using 'base'")
            self.model_name = "base"

    def _load_model(self):
        """Load Whisper model (lazy loading)."""
        if self._model is not None:
            return

        try:
            import whisper

            logger.info(f"Loading Whisper model: {self.model_name}")

            device = self.device
            if device == "auto":
                import torch

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

            result = self._model.transcribe(str(audio_path), **options)

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

        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            return []

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            audio_path = temp_path / "audio.wav"

            try:
                self.extract_audio(video_path, audio_path)
                segments = self.transcribe(audio_path)
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

        data = {
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
