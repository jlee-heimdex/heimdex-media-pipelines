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

WHISPER_API_MAX_BYTES = 25 * 1024 * 1024


class APISTTProcessor:
    def __init__(self, language: str | None = None, api_key: str | None = None):
        self.language = language
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY or pass api_key.")

        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        try:
            openai_module = importlib.import_module("openai")
            openai_client = getattr(openai_module, "OpenAI")
        except ImportError as e:
            raise ImportError("OpenAI SDK is not installed. Install with: pip install openai") from e

        self._client = openai_client(api_key=self.api_key)
        return self._client

    def extract_audio(self, video_path: Path, output_path: Path) -> Path:
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

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        logger.debug(f"ffmpeg output: {result.stderr}")

        if not output_path.exists():
            raise FileNotFoundError(f"Audio extraction failed: {output_path} not created")

        return output_path

    def compress_audio_to_mp3(self, input_path: Path, output_path: Path) -> Path:
        cmd = [
            "ffmpeg",
            "-i",
            str(input_path),
            "-vn",
            "-acodec",
            "libmp3lame",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-b:a",
            "64k",
            "-y",
            str(output_path),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        logger.debug(f"ffmpeg mp3 output: {result.stderr}")

        if not output_path.exists():
            raise FileNotFoundError(f"MP3 compression failed: {output_path} not created")

        return output_path

    def _audio_for_upload(self, wav_path: Path) -> Path:
        wav_size = wav_path.stat().st_size
        if wav_size <= WHISPER_API_MAX_BYTES:
            return wav_path

        mp3_path = wav_path.with_suffix(".mp3")
        logger.warning(
            "Extracted WAV is %.2f MB (over 25MB Whisper API limit); compressing to MP3",
            wav_size / (1024 * 1024),
        )
        return self.compress_audio_to_mp3(wav_path, mp3_path)

    @staticmethod
    def _segment_value(segment: Any, key: str, default: Any = None) -> Any:
        if isinstance(segment, dict):
            return segment.get(key, default)
        return getattr(segment, key, default)

    def _parse_segments(self, response: Any) -> list[TranscriptSegment]:
        segments = []
        response_segments = self._segment_value(response, "segments", []) or []
        for seg in response_segments:
            text = (self._segment_value(seg, "text", "") or "").strip()
            if not text:
                continue
            segments.append(
                TranscriptSegment(
                    start_s=round(float(self._segment_value(seg, "start", 0.0) or 0.0), 3),
                    end_s=round(float(self._segment_value(seg, "end", 0.0) or 0.0), 3),
                    text=text,
                )
            )
        return segments

    @staticmethod
    def _raise_openai_error(exc: Exception) -> None:
        name = exc.__class__.__name__
        status_code = getattr(exc, "status_code", None)
        code = getattr(exc, "code", None)
        message = str(exc)

        if name == "AuthenticationError" or status_code == 401:
            raise RuntimeError("OpenAI authentication failed. Check OPENAI_API_KEY.") from exc
        if name == "RateLimitError" or status_code == 429:
            raise RuntimeError("OpenAI rate limit exceeded. Retry later.") from exc
        if name == "BadRequestError" and ("25" in message and "MB" in message):
            raise RuntimeError("OpenAI rejected audio: file too large even after compression.") from exc
        if code == "context_length_exceeded" or "file size" in message.lower():
            raise RuntimeError("OpenAI rejected audio due to file size constraints.") from exc
        raise RuntimeError(f"OpenAI transcription failed: {message}") from exc

    def transcribe(self, audio_path: Path) -> list[TranscriptSegment]:
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        client = self._get_client()

        request_kwargs: dict[str, Any] = {
            "model": "whisper-1",
            "response_format": "verbose_json",
            "timestamp_granularities": ["segment"],
        }
        if self.language:
            request_kwargs["language"] = self.language

        response: Any = None
        try:
            with audio_path.open("rb") as audio_file:
                response = client.audio.transcriptions.create(
                    file=audio_file,
                    **request_kwargs,
                )
        except Exception as exc:
            self._raise_openai_error(exc)

        if response is None:
            return []

        return self._parse_segments(response)

    def process(self, video_path: str | Path) -> list[TranscriptSegment]:
        video_path = Path(video_path)
        self._last_perf: PerfTimings | None = None

        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            return []

        t_total = time.perf_counter()
        perf = PerfTimings()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            wav_path = temp_path / "audio.wav"

            try:
                t0 = time.perf_counter()
                self.extract_audio(video_path, wav_path)
                perf.ffmpeg_extract_s = round(time.perf_counter() - t0, 3)

                t0 = time.perf_counter()
                upload_path = self._audio_for_upload(wav_path)
                perf.model_load_s = 0.0  # API has no model load

                t0 = time.perf_counter()
                segments = self.transcribe(upload_path)
                perf.transcribe_s = round(time.perf_counter() - t0, 3)

                perf.postprocess_s = 0.0
                perf.total_s = round(time.perf_counter() - t_total, 3)
                self._last_perf = perf
                perf.log(
                    video_path=str(video_path),
                    backend="api",
                    model_name="whisper-1",
                    device="cloud",
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
                logger.error(f"API STT processing failed: {e}")
                return []
