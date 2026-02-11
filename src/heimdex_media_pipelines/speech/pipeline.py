"""Speech Segments pipeline - orchestrates STT -> tagging -> ranking."""

import logging
import time
from pathlib import Path

from heimdex_media_contracts.speech.schemas import PipelineResult, RankedSegment, SpeechSegment
from heimdex_media_contracts.speech.tagger import SpeechTagger
from heimdex_media_contracts.speech.ranker import SegmentRanker

from heimdex_media_pipelines.speech.stt import STTProcessor, convert_to_speech_segments, create_stt_processor

logger = logging.getLogger(__name__)


class SpeechSegmentsPipeline:
    """Speech segment extraction -> tagging -> ranking pipeline."""

    def __init__(
        self,
        stt_processor: STTProcessor | None = None,
        tagger: SpeechTagger | None = None,
        ranker: SegmentRanker | None = None,
        whisper_model: str = "base",
        language: str | None = None,
        backend: str = "auto",
        api_key: str | None = None,
        compute_type: str = "auto",
        beam_size: int = 1,
    ):
        """
        Args:
            stt_processor: STT processor (None to auto-create)
            tagger: Tagger (None to auto-create)
            ranker: Ranker (None to auto-create)
            whisper_model: Whisper model name
            language: Language code (e.g. "ko", "en")
        """
        if stt_processor is not None:
            self.stt = stt_processor
        else:
            try:
                self.stt = create_stt_processor(
                    backend=backend,
                    model_name=whisper_model,
                    language=language,
                    api_key=api_key,
                    compute_type=compute_type,
                    beam_size=beam_size,
                )
            except ImportError:
                if backend != "auto":
                    raise
                self.stt = STTProcessor(
                    model_name=whisper_model,
                    language=language,
                )
        self.tagger = tagger or SpeechTagger()
        self.ranker = ranker or SegmentRanker()

    def run(
        self,
        video_path: str,
        save_transcript: bool = True,
        artifacts_dir: str | None = None,
    ) -> PipelineResult:
        """Run full pipeline on a video file.

        Args:
            video_path: Video file path
            save_transcript: Whether to save transcript.json
            artifacts_dir: Artifacts save directory (None -> artifacts/<video_id>/)

        Returns:
            PipelineResult object
        """
        start_time = time.time()

        path = Path(video_path)
        if not path.exists():
            logger.error(f"File not found: {video_path}")
            return PipelineResult(
                video_path=video_path,
                status="error",
                error=f"File not found: {video_path}",
            )

        video_id = path.stem

        if artifacts_dir:
            output_dir = Path(artifacts_dir)
        else:
            output_dir = Path("artifacts") / video_id
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            logger.info(f"Starting pipeline for: {video_path}")

            # Step 1: STT
            logger.info("Step 1: Running STT...")
            transcript_segments = self.stt.process(video_path)

            if not transcript_segments:
                logger.warning("No speech segments detected")
                return PipelineResult(
                    video_path=video_path,
                    segments=[],
                    total_duration=0.0,
                    processing_time=time.time() - start_time,
                    status="success",
                    error="No speech detected in video",
                )

            logger.info(f"STT complete: {len(transcript_segments)} segments")

            if save_transcript:
                transcript_path = output_dir / "transcript.json"
                STTProcessor.save_transcript(
                    transcript_segments,
                    transcript_path,
                    video_path=video_path,
                )
                logger.info(f"Transcript saved: {transcript_path}")

            segments = convert_to_speech_segments(transcript_segments)

            # Step 2: Tagging
            logger.info("Step 2: Tagging segments...")
            tagged_segments = self.tagger.tag(segments)
            logger.info(f"Tagging complete: {len(tagged_segments)} segments")

            # Step 3: Ranking
            logger.info("Step 3: Ranking segments...")
            ranked_segments = self.ranker.rank(tagged_segments)
            logger.info(f"Ranking complete: {len(ranked_segments)} segments")

            processing_time = time.time() - start_time
            total_duration = sum(s.duration for s in ranked_segments) if ranked_segments else 0.0

            logger.info(
                f"Pipeline complete: {len(ranked_segments)} segments, "
                f"{total_duration:.1f}s total, "
                f"{processing_time:.1f}s processing time"
            )

            return PipelineResult(
                video_path=video_path,
                segments=ranked_segments,
                total_duration=total_duration,
                processing_time=processing_time,
                status="success",
            )

        except FileNotFoundError as e:
            error_msg = f"File not found: {e}"
            logger.error(error_msg)
            return PipelineResult(
                video_path=video_path,
                status="error",
                error=error_msg,
                processing_time=time.time() - start_time,
            )

        except Exception as e:
            error_msg = f"Pipeline failed: {str(e)}"
            logger.error(error_msg, exc_info=True)

            logger.error("=== Retry Guide ===")
            logger.error("1. Check if ffmpeg is installed: ffmpeg -version")
            logger.error("2. Check if whisper is installed: pip install openai-whisper")
            logger.error("3. Check video file is valid: ffprobe <video_path>")
            logger.error("4. Try with a smaller model: --model tiny")
            logger.error("==================")

            return PipelineResult(
                video_path=video_path,
                status="error",
                error=error_msg,
                processing_time=time.time() - start_time,
            )
