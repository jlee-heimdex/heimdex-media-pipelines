"""Tests for STT performance instrumentation and backend selection."""

import types
from pathlib import Path

import pytest

from heimdex_media_pipelines.speech.stt import (
    PerfTimings,
    STTProcessor,
    TranscriptSegment,
    create_stt_processor,
)


class TestPerfTimings:
    def test_perf_timings_to_dict_has_all_keys(self):
        perf = PerfTimings(
            ffmpeg_extract_s=1.1,
            model_load_s=2.2,
            transcribe_s=3.3,
            postprocess_s=0.01,
            total_s=6.61,
        )
        d = perf.to_dict()
        assert set(d.keys()) == {
            "ffmpeg_extract_s",
            "model_load_s",
            "transcribe_s",
            "postprocess_s",
            "total_s",
        }
        assert d["ffmpeg_extract_s"] == 1.1
        assert d["total_s"] == 6.61

    def test_perf_timings_defaults_to_zero(self):
        perf = PerfTimings()
        d = perf.to_dict()
        assert all(v == 0.0 for v in d.values())

    def test_perf_timings_log_emits_json(self, caplog):
        import logging

        with caplog.at_level(logging.INFO, logger="heimdex_media_pipelines.speech.stt"):
            perf = PerfTimings(total_s=5.0)
            perf.log(video_path="/tmp/v.mp4", backend="whisper", model_name="base", device="cpu")

        assert any("stt_perf" in rec.message for rec in caplog.records)
        assert any('"backend": "whisper"' in rec.message for rec in caplog.records)


class TestBackendSelection:
    def test_faster_whisper_backend_explicit(self, monkeypatch):
        monkeypatch.setattr(
            "heimdex_media_pipelines.speech.stt._is_importable",
            lambda m: m == "faster_whisper",
        )
        proc = create_stt_processor(backend="faster-whisper")
        assert type(proc).__name__ == "FasterWhisperSTTProcessor"

    def test_auto_prefers_faster_whisper_over_openai_whisper(self, monkeypatch):
        monkeypatch.setattr(
            "heimdex_media_pipelines.speech.stt._is_importable",
            lambda m: m in {"faster_whisper", "whisper", "torch"},
        )
        proc = create_stt_processor(backend="auto")
        assert type(proc).__name__ == "FasterWhisperSTTProcessor"

    def test_auto_falls_back_to_whisper_when_no_faster(self, monkeypatch):
        monkeypatch.setattr(
            "heimdex_media_pipelines.speech.stt._is_importable",
            lambda m: m in {"whisper", "torch"},
        )
        proc = create_stt_processor(backend="auto")
        assert isinstance(proc, STTProcessor)

    def test_whisper_backend_explicit(self, monkeypatch):
        monkeypatch.setattr(
            "heimdex_media_pipelines.speech.stt._is_importable",
            lambda m: m == "whisper",
        )
        proc = create_stt_processor(backend="whisper")
        assert isinstance(proc, STTProcessor)

    def test_env_var_overrides_auto(self, monkeypatch):
        monkeypatch.setattr(
            "heimdex_media_pipelines.speech.stt._is_importable",
            lambda m: m in {"whisper", "torch", "faster_whisper"},
        )
        monkeypatch.setenv("HEIMDEX_STT_BACKEND", "whisper")
        proc = create_stt_processor(backend="auto")
        assert isinstance(proc, STTProcessor)

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="Invalid backend"):
            create_stt_processor(backend="nonexistent")

    def test_compute_type_and_beam_size_forwarded(self, monkeypatch):
        monkeypatch.setattr(
            "heimdex_media_pipelines.speech.stt._is_importable",
            lambda m: m == "faster_whisper",
        )
        proc = create_stt_processor(
            backend="faster-whisper",
            compute_type="int8",
            beam_size=3,
            best_of=2,
        )
        assert proc.compute_type == "int8"
        assert proc.beam_size == 3
        assert proc.best_of == 2


class TestFasterWhisperProcessor:
    def test_init_defaults(self):
        from heimdex_media_pipelines.speech.stt_faster import FasterWhisperSTTProcessor

        proc = FasterWhisperSTTProcessor()
        assert proc.model_name == "base"
        assert proc.beam_size == 1
        assert proc.best_of == 1
        assert proc.compute_type == "auto"
        assert proc._model is None
        assert proc._last_perf is None

    def test_unknown_model_fallback(self):
        from heimdex_media_pipelines.speech.stt_faster import FasterWhisperSTTProcessor

        proc = FasterWhisperSTTProcessor(model_name="nonexistent")
        assert proc.model_name == "base"

    def test_resolve_compute_type_cpu(self):
        from heimdex_media_pipelines.speech.stt_faster import FasterWhisperSTTProcessor

        proc = FasterWhisperSTTProcessor(device="cpu", compute_type="auto")
        assert proc._resolve_compute_type() == "int8"

    def test_resolve_compute_type_explicit(self):
        from heimdex_media_pipelines.speech.stt_faster import FasterWhisperSTTProcessor

        proc = FasterWhisperSTTProcessor(compute_type="float32")
        assert proc._resolve_compute_type() == "float32"

    def test_process_nonexistent_file_returns_empty(self):
        from heimdex_media_pipelines.speech.stt_faster import FasterWhisperSTTProcessor

        proc = FasterWhisperSTTProcessor()
        result = proc.process("/nonexistent/video.mp4")
        assert result == []

    def test_last_perf_set_after_process(self, monkeypatch, tmp_path):
        from heimdex_media_pipelines.speech.stt_faster import FasterWhisperSTTProcessor

        proc = FasterWhisperSTTProcessor()

        fake_segments = [
            types.SimpleNamespace(start=0.0, end=1.5, text=" hello "),
            types.SimpleNamespace(start=1.5, end=3.0, text="world"),
        ]
        fake_info = types.SimpleNamespace(language="en", language_probability=0.95)

        def fake_transcribe(audio_path, **kwargs):
            return iter(fake_segments), fake_info

        proc._model = types.SimpleNamespace(transcribe=fake_transcribe)

        def fake_extract(video_path, output_path):
            output_path.write_bytes(b"fake-audio")
            return output_path

        monkeypatch.setattr(proc, "extract_audio", fake_extract)

        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake-video")

        result = proc.process(str(video))
        assert len(result) == 2
        assert result[0].text == "hello"
        assert result[1].text == "world"

        assert proc._last_perf is not None
        perf = proc._last_perf.to_dict()
        assert "ffmpeg_extract_s" in perf
        assert "model_load_s" in perf
        assert "transcribe_s" in perf
        assert "total_s" in perf
        assert perf["total_s"] >= 0
