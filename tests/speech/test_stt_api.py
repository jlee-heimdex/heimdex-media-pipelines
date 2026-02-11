import types
from pathlib import Path

import pytest

from heimdex_media_pipelines.speech.stt import STTProcessor, create_stt_processor
from heimdex_media_pipelines.speech.stt_api import APISTTProcessor, WHISPER_API_MAX_BYTES


def test_api_processor_builds_openai_client(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, api_key):
            captured["api_key"] = api_key

    fake_module = types.SimpleNamespace(OpenAI=FakeOpenAI)
    monkeypatch.setattr(
        "heimdex_media_pipelines.speech.stt_api.importlib.import_module",
        lambda name: fake_module if name == "openai" else __import__(name),
    )

    proc = APISTTProcessor(api_key="abc")
    proc._get_client()

    assert captured["api_key"] == "abc"


def test_api_processor_transcribe_parses_segments(tmp_path):
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"fake")

    captured = {}

    class FakeTranscriptions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return {
                "segments": [
                    {"start": 0.0, "end": 1.25, "text": " hello "},
                    {"start": 1.25, "end": 2.5, "text": "world"},
                    {"start": 2.5, "end": 3.0, "text": "   "},
                ]
            }

    fake_client = types.SimpleNamespace(
        audio=types.SimpleNamespace(transcriptions=FakeTranscriptions())
    )

    proc = APISTTProcessor(language="en", api_key="test-key")
    proc._client = fake_client

    segments = proc.transcribe(audio_path)

    assert [s.text for s in segments] == ["hello", "world"]
    assert segments[0].start_s == 0.0
    assert segments[0].end_s == 1.25
    assert captured["model"] == "whisper-1"
    assert captured["response_format"] == "verbose_json"
    assert captured["timestamp_granularities"] == ["segment"]
    assert captured["language"] == "en"


def test_api_key_resolution_precedence(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")

    from_arg = APISTTProcessor(api_key="arg-key")
    assert from_arg.api_key == "arg-key"

    from_env = APISTTProcessor()
    assert from_env.api_key == "env-key"

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError):
        APISTTProcessor()


def test_audio_compression_path_for_large_file(tmp_path, monkeypatch):
    wav_path = tmp_path / "audio.wav"
    wav_path.write_bytes(b"a")

    proc = APISTTProcessor(api_key="k")

    class FakeStat:
        st_size = WHISPER_API_MAX_BYTES + 1

    monkeypatch.setattr(Path, "stat", lambda self: FakeStat())

    expected_mp3 = wav_path.with_suffix(".mp3")
    called = {}

    def fake_compress(input_path, output_path):
        called["input"] = input_path
        called["output"] = output_path
        return output_path

    monkeypatch.setattr(proc, "compress_audio_to_mp3", fake_compress)

    selected = proc._audio_for_upload(wav_path)

    assert selected == expected_mp3
    assert called["input"] == wav_path
    assert called["output"] == expected_mp3


def test_create_stt_processor_local_with_whisper(monkeypatch):
    monkeypatch.setattr(
        "heimdex_media_pipelines.speech.stt._is_importable",
        lambda module_name: module_name == "whisper",
    )

    proc = create_stt_processor(backend="local")
    assert isinstance(proc, STTProcessor)


def test_create_stt_processor_api_with_openai_and_key(monkeypatch):
    monkeypatch.setattr(
        "heimdex_media_pipelines.speech.stt._is_importable",
        lambda module_name: module_name == "openai",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")

    proc = create_stt_processor(backend="api")
    assert isinstance(proc, APISTTProcessor)


def test_create_stt_processor_auto_prefers_local(monkeypatch):
    monkeypatch.setattr(
        "heimdex_media_pipelines.speech.stt._is_importable",
        lambda module_name: module_name in {"whisper", "torch", "openai"},
    )
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")

    proc = create_stt_processor(backend="auto")
    assert isinstance(proc, STTProcessor)


def test_create_stt_processor_auto_uses_api_when_local_unavailable(monkeypatch):
    monkeypatch.setattr(
        "heimdex_media_pipelines.speech.stt._is_importable",
        lambda module_name: module_name == "openai",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")

    proc = create_stt_processor(backend="auto")
    assert isinstance(proc, APISTTProcessor)


def test_create_stt_processor_auto_raises_when_none_available(monkeypatch):
    monkeypatch.setattr(
        "heimdex_media_pipelines.speech.stt._is_importable",
        lambda module_name: False,
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ImportError):
        create_stt_processor(backend="auto")
