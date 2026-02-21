import json
import logging
import sys
import types
from email.message import Message
import urllib.error

import pytest

from heimdex_media_pipelines.vision.engine import (
    CaptionPerfTimings,
    CaptionResult,
    Florence2CaptionEngine,
    InternVL2CaptionEngine,
    LlamaCppCaptionEngine,
    create_caption_engine,
)


def _install_fake_torch(monkeypatch, cuda_available=False):
    class NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return None

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: cuda_available),
        bfloat16="bf16",
        float32="f32",
        float16="f16",
        no_grad=lambda: NoGrad(),
        inference_mode=lambda: NoGrad(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)


def _install_fake_pil(monkeypatch):
    class FakeImage:
        def convert(self, _):
            return self

        def resize(self, _):
            return self

    image_mod = types.SimpleNamespace(open=lambda _: FakeImage())
    pil_mod = types.SimpleNamespace(Image=image_mod)
    monkeypatch.setitem(sys.modules, "PIL", pil_mod)
    monkeypatch.setitem(sys.modules, "PIL.Image", image_mod)


def _install_fake_torchvision(monkeypatch):
    class FakeTensor:
        def unsqueeze(self, _):
            return self

        def to(self, **kwargs):
            return self

    class FakeCompose:
        def __init__(self, _steps):
            self._steps = _steps

        def __call__(self, _image):
            return FakeTensor()

    transforms_mod = types.SimpleNamespace(
        Compose=FakeCompose,
        ToTensor=lambda: object(),
        Normalize=lambda **kwargs: kwargs,
    )
    monkeypatch.setitem(sys.modules, "torchvision.transforms", transforms_mod)


def _install_fake_transformers(monkeypatch):
    calls = {"intern_model": 0, "intern_tokenizer": 0, "florence_model": 0, "florence_processor": 0}

    class FakeParam:
        device = "cpu"

    class FakeInternModel:
        def __init__(self):
            self.dtype = "f32"

        def to(self, _device):
            return self

        def eval(self):
            return self

        def chat(self, _tokenizer, _pixel_values, _prompt, _generation_config):
            return "x" * 600

        def parameters(self):
            return iter([FakeParam()])

    class FakeFlorenceModel:
        def to(self, _device):
            return self

        def eval(self):
            return self

        def parameters(self):
            return iter([FakeParam()])

        def generate(self, **kwargs):
            return [[1, 2, 3]]

    class FakeProcessor:
        def __call__(self, text, images, return_tensors):
            class FakeInputTensor:
                def to(self, _device):
                    return self

            return {"input_ids": FakeInputTensor()}

        def batch_decode(self, _ids, skip_special_tokens=True):
            return ["y" * 600]

    def auto_model_from_pretrained(*args, **kwargs):
        calls["intern_model"] += 1
        return FakeInternModel()

    def auto_tokenizer_from_pretrained(*args, **kwargs):
        calls["intern_tokenizer"] += 1
        return object()

    def auto_model_for_causal_lm_from_pretrained(*args, **kwargs):
        calls["florence_model"] += 1
        return FakeFlorenceModel()

    def auto_processor_from_pretrained(*args, **kwargs):
        calls["florence_processor"] += 1
        return FakeProcessor()

    transformers_mod = types.SimpleNamespace(
        AutoModel=types.SimpleNamespace(from_pretrained=auto_model_from_pretrained),
        AutoTokenizer=types.SimpleNamespace(from_pretrained=auto_tokenizer_from_pretrained),
        AutoModelForCausalLM=types.SimpleNamespace(
            from_pretrained=auto_model_for_causal_lm_from_pretrained
        ),
        AutoProcessor=types.SimpleNamespace(from_pretrained=auto_processor_from_pretrained),
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers_mod)
    return calls


def test_perf_timings_defaults_to_zero():
    perf = CaptionPerfTimings()
    values = perf.to_dict()
    assert values["model_load_s"] == 0.0
    assert values["inference_s"] == 0.0
    assert values["total_s"] == 0.0
    assert values["frames_processed"] == 0
    assert values["frames_skipped"] == 0


def test_perf_timings_to_dict_has_all_keys():
    perf = CaptionPerfTimings()
    assert set(perf.to_dict().keys()) == {
        "model_load_s",
        "inference_s",
        "total_s",
        "frames_processed",
        "frames_skipped",
    }


def test_perf_timings_log_emits_json(caplog):
    with caplog.at_level(logging.INFO, logger="heimdex_media_pipelines.vision.engine"):
        CaptionPerfTimings(total_s=1.23456, frames_processed=3).log(
            video_id="v1", model="internvl2"
        )
    msg = next(r.message for r in caplog.records if "caption_perf" in r.message)
    payload = json.loads(msg.split("caption_perf ", 1)[1])
    assert payload["event"] == "caption_perf"
    assert payload["video_id"] == "v1"
    assert payload["model"] == "internvl2"
    assert payload["total_s"] == 1.235


def test_create_caption_engine_valid_models():
    assert isinstance(create_caption_engine(model="internvl2"), InternVL2CaptionEngine)
    assert isinstance(create_caption_engine(model="florence2"), Florence2CaptionEngine)


def test_create_caption_engine_invalid_model_raises():
    with pytest.raises(ValueError, match="Unknown caption model"):
        create_caption_engine(model="unknown")


def test_internvl2_engine_init_stores_params():
    engine = InternVL2CaptionEngine(
        model_name="OpenGVLab/InternVL2-1B",
        use_gpu=True,
        max_new_tokens=33,
        cache_dir="/tmp/cache",
    )
    assert engine.model_name == "OpenGVLab/InternVL2-1B"
    assert engine.use_gpu is True
    assert engine.max_new_tokens == 33
    assert engine.cache_dir == "/tmp/cache"
    assert engine._model is None
    assert engine._tokenizer is None


def test_florence2_engine_init_stores_params():
    engine = Florence2CaptionEngine(
        model_name="microsoft/Florence-2-base",
        use_gpu=True,
        max_new_tokens=44,
        cache_dir="/tmp/cache",
    )
    assert engine.model_name == "microsoft/Florence-2-base"
    assert engine.use_gpu is True
    assert engine.max_new_tokens == 44
    assert engine.cache_dir == "/tmp/cache"
    assert engine._model is None
    assert engine._processor is None


def test_internvl2_caption_returns_caption_result_with_truncation(monkeypatch, tmp_path):
    _install_fake_torch(monkeypatch)
    _install_fake_pil(monkeypatch)
    _install_fake_torchvision(monkeypatch)
    calls = _install_fake_transformers(monkeypatch)

    image_path = tmp_path / "f.jpg"
    image_path.write_bytes(b"x")

    engine = InternVL2CaptionEngine()
    result = engine.caption(image_path)
    assert isinstance(result, CaptionResult)
    assert result.model == "OpenGVLab/InternVL2-1B"
    assert len(result.caption) == 500
    assert calls["intern_model"] == 1
    assert calls["intern_tokenizer"] == 1


def test_florence2_caption_returns_caption_result_with_truncation(monkeypatch, tmp_path):
    _install_fake_torch(monkeypatch)
    _install_fake_pil(monkeypatch)
    calls = _install_fake_transformers(monkeypatch)

    image_path = tmp_path / "f.jpg"
    image_path.write_bytes(b"x")

    engine = Florence2CaptionEngine()
    result = engine.caption(image_path)
    assert isinstance(result, CaptionResult)
    assert result.model == "microsoft/Florence-2-base"
    assert len(result.caption) == 500
    assert calls["florence_model"] == 1
    assert calls["florence_processor"] == 1


def test_caption_returns_empty_result_on_error(monkeypatch):
    _install_fake_torch(monkeypatch)
    _install_fake_pil(monkeypatch)
    _install_fake_torchvision(monkeypatch)
    _install_fake_transformers(monkeypatch)

    class BrokenIntern(InternVL2CaptionEngine):
        def _load_image(self, image_path):
            raise RuntimeError("boom")

    intern_result = BrokenIntern().caption("/tmp/nope.jpg")
    assert intern_result.caption == ""
    assert intern_result.model == "OpenGVLab/InternVL2-1B"
    assert intern_result.inference_s == 0.0

    class BrokenFlorence(Florence2CaptionEngine):
        def _load_model(self):
            raise RuntimeError("boom")

    florence_result = BrokenFlorence().caption("/tmp/nope.jpg")
    assert florence_result.caption == ""
    assert florence_result.model == "microsoft/Florence-2-base"
    assert florence_result.inference_s == 0.0


def test_lazy_load_model_once_for_both_engines(monkeypatch, tmp_path):
    _install_fake_torch(monkeypatch)
    _install_fake_pil(monkeypatch)
    _install_fake_torchvision(monkeypatch)
    calls = _install_fake_transformers(monkeypatch)

    image_path = tmp_path / "i.jpg"
    image_path.write_bytes(b"x")

    intern_engine = InternVL2CaptionEngine()
    intern_engine.caption(image_path)
    intern_engine.caption(image_path)
    assert calls["intern_model"] == 1
    assert calls["intern_tokenizer"] == 1

    florence_engine = Florence2CaptionEngine()
    florence_engine.caption(image_path)
    florence_engine.caption(image_path)
    assert calls["florence_model"] == 1
    assert calls["florence_processor"] == 1


class _FakeHTTPResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status = status

    def read(self):
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None


def test_llama_http_engine_init_stores_params():
    engine = LlamaCppCaptionEngine(
        base_url="http://x:8089",
        api_key="k",
        max_new_tokens=77,
        timeout_s=30.5,
        max_retries=4,
        retry_backoff_s=1.5,
    )
    assert engine.base_url == "http://x:8089"
    assert engine.api_key == "k"
    assert engine.max_new_tokens == 77
    assert engine.timeout_s == 30.5
    assert engine.max_retries == 4
    assert engine.retry_backoff_s == 1.5


def test_llama_http_caption_success(monkeypatch, tmp_path):
    image_path = tmp_path / "f.jpg"
    image_path.write_bytes(b"abc")

    def fake_urlopen(request, timeout):
        payload = {
            "choices": [{"message": {"content": "이 장면은 제품을 소개합니다."}}],
        }
        return _FakeHTTPResponse(payload, status=200)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    engine = LlamaCppCaptionEngine(base_url="http://x:8089", timeout_s=2.0)
    result = engine.caption(image_path)

    assert result.caption == "제품을 소개합니다"
    assert result.model == "llama-qwen2-vl-2b"
    assert result.inference_s >= 0.0


def test_llama_http_caption_http_error_returns_empty(monkeypatch, tmp_path):
    image_path = tmp_path / "f.jpg"
    image_path.write_bytes(b"abc")

    def fake_urlopen(request, timeout):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", lambda _s: None)
    engine = LlamaCppCaptionEngine(max_retries=0)
    result = engine.caption(image_path)

    assert result.caption == ""
    assert result.model == "llama-qwen2-vl-2b"
    assert result.inference_s == 0.0


def test_llama_http_circuit_breaker_opens_after_failures(monkeypatch, tmp_path):
    image_path = tmp_path / "f.jpg"
    image_path.write_bytes(b"abc")
    calls = {"count": 0}

    def fake_urlopen(request, timeout):
        calls["count"] += 1
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", lambda _s: None)
    engine = LlamaCppCaptionEngine(max_retries=0)

    for _ in range(5):
        result = engine.caption(image_path)
        assert result.caption == ""

    result = engine.caption(image_path)
    assert result.caption == ""
    assert calls["count"] == 5


def test_llama_http_circuit_breaker_resets_on_success(monkeypatch, tmp_path):
    image_path = tmp_path / "f.jpg"
    image_path.write_bytes(b"abc")
    calls = {"count": 0}

    def fake_urlopen(request, timeout):
        calls["count"] += 1
        if calls["count"] <= 4:
            raise urllib.error.URLError("connection refused")
        payload = {"choices": [{"message": {"content": "정상 캡션"}}]}
        return _FakeHTTPResponse(payload, status=200)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", lambda _s: None)
    engine = LlamaCppCaptionEngine(max_retries=0)

    for _ in range(4):
        engine.caption(image_path)
    result = engine.caption(image_path)

    assert result.caption == "정상 캡션"
    assert engine._consecutive_failures == 0
    assert engine._circuit_open_until == 0.0


def test_create_caption_engine_llama_http():
    engine = create_caption_engine(model="llama_http", base_url="http://x:8089")
    assert isinstance(engine, LlamaCppCaptionEngine)
    assert engine.base_url == "http://x:8089"


def test_llama_http_retry_on_server_error(monkeypatch, tmp_path):
    image_path = tmp_path / "f.jpg"
    image_path.write_bytes(b"abc")
    calls = {"count": 0}

    def fake_urlopen(request, timeout):
        calls["count"] += 1
        if calls["count"] <= 2:
            raise urllib.error.HTTPError(
                url="http://x:8089/v1/chat/completions",
                code=500,
                msg="server error",
                hdrs=Message(),
                fp=None,
            )
        payload = {"choices": [{"message": {"content": "성공"}}]}
        return _FakeHTTPResponse(payload, status=200)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", lambda _s: None)
    engine = LlamaCppCaptionEngine(max_retries=3)
    result = engine.caption(image_path)

    assert result.caption == "성공"
    assert calls["count"] == 3


def test_llama_http_no_retry_on_client_error(monkeypatch, tmp_path):
    image_path = tmp_path / "f.jpg"
    image_path.write_bytes(b"abc")
    calls = {"count": 0}

    def fake_urlopen(request, timeout):
        calls["count"] += 1
        raise urllib.error.HTTPError(
            url="http://x:8089/v1/chat/completions",
            code=400,
            msg="bad request",
            hdrs=Message(),
            fp=None,
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", lambda _s: None)
    engine = LlamaCppCaptionEngine(max_retries=3)
    result = engine.caption(image_path)

    assert result.caption == ""
    assert result.model == "llama-qwen2-vl-2b"
    assert calls["count"] == 1
