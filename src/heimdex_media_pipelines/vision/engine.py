from __future__ import annotations

import json
import importlib
import logging
import re
import time
import base64
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)

DEFAULT_MAX_NEW_TOKENS = 80
DEFAULT_CAPTION_PROMPT = "이 장면을 한 문장으로 서술:"


@dataclass
class CaptionResult:
    caption: str = ""
    model: str = ""
    inference_s: float = 0.0


@dataclass
class CaptionPerfTimings:
    model_load_s: float = 0.0
    inference_s: float = 0.0
    total_s: float = 0.0
    frames_processed: int = 0
    frames_skipped: int = 0

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)

    def log(self, video_id: str = "", model: str = "") -> None:
        payload = {
            "event": "caption_perf",
            "video_id": video_id,
            "model": model,
            **{k: round(v, 3) if isinstance(v, float) else v for k, v in self.to_dict().items()},
        }
        logger.info("caption_perf %s", json.dumps(payload, ensure_ascii=False))


# Regex to strip Chinese (CJK Unified), Japanese kana, and stray non-Korean fragments.
# Keeps: Korean (Hangul), ASCII, basic Latin punctuation, digits, whitespace.
_CJK_NOISE_RE = re.compile(
    r"[\u4e00-\u9fff"         # CJK Unified Ideographs (Chinese)
    r"\u3040-\u309f"          # Hiragana
    r"\u30a0-\u30ff"          # Katakana
    r"\u3400-\u4dbf"          # CJK Extension A
    r"\uf900-\ufaff"          # CJK Compatibility Ideographs
    r"\U00020000-\U0002a6df"  # CJK Extension B
    r"]+"
)


_FORMULAIC_PREFIX_RE = re.compile(
    r"^(?:이 (?:장면은|이미지는|사진은|영상은|화면은)\s*)"
    r"|^(?:이 (?:장면|이미지|사진|영상|화면)(?:에서는?|을|를|의)\s*)"
    r"|^(?:다음은\s*)"
    r"|^(?:그림에 있는\s*)"
)


def _clean_caption(text: str) -> str:
    """Remove non-Korean CJK noise and truncate at first degeneration sign."""
    text = text.strip().strip('"').strip("'").strip()
    text = _CJK_NOISE_RE.sub("", text)
    text = re.sub(r"\([^)]*[a-zA-Z]{3,}[^)]*\)", "", text)
    text = _FORMULAIC_PREFIX_RE.sub("", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    for pattern in (r"\n\s*[-*]\s", r"\n\s*\d+[.)]\s", r"\s*#\S", r"\*\*"):
        match = re.search(pattern, text)
        if match:
            text = text[:match.start()].rstrip()
    text = re.sub(r"\s*[,.]?\s*$", "", text)
    if text and text[-1] not in ".!?다요죠니까":
        last_sent = max(text.rfind(". "), text.rfind("다. "), text.rfind("니다."), text.rfind("요."))
        if last_sent > len(text) // 3:
            text = text[:last_sent + 1 + (2 if text[last_sent:last_sent+3] == "다. " else 0)].rstrip()
    return text.strip()


class CaptionEngine(Protocol):
    def caption(self, image_path: str | Path, prompt: str = "") -> CaptionResult: ...


class InternVL2CaptionEngine:
    """InternVL2-1B implementation. Native Korean output. Lazy-loads model on first call."""

    def __init__(
        self,
        model_name: str = "OpenGVLab/InternVL2-1B",
        use_gpu: bool = False,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        cache_dir: str | None = None,
    ):
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.max_new_tokens = max_new_tokens
        self.cache_dir = cache_dir
        self._model = None
        self._tokenizer = None

    def _load_model(self) -> None:
        if self._model is not None:
            return

        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")

        device = "cuda" if self.use_gpu and torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        self._model = transformers.AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            cache_dir=self.cache_dir,
            low_cpu_mem_usage=False,
            device_map=None,
        ).to(device).eval()

        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            cache_dir=self.cache_dir,
        )

    def caption(self, image_path: str | Path, prompt: str = "") -> CaptionResult:
        try:
            if self._model is None:
                self._load_model()
            if self._model is None or self._tokenizer is None:
                raise RuntimeError("InternVL2 model failed to load")

            torch = importlib.import_module("torch")

            if not prompt:
                prompt = f"<image>\n{DEFAULT_CAPTION_PROMPT}"

            pixel_values = self._load_image(image_path)
            generation_config = {
                "max_new_tokens": self.max_new_tokens,
                "num_beams": 1,
                "do_sample": False,
                "repetition_penalty": 1.5,
            }

            t0 = time.monotonic()
            with torch.inference_mode():
                response = self._model.chat(self._tokenizer, pixel_values, prompt, generation_config)
            elapsed = time.monotonic() - t0

            text = _clean_caption(response.strip()[:500])
            return CaptionResult(caption=text, model=self.model_name, inference_s=elapsed)

        except Exception as e:
            logger.warning("Caption failed for %s: %s", image_path, e)
            return CaptionResult(model=self.model_name)

    def _load_image(self, image_path: str | Path):
        torchvision_transforms = importlib.import_module("torchvision.transforms")
        pil_image = importlib.import_module("PIL.Image")
        if self._model is None:
            raise RuntimeError("InternVL2 model is not loaded")

        image = pil_image.open(str(image_path)).convert("RGB")
        image = image.resize((448, 448))

        transform = torchvision_transforms.Compose(
            [
                torchvision_transforms.ToTensor(),
                torchvision_transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        pixel_values: Any = transform(image).unsqueeze(0)

        device = next(self._model.parameters()).device
        pixel_values = pixel_values.to(device=device, dtype=self._model.dtype)
        return pixel_values


class Florence2CaptionEngine:
    """Florence-2-base fallback. English-only. Fastest CPU option."""

    def __init__(
        self,
        model_name: str = "microsoft/Florence-2-base",
        use_gpu: bool = False,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        cache_dir: str | None = None,
    ):
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.max_new_tokens = max_new_tokens
        self.cache_dir = cache_dir
        self._model = None
        self._processor = None

    def _load_model(self) -> None:
        if self._model is not None:
            return

        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")

        device = "cuda" if self.use_gpu and torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        self._model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            cache_dir=self.cache_dir,
        ).to(device).eval()

        self._processor = transformers.AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            cache_dir=self.cache_dir,
        )

    def caption(self, image_path: str | Path, prompt: str = "") -> CaptionResult:
        try:
            if self._model is None:
                self._load_model()
            if self._model is None or self._processor is None:
                raise RuntimeError("Florence2 model failed to load")

            torch = importlib.import_module("torch")
            pil_image = importlib.import_module("PIL.Image")

            task_prompt = "<MORE_DETAILED_CAPTION>" if not prompt else prompt
            image = pil_image.open(str(image_path)).convert("RGB")

            inputs = self._processor(
                text=[task_prompt], images=[image], return_tensors="pt"
            )
            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            t0 = time.monotonic()
            with torch.inference_mode():
                generated_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    num_beams=1,
                    do_sample=False,
                )
            elapsed = time.monotonic() - t0

            text = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            text = text.strip()[:500]
            return CaptionResult(caption=text, model=self.model_name, inference_s=elapsed)

        except Exception as e:
            logger.warning("Caption failed for %s: %s", image_path, e)
            return CaptionResult(model=self.model_name)


class LlamaCppCaptionEngine:

    def __init__(
        self,
        base_url: str = "http://localhost:8089",
        api_key: str = "",
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        timeout_s: float = 300.0,
        max_retries: int = 3,
        retry_backoff_s: float = 2.0,
        max_image_dim: int = 448,
        temperature: float = 0.1,
        repeat_penalty: float = 1.15,
        presence_penalty: float = 0.6,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.max_new_tokens = max_new_tokens
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s
        self.max_image_dim = max_image_dim
        self.temperature = temperature
        self.repeat_penalty = repeat_penalty
        self.presence_penalty = presence_penalty
        self.model_name = "llama-qwen2-vl-2b"
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0

    def _record_success(self) -> None:
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0

    def _record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= 5:
            self._circuit_open_until = time.monotonic() + 60.0
            logger.warning("Llama caption circuit breaker opened for 60s")

    def _is_circuit_open(self) -> bool:
        return time.monotonic() < self._circuit_open_until

    def _maybe_resize(self, image_path: Path) -> bytes:
        """Resize image if larger than max_image_dim, preserving aspect ratio.

        Returns JPEG bytes. Resizing reduces vision encoder tokens
        (e.g. 720x1280 → 252x448 cuts tokens from ~1196 to ~144).
        """
        import io

        try:
            PIL_Image = importlib.import_module("PIL.Image")
            img = PIL_Image.open(image_path)
            w, h = img.size
            max_dim = self.max_image_dim

            if max(w, h) > max_dim:
                if w >= h:
                    new_w = max_dim
                    new_h = int(h * max_dim / w)
                else:
                    new_h = max_dim
                    new_w = int(w * max_dim / h)
                img = img.resize((new_w, new_h), PIL_Image.LANCZOS)
                logger.debug("Resized %s from %dx%d to %dx%d", image_path.name, w, h, new_w, new_h)

            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            return buf.getvalue()
        except Exception:
            return image_path.read_bytes()

    def caption(self, image_path: str | Path, prompt: str = "") -> CaptionResult:
        if self._is_circuit_open():
            logger.warning("Llama caption circuit breaker open; skipping HTTP call")
            return CaptionResult(model=self.model_name)

        if not prompt:
            prompt = DEFAULT_CAPTION_PROMPT

        t0 = time.monotonic()
        image_path_obj = Path(image_path)

        try:
            image_bytes = self._maybe_resize(image_path_obj)
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        except Exception as e:
            logger.warning("Caption failed for %s: %s", image_path, e)
            self._record_failure()
            return CaptionResult(model=self.model_name)

        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                        },
                    ],
                }
            ],
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "repeat_penalty": self.repeat_penalty,
            "presence_penalty": self.presence_penalty,
        }

        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = f"{self.base_url}/v1/chat/completions"

        for attempt in range(self.max_retries + 1):
            try:
                request = urllib.request.Request(url=url, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    status = getattr(response, "status", 200)
                    response_body = response.read().decode("utf-8")

                if status >= 500:
                    if attempt < self.max_retries:
                        time.sleep(self.retry_backoff_s * (2 ** attempt))
                        continue
                    logger.warning("Caption failed for %s: HTTP %s", image_path, status)
                    self._record_failure()
                    return CaptionResult(model=self.model_name)

                response_json = json.loads(response_body)
                raw_text = (
                    response_json.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                text = _clean_caption(str(raw_text).strip()[:500])
                elapsed = time.monotonic() - t0
                self._record_success()
                return CaptionResult(caption=text, model=self.model_name, inference_s=elapsed)
            except urllib.error.HTTPError as e:
                if 500 <= e.code < 600:
                    if attempt < self.max_retries:
                        time.sleep(self.retry_backoff_s * (2 ** attempt))
                        continue
                logger.warning("Caption failed for %s: %s", image_path, e)
                self._record_failure()
                return CaptionResult(model=self.model_name)
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff_s * (2 ** attempt))
                    continue
                logger.warning("Caption failed for %s: %s", image_path, e)
                self._record_failure()
                return CaptionResult(model=self.model_name)
            except Exception as e:
                logger.warning("Caption failed for %s: %s", image_path, e)
                self._record_failure()
                return CaptionResult(model=self.model_name)

        self._record_failure()
        return CaptionResult(model=self.model_name)


_ENGINE_REGISTRY: dict[str, type] = {
    "internvl2": InternVL2CaptionEngine,
    "florence2": Florence2CaptionEngine,
    "llama_http": LlamaCppCaptionEngine,
}


def create_caption_engine(
    model: str = "internvl2",
    use_gpu: bool = False,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    cache_dir: str | None = None,
    **kwargs: Any,
) -> CaptionEngine:
    engine_cls = _ENGINE_REGISTRY.get(model)
    if engine_cls is None:
        raise ValueError(f"Unknown caption model: {model!r}. Available: {sorted(_ENGINE_REGISTRY)}")
    if model == "llama_http":
        return engine_cls(max_new_tokens=max_new_tokens, **kwargs)
    return engine_cls(use_gpu=use_gpu, max_new_tokens=max_new_tokens, cache_dir=cache_dir)
