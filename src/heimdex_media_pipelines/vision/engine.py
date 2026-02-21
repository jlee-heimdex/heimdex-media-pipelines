from __future__ import annotations

import json
import importlib
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)

DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_CAPTION_PROMPT = (
    "당신은 영상 장면 분석 전문가입니다. "
    "이 프레임을 보고, 보이는 내용을 한국어로 2~3문장으로 구체적으로 묘사하세요. "
    "등장하는 사람, 사물, 배경, 행동, 화면에 보이는 텍스트를 빠짐없이 포함하세요. "
    "이 설명만으로 시청자가 장면을 상상할 수 있어야 합니다. "
    "반드시 한국어로만 답하세요."
)


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
                "do_sample": True,
                "temperature": 0.3,
                "repetition_penalty": 1.5,
            }

            t0 = time.monotonic()
            with torch.no_grad():
                response = self._model.chat(self._tokenizer, pixel_values, prompt, generation_config)
            elapsed = time.monotonic() - t0

            text = response.strip()[:500]
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
            with torch.no_grad():
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


_ENGINE_REGISTRY: dict[str, type] = {
    "internvl2": InternVL2CaptionEngine,
    "florence2": Florence2CaptionEngine,
}


def create_caption_engine(
    model: str = "internvl2",
    use_gpu: bool = False,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    cache_dir: str | None = None,
) -> CaptionEngine:
    engine_cls = _ENGINE_REGISTRY.get(model)
    if engine_cls is None:
        raise ValueError(f"Unknown caption model: {model!r}. Available: {sorted(_ENGINE_REGISTRY)}")
    return engine_cls(use_gpu=use_gpu, max_new_tokens=max_new_tokens, cache_dir=cache_dir)
