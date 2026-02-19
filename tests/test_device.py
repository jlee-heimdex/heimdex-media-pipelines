"""Tests for the shared GPU / device detection module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from heimdex_media_pipelines.device import (
    detect_onnx_providers,
    detect_paddle_gpu,
    detect_whisper_device,
    get_onnx_provider_names,
)


class TestDetectOnnxProviders:

    def test_cpu_fallback_when_onnxruntime_missing(self):
        with patch.dict("sys.modules", {"onnxruntime": None}):
            import importlib
            import heimdex_media_pipelines.device as dev
            importlib.reload(dev)
            result = dev.detect_onnx_providers()
            assert result == ["CPUExecutionProvider"]
            importlib.reload(dev)

    def test_coreml_on_macos_arm64(self):
        mock_ort = MagicMock()
        mock_ort.get_available_providers.return_value = [
            "CoreMLExecutionProvider",
            "AzureExecutionProvider",
            "CPUExecutionProvider",
        ]
        with patch("heimdex_media_pipelines.device.platform") as mock_platform, \
             patch.dict("sys.modules", {"onnxruntime": mock_ort}):
            mock_platform.system.return_value = "Darwin"
            mock_platform.machine.return_value = "arm64"

            import importlib
            import heimdex_media_pipelines.device as dev
            importlib.reload(dev)
            result = dev.detect_onnx_providers()
            importlib.reload(dev)

        assert len(result) == 2
        assert result[0][0] == "CoreMLExecutionProvider"
        assert result[0][1]["MLComputeUnits"] == "ALL"
        assert result[1] == "CPUExecutionProvider"

    def test_cuda_on_windows(self):
        mock_ort = MagicMock()
        mock_ort.get_available_providers.return_value = [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        with patch("heimdex_media_pipelines.device.platform") as mock_platform, \
             patch.dict("sys.modules", {"onnxruntime": mock_ort}):
            mock_platform.system.return_value = "Windows"
            mock_platform.machine.return_value = "AMD64"

            import importlib
            import heimdex_media_pipelines.device as dev
            importlib.reload(dev)
            result = dev.detect_onnx_providers()
            importlib.reload(dev)

        assert result == ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def test_dml_on_windows_without_cuda(self):
        mock_ort = MagicMock()
        mock_ort.get_available_providers.return_value = [
            "DmlExecutionProvider",
            "CPUExecutionProvider",
        ]
        with patch("heimdex_media_pipelines.device.platform") as mock_platform, \
             patch.dict("sys.modules", {"onnxruntime": mock_ort}):
            mock_platform.system.return_value = "Windows"
            mock_platform.machine.return_value = "AMD64"

            import importlib
            import heimdex_media_pipelines.device as dev
            importlib.reload(dev)
            result = dev.detect_onnx_providers()
            importlib.reload(dev)

        assert result == ["DmlExecutionProvider", "CPUExecutionProvider"]

    def test_cpu_only_when_no_gpu_provider(self):
        mock_ort = MagicMock()
        mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]
        with patch("heimdex_media_pipelines.device.platform") as mock_platform, \
             patch.dict("sys.modules", {"onnxruntime": mock_ort}):
            mock_platform.system.return_value = "Linux"
            mock_platform.machine.return_value = "x86_64"

            import importlib
            import heimdex_media_pipelines.device as dev
            importlib.reload(dev)
            result = dev.detect_onnx_providers()
            importlib.reload(dev)

        assert result == ["CPUExecutionProvider"]

    def test_macos_intel_uses_coreml_if_available(self):
        mock_ort = MagicMock()
        mock_ort.get_available_providers.return_value = [
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        ]
        with patch("heimdex_media_pipelines.device.platform") as mock_platform, \
             patch.dict("sys.modules", {"onnxruntime": mock_ort}):
            mock_platform.system.return_value = "Darwin"
            mock_platform.machine.return_value = "x86_64"

            import importlib
            import heimdex_media_pipelines.device as dev
            importlib.reload(dev)
            result = dev.detect_onnx_providers()
            importlib.reload(dev)

        assert len(result) == 2
        assert result[0][0] == "CoreMLExecutionProvider"
        assert result[1] == "CPUExecutionProvider"


class TestDetectWhisperDevice:

    def test_macos_always_cpu(self):
        with patch("heimdex_media_pipelines.device.platform") as mock_platform:
            mock_platform.system.return_value = "Darwin"
            device, compute = detect_whisper_device()
        assert device == "cpu"
        assert compute == "int8"

    def test_windows_with_cuda(self):
        mock_ct2 = MagicMock()
        mock_ct2.get_cuda_device_count.return_value = 1
        with patch("heimdex_media_pipelines.device.platform") as mock_platform, \
             patch.dict("sys.modules", {"ctranslate2": mock_ct2}):
            mock_platform.system.return_value = "Windows"
            device, compute = detect_whisper_device()
        assert device == "cuda"
        assert compute == "float16"

    def test_windows_without_cuda(self):
        mock_ct2 = MagicMock()
        mock_ct2.get_cuda_device_count.return_value = 0
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch("heimdex_media_pipelines.device.platform") as mock_platform, \
             patch.dict("sys.modules", {"ctranslate2": mock_ct2, "torch": mock_torch}):
            mock_platform.system.return_value = "Windows"
            device, compute = detect_whisper_device()
        assert device == "cpu"
        assert compute == "int8"

    def test_torch_fallback_when_ctranslate2_missing(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        def fake_import(name, *args, **kwargs):
            if name == "ctranslate2":
                raise ImportError("mocked: no ctranslate2")
            if name == "torch":
                return mock_torch
            return original_import(name, *args, **kwargs)

        import builtins
        original_import = builtins.__import__
        with patch("heimdex_media_pipelines.device.platform") as mock_platform, \
             patch("builtins.__import__", side_effect=fake_import):
            mock_platform.system.return_value = "Windows"
            device, compute = detect_whisper_device()
        assert device == "cuda"
        assert compute == "float16"


class TestDetectPaddleGpu:

    def test_macos_always_false(self):
        with patch("heimdex_media_pipelines.device.platform") as mock_platform:
            mock_platform.system.return_value = "Darwin"
            assert detect_paddle_gpu() is False

    def test_windows_with_paddle_gpu(self):
        mock_paddle = MagicMock()
        mock_paddle.device.is_compiled_with_cuda.return_value = True
        mock_paddle.device.cuda.device_count.return_value = 1
        with patch("heimdex_media_pipelines.device.platform") as mock_platform, \
             patch.dict("sys.modules", {"paddle": mock_paddle}):
            mock_platform.system.return_value = "Windows"
            assert detect_paddle_gpu() is True

    def test_windows_without_paddle(self):
        with patch("heimdex_media_pipelines.device.platform") as mock_platform, \
             patch.dict("sys.modules", {"paddle": None}):
            mock_platform.system.return_value = "Windows"

            import importlib
            import heimdex_media_pipelines.device as dev
            importlib.reload(dev)
            result = dev.detect_paddle_gpu()
            importlib.reload(dev)
        assert result is False

    def test_paddle_cpu_only(self):
        mock_paddle = MagicMock()
        mock_paddle.device.is_compiled_with_cuda.return_value = False
        with patch("heimdex_media_pipelines.device.platform") as mock_platform, \
             patch.dict("sys.modules", {"paddle": mock_paddle}):
            mock_platform.system.return_value = "Windows"
            assert detect_paddle_gpu() is False


class TestGetOnnxProviderNames:

    def test_returns_list_of_strings(self):
        result = get_onnx_provider_names()
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, str)

    def test_empty_when_onnxruntime_missing(self):
        with patch.dict("sys.modules", {"onnxruntime": None}):
            import importlib
            import heimdex_media_pipelines.device as dev
            importlib.reload(dev)
            result = dev.get_onnx_provider_names()
            importlib.reload(dev)
        assert result == []
