# Heimdex Media Pipelines

Heavy ML processing library for video analysis: face detection/embedding, speech-to-text, diarization, OCR, vision captioning, scene detection, and transcoding.

## Quick Reference

```bash
# Install (core only)
pip install -e .

# Install with specific extras
pip install -e ".[faces]"          # insightface + onnxruntime
pip install -e ".[speech]"         # openai-whisper + torch
pip install -e ".[speech-fast]"    # faster-whisper (CTranslate2)
pip install -e ".[diarization]"    # pyannote-audio + torch
pip install -e ".[ocr]"           # paddleocr + paddlepaddle
pip install -e ".[vision]"        # transformers + torch
pip install -e ".[all]"           # Everything

# Run tests
make test                          # Quick (skip golden)
make test-golden                   # Full (requires models)

# CLI
python -m heimdex_media_pipelines doctor --json
python -m heimdex_media_pipelines faces detect --video VIDEO --fps 1.0 --out detections.json
python -m heimdex_media_pipelines speech pipeline --video VIDEO --out result.json
python -m heimdex_media_pipelines ocr detect --video VIDEO --out ocr.json
python -m heimdex_media_pipelines scenes detect --video VIDEO --speech-result transcript.json --out scenes.json
python -m heimdex_media_pipelines pipeline --video VIDEO --split-preset default --split-target-duration-ms 30000 --out result.json
```

## Architecture

```
src/heimdex_media_pipelines/
├── __main__.py        # CLI entry (python -m ...)
├── cli.py             # Typer CLI root
├── device.py          # GPU/CUDA/Metal detection
├── faces/
│   ├── detect.py      # InsightFace wrapper
│   ├── embed.py       # Face embedding extraction
│   ├── register.py    # Identity template building
│   ├── pipeline.py    # Full faces workflow
│   └── cli.py         # Face CLI commands
├── speech/
│   ├── stt.py         # Whisper/faster-whisper wrapper
│   ├── diarization.py # pyannote speaker diarization
│   ├── pipeline.py    # STT -> tag -> rank workflow
│   └── cli.py         # Speech CLI commands
├── scenes/
│   ├── signals.py     # extract_speech_pauses(), extract_speaker_turns()
│   └── splitter.py    # split_scenes() orchestrator (speech-aware)
├── ocr/               # PaddleOCR wrapper
├── vision/            # Transformers-based image understanding
└── transcoding/       # FFmpeg video reencoding
```

## Dependencies

**Core (always installed):**
- `heimdex-media-contracts>=0.7.0`
- `opencv-python-headless>=4.8`
- `typer>=0.9`

**Optional extras (installed per feature):**

| Extra | Key Libraries | Use Case |
|---|---|---|
| `faces` | insightface, onnxruntime | Face detection + embedding |
| `speech` | openai-whisper, torch | Speech-to-text |
| `speech-fast` | faster-whisper | Optimized STT (CTranslate2) |
| `speech-api` | openai | Cloud-based STT |
| `diarization` | pyannote-audio 3.3.2, torch | Speaker identification |
| `ocr` | paddleocr, paddlepaddle | On-screen text extraction |
| `vision` | transformers, torch, torchvision, autoawq | Image understanding (Qwen2.5-VL-7B-AWQ default) |

## GPU / Cross-Platform Detection

`device.py` handles automatic GPU detection:

| Platform | GPU Support | Fallback |
|---|---|---|
| macOS | CoreML via ONNX CoreMLExecutionProvider | CPU (always) |
| Linux/Windows | CUDA via onnxruntime or torch | CPU |
| Whisper on macOS | CPU + int8 only (no Metal backend for CTranslate2) | — |
| PaddleOCR on macOS | CPU only (no GPU package) | — |

**All operations degrade gracefully to CPU.** Never assume GPU availability.

### Model Cache Locations

| Model | Cache Path |
|---|---|
| InsightFace | `~/.insightface/` |
| Whisper/HuggingFace | `~/.cache/huggingface/hub/` |
| pyannote (needs HF token) | `~/.cache/huggingface/hub/` |
| PaddleOCR | `~/.paddleocr/` |

**Aircloud note:** Some containers have unwritable `TORCH_HOME`. The diarization module handles this with explicit `cache_dir` parameter.

## Consumers

| Consumer | How it uses pipelines | Context |
|---|---|---|
| `heimdex-agent` | Subprocess: `python -m heimdex_media_pipelines <cmd>` | Bundled in .app/.exe |
| `dev-heimdex-for-livecommerce` | Docker volume mount, editable install | API + drive-worker |
| Aircloud GPU workers | GHCR Docker images with specific extras | STT, OCR, caption, visual-embed |

**Changes here affect livecommerce (via volume mount) and agent (via bundled distribution).** Always verify downstream impact.

## Testing

33+ test files covering CLI smoke tests, device detection, face/speech/scene/OCR/vision/transcoding pipelines, and speech-aware scene splitting signals/splitter — 35 new tests.

```bash
make test              # Quick tests (skip golden dataset)
make test-golden       # Full OCR golden tests (requires models)
make ocr-audit         # Security audit via pip-audit
```

Docker test: `docker build -f Dockerfile.test -t pipelines-test . && docker run pipelines-test`

## Deployment

### To Livecommerce (EC2)
- Volume-mounted into Docker containers
- `.github/workflows/deploy-staging.yml` syncs to EC2 on push to `main`
- Only `drive-worker` runs on EC2 and mounts pipelines
- Enrichment workers (STT, OCR, caption) run on Aircloud+ GPU

### To PyPI
- Tag `v*` triggers release workflow
- Build wheel + sdist → publish to PyPI
- Current version: 0.8.0

### GPU Worker Images (Aircloud)
- Built via `build-gpu-images.yml` in livecommerce repo
- Published to GHCR with `latest` + git SHA tags

## Rules

- Always maintain CPU fallback for every operation
- Optional dependencies stay optional (use extras)
- CLI output is JSON (consumed by agent via subprocess)
- FFmpeg is required for video probing/transcoding (bundled in agent, available in Docker)
- Test with `make test` before pushing (skip golden for fast iteration)
- Handle `TORCH_HOME` write failures gracefully (Aircloud containers)
