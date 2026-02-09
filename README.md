# heimdex-media-pipelines

Heavy ML pipeline code for Heimdex media processing. Contains face detection/embedding/registration and speech-to-text pipelines.

## Installation

```bash
# Core (cv2 + CLI only)
pip install -e .

# With face detection support
pip install -e ".[faces]"

# With speech-to-text support
pip install -e ".[speech]"

# Everything
pip install -e ".[all]"
```

## CLI Usage

```bash
# Check system dependencies
heimdex-pipelines doctor --json --out doctor.json

# Face detection
heimdex-pipelines faces detect --video video.mp4 --fps 1.0 --out detections.json

# Face embeddings
heimdex-pipelines faces embed --video video.mp4 --detections detections.jsonl --out embeddings.json

# Face identity registration
heimdex-pipelines faces register --identity-id person1 --ref-dir ./refs --out template.json

# Speech-to-text
heimdex-pipelines speech transcribe --video video.mp4 --out transcript.json

# Full speech pipeline (STT + tag + rank)
heimdex-pipelines speech pipeline --video video.mp4 --out result.json
```

## Python API

```python
from heimdex_media_pipelines.faces.detect import detect_faces
from heimdex_media_pipelines.faces.embed import extract_embeddings
from heimdex_media_pipelines.faces.register import build_identity_template
from heimdex_media_pipelines.faces.pipeline import run_pipeline as run_faces_pipeline
from heimdex_media_pipelines.speech.stt import STTProcessor
from heimdex_media_pipelines.speech.pipeline import SpeechSegmentsPipeline
```
