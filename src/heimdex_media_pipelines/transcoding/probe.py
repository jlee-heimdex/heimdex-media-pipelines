import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ProbeResult:
    width: int
    height: int
    codec_name: str
    bitrate_kbps: int
    duration_ms: int
    has_audio: bool
    audio_codec: Optional[str] = None
    audio_bitrate_kbps: Optional[int] = None


def probe_video(path: Path) -> ProbeResult:
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)

    video_stream = next(
        (s for s in data.get("streams", []) if s.get("codec_type") == "video"), None
    )
    audio_stream = next(
        (s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None
    )

    if not video_stream:
        raise ValueError(f"No video stream found in {path}")

    fmt = data.get("format", {})
    total_bitrate = int(fmt.get("bit_rate", 0)) // 1000
    duration_s = float(fmt.get("duration", 0))

    video_bitrate = int(video_stream.get("bit_rate", 0)) // 1000
    if video_bitrate == 0:
        video_bitrate = total_bitrate - (int(audio_stream.get("bit_rate", 0)) // 1000 if audio_stream else 0)
        video_bitrate = max(video_bitrate, 0)

    return ProbeResult(
        width=int(video_stream.get("width", 0)),
        height=int(video_stream.get("height", 0)),
        codec_name=video_stream.get("codec_name", "unknown"),
        bitrate_kbps=video_bitrate,
        duration_ms=int(duration_s * 1000),
        has_audio=audio_stream is not None,
        audio_codec=audio_stream.get("codec_name") if audio_stream else None,
        audio_bitrate_kbps=int(audio_stream.get("bit_rate", 0)) // 1000 if audio_stream else None,
    )
