import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _parse_frame_rate(r_frame_rate: str) -> float:
    """Parse ffprobe r_frame_rate string to a float.

    ffprobe returns frame rate as a rational fraction (e.g. "30000/1001"
    for 29.97fps, "25/1" for 25fps) or occasionally as a plain float.
    Mirrors the Go agent's parseFrameRate() in ffmpeg.go.

    Returns 0.0 if the value cannot be parsed.
    """
    if not r_frame_rate:
        return 0.0
    if "/" in r_frame_rate:
        parts = r_frame_rate.split("/", 1)
        try:
            num, den = float(parts[0]), float(parts[1])
            return round(num / den, 3) if den != 0 else 0.0
        except (ValueError, ZeroDivisionError):
            return 0.0
    try:
        return round(float(r_frame_rate), 3)
    except ValueError:
        return 0.0


@dataclass
class ProbeResult:
    """Video metadata extracted via ffprobe.

    Attributes:
        width: Video width in pixels.
        height: Video height in pixels.
        frame_rate: Frames per second (e.g. 29.97, 25.0, 60.0).
            Parsed from ffprobe's r_frame_rate field.
        codec_name: Video codec (e.g. "h264", "hevc").
        bitrate_kbps: Video stream bitrate in kbps.
        duration_ms: Total duration in milliseconds.
        has_audio: Whether the file contains an audio stream.
        audio_codec: Audio codec name (e.g. "aac"), or None.
        audio_bitrate_kbps: Audio bitrate in kbps, or None.
    """

    width: int
    height: int
    frame_rate: float
    codec_name: str
    bitrate_kbps: int
    duration_ms: int
    has_audio: bool
    audio_codec: Optional[str] = None
    audio_bitrate_kbps: Optional[int] = None


def probe_video(path: Path) -> ProbeResult:
    """Run ffprobe on a video file and return structured metadata.

    Raises:
        subprocess.CalledProcessError: If ffprobe exits with a non-zero code.
        ValueError: If the file contains no video stream.
    """
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
        frame_rate=_parse_frame_rate(video_stream.get("r_frame_rate", "")),
        codec_name=video_stream.get("codec_name", "unknown"),
        bitrate_kbps=video_bitrate,
        duration_ms=int(duration_s * 1000),
        has_audio=audio_stream is not None,
        audio_codec=audio_stream.get("codec_name") if audio_stream else None,
        audio_bitrate_kbps=int(audio_stream.get("bit_rate", 0)) // 1000 if audio_stream else None,
    )
