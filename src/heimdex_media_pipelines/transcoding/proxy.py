import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

from heimdex_media_pipelines.transcoding.probe import ProbeResult

logger = logging.getLogger(__name__)


@dataclass
class TranscodeDecision:
    should_transcode: bool
    reason: str
    should_cap_bitrate: bool = False


def make_transcode_decision(
    probe: ProbeResult,
    max_height: int = 720,
    max_bitrate_kbps: int = 2500,
) -> TranscodeDecision:
    is_h264 = probe.codec_name in ("h264", "h264_qsv", "h264_nvenc")
    is_within_resolution = probe.height <= max_height
    is_within_bitrate = probe.bitrate_kbps <= max_bitrate_kbps

    if is_h264 and is_within_resolution and is_within_bitrate:
        return TranscodeDecision(
            should_transcode=False,
            reason=f"Already H.264 {probe.width}x{probe.height} @ {probe.bitrate_kbps}kbps (within {max_bitrate_kbps}kbps cap)",
        )

    if is_h264 and is_within_resolution and not is_within_bitrate:
        return TranscodeDecision(
            should_transcode=True,
            should_cap_bitrate=True,
            reason=f"H.264 {probe.width}x{probe.height} but bitrate {probe.bitrate_kbps}kbps exceeds {max_bitrate_kbps}kbps cap",
        )

    reasons = []
    if not is_h264:
        reasons.append(f"codec={probe.codec_name}")
    if not is_within_resolution:
        reasons.append(f"height={probe.height} > {max_height}")
    if not is_within_bitrate:
        reasons.append(f"bitrate={probe.bitrate_kbps}kbps > {max_bitrate_kbps}kbps")

    return TranscodeDecision(
        should_transcode=True,
        reason=f"Requires transcode: {', '.join(reasons)}",
    )


def transcode_to_proxy(
    input_path: Path,
    output_path: Path,
    probe: ProbeResult,
    decision: TranscodeDecision,
    *,
    max_height: int = 720,
    preset: str = "fast",
    crf: int = 23,
    max_bitrate: str = "2500k",
    bufsize: str = "5000k",
    audio_bitrate: str = "128k",
) -> Path:
    cmd = ["ffmpeg", "-y", "-i", str(input_path)]

    if decision.should_cap_bitrate and probe.height <= max_height:
        cmd.extend([
            "-c:v", "libx264",
            "-preset", preset,
            "-maxrate", max_bitrate,
            "-bufsize", bufsize,
            "-crf", str(crf),
        ])
    else:
        cmd.extend([
            "-c:v", "libx264",
            "-preset", preset,
            "-crf", str(crf),
            "-maxrate", max_bitrate,
            "-bufsize", bufsize,
            "-vf", f"scale=-2:{max_height}",
        ])

    if probe.has_audio:
        cmd.extend(["-c:a", "aac", "-b:a", audio_bitrate])
    else:
        cmd.extend(["-an"])

    cmd.extend(["-movflags", "+faststart", str(output_path)])

    logger.info(
        "transcode_started",
        extra={
            "input": str(input_path),
            "output": str(output_path),
            "decision": decision.reason,
            "probe": {
                "width": probe.width,
                "height": probe.height,
                "codec": probe.codec_name,
                "bitrate_kbps": probe.bitrate_kbps,
            },
        },
    )

    subprocess.run(cmd, check=True, capture_output=True, text=True)

    output_size = output_path.stat().st_size
    input_size = input_path.stat().st_size
    logger.info(
        "transcode_complete",
        extra={
            "input_size": input_size,
            "output_size": output_size,
            "size_ratio": round(output_size / input_size, 3) if input_size > 0 else 0,
        },
    )

    return output_path
