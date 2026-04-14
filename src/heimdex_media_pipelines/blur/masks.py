"""Per-category blur mask emission.

The main blur pipeline bakes a full blurred MP4 for runtime playback.
For NLE-compatible export we *also* emit one grayscale **mask video**
per category — 0 where the frame is untouched, 255 where that category
was detected. At export time the worker composites the union of
selected masks against the original source to produce a ProRes 4444
``.mov`` with alpha only on the blur regions. Customers drop the
``.mov`` on top of the original in Premiere / DaVinci / FCP.

Design constraints:

* **FFV1 in MKV container.** Lossless grayscale keeps per-pixel bbox
  precision across the export round-trip. Every frame is a keyframe
  (``-g 1``) so the export worker can seek to any frame without a
  surrounding GOP. FFV1 support ships with stock FFmpeg (>=2.x) — no
  extra deps.
* **Memory-flat.** One pre-allocated ``uint8`` buffer per category
  (H * W bytes). Reset + paint + write per frame. At 1080p with 5
  categories that's ~10 MB resident, regardless of video length.
* **Subprocess, not cv2.** ``cv2.VideoWriter`` only speaks lossy
  RGB codecs; FFV1 + grayscale + single-channel is a non-starter.
  We open one ``ffmpeg`` subprocess per category with ``stdin=PIPE``
  and feed it raw bytes. No new Python dependency.
* **Lazily opened.** If ``cfg.emit_masks`` is False the pipeline never
  touches this module — callers that don't need masks pay zero cost.

The pipeline library stays dependency-free: the only things we import
are ``numpy`` (already a transitive of opencv), ``subprocess`` (stdlib),
and ``shutil`` (stdlib). No pydantic, no contracts — the worker does
boundary validation after collecting ``BlurResult.mask_paths``.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# ---------- progress events (caller-supplied callback) ----------


@dataclass
class BlurProgressEvent:
    """One heartbeat emitted by the pipeline while processing a video.

    The worker translates each event into a ``BlurJobProgress`` pydantic
    model (from ``heimdex_media_contracts.blur``) and POSTs it to the
    API's ``/internal/blur/{job_id}/progress`` endpoint. The pipeline
    library itself has no HTTP client — it just calls the callback.

    ``phase`` values match the ``BlurJobPhase`` literal on the contracts
    side: one of ``queued``, ``initializing``, ``detecting``,
    ``encoding``, ``uploading``, ``finalizing``. The pipeline only
    emits the first four; ``uploading`` and ``finalizing`` are the
    worker's responsibility after ``process_video`` returns.
    """

    phase: str
    progress_pct: float
    message: str | None = None


ProgressCallback = Callable[[BlurProgressEvent], None]


class ProgressThrottler:
    """Debounces progress events to at most one per ``min_interval_s``
    AND one per ``min_pct_delta`` change, whichever fires less often.

    Phase transitions always flush immediately — that's how the frontend
    moves between progress-bar stages without waiting for the next
    frame-loop tick.
    """

    def __init__(
        self,
        callback: ProgressCallback | None,
        *,
        min_interval_s: float = 0.5,
        min_pct_delta: float = 1.0,
    ) -> None:
        self._callback = callback
        self._min_interval_s = min_interval_s
        self._min_pct_delta = min_pct_delta
        self._last_time = 0.0
        self._last_pct = -1.0  # forces first emit
        self._last_phase: str | None = None

    def emit(
        self,
        phase: str,
        progress_pct: float,
        message: str | None = None,
    ) -> None:
        if self._callback is None:
            return

        now = time.time()
        phase_changed = phase != self._last_phase
        enough_pct = progress_pct - self._last_pct >= self._min_pct_delta
        enough_time = now - self._last_time >= self._min_interval_s

        if not phase_changed and not (enough_pct and enough_time):
            return

        try:
            self._callback(BlurProgressEvent(
                phase=phase,
                progress_pct=round(progress_pct, 2),
                message=message,
            ))
        except Exception as exc:
            # A broken callback must NOT break the pipeline — log and
            # suppress. The worker wraps its HTTP client in try/except
            # anyway; this is belt-and-suspenders.
            logger.warning(
                "progress_callback_failed phase=%s pct=%.1f error=%s",
                phase, progress_pct, exc,
            )

        self._last_time = now
        self._last_pct = progress_pct
        self._last_phase = phase


# ---------- per-category mask writer ----------


_MASK_CONTAINER_EXT = ".mkv"  # FFV1 lives in Matroska, never .mp4


def _resolve_ffmpeg_binary() -> str:
    """Return the absolute path to the ffmpeg binary, or raise.

    The pipeline is allowed to depend on ffmpeg being present (it's
    already required for other transcoding paths — see the parent
    CLAUDE.md note about "FFmpeg is required for video probing /
    transcoding"). We resolve it once so every mask writer in a single
    aggregator uses the exact same binary.
    """
    binary = shutil.which("ffmpeg")
    if not binary:
        raise RuntimeError(
            "ffmpeg binary not found on PATH; install ffmpeg to enable "
            "per-category blur mask emission"
        )
    return binary


class CategoryMaskWriter:
    """Wraps a single ``ffmpeg`` subprocess that encodes one category's
    grayscale mask stream into an FFV1-in-MKV file.

    The caller is responsible for writing exactly ``frame_count`` raw
    uint8 frames, one per source frame. Shorter streams produce a
    truncated mask video that won't line up with the source on export
    — verified by ``test_mask_writer_writes_every_frame``.
    """

    def __init__(
        self,
        *,
        category: str,
        output_path: Path,
        width: int,
        height: int,
        fps: float,
        ffmpeg_binary: str,
    ) -> None:
        if not category:
            raise ValueError("category must be a non-empty string")
        if width <= 0 or height <= 0:
            raise ValueError(f"invalid mask dimensions {width}x{height}")
        if fps <= 0:
            raise ValueError(f"invalid mask fps {fps}")

        self.category = category
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self._ffmpeg_binary = ffmpeg_binary
        self._process: subprocess.Popen[bytes] | None = None
        self._frames_written = 0

    def _build_command(self) -> list[str]:
        return [
            self._ffmpeg_binary,
            "-y",
            "-hide_banner",
            "-loglevel", "error",
            "-f", "rawvideo",
            "-pix_fmt", "gray",
            "-s", f"{self.width}x{self.height}",
            "-r", f"{self.fps:.6f}",
            "-i", "-",                          # stdin
            "-c:v", "ffv1",
            "-level", "3",
            "-g", "1",                          # every frame = keyframe
            "-pix_fmt", "gray",
            "-an",                              # no audio track
            str(self.output_path),
        ]

    def open(self) -> None:
        if self._process is not None:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._process = subprocess.Popen(  # noqa: S603 — args are static
            self._build_command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write_frame(self, mask_frame: np.ndarray) -> None:
        if self._process is None or self._process.stdin is None:
            raise RuntimeError(
                f"mask writer for '{self.category}' is not open"
            )
        if mask_frame.dtype != np.uint8:
            raise ValueError(
                f"mask frame must be uint8, got {mask_frame.dtype}"
            )
        if mask_frame.shape != (self.height, self.width):
            raise ValueError(
                f"mask frame shape {mask_frame.shape} does not match "
                f"({self.height}, {self.width})"
            )
        try:
            self._process.stdin.write(mask_frame.tobytes())
        except BrokenPipeError as exc:
            # ffmpeg died mid-stream. Collect its stderr so the caller
            # gets a useful error message instead of "Broken pipe".
            stderr = b""
            if self._process.stderr is not None:
                try:
                    stderr = self._process.stderr.read()
                except Exception:
                    stderr = b""
            raise RuntimeError(
                f"ffmpeg exited while encoding '{self.category}' mask: "
                f"{stderr.decode('utf-8', errors='replace').strip()}"
            ) from exc
        self._frames_written += 1

    def close(self) -> Path:
        if self._process is None:
            return self.output_path

        proc = self._process
        self._process = None  # prevent double-close

        if proc.stdin is not None:
            try:
                proc.stdin.close()
            except BrokenPipeError:
                pass

        returncode = proc.wait(timeout=60)

        stderr = b""
        if proc.stderr is not None:
            try:
                stderr = proc.stderr.read() or b""
            finally:
                proc.stderr.close()

        if returncode != 0:
            raise RuntimeError(
                f"ffmpeg mask encode for '{self.category}' failed "
                f"(rc={returncode}): "
                f"{stderr.decode('utf-8', errors='replace').strip()}"
            )

        return self.output_path

    @property
    def frames_written(self) -> int:
        return self._frames_written


# ---------- aggregator: holds one writer per category ----------


class CategoryMaskAggregator:
    """Holds one :class:`CategoryMaskWriter` per category and exposes a
    single ``paint`` call that accepts a mapping of
    ``category -> list of pixel bboxes`` for the current frame.

    Empty categories still receive an all-black frame on every paint to
    keep the mask stream frame-locked to the source video. The aggregator
    is a context manager — exiting flushes and closes every underlying
    ffmpeg subprocess, and the final mask paths are exposed via
    :attr:`mask_paths`.

    Use::

        with CategoryMaskAggregator(
            mask_dir=tmp,
            width=w, height=h, fps=fps,
            categories=("face", "license_plate"),
        ) as masks:
            for frame_idx in range(n_frames):
                regions = {"face": [(x1,y1,x2,y2)], "license_plate": []}
                masks.paint(regions)
        paths = masks.mask_paths  # {"face": Path, "license_plate": Path}
    """

    def __init__(
        self,
        *,
        mask_dir: Path,
        width: int,
        height: int,
        fps: float,
        categories: Sequence[str],
    ) -> None:
        if not categories:
            raise ValueError("CategoryMaskAggregator requires >=1 category")
        self._mask_dir = Path(mask_dir)
        self._width = int(width)
        self._height = int(height)
        self._fps = float(fps)
        self._categories: tuple[str, ...] = tuple(dict.fromkeys(categories))
        self._ffmpeg_binary: str | None = None
        self._writers: dict[str, CategoryMaskWriter] = {}
        self._buffers: dict[str, np.ndarray] = {}
        self._mask_paths: dict[str, Path] = {}
        self._closed = False

    def __enter__(self) -> "CategoryMaskAggregator":
        self._ffmpeg_binary = _resolve_ffmpeg_binary()
        self._mask_dir.mkdir(parents=True, exist_ok=True)
        for category in self._categories:
            output_path = self._mask_dir / f"{category}{_MASK_CONTAINER_EXT}"
            writer = CategoryMaskWriter(
                category=category,
                output_path=output_path,
                width=self._width,
                height=self._height,
                fps=self._fps,
                ffmpeg_binary=self._ffmpeg_binary,
            )
            writer.open()
            self._writers[category] = writer
            self._buffers[category] = np.zeros(
                (self._height, self._width), dtype=np.uint8,
            )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ----- per-frame painting -----

    def paint(
        self,
        regions_by_category: dict[str, list[tuple[int, int, int, int]]],
    ) -> None:
        """Write one frame to every open mask writer.

        ``regions_by_category`` maps a category name to a list of pixel
        bboxes ``(x1, y1, x2, y2)`` with top-left origin. Unknown
        categories are silently ignored — the aggregator only writes to
        the categories it was constructed with. Categories present in
        the aggregator but absent from the mapping receive an all-black
        frame (still written to keep the stream frame-locked).
        """
        if self._closed or not self._writers:
            raise RuntimeError("cannot paint on a closed aggregator")

        for category, writer in self._writers.items():
            buf = self._buffers[category]
            buf.fill(0)
            regions = regions_by_category.get(category)
            if regions:
                for (x1, y1, x2, y2) in regions:
                    self._paint_rect(buf, x1, y1, x2, y2)
            writer.write_frame(buf)

    def _paint_rect(
        self,
        buf: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> None:
        H, W = buf.shape
        ax1 = max(0, min(W, int(x1)))
        ay1 = max(0, min(H, int(y1)))
        ax2 = max(0, min(W, int(x2)))
        ay2 = max(0, min(H, int(y2)))
        if ax2 <= ax1 or ay2 <= ay1:
            return
        buf[ay1:ay2, ax1:ax2] = 255

    # ----- lifecycle -----

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors: list[str] = []
        for category, writer in self._writers.items():
            try:
                path = writer.close()
                self._mask_paths[category] = path
            except Exception as exc:
                errors.append(f"{category}: {exc}")
        if errors:
            raise RuntimeError(
                "one or more mask writers failed: " + "; ".join(errors)
            )

    @property
    def mask_paths(self) -> dict[str, Path]:
        return dict(self._mask_paths)

    @property
    def categories(self) -> tuple[str, ...]:
        return self._categories


__all__ = [
    "BlurProgressEvent",
    "CategoryMaskAggregator",
    "CategoryMaskWriter",
    "ProgressCallback",
    "ProgressThrottler",
]
