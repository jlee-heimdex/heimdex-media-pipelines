"""Signal extraction from speech data for multi-signal scene splitting.

Extracts :class:`SplitSignal` objects from STT output (speech segments
with timing, text, and optional speaker_id).  These are pure transforms
on in-memory data — no I/O, no subprocess calls.
"""

from __future__ import annotations

from typing import Any, Sequence

from heimdex_media_contracts.scenes.splitting import SplitSignal


def extract_speech_pauses(
    speech_segments: Sequence[dict[str, Any]],
    min_gap_ms: int = 300,
) -> list[SplitSignal]:
    """Extract speech pause signals from STT segments.

    A pause is a gap between consecutive speech segments where no one is
    talking.  The split point is placed at the midpoint of the gap.
    Strength is normalised by gap duration: longer pauses get higher
    strength (capped at 1.0 for gaps >= 2000ms).

    Args:
        speech_segments: Dicts with ``"start"`` and ``"end"`` in seconds.
        min_gap_ms: Minimum silence gap (ms) to qualify as a candidate.

    Returns:
        List of :class:`SplitSignal` with ``source="speech_pause"``.
    """
    if len(speech_segments) < 2:
        return []

    sorted_segs = sorted(speech_segments, key=lambda s: s.get("start", 0.0))
    signals: list[SplitSignal] = []

    for i in range(len(sorted_segs) - 1):
        end_ms = int(sorted_segs[i].get("end", 0.0) * 1000)
        next_start_ms = int(sorted_segs[i + 1].get("start", 0.0) * 1000)
        gap_ms = next_start_ms - end_ms

        if gap_ms >= min_gap_ms:
            midpoint = end_ms + gap_ms // 2
            # Normalise: 300ms gap -> ~0.15, 1000ms -> 0.5, 2000ms+ -> 1.0
            strength = min(1.0, gap_ms / 2000.0)
            signals.append(SplitSignal(
                timestamp_ms=midpoint,
                source="speech_pause",
                strength=strength,
            ))

    return signals


def extract_speaker_turns(
    speech_segments: Sequence[dict[str, Any]],
) -> list[SplitSignal]:
    """Extract speaker turn change signals from diarised STT segments.

    A turn change occurs when ``speaker_id`` changes between consecutive
    segments.  The split point is placed at the midpoint between the
    previous segment's end and the next segment's start.

    Strength is 1.0 for clean turns (gap between speakers) and 0.7
    for overlapping turns (next speaker starts before previous ends).

    Segments without ``speaker_id`` (None) are skipped.

    Args:
        speech_segments: Dicts with ``"start"``, ``"end"`` (seconds),
            and optional ``"speaker_id"`` (str or None).

    Returns:
        List of :class:`SplitSignal` with ``source="speaker_turn"``.
    """
    # Filter to segments that have speaker_id
    with_speaker = [
        s for s in speech_segments
        if s.get("speaker_id") is not None
    ]
    if len(with_speaker) < 2:
        return []

    sorted_segs = sorted(with_speaker, key=lambda s: s.get("start", 0.0))
    signals: list[SplitSignal] = []

    for i in range(len(sorted_segs) - 1):
        current = sorted_segs[i]
        next_seg = sorted_segs[i + 1]

        if current["speaker_id"] == next_seg["speaker_id"]:
            continue

        end_ms = int(current.get("end", 0.0) * 1000)
        next_start_ms = int(next_seg.get("start", 0.0) * 1000)
        gap_ms = next_start_ms - end_ms

        if gap_ms >= 0:
            # Clean turn — split at midpoint of gap
            midpoint = end_ms + gap_ms // 2
            strength = 1.0
        else:
            # Overlapping turn — split at next speaker's start
            midpoint = next_start_ms
            strength = 0.7

        signals.append(SplitSignal(
            timestamp_ms=midpoint,
            source="speaker_turn",
            strength=strength,
        ))

    return signals
