"""Tests for speech signal extraction."""

import pytest

from heimdex_media_pipelines.scenes.signals import (
    extract_speech_pauses,
    extract_speaker_turns,
)


def _seg(start: float, end: float, speaker_id: str | None = None) -> dict:
    return {"start": start, "end": end, "text": "hello", "speaker_id": speaker_id}


# ---------------------------------------------------------------------------
# extract_speech_pauses
# ---------------------------------------------------------------------------

class TestExtractSpeechPauses:
    def test_basic_gap(self):
        segs = [_seg(0.0, 5.0), _seg(6.0, 10.0)]  # 1s gap
        result = extract_speech_pauses(segs, min_gap_ms=300)
        assert len(result) == 1
        assert result[0].source == "speech_pause"
        assert result[0].timestamp_ms == 5500  # midpoint of 5000-6000

    def test_gap_below_threshold_ignored(self):
        segs = [_seg(0.0, 5.0), _seg(5.1, 10.0)]  # 100ms gap
        result = extract_speech_pauses(segs, min_gap_ms=300)
        assert len(result) == 0

    def test_exact_threshold(self):
        segs = [_seg(0.0, 5.0), _seg(5.3, 10.0)]  # 300ms gap
        result = extract_speech_pauses(segs, min_gap_ms=300)
        assert len(result) == 1

    def test_strength_normalisation(self):
        segs = [_seg(0.0, 5.0), _seg(7.0, 10.0)]  # 2s gap -> strength 1.0
        result = extract_speech_pauses(segs, min_gap_ms=300)
        assert result[0].strength == 1.0

    def test_short_gap_lower_strength(self):
        segs = [_seg(0.0, 5.0), _seg(5.5, 10.0)]  # 500ms gap
        result = extract_speech_pauses(segs, min_gap_ms=300)
        assert 0.0 < result[0].strength < 0.5

    def test_multiple_gaps(self):
        segs = [_seg(0.0, 5.0), _seg(6.0, 10.0), _seg(12.0, 15.0)]
        result = extract_speech_pauses(segs, min_gap_ms=300)
        assert len(result) == 2

    def test_unsorted_input(self):
        segs = [_seg(6.0, 10.0), _seg(0.0, 5.0)]  # reversed
        result = extract_speech_pauses(segs, min_gap_ms=300)
        assert len(result) == 1

    def test_single_segment(self):
        result = extract_speech_pauses([_seg(0.0, 5.0)], min_gap_ms=300)
        assert len(result) == 0

    def test_empty_input(self):
        result = extract_speech_pauses([], min_gap_ms=300)
        assert len(result) == 0

    def test_overlapping_segments_no_gap(self):
        segs = [_seg(0.0, 5.0), _seg(4.5, 10.0)]  # overlap
        result = extract_speech_pauses(segs, min_gap_ms=300)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# extract_speaker_turns
# ---------------------------------------------------------------------------

class TestExtractSpeakerTurns:
    def test_basic_turn(self):
        segs = [
            _seg(0.0, 5.0, "SPEAKER_00"),
            _seg(5.5, 10.0, "SPEAKER_01"),
        ]
        result = extract_speaker_turns(segs)
        assert len(result) == 1
        assert result[0].source == "speaker_turn"
        assert result[0].strength == 1.0  # clean turn (gap)

    def test_overlapping_turn(self):
        segs = [
            _seg(0.0, 5.5, "SPEAKER_00"),
            _seg(5.0, 10.0, "SPEAKER_01"),  # starts before prev ends
        ]
        result = extract_speaker_turns(segs)
        assert len(result) == 1
        assert result[0].strength == 0.7  # overlapping

    def test_same_speaker_no_turn(self):
        segs = [
            _seg(0.0, 5.0, "SPEAKER_00"),
            _seg(6.0, 10.0, "SPEAKER_00"),
        ]
        result = extract_speaker_turns(segs)
        assert len(result) == 0

    def test_none_speaker_id_skipped(self):
        segs = [
            _seg(0.0, 5.0, None),
            _seg(6.0, 10.0, None),
        ]
        result = extract_speaker_turns(segs)
        assert len(result) == 0

    def test_mixed_none_and_real_speakers(self):
        segs = [
            _seg(0.0, 3.0, None),
            _seg(3.0, 5.0, "SPEAKER_00"),
            _seg(6.0, 10.0, "SPEAKER_01"),
        ]
        result = extract_speaker_turns(segs)
        # Only the turn between SPEAKER_00 and SPEAKER_01
        assert len(result) == 1

    def test_multiple_turns(self):
        segs = [
            _seg(0.0, 5.0, "SPEAKER_00"),
            _seg(6.0, 10.0, "SPEAKER_01"),
            _seg(11.0, 15.0, "SPEAKER_00"),
        ]
        result = extract_speaker_turns(segs)
        assert len(result) == 2

    def test_single_segment(self):
        result = extract_speaker_turns([_seg(0.0, 5.0, "SPEAKER_00")])
        assert len(result) == 0

    def test_empty_input(self):
        result = extract_speaker_turns([])
        assert len(result) == 0

    def test_turn_timestamp_at_midpoint(self):
        segs = [
            _seg(0.0, 5.0, "SPEAKER_00"),
            _seg(7.0, 10.0, "SPEAKER_01"),  # 2s gap
        ]
        result = extract_speaker_turns(segs)
        assert result[0].timestamp_ms == 6000  # midpoint of 5000-7000
