import pytest

from heimdex_media_pipelines.speech.diarization import (
    SpeakerTurn,
    assign_speakers_to_segments,
)
from heimdex_media_pipelines.speech.stt import TranscriptSegment


class TestSpeakerTurn:
    def test_speaker_turn_creation(self):
        turn = SpeakerTurn(start_s=0.0, end_s=1.5, speaker_id="SPEAKER_00")
        assert turn.start_s == 0.0
        assert turn.end_s == 1.5
        assert turn.speaker_id == "SPEAKER_00"

    def test_speaker_turn_with_different_speaker_id(self):
        turn = SpeakerTurn(start_s=1.5, end_s=3.0, speaker_id="SPEAKER_01")
        assert turn.speaker_id == "SPEAKER_01"


class TestAssignSpeakersToSegments:
    def test_empty_speaker_turns_returns_segments_unchanged(self):
        segments = [
            TranscriptSegment(start_s=0.0, end_s=1.0, text="hello"),
            TranscriptSegment(start_s=1.0, end_s=2.0, text="world"),
        ]
        result = assign_speakers_to_segments(segments, [])
        assert len(result) == 2
        assert result[0].speaker_id is None
        assert result[1].speaker_id is None
        assert result is segments

    def test_single_speaker_covering_all_segments(self):
        segments = [
            TranscriptSegment(start_s=0.0, end_s=1.0, text="hello"),
            TranscriptSegment(start_s=1.0, end_s=2.0, text="world"),
        ]
        speaker_turns = [SpeakerTurn(start_s=0.0, end_s=3.0, speaker_id="SPEAKER_00")]
        result = assign_speakers_to_segments(segments, speaker_turns)
        assert result[0].speaker_id == "SPEAKER_00"
        assert result[1].speaker_id == "SPEAKER_00"

    def test_two_non_overlapping_speakers(self):
        segments = [
            TranscriptSegment(start_s=0.0, end_s=1.0, text="hello"),
            TranscriptSegment(start_s=2.0, end_s=3.0, text="world"),
        ]
        speaker_turns = [
            SpeakerTurn(start_s=0.0, end_s=1.5, speaker_id="SPEAKER_00"),
            SpeakerTurn(start_s=1.5, end_s=3.5, speaker_id="SPEAKER_01"),
        ]
        result = assign_speakers_to_segments(segments, speaker_turns)
        assert result[0].speaker_id == "SPEAKER_00"
        assert result[1].speaker_id == "SPEAKER_01"

    def test_segment_spanning_two_speaker_turns_assigned_to_max_overlap(self):
        segments = [
            TranscriptSegment(start_s=0.5, end_s=2.5, text="spanning"),
        ]
        speaker_turns = [
            SpeakerTurn(start_s=0.0, end_s=1.0, speaker_id="SPEAKER_00"),
            SpeakerTurn(start_s=1.0, end_s=3.0, speaker_id="SPEAKER_01"),
        ]
        result = assign_speakers_to_segments(segments, speaker_turns)
        assert result[0].speaker_id == "SPEAKER_01"

    def test_segment_with_equal_overlap_to_two_speakers_picks_one(self):
        segments = [
            TranscriptSegment(start_s=1.0, end_s=2.0, text="equal"),
        ]
        speaker_turns = [
            SpeakerTurn(start_s=0.0, end_s=1.5, speaker_id="SPEAKER_00"),
            SpeakerTurn(start_s=1.5, end_s=3.0, speaker_id="SPEAKER_01"),
        ]
        result = assign_speakers_to_segments(segments, speaker_turns)
        assert result[0].speaker_id in ["SPEAKER_00", "SPEAKER_01"]

    def test_segment_with_no_overlap_keeps_none(self):
        segments = [
            TranscriptSegment(start_s=5.0, end_s=6.0, text="isolated"),
        ]
        speaker_turns = [
            SpeakerTurn(start_s=0.0, end_s=1.0, speaker_id="SPEAKER_00"),
        ]
        result = assign_speakers_to_segments(segments, speaker_turns)
        assert result[0].speaker_id is None

    def test_multiple_segments_with_multiple_speakers(self):
        segments = [
            TranscriptSegment(start_s=0.0, end_s=1.0, text="hello"),
            TranscriptSegment(start_s=1.2, end_s=2.0, text="hi"),
            TranscriptSegment(start_s=2.2, end_s=3.0, text="hey"),
            TranscriptSegment(start_s=3.6, end_s=4.0, text="bye"),
        ]
        speaker_turns = [
            SpeakerTurn(start_s=0.0, end_s=1.0, speaker_id="SPEAKER_00"),
            SpeakerTurn(start_s=1.0, end_s=3.5, speaker_id="SPEAKER_01"),
            SpeakerTurn(start_s=3.5, end_s=5.0, speaker_id="SPEAKER_00"),
        ]
        result = assign_speakers_to_segments(segments, speaker_turns)
        assert result[0].speaker_id == "SPEAKER_00"
        assert result[1].speaker_id == "SPEAKER_01"
        assert result[2].speaker_id == "SPEAKER_01"
        assert result[3].speaker_id == "SPEAKER_00"

    def test_mutates_segments_in_place(self):
        segments = [
            TranscriptSegment(start_s=0.0, end_s=1.0, text="hello"),
        ]
        speaker_turns = [SpeakerTurn(start_s=0.0, end_s=2.0, speaker_id="SPEAKER_00")]
        result = assign_speakers_to_segments(segments, speaker_turns)
        assert result is segments
        assert segments[0].speaker_id == "SPEAKER_00"

    def test_segment_at_exact_boundary_of_speaker_turn(self):
        segments = [
            TranscriptSegment(start_s=1.0, end_s=2.0, text="boundary"),
        ]
        speaker_turns = [
            SpeakerTurn(start_s=0.0, end_s=1.0, speaker_id="SPEAKER_00"),
            SpeakerTurn(start_s=1.0, end_s=3.0, speaker_id="SPEAKER_01"),
        ]
        result = assign_speakers_to_segments(segments, speaker_turns)
        assert result[0].speaker_id == "SPEAKER_01"

    def test_three_speakers_with_overlapping_turns(self):
        segments = [
            TranscriptSegment(start_s=0.5, end_s=1.5, text="seg1"),
            TranscriptSegment(start_s=1.5, end_s=2.5, text="seg2"),
        ]
        speaker_turns = [
            SpeakerTurn(start_s=0.0, end_s=1.0, speaker_id="SPEAKER_00"),
            SpeakerTurn(start_s=0.8, end_s=2.0, speaker_id="SPEAKER_01"),
            SpeakerTurn(start_s=1.8, end_s=3.0, speaker_id="SPEAKER_02"),
        ]
        result = assign_speakers_to_segments(segments, speaker_turns)
        assert result[0].speaker_id == "SPEAKER_01"
        assert result[1].speaker_id == "SPEAKER_02"
