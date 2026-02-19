from heimdex_media_pipelines.transcoding.probe import ProbeResult
from heimdex_media_pipelines.transcoding.proxy import make_transcode_decision


def _probe(*, width=1920, height=1080, codec="h264", bitrate=3000, has_audio=True):
    return ProbeResult(
        width=width, height=height, codec_name=codec,
        bitrate_kbps=bitrate, duration_ms=60000, has_audio=has_audio,
    )


class TestMakeTranscodeDecision:
    def test_skip_when_h264_within_resolution_and_bitrate(self):
        decision = make_transcode_decision(_probe(height=720, bitrate=1500))
        assert not decision.should_transcode
        assert "Already H.264" in decision.reason

    def test_cap_bitrate_when_h264_within_resolution_but_over_bitrate(self):
        decision = make_transcode_decision(_probe(height=720, bitrate=4000))
        assert decision.should_transcode
        assert decision.should_cap_bitrate
        assert "exceeds" in decision.reason

    def test_transcode_when_wrong_codec(self):
        decision = make_transcode_decision(_probe(codec="hevc", height=720, bitrate=1500))
        assert decision.should_transcode
        assert not decision.should_cap_bitrate
        assert "codec=hevc" in decision.reason

    def test_transcode_when_over_resolution(self):
        decision = make_transcode_decision(_probe(height=1080, bitrate=1500))
        assert decision.should_transcode
        assert "height=1080" in decision.reason

    def test_transcode_when_over_resolution_and_bitrate(self):
        decision = make_transcode_decision(_probe(height=1080, bitrate=5000))
        assert decision.should_transcode
        assert "height=1080" in decision.reason
        assert "bitrate=5000" in decision.reason

    def test_custom_thresholds(self):
        decision = make_transcode_decision(
            _probe(height=480, bitrate=800),
            max_height=480,
            max_bitrate_kbps=1000,
        )
        assert not decision.should_transcode

    def test_h264_variants_accepted(self):
        for codec in ("h264", "h264_qsv", "h264_nvenc"):
            decision = make_transcode_decision(_probe(codec=codec, height=720, bitrate=1000))
            assert not decision.should_transcode
