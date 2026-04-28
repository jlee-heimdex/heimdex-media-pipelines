"""Tests for composition rendering pipeline.

Unit tests mock subprocess to avoid requiring ffmpeg.
"""

from __future__ import annotations

import os
import subprocess
from unittest.mock import MagicMock, call, patch

import pytest

from heimdex_media_contracts.composition import (
    BackgroundOverlaySpec,
    CompositionSpec,
    OutputSpec,
    SceneClipSpec,
    SubtitleSpec,
    SubtitleStyleSpec,
    TextOverlaySpec,
    TransformSpec,
)
from heimdex_media_pipelines.composition.render import (
    RenderResult,
    extract_clip,
    render_composition,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def single_clip_spec() -> CompositionSpec:
    return CompositionSpec(
        output=OutputSpec(),
        scene_clips=[
            SceneClipSpec(
                scene_id="s001",
                video_id="vid_1",
                start_ms=0,
                end_ms=10000,
                timeline_start_ms=0,
            ),
        ],
    )


@pytest.fixture
def two_clip_spec() -> CompositionSpec:
    return CompositionSpec(
        output=OutputSpec(),
        scene_clips=[
            SceneClipSpec(
                scene_id="s001",
                video_id="vid_1",
                start_ms=0,
                end_ms=10000,
                timeline_start_ms=0,
            ),
            SceneClipSpec(
                scene_id="s002",
                video_id="vid_2",
                start_ms=5000,
                end_ms=15000,
                timeline_start_ms=10000,
            ),
        ],
    )


@pytest.fixture
def spec_with_subtitles() -> CompositionSpec:
    return CompositionSpec(
        output=OutputSpec(),
        scene_clips=[
            SceneClipSpec(
                scene_id="s001",
                video_id="vid_1",
                start_ms=0,
                end_ms=10000,
                timeline_start_ms=0,
            ),
        ],
        subtitles=[
            SubtitleSpec(
                text="Hello",
                start_ms=0,
                end_ms=5000,
                style=SubtitleStyleSpec(),
            ),
            SubtitleSpec(
                text="World",
                start_ms=5000,
                end_ms=10000,
                style=SubtitleStyleSpec(),
            ),
        ],
    )


# ---------------------------------------------------------------------------
# extract_clip tests
# ---------------------------------------------------------------------------

class TestExtractClip:
    @patch("heimdex_media_pipelines.composition.render.subprocess.run")
    def test_extract_clip_args(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=0)
        extract_clip("/input.mp4", "/output.mp4", 5000, 15000)

        args = mock_run.call_args[0][0]
        assert "ffmpeg" in args
        assert "-ss" in args
        assert "5.0" == args[args.index("-ss") + 1]
        assert "-t" in args
        assert "10.0" == args[args.index("-t") + 1]
        assert "-c:v" in args
        assert "libx264" == args[args.index("-c:v") + 1]
        assert "-c:a" in args
        assert "aac" == args[args.index("-c:a") + 1]
        assert "-i" in args
        assert "/input.mp4" == args[args.index("-i") + 1]

    @patch("heimdex_media_pipelines.composition.render.subprocess.run")
    def test_extract_clip_avoid_negative_ts(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=0)
        extract_clip("/input.mp4", "/output.mp4", 0, 10000)

        args = mock_run.call_args[0][0]
        assert "-avoid_negative_ts" in args
        assert "make_zero" == args[args.index("-avoid_negative_ts") + 1]

    @patch("heimdex_media_pipelines.composition.render.subprocess.run")
    def test_extract_clip_failure(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(
            returncode=1,
            stderr="Error: invalid input",
        )
        with pytest.raises(RuntimeError, match="extract_clip failed"):
            extract_clip("/input.mp4", "/output.mp4", 0, 10000)


# ---------------------------------------------------------------------------
# render_composition tests
# ---------------------------------------------------------------------------

class TestRenderComposition:
    @patch("heimdex_media_pipelines.composition.render.os.path.getsize", return_value=1024000)
    @patch("heimdex_media_pipelines.composition.render.subprocess.run")
    @patch("heimdex_media_pipelines.composition.render.extract_clip")
    def test_calls_extract_clip_per_clip(
        self,
        mock_extract: MagicMock,
        mock_run: MagicMock,
        mock_getsize: MagicMock,
        two_clip_spec: CompositionSpec,
    ) -> None:
        mock_run.return_value = MagicMock(returncode=0)

        render_composition(
            spec=two_clip_spec,
            media_paths={"vid_1": "/v1.mp4", "vid_2": "/v2.mp4"},
            output_path="/out.mp4",
            font_dir="/fonts",
        )

        assert mock_extract.call_count == 2
        # First clip: start=0, end=10000
        first_call = mock_extract.call_args_list[0]
        assert first_call[0][0] == "/v1.mp4"
        assert first_call[0][2] == 0
        assert first_call[0][3] == 10000
        # Second clip: start=5000, end=15000
        second_call = mock_extract.call_args_list[1]
        assert second_call[0][0] == "/v2.mp4"
        assert second_call[0][2] == 5000
        assert second_call[0][3] == 15000

    @patch("heimdex_media_pipelines.composition.render.os.path.getsize", return_value=1024000)
    @patch("heimdex_media_pipelines.composition.render.subprocess.run")
    @patch("heimdex_media_pipelines.composition.render.extract_clip")
    @patch("heimdex_media_pipelines.composition.render.build_filter_graph")
    def test_calls_build_filter_graph(
        self,
        mock_build_fg: MagicMock,
        mock_extract: MagicMock,
        mock_run: MagicMock,
        mock_getsize: MagicMock,
        single_clip_spec: CompositionSpec,
    ) -> None:
        mock_build_fg.return_value = "[base]overlay[final]"
        mock_run.return_value = MagicMock(returncode=0)

        render_composition(
            spec=single_clip_spec,
            media_paths={"vid_1": "/v1.mp4"},
            output_path="/out.mp4",
            font_dir="/fonts",
        )

        mock_build_fg.assert_called_once()
        call_kwargs = mock_build_fg.call_args
        assert call_kwargs[1]["font_dir"] == "/fonts"

    @patch("heimdex_media_pipelines.composition.render.os.path.getsize", return_value=1024000)
    @patch("heimdex_media_pipelines.composition.render.subprocess.run")
    @patch("heimdex_media_pipelines.composition.render.extract_clip")
    def test_cpu_encoding_args(
        self,
        mock_extract: MagicMock,
        mock_run: MagicMock,
        mock_getsize: MagicMock,
        single_clip_spec: CompositionSpec,
    ) -> None:
        mock_run.return_value = MagicMock(returncode=0)

        render_composition(
            spec=single_clip_spec,
            media_paths={"vid_1": "/v1.mp4"},
            output_path="/out.mp4",
            font_dir="/fonts",
            use_gpu=False,
        )

        args = mock_run.call_args[0][0]
        # CPU encoding
        idx = args.index("-c:v")
        assert args[idx + 1] == "libx264"
        assert "-preset" in args
        assert args[args.index("-preset") + 1] == "medium"
        assert "-crf" in args
        assert args[args.index("-crf") + 1] == "23"

    @patch("heimdex_media_pipelines.composition.render.os.path.getsize", return_value=1024000)
    @patch("heimdex_media_pipelines.composition.render.subprocess.run")
    @patch("heimdex_media_pipelines.composition.render.extract_clip")
    def test_gpu_encoding_args(
        self,
        mock_extract: MagicMock,
        mock_run: MagicMock,
        mock_getsize: MagicMock,
        single_clip_spec: CompositionSpec,
    ) -> None:
        mock_run.return_value = MagicMock(returncode=0)

        render_composition(
            spec=single_clip_spec,
            media_paths={"vid_1": "/v1.mp4"},
            output_path="/out.mp4",
            font_dir="/fonts",
            use_gpu=True,
        )

        args = mock_run.call_args[0][0]
        idx = args.index("-c:v")
        assert args[idx + 1] == "h264_nvenc"
        assert "-preset" in args
        assert args[args.index("-preset") + 1] == "p4"
        assert "-cq" in args
        assert args[args.index("-cq") + 1] == "23"

    @patch("heimdex_media_pipelines.composition.render.os.path.getsize", return_value=1024000)
    @patch("heimdex_media_pipelines.composition.render.subprocess.run")
    @patch("heimdex_media_pipelines.composition.render.extract_clip")
    def test_common_audio_args(
        self,
        mock_extract: MagicMock,
        mock_run: MagicMock,
        mock_getsize: MagicMock,
        single_clip_spec: CompositionSpec,
    ) -> None:
        mock_run.return_value = MagicMock(returncode=0)

        render_composition(
            spec=single_clip_spec,
            media_paths={"vid_1": "/v1.mp4"},
            output_path="/out.mp4",
            font_dir="/fonts",
        )

        args = mock_run.call_args[0][0]
        assert "-c:a" in args
        assert args[args.index("-c:a") + 1] == "aac"
        assert "-b:a" in args
        assert args[args.index("-b:a") + 1] == "128k"
        assert "-movflags" in args
        assert args[args.index("-movflags") + 1] == "+faststart"

    def test_render_result_fields(self) -> None:
        r = RenderResult(
            output_path="/out.mp4",
            duration_ms=10000,
            size_bytes=1024,
            render_time_ms=500,
        )
        assert r.output_path == "/out.mp4"
        assert r.duration_ms == 10000
        assert r.size_bytes == 1024
        assert r.render_time_ms == 500

    @patch("heimdex_media_pipelines.composition.render.os.path.getsize", return_value=1024000)
    @patch("heimdex_media_pipelines.composition.render.subprocess.run")
    @patch("heimdex_media_pipelines.composition.render.extract_clip")
    def test_ffmpeg_failure_raises(
        self,
        mock_extract: MagicMock,
        mock_run: MagicMock,
        mock_getsize: MagicMock,
        single_clip_spec: CompositionSpec,
    ) -> None:
        mock_run.return_value = MagicMock(
            returncode=1,
            stderr="ffmpeg error: codec not found",
        )

        with pytest.raises(RuntimeError, match="ffmpeg render failed"):
            render_composition(
                spec=single_clip_spec,
                media_paths={"vid_1": "/v1.mp4"},
                output_path="/out.mp4",
                font_dir="/fonts",
            )

    @patch("heimdex_media_pipelines.composition.render.os.path.getsize", return_value=2048)
    @patch("heimdex_media_pipelines.composition.render.subprocess.run")
    @patch("heimdex_media_pipelines.composition.render.extract_clip")
    def test_render_time_ms_populated(
        self,
        mock_extract: MagicMock,
        mock_run: MagicMock,
        mock_getsize: MagicMock,
        single_clip_spec: CompositionSpec,
    ) -> None:
        mock_run.return_value = MagicMock(returncode=0)

        result = render_composition(
            spec=single_clip_spec,
            media_paths={"vid_1": "/v1.mp4"},
            output_path="/out.mp4",
            font_dir="/fonts",
        )

        assert result.render_time_ms >= 0
        assert isinstance(result.render_time_ms, int)

    @patch("heimdex_media_pipelines.composition.render.os.path.getsize", return_value=5000)
    @patch("heimdex_media_pipelines.composition.render.subprocess.run")
    @patch("heimdex_media_pipelines.composition.render.extract_clip")
    def test_video_label_with_subtitles(
        self,
        mock_extract: MagicMock,
        mock_run: MagicMock,
        mock_getsize: MagicMock,
        spec_with_subtitles: CompositionSpec,
    ) -> None:
        """When subtitles are present, video output label should be 'final'."""
        mock_run.return_value = MagicMock(returncode=0)

        render_composition(
            spec=spec_with_subtitles,
            media_paths={"vid_1": "/v1.mp4"},
            output_path="/out.mp4",
            font_dir="/fonts",
        )

        args = mock_run.call_args[0][0]
        map_idx = args.index("-map")
        assert args[map_idx + 1] == "[final]"

    @patch("heimdex_media_pipelines.composition.render.os.path.getsize", return_value=5000)
    @patch("heimdex_media_pipelines.composition.render.subprocess.run")
    @patch("heimdex_media_pipelines.composition.render.extract_clip")
    def test_video_label_without_subtitles(
        self,
        mock_extract: MagicMock,
        mock_run: MagicMock,
        mock_getsize: MagicMock,
        single_clip_spec: CompositionSpec,
    ) -> None:
        """Without subtitles, video output label should be 'canvas{N}'."""
        mock_run.return_value = MagicMock(returncode=0)

        render_composition(
            spec=single_clip_spec,
            media_paths={"vid_1": "/v1.mp4"},
            output_path="/out.mp4",
            font_dir="/fonts",
        )

        args = mock_run.call_args[0][0]
        map_idx = args.index("-map")
        assert args[map_idx + 1] == "[canvas1]"


# ---------------------------------------------------------------------------
# V2 overlay tests
# ---------------------------------------------------------------------------

class TestRenderCompositionOverlays:
    """V2 overlay rendering — bake to PNG, overlay= filter chain.

    All tests mock both subprocess and bake_overlay_png so they don't need
    real fonts or PIL. The goal is to verify the ffmpeg command shape:
    - Overlay PNG inputs are appended (with -loop 1)
    - Filter graph includes the overlay= chain
    - Output -map routes through the overlay chain's final label
    """

    def _spec_with_overlay(
        self, *, kind: str = "text", subtitles: bool = False
    ) -> CompositionSpec:
        clip = SceneClipSpec(
            scene_id="s001",
            video_id="vid_1",
            start_ms=0,
            end_ms=10000,
            timeline_start_ms=0,
        )
        if kind == "text":
            ov = TextOverlaySpec(
                id="t1",
                start_ms=0,
                end_ms=5000,
                text="라이브",
                italic=True,
            )
        else:
            ov = BackgroundOverlaySpec(
                id="b1",
                start_ms=0,
                end_ms=3000,
                fill_color="#1A1A1A",
                transform=TransformSpec(width_px=200, height_px=80),
            )
        sub_kwargs = {}
        if subtitles:
            sub_kwargs = {
                "subtitles": [
                    SubtitleSpec(
                        text="legacy",
                        start_ms=0,
                        end_ms=2000,
                        style=SubtitleStyleSpec(),
                    )
                ]
            }
        return CompositionSpec(
            output=OutputSpec(),
            scene_clips=[clip],
            overlays=[ov],
            **sub_kwargs,
        )

    @patch("heimdex_media_pipelines.composition.render.os.path.getsize", return_value=1024)
    @patch("heimdex_media_pipelines.composition.render.subprocess.run")
    @patch("heimdex_media_pipelines.composition.render.extract_clip")
    @patch(
        "heimdex_media_pipelines.composition.overlay_render.bake_overlay_png"
    )
    def test_text_overlay_adds_png_input_and_overlay_filter(
        self,
        mock_bake: MagicMock,
        mock_extract: MagicMock,
        mock_run: MagicMock,
        mock_getsize: MagicMock,
    ) -> None:
        # Mock the bake to return a fake image whose .save() is a no-op
        mock_img = MagicMock()
        mock_bake.return_value = mock_img
        mock_run.return_value = MagicMock(returncode=0)

        spec = self._spec_with_overlay(kind="text", subtitles=False)
        render_composition(
            spec=spec,
            media_paths={"vid_1": "/v1.mp4"},
            output_path="/out.mp4",
            font_dir="/fonts",
        )

        # Bake was called once per overlay
        assert mock_bake.call_count == 1
        # Image.save was called to materialize the PNG
        mock_img.save.assert_called_once()

        args = mock_run.call_args[0][0]
        # Two -i flags: 1 clip + 1 overlay PNG
        assert args.count("-i") == 2
        # -loop 1 precedes the overlay PNG input
        assert "-loop" in args
        loop_idx = args.index("-loop")
        assert args[loop_idx + 1] == "1"

        # Filter graph includes the overlay= chain mapping into [vout]
        filter_idx = args.index("-filter_complex")
        graph = args[filter_idx + 1]
        assert "overlay=" in graph
        assert "[vout]" in graph

        # Final video map routes through [vout]
        map_idx = args.index("-map")
        assert args[map_idx + 1] == "[vout]"

    @patch("heimdex_media_pipelines.composition.render.os.path.getsize", return_value=1024)
    @patch("heimdex_media_pipelines.composition.render.subprocess.run")
    @patch("heimdex_media_pipelines.composition.render.extract_clip")
    @patch(
        "heimdex_media_pipelines.composition.overlay_render.bake_overlay_png"
    )
    def test_background_overlay_works(
        self,
        mock_bake: MagicMock,
        mock_extract: MagicMock,
        mock_run: MagicMock,
        mock_getsize: MagicMock,
    ) -> None:
        mock_bake.return_value = MagicMock()
        mock_run.return_value = MagicMock(returncode=0)

        spec = self._spec_with_overlay(kind="background", subtitles=False)
        render_composition(
            spec=spec,
            media_paths={"vid_1": "/v1.mp4"},
            output_path="/out.mp4",
            font_dir="/fonts",
        )

        # Bake fired once for the background overlay
        assert mock_bake.call_count == 1
        # Bake was called with the contracts BackgroundOverlaySpec
        ov_arg = mock_bake.call_args[0][0]
        assert isinstance(ov_arg, BackgroundOverlaySpec)

    @patch("heimdex_media_pipelines.composition.render.os.path.getsize", return_value=1024)
    @patch("heimdex_media_pipelines.composition.render.subprocess.run")
    @patch("heimdex_media_pipelines.composition.render.extract_clip")
    @patch(
        "heimdex_media_pipelines.composition.overlay_render.bake_overlay_png"
    )
    def test_overlay_chain_after_subtitles(
        self,
        mock_bake: MagicMock,
        mock_extract: MagicMock,
        mock_run: MagicMock,
        mock_getsize: MagicMock,
    ) -> None:
        # The legacy drawtext subtitles ALSO need /fonts because the
        # contracts strict resolver runs at filter-graph build time. Skip
        # asserting on the chain when /fonts isn't present — pre-existing
        # test isolation issue tracked separately.
        if not os.path.exists("/fonts/Pretendard-Regular.ttf"):
            pytest.skip("legacy /fonts stub not available")
        mock_bake.return_value = MagicMock()
        mock_run.return_value = MagicMock(returncode=0)

        spec = self._spec_with_overlay(kind="text", subtitles=True)
        render_composition(
            spec=spec,
            media_paths={"vid_1": "/v1.mp4"},
            output_path="/out.mp4",
            font_dir="/fonts",
        )

        args = mock_run.call_args[0][0]
        graph = args[args.index("-filter_complex") + 1]
        # Subtitle drawtext writes to [final], overlay chain reads [final]
        # and writes to [vout].
        assert "drawtext" in graph
        assert "[final]" in graph
        assert "[vout]" in graph
        map_idx = args.index("-map")
        assert args[map_idx + 1] == "[vout]"

    @patch("heimdex_media_pipelines.composition.render.os.path.getsize", return_value=1024)
    @patch("heimdex_media_pipelines.composition.render.subprocess.run")
    @patch("heimdex_media_pipelines.composition.render.extract_clip")
    def test_no_overlays_does_not_import_pil(
        self,
        mock_extract: MagicMock,
        mock_run: MagicMock,
        mock_getsize: MagicMock,
        single_clip_spec: CompositionSpec,
    ) -> None:
        """Overlays default to []; the PIL import must stay lazy so legacy
        callers without the [composition] extra still render successfully.
        Asserted indirectly: no bake_overlay_png attempted, just the
        normal ffmpeg call shape."""
        mock_run.return_value = MagicMock(returncode=0)

        render_composition(
            spec=single_clip_spec,
            media_paths={"vid_1": "/v1.mp4"},
            output_path="/out.mp4",
            font_dir="/fonts",
        )

        args = mock_run.call_args[0][0]
        # Single -i for the single clip; no overlay PNG inputs added
        assert args.count("-i") == 1
        assert "-loop" not in args
        # Final map routes through canvas1, not [vout]
        map_idx = args.index("-map")
        assert args[map_idx + 1] == "[canvas1]"
