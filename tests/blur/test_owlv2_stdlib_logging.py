"""Regression tests for stdlib-compatible logging in the blur library.

The library uses ``logging.getLogger(__name__)`` (stdlib), not structlog.
Passing arbitrary kwargs like ``logger.info("msg", foo=1)`` raises
``TypeError: Logger._log() got an unexpected keyword argument`` at
runtime — but only if the code path is actually executed. Every prior
smoke test stubbed out OWLv2Detector via the ``Detector`` protocol and
never ran the ``__init__`` logger call; the bug only surfaced on
Aircloud at cold start.

These tests exercise the log-emitting paths directly with a fake
``transformers`` + ``torch`` import so CI catches regressions without
needing GPU or HuggingFace download.
"""

from __future__ import annotations

import logging
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _install_fake_transformers(monkeypatch):
    """Stub transformers.Owlv2Processor / Owlv2ForObjectDetection so
    the OWLv2Detector constructor runs without downloading weights."""
    processor_cls = MagicMock(name="Owlv2Processor")
    processor_cls.from_pretrained = MagicMock(return_value=MagicMock(name="proc"))

    model_instance = MagicMock(name="Owlv2Model")
    model_instance.to = MagicMock(return_value=model_instance)
    model_instance.eval = MagicMock(return_value=None)
    model_cls = MagicMock(name="Owlv2ForObjectDetection")
    model_cls.from_pretrained = MagicMock(return_value=model_instance)

    fake_transformers = SimpleNamespace(
        Owlv2Processor=processor_cls,
        Owlv2ForObjectDetection=model_cls,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    # torch is also imported inside OWLv2Detector.__init__
    fake_torch_cuda = SimpleNamespace(is_available=lambda: False)
    fake_torch = SimpleNamespace(cuda=fake_torch_cuda, no_grad=MagicMock(), tensor=MagicMock())
    monkeypatch.setitem(sys.modules, "torch", fake_torch)


def test_owlv2_constructor_emits_stdlib_compatible_logs(monkeypatch, caplog):
    """OWLv2Detector.__init__ must log through stdlib logger without
    raising. Regression against the structlog-style kwargs bug that
    crashed the worker on Aircloud cold start."""
    _install_fake_transformers(monkeypatch)

    caplog.set_level(logging.INFO, logger="heimdex_media_pipelines.blur.owlv2")

    from heimdex_media_pipelines.blur.owlv2 import OWLv2Detector

    # The constructor used to raise TypeError because it called
    # logger.info("owlv2_loading", model=..., device=...) — kwargs that
    # stdlib Logger rejects.
    det = OWLv2Detector(model_id="fake/model", device="cpu", score_threshold=0.3)
    assert det.device == "cpu"
    assert det.model_id == "fake/model"

    # Both log lines must land — proves the constructor reached
    # ``_model.eval()`` without TypeError.
    messages = [r.getMessage() for r in caplog.records]
    assert any("owlv2 loading" in m for m in messages), messages
    assert any("owlv2 loaded" in m for m in messages), messages


def test_pipeline_logger_calls_are_stdlib_safe():
    """Scan blur/pipeline.py and blur/owlv2.py for the structlog-style
    kwarg pattern. Any match is an automatic regression.

    This is a fast static check — no imports needed — that catches the
    bug earlier than the constructor test above.
    """
    from pathlib import Path

    import heimdex_media_pipelines.blur.pipeline as pipeline_mod
    import heimdex_media_pipelines.blur.owlv2 as owlv2_mod

    bad_patterns = [
        "logger.info(",
        "logger.warning(",
        "logger.error(",
        "logger.debug(",
        "logger.exception(",
    ]

    for mod in (pipeline_mod, owlv2_mod):
        src = Path(mod.__file__).read_text()
        for line_no, line in enumerate(src.splitlines(), 1):
            stripped = line.strip()
            if not any(p in stripped for p in bad_patterns):
                continue
            # A logger call. The first arg must be a string literal; no
            # bare keyword args of the form ``field=value`` after it
            # except for ``exc_info``, ``extra``, ``stack_info``.
            # Cheap heuristic: if the call line (or the multi-line call)
            # contains a bare ``<identifier>=`` that isn't one of the
            # allowed ones, flag it.
            # Accumulate parens-balanced call text first.
            buf = stripped
            depth = buf.count("(") - buf.count(")")
            i = line_no
            while depth > 0 and i < len(src.splitlines()):
                i += 1
                nxt = src.splitlines()[i - 1].strip()
                buf += " " + nxt
                depth += nxt.count("(") - nxt.count(")")

            allowed = ("exc_info=", "extra=", "stack_info=", "stacklevel=")
            import re
            # Remove string literals so ``foo="bar=baz"`` doesn't trip.
            clean = re.sub(r'"[^"]*"', '""', buf)
            clean = re.sub(r"'[^']*'", "''", clean)
            for match in re.finditer(r"\b([a-zA-Z_]\w*)\s*=", clean):
                kwarg = match.group(0)
                if any(a in kwarg for a in allowed):
                    continue
                # Any other bare kwarg inside a logger call is the bug.
                pytest.fail(
                    f"{mod.__name__}:{line_no}: stdlib logger called with "
                    f"unsupported kwarg `{kwarg}` — use %-formatting instead. "
                    f"Full call: {buf[:200]}"
                )
