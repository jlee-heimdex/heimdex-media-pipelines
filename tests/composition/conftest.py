"""Pytest plugin for composition tests.

Adds ``--regen-goldens`` flag for refreshing baked-overlay PNG fixtures.
Run with ``pytest tests/composition/ --regen-goldens`` after intentional
visual changes; the test then writes the new image to the goldens dir
and skips comparison for that run. Re-run without the flag to verify.
"""

from __future__ import annotations


def pytest_addoption(parser):
    parser.addoption(
        "--regen-goldens",
        action="store_true",
        default=False,
        help="Regenerate golden PNG fixtures for overlay bake tests.",
    )
