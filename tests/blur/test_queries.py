"""Query map integrity + label → category reverse lookup."""

from __future__ import annotations

from heimdex_media_pipelines.blur import DEFAULT_OWL_QUERIES, label_to_category
from heimdex_media_pipelines.blur.queries import DIRECT_BLUR_CATEGORIES


def test_default_categories_present():
    assert "license_plate" in DEFAULT_OWL_QUERIES
    assert "card_object" in DEFAULT_OWL_QUERIES
    assert "logo" in DEFAULT_OWL_QUERIES


def test_no_duplicate_queries_within_category():
    for cat, qs in DEFAULT_OWL_QUERIES.items():
        assert len(qs) == len(set(qs)), f"duplicates in {cat}"


def test_direct_blur_categories_match_defaults():
    assert DIRECT_BLUR_CATEGORIES == frozenset({"license_plate", "logo", "card_object"})


def test_label_to_category_roundtrip():
    for cat, qs in DEFAULT_OWL_QUERIES.items():
        for q in qs:
            assert label_to_category(q) == cat


def test_label_to_category_case_insensitive():
    assert label_to_category("CREDIT CARD") == "card_object"
    assert label_to_category(" passport ") == "card_object"


def test_label_to_category_fallback():
    assert label_to_category("something nonsense") == "object"
