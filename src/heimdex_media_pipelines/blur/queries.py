"""Default OWLv2 text queries, grouped by PII category.

Each category is a prompt bundle that the OWLv2 open-vocabulary detector
consumes. The category → query mapping is the reverse lookup used when
tagging detections in the manifest, so categories here must match the
literals declared in ``heimdex_media_contracts.blur``.

Logo queries ship in this file but ``BlurConfig.categories`` **excludes**
``logo`` by default — in livecommerce footage, blurring logos erases the
product being sold. Turning logo blur on is an explicit tenant opt-in.
"""

from __future__ import annotations

DEFAULT_OWL_QUERIES: dict[str, list[str]] = {
    "license_plate": [
        "car license plate",
        "vehicle number plate on bumper",
        "white rectangular plate with numbers on car",
        "korean license plate",
        "small license plate on vehicle",
        "white license plate with korean characters",
        "yellow license plate on vehicle",
        "green license plate on vehicle",
        "blue license plate on vehicle",
        "license plate on front bumper",
        "license plate on rear bumper",
    ],
    "logo": [
        "brand logo",
        "company logo",
        "brand logo on product",
        "brand logo on product surface",
        "logo mark on product",
        "brand symbol on product",
        "product branding",
        "brand marking on packaging",
        "logo printed on product packaging",
        "logo on paper packaging",
        "logo on metallic packaging",
        "brand name on foil pouch",
        "label with brand name on product",
        "logo on product label",
        "circular logo on product label",
        "sticker label on product",
        "colorful label patch on product",
        "brand name on metal container",
        "logo on tin can",
        "gold logo on dark container",
        "logo on glass container",
        "logo on plastic product",
        "brand name on device surface",
        "logo engraved on product",
    ],
    "card_object": [
        "credit card",
        "debit card",
        "bank card",
        "ID card",
        "resident registration card",
        "driver license",
        "passport",
        "health insurance card",
        "business card",
        "name card",
        "membership card",
    ],
}

# Categories whose detections are blurred directly (as opposed to
# multi-pass strategies like "detect card → detect PII region within
# card"). For v1 all are direct-blur.
DIRECT_BLUR_CATEGORIES: frozenset[str] = frozenset({
    "license_plate",
    "logo",
    "card_object",
})


def label_to_category(label: str, query_map: dict[str, list[str]] | None = None) -> str:
    """Reverse-lookup: OWLv2 query text → category key.

    Falls back to ``"OBJECT"`` when the label is not found — this
    happens when callers pass custom query lists at runtime.
    """
    query_map = query_map if query_map is not None else DEFAULT_OWL_QUERIES
    label_lower = label.strip().lower()
    for cat, queries in query_map.items():
        for q in queries:
            if q.lower() == label_lower:
                return cat
    return "object"


__all__ = [
    "DEFAULT_OWL_QUERIES",
    "DIRECT_BLUR_CATEGORIES",
    "label_to_category",
]
