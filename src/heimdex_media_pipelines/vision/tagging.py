"""VLM prompt construction for caption + structured tag extraction.

Builds prompts that instruct the VLM to produce both a natural language
caption and structured tags in a single inference pass.  The output
format is line-based (not JSON) for reliability with small models.
"""

from __future__ import annotations

from heimdex_media_contracts.tags.vocabulary import VLM_KEYWORD_TAGS, VLM_PRODUCT_TAGS

TAG_MAX_NEW_TOKENS = 200

_KEYWORD_TAG_LIST = ", ".join(sorted(VLM_KEYWORD_TAGS.keys()))
_PRODUCT_TAG_LIST = ", ".join(sorted(VLM_PRODUCT_TAGS.keys()))

_PROMPT_WITH_TRANSCRIPT = """이미지와 자막을 보고 다음 형식으로 응답하세요.

자막: {transcript}

설명: (이 장면을 한국어 한 문장으로 설명)
콘텐츠태그: (다음 중 해당하는 것만 선택: {keyword_tags})
상품태그: (다음 중 해당하는 것만 선택: {product_tags})
상품명: (장면에 보이는 구체적 상품명 1-3개, 없으면 "없음")"""

_PROMPT_IMAGE_ONLY = """이미지를 보고 다음 형식으로 응답하세요.

설명: (이 장면을 한국어 한 문장으로 설명)
콘텐츠태그: (다음 중 해당하는 것만 선택: {keyword_tags})
상품태그: (다음 중 해당하는 것만 선택: {product_tags})
상품명: (장면에 보이는 구체적 상품명 1-3개, 없으면 "없음")"""


def build_tag_prompt(
    transcript: str = "",
    max_transcript_chars: int = 300,
) -> str:
    """Build the VLM prompt for caption + tag extraction.

    Args:
        transcript: Raw transcript text for the scene.  Empty string
            produces an image-only prompt.
        max_transcript_chars: Truncate transcript to this length to
            avoid blowing up the VLM context window.

    Returns:
        Formatted prompt string ready for ``CaptionEngine.caption()``.
    """
    transcript_trimmed = transcript.strip()[:max_transcript_chars]

    if transcript_trimmed:
        return _PROMPT_WITH_TRANSCRIPT.format(
            transcript=transcript_trimmed,
            keyword_tags=_KEYWORD_TAG_LIST,
            product_tags=_PRODUCT_TAG_LIST,
        )

    return _PROMPT_IMAGE_ONLY.format(
        keyword_tags=_KEYWORD_TAG_LIST,
        product_tags=_PRODUCT_TAG_LIST,
    )


def get_tag_max_tokens() -> int:
    """Recommended ``max_new_tokens`` for the tag extraction prompt."""
    return TAG_MAX_NEW_TOKENS
