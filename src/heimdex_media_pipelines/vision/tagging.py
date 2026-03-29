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

# Few-shot example teaches the model the exact output format.
# The 2B model needs concrete examples, not abstract instructions.
_EXAMPLE = """설명: 호스트가 수분크림을 손등에 발라 텍스처를 보여주고 있다
콘텐츠태그: swatch_test, texture_show
상품태그: skincare
상품명: 수분크림"""

_PROMPT_WITH_TRANSCRIPT = """아래 예시처럼 이미지와 자막을 분석해서 답하세요.

[예시]
{example}

[자막]
{transcript}

[답변]
설명:"""

_PROMPT_IMAGE_ONLY = """아래 예시처럼 이미지를 분석해서 답하세요.

[예시]
{example}

[답변]
설명:"""

_TAG_SUFFIX = """
콘텐츠태그 선택지: {keyword_tags}
상품태그 선택지: {product_tags}"""


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

    # Build the tag selection reference (appended as system context)
    tag_ref = _TAG_SUFFIX.format(
        keyword_tags=_KEYWORD_TAG_LIST,
        product_tags=_PRODUCT_TAG_LIST,
    )

    if transcript_trimmed:
        prompt = _PROMPT_WITH_TRANSCRIPT.format(
            example=_EXAMPLE,
            transcript=transcript_trimmed,
        )
    else:
        prompt = _PROMPT_IMAGE_ONLY.format(
            example=_EXAMPLE,
        )

    return prompt + tag_ref


def get_tag_max_tokens() -> int:
    """Recommended ``max_new_tokens`` for the tag extraction prompt."""
    return TAG_MAX_NEW_TOKENS
