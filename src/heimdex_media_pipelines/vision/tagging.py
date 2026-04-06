"""VLM prompt construction for caption + structured tag extraction.

Builds prompts that instruct the VLM to produce both a natural language
caption and structured tags in a single inference pass.  The output
format is line-based (not JSON) for reliability with small models.
"""

from __future__ import annotations

from heimdex_media_contracts.tags.vocabulary import VLM_KEYWORD_TAGS, VLM_PRODUCT_TAGS

TAG_MAX_NEW_TOKENS = 256

_KEYWORD_TAG_LIST = ", ".join(sorted(VLM_KEYWORD_TAGS.keys()))
_PRODUCT_TAG_LIST = ", ".join(sorted(VLM_PRODUCT_TAGS.keys()))

# Format specification teaches the model the expected output structure
# without injecting any specific product content that could be parroted.
_FORMAT_SPEC = """설명: (이미지에 보이는 장면을 한 문장으로 서술)
콘텐츠태그: (아래 선택지에서 해당하는 태그를 콤마로 구분)
상품태그: (아래 선택지에서 해당하는 태그를 콤마로 구분)
상품명: (이미지에 보이는 상품명, 없으면 없음)
AI태그: (장면을 설명하는 한국어 키워드 2~7개, 콤마로 구분)"""

_TAG_REF = """
[태그 선택지]
콘텐츠태그: {keyword_tags}
상품태그: {product_tags}"""

_PROMPT_WITH_TRANSCRIPT = """아래 형식에 맞춰 이미지와 자막을 분석해서 답하세요.

[출력 형식]
{format_spec}
{tag_ref}

[자막]
{transcript}

[답변]
설명:"""

_PROMPT_IMAGE_ONLY = """아래 형식에 맞춰 이미지를 분석해서 답하세요.

[출력 형식]
{format_spec}
{tag_ref}

[답변]
설명:"""


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

    tag_ref = _TAG_REF.format(
        keyword_tags=_KEYWORD_TAG_LIST,
        product_tags=_PRODUCT_TAG_LIST,
    )

    if transcript_trimmed:
        prompt = _PROMPT_WITH_TRANSCRIPT.format(
            format_spec=_FORMAT_SPEC,
            tag_ref=tag_ref,
            transcript=transcript_trimmed,
        )
    else:
        prompt = _PROMPT_IMAGE_ONLY.format(
            format_spec=_FORMAT_SPEC,
            tag_ref=tag_ref,
        )

    return prompt


def get_tag_max_tokens() -> int:
    """Recommended ``max_new_tokens`` for the tag extraction prompt."""
    return TAG_MAX_NEW_TOKENS
