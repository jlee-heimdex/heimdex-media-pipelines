"""VLM prompt construction for caption + structured tag extraction.

Builds prompts that instruct the VLM to produce both a natural language
caption and structured tags in a single inference pass.  The output
format is line-based (not JSON) for reliability with small models.

IMPORTANT: The 2B VLM (Qwen2-VL-2B-Instruct) requires concrete few-shot
examples to produce structured multi-line output.  Abstract format specs
cause the model to generate only the caption and stop.  Use 3+ diverse
examples from different categories and randomly sample one per request
to prevent single-example poisoning (hallucinating one product across
all videos).
"""

from __future__ import annotations

import random

from heimdex_media_contracts.tags.vocabulary import VLM_KEYWORD_TAGS, VLM_PRODUCT_TAGS

TAG_MAX_NEW_TOKENS = 256

_KEYWORD_TAG_LIST = ", ".join(sorted(VLM_KEYWORD_TAGS.keys()))
_PRODUCT_TAG_LIST = ", ".join(sorted(VLM_PRODUCT_TAGS.keys()))

# Diverse few-shot examples from different product categories.
# One is randomly sampled per request to prevent single-example poisoning.
_EXAMPLES = [
    """설명: 호스트가 운동화를 신고 걸어보며 착화감을 설명하고 있다
콘텐츠태그: wearing_show, product_demo
상품태그: shoes
상품명: 나이키 에어맥스
AI태그: 운동화, 착화감, 워킹 테스트, 쿠셔닝""",
    """설명: 호스트가 프라이팬에 계란을 부치며 논스틱 코팅을 보여주고 있다
콘텐츠태그: cooking_show, product_demo
상품태그: kitchenware
상품명: 테팔 프라이팬
AI태그: 프라이팬, 논스틱, 계란 요리, 코팅 시연""",
    """설명: 호스트가 가방 내부 수납공간을 하나씩 열어 보여주고 있다
콘텐츠태그: closeup_detail, product_demo
상품태그: bag
상품명: 캔버스 토트백
AI태그: 토트백, 수납공간, 내부 구조, 데일리백""",
]

_TAG_REF = """
[태그 선택지]
콘텐츠태그: {keyword_tags}
상품태그: {product_tags}"""

_PROMPT_WITH_TRANSCRIPT = """아래 예시처럼 이미지와 자막을 분석해서 답하세요.

[예시]
{example}
{tag_ref}

[자막]
{transcript}

[답변]
설명:"""

_PROMPT_IMAGE_ONLY = """아래 예시처럼 이미지를 분석해서 답하세요.

[예시]
{example}
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

    example = random.choice(_EXAMPLES)

    tag_ref = _TAG_REF.format(
        keyword_tags=_KEYWORD_TAG_LIST,
        product_tags=_PRODUCT_TAG_LIST,
    )

    if transcript_trimmed:
        prompt = _PROMPT_WITH_TRANSCRIPT.format(
            example=example,
            tag_ref=tag_ref,
            transcript=transcript_trimmed,
        )
    else:
        prompt = _PROMPT_IMAGE_ONLY.format(
            example=example,
            tag_ref=tag_ref,
        )

    return prompt


def get_tag_max_tokens() -> int:
    """Recommended ``max_new_tokens`` for the tag extraction prompt."""
    return TAG_MAX_NEW_TOKENS
