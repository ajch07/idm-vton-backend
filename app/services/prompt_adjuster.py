"""Prompt builders for virtual try-on services."""

from typing import Optional
from .tryon_interface import TryOnMetadata


DEFAULT_PROMPT_TEMPLATE = (
    "Edit the person in image 1 so they wear the exact {garment_description} from image 2. "
    "Preserve identity, face, hair, hands, pose, body shape, background, framing, and lighting. "
    "Match color, print, silhouette, fabric texture, fit, folds, and shadows. "
    "Output a realistic ecommerce try-on photo."
)

DEFAULT_NEGATIVE_PROMPT = (
    "Do not change face, eyes, nose, lips, jawline, skin tone, hair, hands, pose, body proportions, "
    "background, framing, or lighting. No beauty retouching, age change, identity drift, extra people, "
    "extra limbs, duplicate garments, artifacts, blur, text, or watermark."
)


def _garment_description(metadata: TryOnMetadata) -> str:
    category = (metadata.category or "").strip()
    garment_name = (metadata.garment_name or "garment").strip()
    if category and category.lower() not in garment_name.lower():
        return f"{garment_name} ({category})"
    return garment_name


def _normalize(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _edit_scope(metadata: TryOnMetadata) -> str:
    category = (metadata.category or "").lower()
    garment_name = _garment_description(metadata).lower()
    text = f"{garment_name} {category}"

    if any(keyword in text for keyword in ("skirt", "pant", "pants", "trouser", "jean", "short", "bottom", "palazzo")):
        return (
            "Edit only the lower-body clothing from the waist down. Do not alter the face, hair, upper body, "
            "arms, hands, or background."
        )
    if any(keyword in text for keyword in ("dress", "gown", "jumpsuit", "romper", "saree", "sari")):
        return (
            "Edit only the outfit. Preserve the person's face, hair, hands, body shape, pose, and background."
        )
    return (
        "Edit only the clothing on the body. Do not alter the face, hair, hands, body shape, pose, or background."
    )


def build_universal_prompt(metadata: TryOnMetadata) -> tuple[str, str]:
    """
    Build simple, universal prompt for try-on.
    
    Args:
        metadata: TryOnMetadata (garment info, etc.)
        
    Returns:
        Tuple of (prompt, negative_prompt)
    """
    base_prompt = _normalize(metadata.user_prompt) or DEFAULT_PROMPT_TEMPLATE.format(
        garment_description=_garment_description(metadata)
    )
    negative_prompt = _normalize(metadata.user_negative_prompt) or DEFAULT_NEGATIVE_PROMPT
    return base_prompt, negative_prompt


def build_fal_compatible_prompt(metadata: TryOnMetadata) -> tuple[str, str]:
    """
    Build prompts compatible with FAL API format.
    
    Args:
        metadata: TryOnMetadata
        
    Returns:
        Tuple of (prompt, negative_prompt) for FAL API
    """
    prompt, negative_prompt = build_universal_prompt(metadata)
    
    prompt = (
        "Image 1 is the person. Image 2 is the garment. "
        "Use only the garment from image 2. Do not copy the model, mannequin, legs, pose, or background. "
        "Preserve facial identity and skin texture. "
        f"{_edit_scope(metadata)} "
        f"{prompt}"
    )
    
    return prompt, negative_prompt


def build_openai_prompt(metadata: TryOnMetadata) -> tuple[str, str]:
    """
    Build prompt optimized for OpenAI image editing.

    Keeps the instructions short and explicit so the provider can be swapped
    without changing the rest of the pipeline.
    """
    prompt, negative_prompt = build_universal_prompt(metadata)
    prompt = (
        "Edit image 1 so the person wears the garment from image 2. "
        "Do not change identity, face, hair, pose, hands, background, framing, or lighting. "
        "Use realistic fabric texture, folds, shadows, and fit. "
        f"{_edit_scope(metadata)} "
        f"{prompt}"
    )
    return prompt, negative_prompt


def build_runpod_prompt(metadata: TryOnMetadata) -> tuple[str, str]:
    """
    Build prompt for the RunPod editing handler.
    
    Args:
        metadata: TryOnMetadata
        
    Returns:
        Tuple of (prompt, negative_prompt)
    """
    return build_universal_prompt(metadata)
