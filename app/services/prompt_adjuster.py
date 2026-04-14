"""Prompt builders for virtual try-on services."""

from typing import Optional
from .tryon_interface import TryOnMetadata


DEFAULT_PROMPT_TEMPLATE = (
    "Edit the input person photo so the person is wearing the exact {garment_description} "
    "from the reference image. Keep the person's identity, face, hair, hands, pose, body shape, "
    "background, framing, and lighting unchanged. Match the garment color, print, silhouette, "
    "fabric texture, fit, folds, and shadows realistically. Output a realistic ecommerce try-on photo."
)

DEFAULT_NEGATIVE_PROMPT = (
    "Do not change face, hair, hands, pose, body proportions, background, framing, or lighting. "
    "No extra people, extra limbs, duplicate garments, artifacts, blur, text, or watermark."
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
        f"{prompt}\n\nUse the first image as the person photo and the second "
        "image as the garment reference."
    )
    
    return prompt, negative_prompt


def build_runpod_prompt(metadata: TryOnMetadata) -> tuple[str, str]:
    """
    Build prompt for Runpod handler using FLUX.1-Kontext-dev.
    
    Args:
        metadata: TryOnMetadata
        
    Returns:
        Tuple of (prompt, negative_prompt)
    """
    return build_universal_prompt(metadata)
