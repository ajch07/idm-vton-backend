"""Prompt builders for virtual try-on services."""

from typing import Optional
from .tryon_interface import TryOnMetadata


DEFAULT_PROMPT_TEMPLATE = (
    """ 1. Edit the input person photo so the person is wearing the **exact** {garment_description} from the reference image.
        2. You **MUST** preserve the person's identity, face, hair, hands, pose, body shape, background, framing, and lighting exactly.
        3. You **MUST** match the garment color, print,silhouette, fabric texture, fit, folds, and shadows realistically. Output a realistic ecommerce try-on photo."""
)

DEFAULT_NEGATIVE_PROMPT = (
    """ 1. *You **MUST NOT** change face, eyes, nose, lips, jawline, skin tone, hair, hands, pose, body proportions, background, framing, or lighting*. 
        2. *You **MUST NOT** apply beauty retouching, age change, identity drift, extra people, extra limbs, duplicate garments, artifacts, blur, text, or watermark.*"""
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
            "You MUST edit only the lower-body clothing region from the waist down. You MUST NOT alter "
            "the face, hair, upper body, arms, hands, or background."
        )
    if any(keyword in text for keyword in ("dress", "gown", "jumpsuit", "romper", "saree", "sari")):
        return (
            "You MUST edit only the outfit region. You MUST preserve the person's face, hair, hands, "
            "body shape, pose, and background exactly."
        )
    return (
        "You MUST edit only the clothing region on the person's body. You MUST NOT alter the face, "
        "hair, hands, body shape, pose, or background."
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
        "Image 1 is the base person photo and MUST remain the same person. "
        "Image 2 is the garment reference. "
        f"{_edit_scope(metadata)} "
        "You MUST use only the garment from image 2. You MUST NOT copy the model, mannequin, legs, pose, or background from image 2. "
        "You MUST preserve facial identity exactly with the same eyes, nose, lips, jawline, expression, and skin texture. "
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
