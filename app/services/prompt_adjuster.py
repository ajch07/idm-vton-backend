"""
Universal prompt generation for virtual try-on.
Keeps it simple - no edge case hardcoding.
Will optimize after testing if needed.
"""

from typing import Optional
from .tryon_interface import TryOnMetadata


# Universal base prompt - works for all cases
# Model will figure out the details during generation
DEFAULT_PROMPT = (
    "Photorealistic virtual try-on. Replace the outfit in the person photo with the "
    "garment from the reference image. Preserve the same face identity, facial features, "
    "skin tone, hair, body shape, pose, background, camera angle, and lighting. Keep fabric "
    "texture, folds, and shadows realistic. Ensure the garment fits naturally on the person's body. "
    "Show the garment as it would actually look when worn."
)

DEFAULT_NEGATIVE_PROMPT = (
    "Do not change face or hair. Do not alter body proportions or pose. Do not change background, "
    "lighting, or camera angle. No artifacts, text, watermark, blur, or extra people. "
    "Do not show clothing worn before the garment."
)


def build_universal_prompt(metadata: TryOnMetadata) -> tuple[str, str]:
    """
    Build simple, universal prompt for try-on.
    
    Args:
        metadata: TryOnMetadata (garment info, etc.)
        
    Returns:
        Tuple of (prompt, negative_prompt)
    """
    # Use user's custom prompt if provided, otherwise use universal default
    base_prompt = metadata.user_prompt or DEFAULT_PROMPT
    negative_prompt = metadata.user_negative_prompt or DEFAULT_NEGATIVE_PROMPT
    
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
    
    # Add FAL-specific instruction
    prompt = (
        f"{prompt}\n\nUse the first image as the person photo and the second "
        "image as the garment reference."
    )
    
    return prompt, negative_prompt


def build_runpod_prompt(metadata: TryOnMetadata) -> str:
    """
    Build prompt for Runpod handler (IDM-VTON + Flux).
    
    Args:
        metadata: TryOnMetadata
        
    Returns:
        Complete prompt for Runpod inference
    """
    prompt, negative_prompt = build_universal_prompt(metadata)
    
    # Combine into single prompt for Runpod handler
    return f"{prompt}\n\nNegative constraints: {negative_prompt}"
