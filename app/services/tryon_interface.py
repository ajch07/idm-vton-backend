"""
Abstract interface for Try-On services.
This allows multiple implementations (FAL, Runpod, Hybrid) to be swapped.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class TryOnMetadata:
    """Metadata passed to try-on service."""
    garment_name: str
    garment_id: str
    user_prompt: Optional[str] = None  # Custom prompt from user
    user_negative_prompt: Optional[str] = None  # Custom negative prompt
    # Additional attributes can be added here after testing if needed


@dataclass
class TryOnResult:
    """Result from try-on service."""
    image_bytes: bytes
    mime_type: str
    model_used: Optional[str] = None  # "idm-vton", "flux", "fal", etc.
    processing_time_ms: Optional[int] = None
    error: Optional[str] = None


class TryOnService(ABC):
    """Abstract base class for all try-on services."""

    @abstractmethod
    async def generate(
        self,
        user_image_bytes: bytes,
        user_image_type: str,
        garment_image_bytes: bytes,
        garment_image_type: str,
        metadata: TryOnMetadata,
    ) -> TryOnResult:
        """
        Generate a virtual try-on image.

        Args:
            user_image_bytes: User's photo bytes
            user_image_type: MIME type of user image (e.g., "image/jpeg")
            garment_image_bytes: Product garment photo bytes
            garment_image_type: MIME type of garment image
            metadata: Try-on metadata (garment name, id, prompts)

        Returns:
            TryOnResult containing the generated image and metadata

        Raises:
            HTTPException: On service error
        """
        pass
