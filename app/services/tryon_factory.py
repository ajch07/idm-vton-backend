"""
Factory pattern for creating and switching between Try-On services.
Allows easy switching between FAL, Runpod, and Hybrid modes.
"""

import logging
from typing import Optional

from fastapi import HTTPException

from ..config import get_settings
from .tryon_interface import TryOnService
from .fal_service import FALTryOnService
from .runpod_service import RunpodTryOnService

logger = logging.getLogger(__name__)
settings = get_settings()


class HybridTryOnService(TryOnService):
    """
    Hybrid service: Try Runpod first, fall back to FAL if it fails.
    Provides cost optimization with reliability.
    """

    def __init__(self, runpod_service: TryOnService, fal_service: TryOnService):
        self.runpod_service = runpod_service
        self.fal_service = fal_service

    async def generate(
        self,
        user_image_bytes: bytes,
        user_image_type: str,
        garment_image_bytes: bytes,
        garment_image_type: str,
        metadata,
    ):
        """Try Runpod first, fall back to FAL on failure."""
        try:
            logger.info(f"Attempting Runpod generation | garment={metadata.garment_id}")
            result = await self.runpod_service.generate(
                user_image_bytes,
                user_image_type,
                garment_image_bytes,
                garment_image_type,
                metadata,
            )
            logger.info(f"Runpod succeeded | garment={metadata.garment_id}")
            return result
        except Exception as e:
            logger.warning(
                f"Runpod failed, falling back to FAL | error={str(e)} | "
                f"garment={metadata.garment_id}"
            )
            return await self.fal_service.generate(
                user_image_bytes,
                user_image_type,
                garment_image_bytes,
                garment_image_type,
                metadata,
            )


def get_tryon_service(service_type: Optional[str] = None) -> TryOnService:
    """
    Factory function to get appropriate Try-On service instance.

    Args:
        service_type: "fal", "runpod", "hybrid", or None (uses env variable)

    Returns:
        TryOnService instance

    Raises:
        ValueError: If service type is invalid or required config is missing
        HTTPException: If required configuration is not set
    """
    # Use parameter or fall back to environment variable
    service = service_type or settings.tryon_service

    logger.info(f"Initializing try-on service | type={service}")

    if service == "fal":
        if not settings.fal_api_key:
            raise HTTPException(status_code=500, detail="FAL_API_KEY is not set")
        return FALTryOnService()

    elif service == "runpod":
        if not settings.runpod_endpoint or not settings.runpod_api_key:
            raise HTTPException(
                status_code=500,
                detail="RUNPOD_ENDPOINT and RUNPOD_API_KEY are required for Runpod service",
            )
        return RunpodTryOnService(
            endpoint_url=settings.runpod_endpoint,
            api_key=settings.runpod_api_key,
            timeout_seconds=settings.runpod_timeout_seconds,
        )

    elif service == "hybrid":
        # Hybrid mode requires both FAL and Runpod to be configured
        if not settings.fal_api_key:
            raise HTTPException(status_code=500, detail="FAL_API_KEY is not set (required for hybrid mode)")
        if not settings.runpod_endpoint or not settings.runpod_api_key:
            raise HTTPException(
                status_code=500,
                detail="RUNPOD_ENDPOINT and RUNPOD_API_KEY are required for hybrid mode",
            )

        runpod_svc = RunpodTryOnService(
            endpoint_url=settings.runpod_endpoint,
            api_key=settings.runpod_api_key,
            timeout_seconds=settings.runpod_timeout_seconds,
        )
        fal_svc = FALTryOnService()
        return HybridTryOnService(runpod_service=runpod_svc, fal_service=fal_svc)

    else:
        raise ValueError(
            f"Invalid TRYON_SERVICE: {service}. Must be 'fal', 'runpod', or 'hybrid'"
        )
