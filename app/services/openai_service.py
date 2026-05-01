"""OpenAI image-edit based try-on service."""

import asyncio
import base64
import io
import logging
import time

import httpx
from fastapi import HTTPException
from openai import OpenAI

from ..config import get_settings
from .prompt_adjuster import build_openai_prompt
from .tryon_interface import TryOnMetadata, TryOnResult, TryOnService

logger = logging.getLogger(__name__)
settings = get_settings()


def _image_bytes_to_file(image_bytes: bytes, mime_type: str, name: str) -> io.BytesIO:
    buffer = io.BytesIO(image_bytes)
    suffix = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }.get(mime_type.lower(), ".bin")
    buffer.name = f"{name}{suffix}"  # type: ignore[attr-defined]
    buffer.seek(0)
    return buffer


def _decode_b64_image(value: str) -> tuple[str, bytes]:
    if value.startswith("data:"):
        header, data = value.split(",", 1)
        mime_type = header[5:].split(";", 1)[0] or "image/png"
        return mime_type, base64.b64decode(data)
    return "image/png", base64.b64decode(value)


def _extract_image_result(response) -> tuple[str, bytes]:
    data = getattr(response, "data", None) or []
    if not data:
        raise HTTPException(status_code=502, detail="OpenAI did not return image data.")

    item = data[0]
    b64_json = getattr(item, "b64_json", None)
    url = getattr(item, "url", None)

    if b64_json:
        return _decode_b64_image(b64_json)

    if url:
        resp = httpx.get(url, timeout=60)
        if resp.status_code >= 400:
            raise HTTPException(status_code=502, detail="Failed to download OpenAI image.")
        return resp.headers.get("content-type", "image/png"), resp.content

    raise HTTPException(status_code=502, detail="OpenAI response did not include image bytes.")


class OpenAITryOnService(TryOnService):
    """Try-on service using OpenAI GPT Image 2."""

    def __init__(self) -> None:
        if not settings.openai_api_key:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set.")
        self.client = OpenAI(api_key=settings.openai_api_key)

    async def generate(
        self,
        user_image_bytes: bytes,
        user_image_type: str,
        garment_image_bytes: bytes,
        garment_image_type: str,
        metadata: TryOnMetadata,
    ) -> TryOnResult:
        start_time = time.time()

        try:
            prompt, negative_prompt = build_openai_prompt(metadata)

            user_file = _image_bytes_to_file(user_image_bytes, user_image_type, "user")
            garment_file = _image_bytes_to_file(garment_image_bytes, garment_image_type, "garment")

            response = await asyncio.to_thread(
                self.client.images.edit,
                model=settings.openai_model,
                image=[user_file, garment_file],
                prompt=f"{prompt}\n\nNegative constraints: {negative_prompt}",
                size=settings.openai_image_size,
                quality=settings.openai_quality,
            )

            mime_type, image_bytes = _extract_image_result(response)
            processing_time_ms = int((time.time() - start_time) * 1000)

            logger.info(
                f"OpenAI try-on successful | time={processing_time_ms}ms | "
                f"garment={metadata.garment_id}"
            )

            return TryOnResult(
                image_bytes=image_bytes,
                mime_type=mime_type,
                provider_used="openai",
                model_used=settings.openai_model,
                processing_time_ms=processing_time_ms,
            )
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(f"OpenAI try-on failed | error={str(exc)}")
            raise HTTPException(status_code=502, detail=f"OpenAI service error: {str(exc)}")
