"""
FAL API implementation of Try-On service.
Refactored from original tryon_service.py to follow the TryOnService interface.
"""

import base64
import json
import logging
import time
import urllib.parse
from typing import Optional, Tuple

import httpx
from fastapi import HTTPException

from ..config import get_settings
from .tryon_interface import TryOnService, TryOnMetadata, TryOnResult
from .prompt_adjuster import build_fal_compatible_prompt

logger = logging.getLogger(__name__)
settings = get_settings()


def build_fal_endpoint() -> str:
    if settings.fal_endpoint:
        return settings.fal_endpoint
    if not settings.fal_model:
        raise HTTPException(status_code=500, detail="FAL_MODEL is not set.")
    return f"https://fal.run/{settings.fal_model}"


def to_data_url(data: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def parse_extra_json() -> dict:
    if not settings.fal_extra_json:
        return {}
    try:
        value = json.loads(settings.fal_extra_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"FAL_EXTRA_JSON invalid: {exc}") from exc
    if not isinstance(value, dict):
        raise HTTPException(status_code=500, detail="FAL_EXTRA_JSON must be a JSON object.")
    return value


def build_fal_payload(
    *,
    full_prompt: str,
    user_data_url: str,
    garment_data_url: str,
    negative_prompt: Optional[str],
) -> dict:
    payload: dict = {}

    prompt_key = settings.fal_prompt_field or "prompt"
    payload[prompt_key] = full_prompt

    if negative_prompt:
        negative_key = settings.fal_negative_field or "negative_prompt"
        payload[negative_key] = negative_prompt

    if settings.fal_user_field and settings.fal_garment_field:
        payload[settings.fal_user_field] = user_data_url
        payload[settings.fal_garment_field] = garment_data_url
    else:
        image_key = settings.fal_image_field or "image_urls"
        payload[image_key] = [user_data_url, garment_data_url]

    extra = parse_extra_json()
    if extra:
        payload.update(extra)

    return payload


def extract_fal_image(payload: dict) -> Optional[dict]:
    if not isinstance(payload, dict):
        return None

    if isinstance(payload.get("images"), list) and payload["images"]:
        return payload["images"][0]

    if payload.get("image"):
        return payload["image"]

    if payload.get("output"):
        output = payload["output"]
        if isinstance(output, list) and output:
            return output[0]
        if isinstance(output, dict):
            return output

    if payload.get("image_url"):
        return {"url": payload["image_url"]}

    if payload.get("url"):
        return {"url": payload["url"]}

    return None


def decode_data_url(value: str) -> Optional[Tuple[str, bytes]]:
    if not value.startswith("data:"):
        return None
    header, data = value.split(",", 1)
    mime_type = header[5:].split(";", 1)[0] or "image/png"
    if ";base64" in header:
        return mime_type, base64.b64decode(data)
    return mime_type, urllib.parse.unquote_to_bytes(data)


async def fetch_image_from_url(url: str) -> Tuple[str, bytes]:
    data_url = decode_data_url(url)
    if data_url:
        return data_url

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.get(url)
    if response.status_code >= 400:
        raise HTTPException(status_code=502, detail="Failed to download FAL image.")
    mime_type = response.headers.get("content-type", "image/png")
    return mime_type, response.content


async def resolve_fal_image(image_payload: dict) -> Tuple[str, bytes]:
    if isinstance(image_payload, str):
        return await fetch_image_from_url(image_payload)

    if not isinstance(image_payload, dict):
        raise HTTPException(status_code=502, detail="Unexpected FAL image payload.")

    if image_payload.get("data") or image_payload.get("base64"):
        data = image_payload.get("data") or image_payload.get("base64")
        mime_type = (
            image_payload.get("content_type")
            or image_payload.get("contentType")
            or image_payload.get("mime_type")
            or "image/png"
        )
        return mime_type, base64.b64decode(data)

    url = image_payload.get("url") or image_payload.get("image_url")
    if url:
        return await fetch_image_from_url(url)

    raise HTTPException(status_code=502, detail="FAL response did not include image data.")


class FALTryOnService(TryOnService):
    """Try-on service using FAL (Futuristic AI Language) API."""

    async def generate(
        self,
        user_image_bytes: bytes,
        user_image_type: str,
        garment_image_bytes: bytes,
        garment_image_type: str,
        metadata: TryOnMetadata,
    ) -> TryOnResult:
        """
        Generate try-on image using FAL API.
        """
        start_time = time.time()

        if not settings.fal_api_key:
            raise HTTPException(status_code=500, detail="FAL_API_KEY is not set.")

        try:
            # Build prompt
            prompt, negative_prompt = build_fal_compatible_prompt(metadata)

            # Create data URLs
            user_data_url = to_data_url(user_image_bytes, user_image_type)
            garment_data_url = to_data_url(garment_image_bytes, garment_image_type)

            # Build FAL payload
            payload = build_fal_payload(
                full_prompt=prompt,
                user_data_url=user_data_url,
                garment_data_url=garment_data_url,
                negative_prompt=negative_prompt,
            )

            # Make request
            endpoint = build_fal_endpoint()
            headers = {
                "Authorization": f"Key {settings.fal_api_key}",
                "Content-Type": "application/json",
            }

            async with httpx.AsyncClient(timeout=90) as client:
                response = await client.post(endpoint, json=payload, headers=headers)

            if response.status_code >= 400:
                logger.error(f"FAL request failed | status={response.status_code}")
                raise HTTPException(status_code=response.status_code, detail=response.text)

            # Parse response
            content_type = response.headers.get("content-type", "")
            if "application/json" not in content_type.lower():
                # Direct image response
                processing_time_ms = int((time.time() - start_time) * 1000)
                return TryOnResult(
                    image_bytes=response.content,
                    mime_type=content_type or "image/png",
                    provider_used="fal",
                    model_used="fal",
                    processing_time_ms=processing_time_ms,
                )

            # JSON response
            data = response.json()
            image = extract_fal_image(data)
            if not image:
                raise HTTPException(status_code=502, detail="No image returned by FAL.")

            mime_type, image_bytes = await resolve_fal_image(image)
            processing_time_ms = int((time.time() - start_time) * 1000)

            logger.info(
                f"FAL try-on successful | time={processing_time_ms}ms | "
                f"garment={metadata.garment_id}"
            )

            return TryOnResult(
                image_bytes=image_bytes,
                mime_type=mime_type,
                provider_used="fal",
                model_used="fal",
                processing_time_ms=processing_time_ms,
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"FAL try-on failed | error={str(e)}")
            raise HTTPException(status_code=502, detail=f"FAL service error: {str(e)}")
