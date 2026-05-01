import uuid
from typing import Tuple

import httpx
from fastapi import HTTPException

from ..config import get_settings

settings = get_settings()


def _ensure_supabase_config() -> Tuple[str, str, str]:
    if not settings.supabase_url or not settings.supabase_service_key:
        raise HTTPException(status_code=500, detail="Supabase storage is not configured.")
    if settings.supabase_service_key.startswith("sb_publishable_"):
        raise HTTPException(
            status_code=500,
            detail="Supabase service role key required (do not use publishable/anon key).",
        )
    bucket = settings.supabase_bucket or "product-media"
    return settings.supabase_url.rstrip("/"), settings.supabase_service_key, bucket


async def upload_to_supabase(
    content: bytes,
    content_type: str,
    object_path: str,
) -> str:
    supabase_url, service_key, bucket = _ensure_supabase_config()

    upload_url = f"{supabase_url}/storage/v1/object/{bucket}/{object_path}"
    headers = {
        "Authorization": f"Bearer {service_key}",
        "apikey": service_key,
        "Content-Type": content_type,
    }

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(upload_url, content=content, headers=headers)

    if response.status_code not in {200, 201}:
        detail = response.text.strip()
        if len(detail) > 300:
            detail = detail[:300] + "..."
        raise HTTPException(
            status_code=500,
            detail=f"Supabase upload failed ({response.status_code}): {detail or 'no response body'}",
        )

    return f"{supabase_url}/storage/v1/object/public/{bucket}/{object_path}"


async def create_signed_object_url(object_path: str, expires_in: int = 3600) -> str:
    supabase_url, service_key, bucket = _ensure_supabase_config()

    sign_url = f"{supabase_url}/storage/v1/object/sign/{bucket}/{object_path}"
    headers = {
        "Authorization": f"Bearer {service_key}",
        "apikey": service_key,
        "Content-Type": "application/json",
    }
    payload = {"expiresIn": expires_in}

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(sign_url, json=payload, headers=headers)

    if response.status_code not in {200, 201}:
        detail = response.text.strip()
        if len(detail) > 300:
            detail = detail[:300] + "..."
        raise HTTPException(
            status_code=500,
            detail=f"Supabase signed URL failed ({response.status_code}): {detail or 'no response body'}",
        )

    data = response.json()
    signed_path = data.get("signedURL") or data.get("signedUrl")
    if not signed_path:
        raise HTTPException(
            status_code=500,
            detail="Supabase signed URL response missing signedURL.",
        )

    if signed_path.startswith("http://") or signed_path.startswith("https://"):
        return signed_path
    return f"{supabase_url}/storage/v1{signed_path}"


def build_media_path(product_id: uuid.UUID, extension: str) -> str:
    safe_ext = extension if extension.startswith(".") else f".{extension}"
    return f"products/{product_id}/{uuid.uuid4().hex}{safe_ext}"


def build_generation_path(user_id: uuid.UUID, extension: str) -> str:
    safe_ext = extension if extension.startswith(".") else f".{extension}"
    return f"tryons/{user_id}/{uuid.uuid4().hex}{safe_ext}"
