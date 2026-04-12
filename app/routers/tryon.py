import uuid
import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import get_settings
from ..dependencies import get_current_user, get_session
from ..models import CreditTransaction, User
from ..services.tryon_factory import get_tryon_service
from ..services.tryon_interface import TryOnMetadata

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(tags=["try-on"])


def validate_upload(file: UploadFile, label: str) -> None:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"{label} must be an image.")


def validate_size(data: bytes, label: str) -> None:
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"{label} exceeds {settings.max_upload_mb}MB limit.",
        )


async def debit_credits(
    session: AsyncSession,
    user_id: uuid.UUID,
    amount: int,
    reason: str,
) -> None:
    result = await session.execute(select(User).where(User.id == user_id).with_for_update())
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    if user.credits < amount:
        raise HTTPException(status_code=402, detail="Not enough credits.")

    user.credits -= amount
    session.add(
        CreditTransaction(
            user_id=user.id,
            delta=-amount,
            reason=reason,
            source="try_on",
        )
    )
    await session.commit()


async def refund_credits(
    session: AsyncSession,
    user_id: uuid.UUID,
    amount: int,
    reason: str,
) -> None:
    result = await session.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        return
    user.credits += amount
    session.add(
        CreditTransaction(
            user_id=user.id,
            delta=amount,
            reason=reason,
            source="try_on",
        )
    )
    await session.commit()


@router.post("/api/try-on")
async def try_on(
    userImage: UploadFile = File(...),
    garmentImage: UploadFile = File(...),
    garmentId: str = Form(...),
    garmentName: str = Form(...),
    category: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    negativePrompt: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> Response:
    if settings.credits_per_tryon <= 0:
        raise HTTPException(status_code=500, detail="Invalid credit configuration.")

    validate_upload(userImage, "userImage")
    validate_upload(garmentImage, "garmentImage")

    user_bytes = await userImage.read()
    garment_bytes = await garmentImage.read()
    validate_size(user_bytes, "userImage")
    validate_size(garment_bytes, "garmentImage")

    await debit_credits(session, current_user.id, settings.credits_per_tryon, "try_on")

    try:
        # Create metadata for try-on service (simple, just basic info)
        metadata = TryOnMetadata(
            garment_name=garmentName,
            garment_id=garmentId,
            user_prompt=prompt,
            user_negative_prompt=negativePrompt,
            category=category,
        )
        
        # Get appropriate service (FAL, Runpod, or Hybrid)
        service = get_tryon_service()
        
        result = await service.generate(
            user_image_bytes=user_bytes,
            user_image_type=userImage.content_type or "image/jpeg",
            garment_image_bytes=garment_bytes,
            garment_image_type=garmentImage.content_type or "image/jpeg",
            metadata=metadata,
        )
        
        logger.info(
            f"Try-on succeeded | user={current_user.id} | garment={garmentId} | "
            f"model={result.model_used} | time={result.processing_time_ms}ms"
        )
        
        return Response(content=result.image_bytes, media_type=result.mime_type)
    except HTTPException:
        await refund_credits(session, current_user.id, settings.credits_per_tryon, "try_on_refund")
        raise
    except Exception as exc:
        logger.error(f"Try-on failed | user={current_user.id} | garment={garmentId} | error={str(exc)}")
        await refund_credits(session, current_user.id, settings.credits_per_tryon, "try_on_refund")
        raise HTTPException(status_code=500, detail="Try-on failed.") from exc
