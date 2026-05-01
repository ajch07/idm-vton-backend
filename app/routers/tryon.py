import uuid
import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import get_settings
from ..dependencies import get_current_user, get_session
from ..models import CreditTransaction, TryOnGeneration, User
from ..schemas import TryOnGenerationOut
from ..services.storage import build_generation_path, create_signed_object_url, upload_to_supabase
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


def _guess_extension(mime_type: str) -> str:
    if mime_type == "image/png":
        return ".png"
    if mime_type in {"image/jpeg", "image/jpg"}:
        return ".jpg"
    if mime_type == "image/webp":
        return ".webp"
    return ".bin"


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

        generation_id = uuid.uuid4()
        storage_path = build_generation_path(current_user.id, _guess_extension(result.mime_type))
        image_url = await upload_to_supabase(result.image_bytes, result.mime_type, storage_path)

        generation = TryOnGeneration(
            id=generation_id,
            user_id=current_user.id,
            provider=result.provider_used or settings.tryon_service,
            model_used=result.model_used,
            garment_id=garmentId,
            garment_name=garmentName,
            category=category,
            prompt=prompt,
            negative_prompt=negativePrompt,
            image_url=image_url,
            storage_path=storage_path,
            mime_type=result.mime_type,
            processing_time_ms=result.processing_time_ms,
        )
        session.add(generation)
        await session.commit()
        await session.refresh(generation)
        
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


@router.get("/api/try-on/history", response_model=list[TryOnGenerationOut])
async def try_on_history(
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> list[TryOnGenerationOut]:
    result = await session.execute(
        select(TryOnGeneration)
        .where(TryOnGeneration.user_id == current_user.id)
        .order_by(TryOnGeneration.created_at.desc())
    )
    generations = result.scalars().all()
    signed_generations = []
    for item in generations:
        image_url = item.image_url
        try:
            image_url = await create_signed_object_url(item.storage_path, expires_in=60 * 60)
        except HTTPException:
            logger.warning(
                "Could not create signed history URL | user=%s | path=%s",
                current_user.id,
                item.storage_path,
            )
        signed_generations.append(
            TryOnGenerationOut.model_validate(item).model_copy(update={"image_url": image_url})
        )

    return signed_generations
