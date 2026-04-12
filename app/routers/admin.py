import mimetypes
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..dependencies import get_session, require_admin
from ..models import CreditTransaction, Order, Product, ProductMedia, User
from ..schemas import (
    AdminActivityOut,
    AdminMetrics,
    AdminOrderOut,
    AdminUserOut,
    CreditGrantRequest,
    ProductMediaOut,
)
from ..services.storage import build_media_path, upload_to_supabase
from ..config import get_settings

settings = get_settings()

router = APIRouter(prefix="/api/admin", tags=["admin"], dependencies=[Depends(require_admin)])


def _coerce_uuid(value: str) -> uuid.UUID | None:
    try:
        return uuid.UUID(value)
    except ValueError:
        return None


@router.get("/metrics", response_model=AdminMetrics)
async def metrics(session: AsyncSession = Depends(get_session)) -> AdminMetrics:
    users_count = await session.scalar(select(func.count(User.id)))
    products_count = await session.scalar(select(func.count(Product.id)).where(Product.is_active.is_(True)))
    orders_count = await session.scalar(select(func.count(Order.id)))
    revenue_total = await session.scalar(
        select(func.coalesce(func.sum(Order.amount), 0)).where(Order.status == "paid")
    )
    try_on_count = await session.scalar(
        select(func.count(CreditTransaction.id)).where(CreditTransaction.reason == "try_on")
    )

    return AdminMetrics(
        users=users_count or 0,
        products=products_count or 0,
        orders=orders_count or 0,
        revenue=revenue_total or 0,
        try_ons=try_on_count or 0,
    )


@router.get("/users", response_model=List[AdminUserOut])
async def list_users(session: AsyncSession = Depends(get_session)) -> List[AdminUserOut]:
    result = await session.execute(select(User).order_by(User.created_at.desc()).limit(200))
    users = result.scalars().all()
    return [AdminUserOut.model_validate(user) for user in users]


@router.get("/orders", response_model=List[AdminOrderOut])
async def list_orders(session: AsyncSession = Depends(get_session)) -> List[AdminOrderOut]:
    result = await session.execute(select(Order).order_by(Order.created_at.desc()).limit(200))
    orders = result.scalars().all()

    user_ids = {order.user_id for order in orders}
    users = {}
    if user_ids:
        users_result = await session.execute(select(User).where(User.id.in_(user_ids)))
        users = {user.id: user for user in users_result.scalars().all()}

    response: List[AdminOrderOut] = []
    for order in orders:
        user = users.get(order.user_id)
        payload = AdminOrderOut.model_validate(order)
        payload.user_email = user.email if user else None
        response.append(payload)

    return response


@router.get("/activity", response_model=List[AdminActivityOut])
async def list_activity(session: AsyncSession = Depends(get_session)) -> List[AdminActivityOut]:
    result = await session.execute(
        select(CreditTransaction).order_by(CreditTransaction.created_at.desc()).limit(300)
    )
    transactions = result.scalars().all()

    user_ids = {tx.user_id for tx in transactions}
    users = {}
    if user_ids:
        users_result = await session.execute(select(User).where(User.id.in_(user_ids)))
        users = {user.id: user for user in users_result.scalars().all()}

    response: List[AdminActivityOut] = []
    for tx in transactions:
        user = users.get(tx.user_id)
        payload = AdminActivityOut.model_validate(tx)
        payload.user_email = user.email if user else None
        response.append(payload)

    return response


@router.get("/products/{product_id}/media", response_model=List[ProductMediaOut])
async def list_product_media(product_id: str, session: AsyncSession = Depends(get_session)) -> List[ProductMediaOut]:
    parsed_id = _coerce_uuid(product_id)
    if not parsed_id:
        raise HTTPException(status_code=400, detail="Invalid product id.")

    result = await session.execute(
        select(ProductMedia).where(ProductMedia.product_id == parsed_id).order_by(ProductMedia.order_index.asc())
    )
    media_items = result.scalars().all()
    return [ProductMediaOut.model_validate(item) for item in media_items]


@router.post("/media/upload", response_model=ProductMediaOut)
async def upload_product_media(
    product_id: str = Form(...),
    media_type: str = Form(...),
    order_index: Optional[int] = Form(None),
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
) -> ProductMediaOut:
    parsed_id = _coerce_uuid(product_id)
    if not parsed_id:
        raise HTTPException(status_code=400, detail="Invalid product id.")

    if media_type not in {"image", "video"}:
        raise HTTPException(status_code=400, detail="Invalid media type.")

    if not file.content_type:
        raise HTTPException(status_code=400, detail="Missing content type.")

    if media_type == "image" and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    if media_type == "video" and not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video.")

    result = await session.execute(select(Product).where(Product.id == parsed_id))
    product = result.scalar_one_or_none()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found.")

    data = await file.read()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_upload_mb}MB limit.")

    ext = ""
    if file.filename and "." in file.filename:
        ext = file.filename.rsplit(".", 1)[-1]
    if not ext:
        guessed = mimetypes.guess_extension(file.content_type) or ""
        ext = guessed.lstrip(".")
    if not ext:
        ext = "bin"

    object_path = build_media_path(product.id, ext)
    url = await upload_to_supabase(data, file.content_type, object_path)

    is_primary = False
    if media_type == "image" and not product.image_url:
        product.image_url = url
        is_primary = True

    media = ProductMedia(
        product_id=product.id,
        media_type=media_type,
        url=url,
        order_index=order_index or 0,
        is_primary=is_primary,
    )
    session.add(media)
    await session.commit()
    await session.refresh(media)
    return ProductMediaOut.model_validate(media)


@router.delete("/products/{product_id}/media/{media_id}", response_model=ProductMediaOut)
async def delete_product_media(
    product_id: str,
    media_id: str,
    session: AsyncSession = Depends(get_session),
) -> ProductMediaOut:
    parsed_product = _coerce_uuid(product_id)
    parsed_media = _coerce_uuid(media_id)
    if not parsed_product or not parsed_media:
        raise HTTPException(status_code=400, detail="Invalid id.")

    result = await session.execute(
        select(ProductMedia).where(
            ProductMedia.id == parsed_media,
            ProductMedia.product_id == parsed_product,
        )
    )
    media = result.scalar_one_or_none()
    if not media:
        raise HTTPException(status_code=404, detail="Media not found.")

    if media.is_primary:
        product_result = await session.execute(select(Product).where(Product.id == parsed_product))
        product = product_result.scalar_one_or_none()
        if product and product.image_url == media.url:
            product.image_url = None

    await session.delete(media)
    await session.flush()

    if media.is_primary:
        next_image_result = await session.execute(
            select(ProductMedia)
            .where(
                ProductMedia.product_id == parsed_product,
                ProductMedia.media_type == "image",
            )
            .order_by(ProductMedia.order_index.asc(), ProductMedia.created_at.asc())
            .limit(1)
        )
        next_image = next_image_result.scalar_one_or_none()
        if next_image:
            product_result = await session.execute(select(Product).where(Product.id == parsed_product))
            product = product_result.scalar_one_or_none()
            if product:
                product.image_url = next_image.url
                next_image.is_primary = True

    await session.commit()
    return ProductMediaOut.model_validate(media)


@router.post("/credits/grant", response_model=AdminUserOut)
async def grant_credits(
    payload: CreditGrantRequest,
    session: AsyncSession = Depends(get_session),
) -> AdminUserOut:
    try:
        user_id = uuid.UUID(payload.user_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid user id.") from exc

    result = await session.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    user.credits += payload.delta
    session.add(
        CreditTransaction(
            user_id=user.id,
            delta=payload.delta,
            reason=payload.reason,
            source=payload.source,
            reference_id=payload.reference_id,
        )
    )

    await session.commit()
    await session.refresh(user)
    return AdminUserOut.model_validate(user)
