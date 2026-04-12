import uuid
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..dependencies import get_session, require_admin
from ..models import Product, ProductMedia
from ..schemas import ProductCreate, ProductMediaOut, ProductOut, ProductUpdate
from ..utils.slug import slugify

router = APIRouter(tags=["products"])


def _coerce_uuid(value: str) -> uuid.UUID | None:
    try:
        return uuid.UUID(value)
    except ValueError:
        return None


@router.get("/api/products", response_model=List[ProductOut])
async def list_products(session: AsyncSession = Depends(get_session)) -> List[ProductOut]:
    result = await session.execute(
        select(Product).where(Product.is_active.is_(True)).order_by(Product.created_at.desc())
    )
    products = result.scalars().all()
    return [ProductOut.model_validate(product) for product in products]


@router.get("/api/admin/products", response_model=List[ProductOut], dependencies=[Depends(require_admin)])
async def list_all_products(session: AsyncSession = Depends(get_session)) -> List[ProductOut]:
    result = await session.execute(select(Product).order_by(Product.created_at.desc()))
    products = result.scalars().all()
    return [ProductOut.model_validate(product) for product in products]


@router.get("/api/products/{slug}", response_model=ProductOut)
async def get_product(slug: str, session: AsyncSession = Depends(get_session)) -> ProductOut:
    result = await session.execute(select(Product).where(Product.slug == slug))
    product = result.scalar_one_or_none()
    if not product or not product.is_active:
        raise HTTPException(status_code=404, detail="Product not found.")
    return ProductOut.model_validate(product)


@router.get("/api/products/{slug}/media", response_model=List[ProductMediaOut])
async def get_product_media(slug: str, session: AsyncSession = Depends(get_session)) -> List[ProductMediaOut]:
    result = await session.execute(select(Product).where(Product.slug == slug))
    product = result.scalar_one_or_none()
    if not product or not product.is_active:
        raise HTTPException(status_code=404, detail="Product not found.")

    media_result = await session.execute(
        select(ProductMedia)
        .where(ProductMedia.product_id == product.id)
        .order_by(ProductMedia.order_index.asc(), ProductMedia.created_at.asc())
    )
    media_items = media_result.scalars().all()
    return [ProductMediaOut.model_validate(item) for item in media_items]


@router.post("/api/admin/products", response_model=ProductOut, dependencies=[Depends(require_admin)])
async def create_product(
    payload: ProductCreate,
    session: AsyncSession = Depends(get_session),
) -> ProductOut:
    slug = payload.slug or slugify(payload.name)
    existing = await session.execute(select(Product).where(Product.slug == slug))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Product slug already exists.")

    product = Product(
        slug=slug,
        name=payload.name,
        category=payload.category,
        description=payload.description,
        details=payload.details,
        price=payload.price,
        currency=payload.currency,
        image_url=payload.image_url,
        stock=payload.stock,
        is_active=payload.is_active,
    )
    session.add(product)
    await session.commit()
    await session.refresh(product)
    return ProductOut.model_validate(product)


@router.put("/api/admin/products/{product_id}", response_model=ProductOut, dependencies=[Depends(require_admin)])
async def update_product(
    product_id: str,
    payload: ProductUpdate,
    session: AsyncSession = Depends(get_session),
) -> ProductOut:
    product = None
    parsed_id = _coerce_uuid(product_id)
    if parsed_id:
        result = await session.execute(select(Product).where(Product.id == parsed_id))
        product = result.scalar_one_or_none()
    if not product:
        result = await session.execute(select(Product).where(Product.slug == product_id))
        product = result.scalar_one_or_none()

    if not product:
        raise HTTPException(status_code=404, detail="Product not found.")

    updates = payload.model_dump(exclude_unset=True)
    if "slug" in updates and updates["slug"]:
        slug = updates["slug"]
        existing = await session.execute(select(Product).where(Product.slug == slug, Product.id != product.id))
        if existing.scalar_one_or_none():
            raise HTTPException(status_code=400, detail="Product slug already exists.")

    for key, value in updates.items():
        setattr(product, key, value)

    await session.commit()
    await session.refresh(product)
    return ProductOut.model_validate(product)


@router.delete("/api/admin/products/{product_id}", response_model=ProductOut, dependencies=[Depends(require_admin)])
async def delete_product(product_id: str, session: AsyncSession = Depends(get_session)) -> ProductOut:
    product = None
    parsed_id = _coerce_uuid(product_id)
    if parsed_id:
        result = await session.execute(select(Product).where(Product.id == parsed_id))
        product = result.scalar_one_or_none()
    if not product:
        result = await session.execute(select(Product).where(Product.slug == product_id))
        product = result.scalar_one_or_none()

    if not product:
        raise HTTPException(status_code=404, detail="Product not found.")

    product.is_active = False
    await session.commit()
    await session.refresh(product)
    return ProductOut.model_validate(product)
