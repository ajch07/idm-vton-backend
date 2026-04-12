import uuid
from typing import List

import razorpay
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import get_settings
from ..dependencies import get_current_user, get_session
from ..models import CreditTransaction, Order, Product, User
from ..schemas import CreateOrderRequest, CreateOrderResponse, RazorpayVerifyRequest

settings = get_settings()

router = APIRouter(prefix="/api/payments", tags=["payments"])


def get_razorpay_client() -> razorpay.Client:
    if not settings.razorpay_key_id or not settings.razorpay_key_secret:
        raise HTTPException(status_code=500, detail="Razorpay keys are not set.")
    return razorpay.Client(auth=(settings.razorpay_key_id, settings.razorpay_key_secret))


def _normalize_currency(value: str) -> str:
    return (value or "INR").upper()


@router.post("/razorpay/order", response_model=CreateOrderResponse)
async def create_razorpay_order(
    payload: CreateOrderRequest,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> CreateOrderResponse:
    if not payload.items:
        raise HTTPException(status_code=400, detail="No items provided.")

    product_ids = [item.product_id for item in payload.items]
    result = await session.execute(select(Product).where(Product.slug.in_(product_ids)))
    products = {product.slug: product for product in result.scalars().all()}

    if len(products) != len(set(product_ids)):
        raise HTTPException(status_code=400, detail="One or more products are invalid.")

    total_amount = 0
    order_items: List[dict] = []
    total_quantity = 0

    for item in payload.items:
        product = products[item.product_id]
        if not product.is_active:
            raise HTTPException(status_code=400, detail=f"{product.name} is not available.")
        if product.stock < item.quantity:
            raise HTTPException(status_code=409, detail=f"Insufficient stock for {product.name}.")

        line_total = product.price * item.quantity
        total_amount += line_total
        total_quantity += item.quantity
        order_items.append(
            {
                "product_id": str(product.id),
                "slug": product.slug,
                "name": product.name,
                "price": product.price,
                "quantity": item.quantity,
            }
        )

    if total_amount <= 0:
        raise HTTPException(status_code=400, detail="Invalid order amount.")

    currency = _normalize_currency(payload.currency)
    razorpay_client = get_razorpay_client()

    receipt = f"order_{uuid.uuid4().hex[:10]}"
    razorpay_order = razorpay_client.order.create(
        {
            "amount": total_amount * 100,
            "currency": currency,
            "receipt": receipt,
            "notes": {"user_id": str(current_user.id)},
        }
    )

    order = Order(
        user_id=current_user.id,
        amount=total_amount,
        currency=currency,
        status="created",
        razorpay_order_id=razorpay_order.get("id"),
        receipt=receipt,
        items=order_items,
        credits_awarded=settings.credits_per_purchase * total_quantity,
    )
    session.add(order)
    await session.commit()
    await session.refresh(order)

    return CreateOrderResponse(
        order_id=str(order.id),
        razorpay_order_id=order.razorpay_order_id or "",
        amount=order.amount,
        currency=order.currency,
        key_id=settings.razorpay_key_id,
    )


@router.post("/razorpay/verify")
async def verify_razorpay_payment(
    payload: RazorpayVerifyRequest,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> dict:
    try:
        order_id = uuid.UUID(payload.order_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid order id.") from exc

    result = await session.execute(select(Order).where(Order.id == order_id))
    order = result.scalar_one_or_none()
    if not order or order.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Order not found.")

    if order.status == "paid":
        return {"status": "already_paid"}

    if order.razorpay_order_id != payload.razorpay_order_id:
        raise HTTPException(status_code=400, detail="Order mismatch.")

    razorpay_client = get_razorpay_client()
    try:
        razorpay_client.utility.verify_payment_signature(
            {
                "razorpay_order_id": payload.razorpay_order_id,
                "razorpay_payment_id": payload.razorpay_payment_id,
                "razorpay_signature": payload.razorpay_signature,
            }
        )
    except razorpay.errors.SignatureVerificationError as exc:
        raise HTTPException(status_code=400, detail="Invalid payment signature.") from exc

    order.status = "paid"
    order.razorpay_payment_id = payload.razorpay_payment_id
    order.razorpay_signature = payload.razorpay_signature

    total_quantity = 0
    if order.items:
        for item in order.items:
            slug = item.get("slug")
            quantity = int(item.get("quantity", 0))
            if not slug or quantity <= 0:
                continue
            product_result = await session.execute(select(Product).where(Product.slug == slug))
            product = product_result.scalar_one_or_none()
            if product:
                product.stock = max(product.stock - quantity, 0)
            total_quantity += quantity

    if total_quantity > 0:
        credits_awarded = settings.credits_per_purchase * total_quantity
        current_user.credits += credits_awarded
        order.credits_awarded = credits_awarded
        session.add(
            CreditTransaction(
                user_id=current_user.id,
                delta=credits_awarded,
                reason="purchase",
                source="razorpay",
                reference_id=payload.razorpay_payment_id,
            )
        )

    await session.commit()
    return {"status": "paid"}
