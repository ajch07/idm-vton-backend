import uuid
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, EmailStr, Field


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    name: Optional[str] = None


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class GoogleAuthRequest(BaseModel):
    id_token: str


class UserOut(BaseModel):
    id: uuid.UUID
    email: EmailStr
    name: Optional[str] = None
    credits: int
    is_admin: bool
    created_at: datetime

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class ProductBase(BaseModel):
    slug: Optional[str] = None
    name: str
    category: Optional[str] = None
    description: Optional[str] = None
    details: Optional[List[str]] = None
    price: int
    currency: str = "INR"
    image_url: Optional[str] = None
    stock: int = 0
    is_active: bool = True


class ProductCreate(ProductBase):
    pass


class ProductUpdate(BaseModel):
    slug: Optional[str] = None
    name: Optional[str] = None
    category: Optional[str] = None
    description: Optional[str] = None
    details: Optional[List[str]] = None
    price: Optional[int] = None
    currency: Optional[str] = None
    image_url: Optional[str] = None
    stock: Optional[int] = None
    is_active: Optional[bool] = None


class ProductOut(ProductBase):
    id: uuid.UUID
    created_at: datetime

    class Config:
        from_attributes = True


class ProductMediaOut(BaseModel):
    id: uuid.UUID
    product_id: uuid.UUID
    media_type: str
    url: str
    order_index: int
    is_primary: bool
    created_at: datetime

    class Config:
        from_attributes = True


class OrderItemInput(BaseModel):
    product_id: str
    quantity: int = Field(gt=0)


class CreateOrderRequest(BaseModel):
    items: List[OrderItemInput]
    currency: str = "INR"


class CreateOrderResponse(BaseModel):
    order_id: str
    razorpay_order_id: str
    amount: int
    currency: str
    key_id: str


class RazorpayVerifyRequest(BaseModel):
    order_id: str
    razorpay_order_id: str
    razorpay_payment_id: str
    razorpay_signature: str


class OrderOut(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    amount: int
    currency: str
    status: str
    razorpay_order_id: Optional[str] = None
    razorpay_payment_id: Optional[str] = None
    receipt: Optional[str] = None
    items: Optional[List[dict]] = None
    credits_awarded: int
    created_at: datetime

    class Config:
        from_attributes = True


class AdminMetrics(BaseModel):
    users: int
    products: int
    orders: int
    revenue: int
    try_ons: int


class AdminUserOut(BaseModel):
    id: uuid.UUID
    email: EmailStr
    name: Optional[str] = None
    credits: int
    is_admin: bool
    created_at: datetime

    class Config:
        from_attributes = True


class AdminOrderOut(OrderOut):
    user_email: Optional[EmailStr] = None


class AdminActivityOut(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    user_email: Optional[EmailStr] = None
    delta: int
    reason: str
    source: str
    reference_id: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class CreditGrantRequest(BaseModel):
    user_id: str
    delta: int
    reason: str = "manual"
    source: str = "admin"
    reference_id: Optional[str] = None
