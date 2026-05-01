from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..auth import (
    create_access_token,
    fetch_supabase_user,
    hash_password,
    verify_password,
)
from ..config import get_settings
from ..dependencies import get_current_user, get_session
from ..models import CreditTransaction, User
from ..schemas import (
    TokenResponse,
    SupabaseAuthRequest,
    UserCreate,
    UserLogin,
    UserOut,
)

settings = get_settings()

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _normalize_email(email: str) -> str:
    return email.strip().lower()


async def _issue_token(user: User) -> TokenResponse:
    token = create_access_token(subject=str(user.id))
    return TokenResponse(access_token=token, user=UserOut.model_validate(user))


async def _find_or_create_supabase_user(
    *,
    email: str,
    supabase_user_id: str,
    name: str | None,
    session: AsyncSession,
) -> User:
    result = await session.execute(
        select(User).where(or_(User.supabase_user_id == supabase_user_id, User.email == email))
    )
    user = result.scalar_one_or_none()

    if user:
        if not user.supabase_user_id:
            user.supabase_user_id = supabase_user_id
        if name and not user.name:
            user.name = name
        await session.commit()
        await session.refresh(user)
        return user

    if not settings.enable_signup:
        raise HTTPException(status_code=403, detail="Signup is disabled.")

    user = User(
        email=email,
        supabase_user_id=supabase_user_id,
        name=name,
        is_admin=email in settings.admin_emails,
        credits=settings.credits_signup,
    )
    session.add(user)
    await session.flush()

    if settings.credits_signup:
        session.add(
            CreditTransaction(
                user_id=user.id,
                delta=settings.credits_signup,
                reason="signup",
                source="system",
            )
        )

    await session.commit()
    await session.refresh(user)
    return user


@router.post("/register", response_model=TokenResponse)
async def register(payload: UserCreate, session: AsyncSession = Depends(get_session)) -> TokenResponse:
    if not settings.enable_signup:
        raise HTTPException(status_code=403, detail="Signup is disabled.")

    email = _normalize_email(payload.email)
    existing = await session.execute(select(User).where(User.email == email))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Email already registered.")

    user = User(
        email=email,
        password_hash=hash_password(payload.password),
        name=payload.name,
        is_admin=email in settings.admin_emails,
        credits=settings.credits_signup,
    )
    session.add(user)
    await session.flush()

    if settings.credits_signup:
        session.add(
            CreditTransaction(
                user_id=user.id,
                delta=settings.credits_signup,
                reason="signup",
                source="system",
            )
        )

    await session.commit()
    await session.refresh(user)
    return await _issue_token(user)


@router.post("/login", response_model=TokenResponse)
async def login(payload: UserLogin, session: AsyncSession = Depends(get_session)) -> TokenResponse:
    email = _normalize_email(payload.email)
    result = await session.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    if not user or not user.password_hash:
        raise HTTPException(status_code=401, detail="Invalid credentials.")

    if not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials.")

    return await _issue_token(user)


@router.post("/supabase", response_model=TokenResponse)
async def supabase_auth(
    payload: SupabaseAuthRequest,
    session: AsyncSession = Depends(get_session),
) -> TokenResponse:
    info = await fetch_supabase_user(payload.access_token)

    email = info.get("email")
    supabase_user_id = info.get("id")
    if not email or not supabase_user_id:
        raise HTTPException(status_code=400, detail="Supabase session missing email or user id.")

    metadata = info.get("user_metadata") or {}
    name = (
        metadata.get("full_name")
        or metadata.get("name")
        or metadata.get("preferred_name")
    )

    user = await _find_or_create_supabase_user(
        email=_normalize_email(email),
        supabase_user_id=str(supabase_user_id),
        name=name,
        session=session,
    )
    return await _issue_token(user)


@router.get("/me", response_model=UserOut)
async def me(current_user: User = Depends(get_current_user)) -> UserOut:
    return UserOut.model_validate(current_user)
