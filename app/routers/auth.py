from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..auth import create_access_token, hash_password, verify_google_token, verify_password
from ..config import get_settings
from ..dependencies import get_current_user, get_session
from ..models import CreditTransaction, User
from ..schemas import (
    GoogleAuthRequest,
    TokenResponse,
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


@router.post("/google", response_model=TokenResponse)
async def google_auth(payload: GoogleAuthRequest, session: AsyncSession = Depends(get_session)) -> TokenResponse:
    info = verify_google_token(payload.id_token)

    email = info.get("email")
    if not email:
        raise HTTPException(status_code=400, detail="Google account missing email.")

    email = _normalize_email(email)
    google_sub = info.get("sub")
    name = info.get("name") or info.get("given_name")

    result = await session.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()

    if user:
        if google_sub and not user.google_sub:
            user.google_sub = google_sub
        if name and not user.name:
            user.name = name
        await session.commit()
        await session.refresh(user)
        return await _issue_token(user)

    if not settings.enable_signup:
        raise HTTPException(status_code=403, detail="Signup is disabled.")

    user = User(
        email=email,
        google_sub=google_sub,
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
    return await _issue_token(user)


@router.get("/me", response_model=UserOut)
async def me(current_user: User = Depends(get_current_user)) -> UserOut:
    return UserOut.model_validate(current_user)
