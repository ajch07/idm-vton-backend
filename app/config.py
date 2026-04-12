import os
from functools import lru_cache
from typing import List

from dotenv import load_dotenv

load_dotenv()


def _env_str(key: str, default: str = "") -> str:
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip()


def _env_int(key: str, default: int) -> int:
    value = os.getenv(key)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_bool(key: str, default: bool = False) -> bool:
    value = os.getenv(key)
    if value is None or value == "":
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def parse_origins(raw: str) -> List[str]:
    if not raw:
        return []
    return [
        origin.strip().rstrip("/")
        for origin in raw.split(",")
        if origin.strip()
    ]


class Settings:
    app_title = _env_str("APP_TITLE", "Virtual Try-On API")

    database_url = _env_str("DATABASE_URL")

    jwt_secret = _env_str("JWT_SECRET")
    jwt_alg = _env_str("JWT_ALG", "HS256")
    access_token_expire_min = _env_int("ACCESS_TOKEN_EXPIRE_MIN", 60 * 24 * 7)

    google_client_id = _env_str("GOOGLE_CLIENT_ID")

    razorpay_key_id = _env_str("RAZORPAY_KEY_ID")
    razorpay_key_secret = _env_str("RAZORPAY_KEY_SECRET")

    credits_signup = _env_int("CREDITS_SIGNUP", 5)
    credits_per_tryon = _env_int("CREDITS_PER_TRYON", 1)
    credits_per_purchase = _env_int("CREDITS_PER_PURCHASE", 2)
    credits_story_bonus = _env_int("CREDITS_STORY_BONUS", 3)

    admin_emails = [
        email.strip().lower()
        for email in _env_str("ADMIN_EMAILS", "").split(",")
        if email.strip()
    ]

    cors_origins = parse_origins(_env_str("CORS_ORIGINS", ""))

    # Try-On Service Configuration
    tryon_service = _env_str("TRYON_SERVICE", "fal")  # "fal", "runpod", or "hybrid"

    # FAL Configuration
    fal_api_key = _env_str("FAL_API_KEY")
    fal_model = _env_str("FAL_MODEL", "fal-ai/nano-banana")
    fal_endpoint = _env_str("FAL_ENDPOINT")
    fal_prompt_field = _env_str("FAL_PROMPT_FIELD", "prompt")
    fal_negative_field = _env_str("FAL_NEGATIVE_FIELD", "negative_prompt")
    fal_image_field = _env_str("FAL_IMAGE_FIELD", "image_urls")
    fal_user_field = _env_str("FAL_USER_FIELD")
    fal_garment_field = _env_str("FAL_GARMENT_FIELD")
    fal_extra_json = _env_str("FAL_EXTRA_JSON")

    # Runpod Configuration
    runpod_endpoint = _env_str("RUNPOD_ENDPOINT")
    runpod_api_key = _env_str("RUNPOD_API_KEY")
    runpod_timeout_seconds = _env_int("RUNPOD_TIMEOUT_SECONDS", 60)

    max_upload_mb = _env_int("MAX_UPLOAD_MB", 12)

    supabase_url = _env_str("SUPABASE_URL")
    supabase_service_key = _env_str("SUPABASE_SERVICE_ROLE_KEY")
    supabase_bucket = _env_str("SUPABASE_STORAGE_BUCKET", "product-media")

    enable_signup = _env_bool("ENABLE_SIGNUP", True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
