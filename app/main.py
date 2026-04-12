from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .database import init_db
from .routers import admin, auth, payments, products, tryon

settings = get_settings()


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_title, version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(auth.router)
    app.include_router(products.router)
    app.include_router(tryon.router)
    # app.include_router(payments.router)
    app.include_router(admin.router)

    @app.get("/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "mode": "fal",
            "model": settings.fal_model or settings.fal_endpoint,
        }

    @app.on_event("startup")
    async def startup() -> None:
        await init_db()

    return app


app = create_app()
