"""FastAPI app: sync REST, no jobs/websockets/DB. See lab/README.md."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from vm_lab.config import settings
from vm_lab.routers import calibration, displacement, find, images, measure, models, rectify
from vm_lab.store import store


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    store.rehydrate()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Visual Metrology Lab", version="0.1.0", lifespan=_lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(images.router)
    app.include_router(models.router)
    app.include_router(calibration.router)
    app.include_router(find.router)
    app.include_router(measure.router)
    app.include_router(rectify.router)
    app.include_router(displacement.router)

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
