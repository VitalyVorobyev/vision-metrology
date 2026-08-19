from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse

from vm_lab.media import TIERS, Tier, ensure_cached, etag_for, render
from vm_lab.schemas import ImageOut
from vm_lab.store import store

router = APIRouter(prefix="/api/images", tags=["images"])

_SUFFIXES = {"image/png": ".png", "image/bmp": ".bmp"}


@router.post("", response_model=ImageOut)
async def upload_image(file: UploadFile) -> ImageOut:
    if file.content_type not in ("image/png", "image/bmp", "image/x-ms-bmp"):
        raise HTTPException(415, f"unsupported content type: {file.content_type}")
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "empty upload")
    rec = store.add_image(file.filename or "upload", raw)
    return ImageOut(id=rec.id, filename=rec.filename, width=rec.width, height=rec.height, sha256=rec.sha256)


@router.get("", response_model=list[ImageOut])
async def list_images() -> list[ImageOut]:
    return [
        ImageOut(id=r.id, filename=r.filename, width=r.width, height=r.height, sha256=r.sha256)
        for r in store.images.values()
    ]


@router.get("/{image_id}/{tier}")
async def get_image_tier(image_id: str, tier: Tier, request: Request) -> Response:
    if image_id not in store.images:
        raise HTTPException(404, f"no such image: {image_id}")
    image = store.images[image_id]
    spec = TIERS[tier]
    tag = etag_for(image, tier)
    if request.headers.get("if-none-match") == tag:
        return Response(status_code=304)

    if spec.cached:
        path = ensure_cached(image, tier)
        return FileResponse(path, media_type=spec.media_type, headers={"ETag": tag})

    payload = render(image, tier)
    return Response(content=payload, media_type=spec.media_type, headers={"ETag": tag})
