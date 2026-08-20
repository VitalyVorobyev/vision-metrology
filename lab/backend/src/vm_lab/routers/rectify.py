"""POST /api/rectify — find + rectify each match into a canonical, model-frame crop
(`vm.ShapeMatch.model_frame_map`). See `crates/vision-metrology/src/matching/crop.rs`
for the Rust-side semantics this mirrors: `crop.rect` is in **model-frame coordinates**
(the same frame `ModelOut.roi` is expressed in), the crop's pixel size is fixed by the
spec alone, and `normalize_scale=True` renders at model scale regardless of the found
scale.

Rectified crops are cached in memory, keyed by `(image_id, model_id, index)` — this is a
single-user local workbench (`store.py`), so the cache always holds the most recent
`rectify` call's crops for that image/model pair, same spirit as `media.py`'s thumbnail
cache but not disk-persisted (purely derived, and specific to one request's crop spec).
`GET` the `crop_url` each match reports to fetch its PNG, `images/{id}/{tier}`-style
with a content-derived ETag.
"""

from __future__ import annotations

import hashlib
import io

from fastapi import APIRouter, HTTPException, Request, Response
from PIL import Image as PILImage

import vision_metrology as vm

from vm_lab.routers.find import run_find
from vm_lab.schemas import FindRequest, RectifyMatchOut, RectifyRequest, RectifyResponse
from vm_lab.store import store

router = APIRouter(prefix="/api/rectify", tags=["rectify"])

# (image_id, model_id, index) -> PNG bytes. In-memory only, like the thumbnail cache,
# and overwritten wholesale by each POST for that (image_id, model_id) pair.
_crop_cache: dict[tuple[str, str, int], bytes] = {}


def _crop_png(dst) -> bytes:  # noqa: ANN001 - a (H, W) uint8 ndarray from Map.apply_with_mask
    buf = io.BytesIO()
    PILImage.fromarray(dst).save(buf, "PNG")
    return buf.getvalue()


def _etag(payload: bytes) -> str:
    return f'"{hashlib.sha256(payload).hexdigest()[:16]}"'


@router.post("", response_model=RectifyResponse)
async def rectify(req: RectifyRequest) -> RectifyResponse:
    if req.image_id not in store.images:
        raise HTTPException(404, f"no such image: {req.image_id}")
    if req.model_id not in store.models:
        raise HTTPException(404, f"no such model: {req.model_id}")

    find_req = FindRequest(
        image_id=req.image_id,
        model_id=req.model_id,
        min_score=req.min_score,
        max_matches=req.max_matches,
    )
    matches = run_find(find_req)

    img = store.load_array(req.image_id)
    spec = vm.CropSpec(req.crop.rect, req.crop.px_per_unit, req.crop.normalize_scale)
    width, height = spec.output_size

    # Drop this pair's stale crops before caching the new set, so a shrinking
    # match count does not leave orphaned indices servable.
    for key in [k for k in _crop_cache if k[0] == req.image_id and k[1] == req.model_id]:
        del _crop_cache[key]

    out: list[RectifyMatchOut] = []
    for i, m in enumerate(matches):
        crop_map = m.model_frame_map(spec)
        dst, mask = crop_map.apply_with_mask(img, border_mode="constant")
        validity = float((mask == 255).mean())
        _crop_cache[(req.image_id, req.model_id, i)] = _crop_png(dst)
        out.append(
            RectifyMatchOut(
                index=i,
                x=m.x,
                y=m.y,
                angle=m.angle,
                scale=m.scale,
                score=m.score,
                support=m.support,
                level=m.level,
                validity=validity,
                crop_url=f"/api/rectify/{req.image_id}/{req.model_id}/{i}",
                width=width,
                height=height,
            )
        )
    return RectifyResponse(width=width, height=height, matches=out)


@router.get("/{image_id}/{model_id}/{index}")
async def get_crop(image_id: str, model_id: str, index: int, request: Request) -> Response:
    key = (image_id, model_id, index)
    payload = _crop_cache.get(key)
    if payload is None:
        raise HTTPException(
            404, "no cached crop for this image/model/index -- POST /api/rectify first"
        )
    tag = _etag(payload)
    if request.headers.get("if-none-match") == tag:
        return Response(status_code=304)
    return Response(content=payload, media_type="image/png", headers={"ETag": tag})
