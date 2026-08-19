from __future__ import annotations

from fastapi import APIRouter, HTTPException

import vision_metrology as vm

from vm_lab.schemas import ModelCreateRequest, ModelOut
from vm_lab.store import store

router = APIRouter(prefix="/api/models", tags=["models"])


def _to_out(rec) -> ModelOut:  # noqa: ANN001 - ModelRecord, avoids a circular import at type level
    return ModelOut(
        id=rec.id,
        image_id=rec.image_id,
        roi=rec.roi,
        min_contrast=rec.min_contrast,
        num_levels=rec.num_levels,
        origin=rec.origin,
        num_levels_built=rec.num_levels_built,
        point_counts=rec.point_counts,
    )


@router.post("", response_model=ModelOut)
async def create_model(req: ModelCreateRequest) -> ModelOut:
    if req.image_id not in store.images:
        raise HTTPException(404, f"no such image: {req.image_id}")
    img = store.load_array(req.image_id)
    config = vm.ShapeModelConfig(
        num_levels=req.num_levels,
        min_contrast=vm.Contrast.fraction_of_range(req.min_contrast),
    )
    try:
        model = vm.ShapeModel(img, req.roi, config)
    except Exception as exc:  # noqa: BLE001 - surfaced verbatim to the caller
        raise HTTPException(400, f"teach failed: {exc}") from None
    rec = store.add_model(req.image_id, req.roi, req.min_contrast, req.num_levels, model)
    return _to_out(rec)


@router.get("", response_model=list[ModelOut])
async def list_models() -> list[ModelOut]:
    return [_to_out(r) for r in store.models.values()]
