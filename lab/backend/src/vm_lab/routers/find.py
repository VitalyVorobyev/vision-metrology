from __future__ import annotations

import math

from fastapi import APIRouter, HTTPException

import vision_metrology as vm

from vm_lab.schemas import FindRequest, FindResponse, MatchOut
from vm_lab.store import store

router = APIRouter(prefix="/api/find", tags=["find"])


def run_find(req: FindRequest) -> list[vm.ShapeMatch]:
    if req.image_id not in store.images:
        raise HTTPException(404, f"no such image: {req.image_id}")
    if req.model_id not in store.models:
        raise HTTPException(404, f"no such model: {req.model_id}")
    img = store.load_array(req.image_id)
    model = store.get_model(req.model_id)
    angle_range = None
    if req.angle_range is not None:
        angle_range = (math.radians(req.angle_range[0]), math.radians(req.angle_range[1]))
    config = vm.ShapeSearchConfig(
        min_score=req.min_score,
        max_matches=req.max_matches,
        roi=req.roi,
        angle_range=angle_range,
    )
    matcher = vm.ShapeMatcher(config)
    return matcher.find(img, model)


@router.post("", response_model=FindResponse)
async def find(req: FindRequest) -> FindResponse:
    matches = run_find(req)
    return FindResponse(
        matches=[
            MatchOut(
                x=m.x, y=m.y, angle=m.angle, scale=m.scale, score=m.score, support=m.support, level=m.level
            )
            for m in matches
        ]
    )
