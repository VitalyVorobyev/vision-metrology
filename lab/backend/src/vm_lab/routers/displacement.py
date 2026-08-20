"""POST /api/displacement — pairwise subpixel inter-frame shift over an ordered image
sequence (`vm.displacement`, `crates/vision-metrology/src/corr/displacement.rs`).

Each consecutive pair in `image_ids` is tracked independently (the same `window` rect,
taken from the *first* image of each pair — i.e. always in the previous frame's own
coordinates, not re-anchored to `image_ids[0]`); the response also reports the running
cumulative trajectory the Motion tab plots. A pair that comes back below `min_score` is
a 422, naming which pair failed, rather than silently dropping it from the trajectory.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

import vision_metrology as vm

from vm_lab.schemas import DisplacementPairOut, DisplacementRequest, DisplacementResponse
from vm_lab.store import store

router = APIRouter(prefix="/api/displacement", tags=["displacement"])


def _refine(req: DisplacementRequest) -> vm.Refine:
    if req.refine == "none":
        return vm.Refine.none()
    return vm.Refine.lucas_kanade(iters=req.lk_iters)


@router.post("", response_model=DisplacementResponse)
async def compute_displacement(req: DisplacementRequest) -> DisplacementResponse:
    for image_id in req.image_ids:
        if image_id not in store.images:
            raise HTTPException(404, f"no such image: {image_id}")

    cfg = vm.DisplacementConfig(
        window=req.window,
        search=(req.search_x, req.search_y),
        refine=_refine(req),
        min_score=req.min_score,
    )

    arrays = [store.load_array(image_id) for image_id in req.image_ids]

    pairs: list[DisplacementPairOut] = []
    cumulative_x = [0.0]
    cumulative_y = [0.0]
    for i in range(len(arrays) - 1):
        prev_id, curr_id = req.image_ids[i], req.image_ids[i + 1]
        try:
            d = vm.displacement(arrays[i], arrays[i + 1], cfg)
        except ValueError as exc:
            raise HTTPException(
                422, f"displacement failed between {prev_id} and {curr_id}: {exc}"
            ) from exc
        pairs.append(
            DisplacementPairOut(from_image_id=prev_id, to_image_id=curr_id, dx=d.dx, dy=d.dy, score=d.score)
        )
        cumulative_x.append(cumulative_x[-1] + d.dx)
        cumulative_y.append(cumulative_y[-1] + d.dy)

    return DisplacementResponse(pairs=pairs, cumulative_x=cumulative_x, cumulative_y=cumulative_y)
