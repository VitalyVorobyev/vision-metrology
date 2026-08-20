"""POST/GET /api/calibration — upload a calibration-rs `RigExtrinsicsExport` or a
`table_calibration` `calibration.json`, either format, detected by shape.

Stored like models/images: the raw JSON on disk plus a metadata sidecar, so a server
restart rebuilds the registry (`store.rehydrate`). The parsed `(CameraModel, pose)` list
is what `/api/measure` reads to convert fitted geometry to millimetres.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, UploadFile

from vm_lab.schemas import CalibrationOut
from vm_lab.store import store

router = APIRouter(prefix="/api/calibration", tags=["calibration"])


def _to_out(rec) -> CalibrationOut:  # noqa: ANN001 - CalibrationRecord, avoids a circular import at type level
    return CalibrationOut(id=rec.id, filename=rec.filename, format=rec.format, n_cameras=rec.n_cameras)


@router.post("", response_model=CalibrationOut)
async def upload_calibration(file: UploadFile) -> CalibrationOut:
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "empty upload")
    try:
        json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(400, f"not valid JSON: {exc}") from None
    try:
        rec = store.add_calibration(file.filename or "calibration.json", raw)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    return _to_out(rec)


@router.get("", response_model=list[CalibrationOut])
async def list_calibrations() -> list[CalibrationOut]:
    return [_to_out(r) for r in store.calibrations.values()]
