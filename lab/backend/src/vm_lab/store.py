"""In-memory registries backed by files on disk under `data/` (no database, no jobs).

Uploaded images and taught models are durable — a PNG/model file plus a JSON sidecar of
metadata, so a server restart rebuilds the registries from disk (`rehydrate`). Tier caches
(thumb/preview) are pure derived data, kept in `data/thumbnails/` and safe to delete.

Content-addressed: an image's id is assigned in upload order, but its `sha256` (of the
grayscale-converted pixel bytes) is what the tier ETag keys on — see `media.py`.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

import vision_metrology as vm

from vm_lab.config import settings


@dataclass(frozen=True)
class ImageRecord:
    id: str
    filename: str
    width: int
    height: int
    sha256: str

    @property
    def path(self) -> Path:
        return settings.images_dir / f"{self.id}.png"

    @property
    def sidecar(self) -> Path:
        return settings.images_dir / f"{self.id}.json"


def detect_calibration_format(doc: dict) -> str:
    """Detect which of the two supported calibration JSON shapes `doc` is,
    by field presence — there is no shared envelope between them.

    `RigExtrinsicsExport` (calibration-rs) always carries a `kind` field;
    `table_calibration`'s `calibration.json` has no such tag, but always
    carries top-level `intrinsic`/`extrinsic` maps.
    """
    if doc.get("kind") == "rig_extrinsics":
        return "rig_extrinsics"
    if "intrinsic" in doc and "extrinsic" in doc:
        return "table_calibration"
    raise ValueError(
        'unrecognized calibration format: expected a RigExtrinsicsExport document '
        '(kind="rig_extrinsics") or a table_calibration calibration.json '
        '(top-level "intrinsic"/"extrinsic")'
    )


@dataclass(frozen=True)
class CalibrationRecord:
    id: str
    filename: str
    format: str
    n_cameras: int

    @property
    def path(self) -> Path:
        return settings.calibrations_dir / f"{self.id}.json"

    @property
    def sidecar(self) -> Path:
        return settings.calibrations_dir / f"{self.id}.meta.json"


@dataclass(frozen=True)
class ModelRecord:
    id: str
    image_id: str
    roi: tuple[float, float, float, float]
    min_contrast: float
    num_levels: int | None
    origin: tuple[float, float]
    num_levels_built: int
    point_counts: list[int]

    @property
    def path(self) -> Path:
        return settings.models_dir / f"{self.id}.bin"

    @property
    def sidecar(self) -> Path:
        return settings.models_dir / f"{self.id}.json"


class Store:
    """Process-wide registries. One instance (`store`, below); FastAPI depends on it
    directly rather than through a DI container — this is a single-user local workbench,
    not a multi-tenant service."""

    def __init__(self) -> None:
        self.images: dict[str, ImageRecord] = {}
        self.models: dict[str, ModelRecord] = {}
        self.calibrations: dict[str, CalibrationRecord] = {}
        self._model_objs: dict[str, vm.ShapeModel] = {}
        # camera/pose pairs, keyed by calibration id: (CameraModel, pose[4,4]).
        self._calibration_objs: dict[str, list[tuple[vm.CameraModel, np.ndarray]]] = {}
        self._next_image = 1
        self._next_model = 1
        self._next_calibration = 1

    # -- images ----------------------------------------------------------------

    def add_image(self, filename: str, raw_bytes: bytes) -> ImageRecord:
        img = PILImage.open(io.BytesIO(raw_bytes))
        img = img.convert("L")  # grayscale-convert on ingest, per the lab's scope
        sha256 = hashlib.sha256(img.tobytes()).hexdigest()
        image_id = f"img-{self._next_image}"
        self._next_image += 1
        rec = ImageRecord(
            id=image_id, filename=filename, width=img.width, height=img.height, sha256=sha256
        )
        rec.path.parent.mkdir(parents=True, exist_ok=True)
        img.save(rec.path, "PNG")
        rec.sidecar.write_text(json.dumps(asdict(rec)))
        self.images[image_id] = rec
        return rec

    def load_array(self, image_id: str) -> np.ndarray:
        rec = self.images[image_id]
        return np.asarray(PILImage.open(rec.path).convert("L"), dtype=np.uint8)

    # -- models ------------------------------------------------------------------

    def add_model(
        self,
        image_id: str,
        roi: tuple[float, float, float, float],
        min_contrast: float,
        num_levels: int | None,
        model: vm.ShapeModel,
    ) -> ModelRecord:
        model_id = f"model-{self._next_model}"
        self._next_model += 1
        rec = ModelRecord(
            id=model_id,
            image_id=image_id,
            roi=roi,
            min_contrast=min_contrast,
            num_levels=num_levels,
            origin=model.origin,
            num_levels_built=model.num_levels,
            point_counts=model.point_counts,
        )
        rec.path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(rec.path))
        rec.sidecar.write_text(json.dumps(asdict(rec)))
        self.models[model_id] = rec
        self._model_objs[model_id] = model
        return rec

    def get_model(self, model_id: str) -> vm.ShapeModel:
        if model_id not in self._model_objs:
            rec = self.models[model_id]
            self._model_objs[model_id] = vm.ShapeModel.load(str(rec.path))
        return self._model_objs[model_id]

    # -- calibrations --------------------------------------------------------------

    def add_calibration(self, filename: str, raw_bytes: bytes) -> CalibrationRecord:
        doc = json.loads(raw_bytes)
        fmt = detect_calibration_format(doc)
        loader = vm.load_rig_extrinsics if fmt == "rig_extrinsics" else vm.load_table_calibration
        cameras = loader(raw_bytes)  # raises ValueError on a malformed document

        calibration_id = f"cal-{self._next_calibration}"
        self._next_calibration += 1
        rec = CalibrationRecord(
            id=calibration_id, filename=filename, format=fmt, n_cameras=len(cameras)
        )
        rec.path.parent.mkdir(parents=True, exist_ok=True)
        rec.path.write_bytes(raw_bytes)
        rec.sidecar.write_text(json.dumps(asdict(rec)))
        self.calibrations[calibration_id] = rec
        self._calibration_objs[calibration_id] = cameras
        return rec

    def get_calibration(self, calibration_id: str) -> list[tuple[vm.CameraModel, np.ndarray]]:
        if calibration_id not in self._calibration_objs:
            rec = self.calibrations[calibration_id]
            raw_bytes = rec.path.read_bytes()
            loader = vm.load_rig_extrinsics if rec.format == "rig_extrinsics" else vm.load_table_calibration
            self._calibration_objs[calibration_id] = loader(raw_bytes)
        return self._calibration_objs[calibration_id]

    # -- startup rehydration -------------------------------------------------------

    def rehydrate(self) -> None:
        settings.ensure()
        for sidecar in sorted(settings.images_dir.glob("img-*.json")):
            data = json.loads(sidecar.read_text())
            rec = ImageRecord(**data)
            if rec.path.is_file():
                self.images[rec.id] = rec
                self._next_image = max(self._next_image, int(rec.id.split("-")[1]) + 1)
        for sidecar in sorted(settings.models_dir.glob("model-*.json")):
            data = json.loads(sidecar.read_text())
            data["roi"] = tuple(data["roi"])
            data["origin"] = tuple(data["origin"])
            rec = ModelRecord(**data)
            if rec.path.is_file():
                self.models[rec.id] = rec
                self._next_model = max(self._next_model, int(rec.id.split("-")[1]) + 1)
        for sidecar in sorted(settings.calibrations_dir.glob("cal-*.meta.json")):
            data = json.loads(sidecar.read_text())
            rec = CalibrationRecord(**data)
            if rec.path.is_file():
                self.calibrations[rec.id] = rec
                self._next_calibration = max(self._next_calibration, int(rec.id.split("-")[1]) + 1)


store = Store()
