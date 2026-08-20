"""Filesystem layout for the lab's in-memory-registry + files-on-disk storage.

No database: registries live in process memory (`store.py`) and are rebuilt from these
directories on startup where content is durable (uploaded images, saved models). Tier
caches (thumb/preview) are pure derived data and are safe to lose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

_LAB_ROOT = Path(__file__).resolve().parents[3]  # lab/backend/src/vm_lab -> lab/


#: The frontend's dev server (vite.config.ts: port 5174, strictPort) and the same host
#: under 127.0.0.1 — browsers treat localhost and 127.0.0.1 as distinct CORS origins.
DEFAULT_CORS_ORIGINS = ("http://localhost:5174", "http://127.0.0.1:5174")


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: _LAB_ROOT / "backend" / "data")
    cors_origins: tuple[str, ...] = DEFAULT_CORS_ORIGINS

    @property
    def images_dir(self) -> Path:
        return self.data_dir / "images"

    @property
    def thumbnails_dir(self) -> Path:
        return self.data_dir / "thumbnails"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def calibrations_dir(self) -> Path:
        return self.data_dir / "calibrations"

    def ensure(self) -> None:
        for d in (
            self.images_dir,
            self.thumbnails_dir / "thumb",
            self.thumbnails_dir / "preview",
            self.models_dir,
            self.calibrations_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
