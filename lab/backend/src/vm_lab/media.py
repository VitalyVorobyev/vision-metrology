"""Rendered image tiers: thumb (256px WebP), preview (1024px WebP), full (native PNG).

Same tier pattern as the anomaly lab: `thumb`/`preview` are cached to disk on first
render and served with a strong ETag derived from content (`sha256-tier`); `full` is
rendered on demand and never cached — it is the pixel-exact source used for
teach/find/measure alignment, viewed rarely enough that caching it is not worth the disk.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from PIL import Image as PILImage

from vm_lab.config import settings
from vm_lab.store import ImageRecord


class Tier(StrEnum):
    THUMB = "thumb"
    PREVIEW = "preview"
    FULL = "full"


@dataclass(frozen=True)
class TierSpec:
    long_edge: int | None
    media_type: str
    cached: bool


TIERS: dict[Tier, TierSpec] = {
    Tier.THUMB: TierSpec(long_edge=256, media_type="image/webp", cached=True),
    Tier.PREVIEW: TierSpec(long_edge=1024, media_type="image/webp", cached=True),
    Tier.FULL: TierSpec(long_edge=None, media_type="image/png", cached=False),
}
_WEBP_QUALITY = {Tier.THUMB: 80, Tier.PREVIEW: 85}


def etag_for(image: ImageRecord, tier: Tier) -> str:
    return f'"{image.sha256[:16]}-{tier.value}"'


def _cache_path(image_id: str, tier: Tier) -> Path:
    return settings.thumbnails_dir / tier.value / f"{image_id}.webp"


def render(image: ImageRecord, tier: Tier) -> bytes:
    spec = TIERS[tier]
    source = PILImage.open(image.path)
    if spec.long_edge is None:
        buf = io.BytesIO()
        source.save(buf, "PNG", optimize=False)
        return buf.getvalue()
    source = source.copy()
    source.thumbnail((spec.long_edge, spec.long_edge), PILImage.Resampling.LANCZOS)
    buf = io.BytesIO()
    source.save(buf, "WEBP", quality=_WEBP_QUALITY[tier])
    return buf.getvalue()


def ensure_cached(image: ImageRecord, tier: Tier) -> Path:
    if not TIERS[tier].cached:
        msg = f"the {tier.value} tier is rendered on demand and never cached"
        raise ValueError(msg)
    path = _cache_path(image.id, tier)
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = render(image, tier)
    scratch = path.with_name(f".{path.name}.partial")
    scratch.write_bytes(payload)
    scratch.replace(path)
    return path
