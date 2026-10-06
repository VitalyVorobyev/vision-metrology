"""`MeasureConfigIn.polarity` over the API: the lab's names reach `vm.MeasureConfig`.

A bright disc on a dark ground, measured with radial calipers that scan outward: every
edge goes from bright to dark. The desktop command maps the same names the same way.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image as PILImage

from vm_lab.app import create_app
from vm_lab.routers.measure import _measure_config
from vm_lab.schemas import MeasureConfigIn, MeasureObjectIn
from vm_lab.store import store

CENTER, RADIUS, SIZE = (64.0, 64.0), 24.0, 128


def _disc_png() -> bytes:
    ys, xs = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    cover = np.clip(RADIUS + 0.5 - np.hypot(xs - CENTER[0], ys - CENTER[1]), 0.0, 1.0)
    pixels = (20.0 + 180.0 * cover).round().astype(np.uint8)
    buf = io.BytesIO()
    PILImage.fromarray(pixels, mode="L").save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture()
def taught(tmp_path, monkeypatch) -> tuple[TestClient, str, str]:
    """A client on an isolated store, the disc's image id and a model taught on it."""
    from vm_lab.config import Settings

    isolated = Settings(data_dir=tmp_path)
    monkeypatch.setattr("vm_lab.store.settings", isolated)
    monkeypatch.setattr("vm_lab.media.settings", isolated)
    store.images.clear()
    store.models.clear()
    store.calibrations.clear()

    client = TestClient(create_app())
    resp = client.post("/api/images", files={"file": ("disc.png", _disc_png(), "image/png")})
    assert resp.status_code == 200, resp.text
    image_id = resp.json()["id"]
    resp = client.post(
        "/api/models", json={"image_id": image_id, "roi": (24.0, 24.0, 80.0, 80.0), "min_contrast": 0.1}
    )
    assert resp.status_code == 200, resp.text
    return client, image_id, resp.json()["id"]


def _measure(taught: tuple[TestClient, str, str], polarity: str | None) -> dict:
    client, image_id, model_id = taught
    measure = {} if polarity is None else {"polarity": polarity}
    resp = client.post(
        "/api/measure",
        json={
            "image_id": image_id,
            "model_id": model_id,
            "min_score": 0.5,
            "objects": [
                {
                    "kind": "circle",
                    "cx": CENTER[0],
                    "cy": CENTER[1],
                    "r": RADIUS,
                    "n_calipers": 8,
                    "caliper_len": 8.0,
                    "caliper_width": 4.0,
                    "measure": measure,
                }
            ],
        },
    )
    assert resp.status_code == 200, resp.text
    (result,) = resp.json()["objects"]
    return result


@pytest.mark.parametrize(
    ("polarity", "native"),
    [("bright_to_dark", "falling"), ("dark_to_bright", "rising"), ("either", "any"), (None, "any")],
)
def test_each_lab_polarity_maps_to_its_vm_name(polarity: str | None, native: str) -> None:
    obj = MeasureObjectIn(kind="circle", measure=MeasureConfigIn(polarity=polarity))
    assert _measure_config(obj).polarity == native


@pytest.mark.parametrize("polarity", ["bright_to_dark", "either", None])
def test_a_polarity_that_admits_the_rim_measures_it(taught, polarity: str | None) -> None:
    result = _measure(taught, polarity)
    assert result["kind"] == "circle"
    assert [c["status"] for c in result["calipers"]] == ["hit"] * 8
    assert all(c["profile"]["edges"][0]["polarity"] == "falling" for c in result["calipers"])
    assert result["circle_r"] == pytest.approx(RADIUS, abs=0.1)


def test_the_opposite_polarity_rejects_every_caliper(taught) -> None:
    result = _measure(taught, "dark_to_bright")
    assert result["kind"] == "error"
    assert [(c["status"], c["reason"]) for c in result["calipers"]] == [
        ("rejected", "wrong_polarity")
    ] * 8
