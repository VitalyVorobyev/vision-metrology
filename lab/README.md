# Visual Metrology Lab

A local interactive workbench over the `vision_metrology` Python package. Where
`examples/python/` shows the library's chain in a script, the lab makes it a workbench:
upload an image, drag a box to **teach** a shape model, **find** it elsewhere in the
image, **measure** circles/lines against the found pose, and **judge** the fit — rms,
max deviation, per-caliper hit/reject, the raw intensity profile behind each hit. It is
the library's own end-to-end chain (`ShapeModel` → `ShapeMatcher` → `MetrologyModel`),
made visible.

## Architecture

```
lab/backend/    FastAPI, sync REST, in-memory registries + files under backend/data/
                (gitignored). No jobs, no websockets, no database — see "Out of scope".
lab/frontend/   Vite + React 19 + TypeScript strict + Tailwind v4, built on the shared
                @vitavision/lab-ui design system (ZoomPanCanvas, MeasureOverlay,
                LineProfile, SchemaForm, ...).
```

The backend does all the geometry: every response carries **source-image pixel
coordinates** — caliper boxes, measured points, fitted circles/lines, per-caliper
profiles — so the frontend only draws what it is told and never reconstructs a pose or
a fixture transform itself. (See `geometry.py` for the one place that transform lives,
and its docstring for a real gotcha it works around — see "Notes" below.)

`lab/` is **not** part of the Cargo workspace; it contains no Rust. It depends on
`vision_metrology`, the PyO3 bindings published from `crates/vm-python`, as an editable
local dependency.

## API

All routes under `/api`.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/images` | Upload a PNG/BMP; grayscale-converted on ingest. Returns `image_id`. |
| `GET` | `/images` | List uploaded images. |
| `GET` | `/images/{id}/{tier}` | `thumb` (256px WebP) / `preview` (1024px WebP) / `full` (native PNG). ETag by content hash; `thumb`/`preview` cached to disk, `full` rendered on demand. |
| `POST` | `/models` | Teach a `ShapeModel` from an ROI on an image (`min_contrast`, `num_levels`). Returns `model_id` + origin/point-count metadata. Saved to disk (`ShapeModel.save`). |
| `GET` | `/models` | List taught models. |
| `POST` | `/find` | `ShapeMatcher.find` against a model, with `roi`/`angle_range`/`min_score`. Returns matches (x, y, angle, scale, score, ...). |
| `POST` | `/measure` | Place a `MetrologyModel` (circle/line objects, nominal geometry in **model frame**) at a fixture pose — explicit, or auto-found — and measure. Returns per-object fit + rms/max_dev/n_used, a source-frame `overlay` array (`@vitavision/lab-ui`'s `MeasurePrimitive` shape), and per-caliper hit/reject reason + raw profile. |
| `GET` | `/health` | Liveness. |

## Run it

Two terminals, from `lab/`:

```bash
cd backend && uv sync && uv run uvicorn vm_lab.app:app --reload   # :8000
cd frontend && bun install && bun run dev                         # :5174, proxies /api -> :8000
```

Open http://localhost:5174. Data lands under `lab/backend/data/` (images, tier cache,
saved models) — gitignored, safe to delete to reset the workbench.

`uv sync` resolves `vision-metrology` as an editable path dependency onto
`crates/vm-python` (`pyproject.toml`'s `[tool.uv.sources]`), which `uv` builds through
maturin's PEP 517 hooks automatically — no separate `maturin develop` step needed. If
that ever fights `uv`'s resolver in some environment, fall back to building a wheel by
hand and installing it:

```bash
maturin build -m ../../crates/vm-python/Cargo.toml --release
uv pip install ../../crates/vm-python/target/wheels/vision_metrology-*.whl
```

## Tests

```bash
cd backend && uv run pytest       # one end-to-end smoke test: upload -> teach -> find -> measure a synthetic disc
cd frontend && bun run typecheck && bun run test && bun run build
```

## Out of MVP scope

- **No jobs, no websockets.** Every endpoint is synchronous REST; teach/find/measure on
  lab-sized images complete well within a request. If frame-scale batch work ever
  becomes a real need, that is a new track, not a retrofit onto these endpoints.
- **No database.** In-memory registries (`store.py`), rebuilt from JSON sidecars on disk
  at startup (`Store.rehydrate`). Fine for a single local user; not a design to scale
  past that without real work.
- **Pixels only, no millimetres.** Everything the lab reports — radius, rms, max_dev,
  caliper positions — is in image pixels. The `vision_metrology::metric` module
  (pixel-to-world calibration) has not landed yet; once it does, the lab is the natural
  place to add a "calibrate" step and per-object unit conversion.
- **No auth, no multi-user.** A local workbench for one person at a time.

## Notes for the next session

- **The origin-correction gotcha** (`backend/src/vm_lab/geometry.py`): the Rust
  `ShapeMatch::pose` maps a model-frame point as `position + scale·R(angle)·(point −
  origin)`, but the PyO3 binding `MetrologyModel.apply(image, x, y, angle, scale)`
  builds its fixture as `scale·R(angle)·point + (x, y)` — it does not subtract `origin`
  first. The two only agree when a model's `origin` happens to be `(0, 0)`. The backend
  works around this itself (`correct_translation`) rather than assuming callers know to;
  worth fixing at the binding level (`crates/vm-python/src/measure_py.rs`) so future
  Python consumers don't have to rediscover it. Filed as a candidate for
  `docs/backlog.md` on the main crate.
- **Per-caliper hit/reject and profiles are recomputed by the lab**, not read off
  `MetrologyModel.apply`'s result — that binding only returns the fit and its used
  `hits`, not what every caliper individually found. `measure.py`'s `_place_calipers`
  mirrors `MetrologyModel::measure_one` (`crates/vision-metrology/src/measure/model.rs`)
  by hand to reconstruct it. If that Rust function's caliper placement ever changes,
  this needs updating in step — there is no shared source of truth today. A cleaner fix
  is a Rust-side "explain" API that returns per-caliper outcomes directly; also a
  backlog candidate.
- **UX left open**: no explicit-fixture entry in the Measure tab (auto-find only in the
  MVP UI, though the backend request schema supports an explicit `fixture`); no way to
  edit/delete a taught model; the ROI-drag layer assumes `full`/`preview` tiers share the
  same aspect ratio as `thumb` (true today, since all tiers letterbox to the same source
  aspect ratio, but worth a comment if that ever changes); line-object fitted-segment
  extent in the overlay is derived from measured hit points, not from the true inlier set
  the Rust fit actually used (cosmetic only — no metrology numbers depend on it).
