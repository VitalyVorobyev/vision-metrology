# Visual Metrology Lab

A local interactive workbench over the `vision_metrology` Python package. Where
`examples/python/` shows the library's chain in a script, the lab makes it a workbench:
upload an image, drag a box to **teach** a shape model, **find** it elsewhere in the
image, **measure** circles/lines against the found pose, and **judge** the fit — rms,
max deviation, per-caliper hit/reject, the raw intensity profile behind each hit. The
**Align** tab closes the last seam: rectify every found instance into a canonical,
model-frame crop (`ShapeMatch.model_frame_map`), the fixed-size tensor an anomaly model
needs across frames regardless of the pose a part was found at. It is the library's own
end-to-end chain (`ShapeModel` → `ShapeMatcher` → `MetrologyModel` / `Map`), made
visible.

## Architecture

```
lab/backend/    FastAPI, sync REST, in-memory registries + files under backend/data/
                (gitignored). No jobs, no websockets, no database — see "Out of scope".
lab/contract/   openapi.json — the committed wire contract, dumped from the FastAPI app
                by `backend/scripts/export_openapi.py`. Source of truth for the
                frontend's typed client; regenerate it whenever a route or schema
                changes and commit the diff (the diff *is* the contract changing).
lab/frontend/   Vite + React 19 + TypeScript strict + Tailwind v4, built on the shared
                @vitavision/lab-ui design system (ZoomPanCanvas, MeasureOverlay,
                LineProfile, SchemaForm, ...). `src/api/generated.ts` (openapi-typescript
                output, also committed) supplies every request/response type; `backend.ts`
                is the one `LabBackend` interface every tab/component calls through
                (`httpBackend` today, over `openapi-fetch`) — no raw `fetch` outside it.
                `baseUrl.ts` resolves where the backend lives (env override, else
                `127.0.0.1:8000`), with an injected-global hook already wired for a future
                Tauri shell so a second `LabBackend` implementation can slot in behind
                `getBackend()` without touching the UI.
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
| `POST` | `/rectify` | `ShapeMatcher.find` against a model, then `ShapeMatch.model_frame_map` per match — a `crop` spec (`rect` in **model-frame coordinates**, `px_per_unit`, `normalize_scale`). Returns, per match, pose/score plus a `crop_url` and a validity fraction (in-scene pixel coverage). |
| `GET` | `/rectify/{image_id}/{model_id}/{index}` | The PNG crop for one match from the most recent `/rectify` call on that `(image_id, model_id)` pair — cached in memory (not disk), `images/{id}/{tier}`-style ETag. 404 until a matching `POST /rectify` has run. |
| `GET` | `/health` | Liveness. |

## Run it

Two terminals, from `lab/`:

```bash
cd backend && uv sync && uv run uvicorn vm_lab.app:app --reload   # :8000
cd frontend && bun install && bun run dev                         # :5174, calls :8000 directly
```

Open http://localhost:5174. Data lands under `lab/backend/data/` (images, tier cache,
saved models) — gitignored, safe to delete to reset the workbench.

The frontend talks to the backend over CORS'd HTTP (`Settings.cors_origins` in
`config.py` allows `:5174`), not a dev-server proxy — `src/api/baseUrl.ts` resolves the
backend's base URL itself (`VITE_API_BASE_URL` env override, else the `:8000` default),
which is what lets the same bundle later run inside a Tauri shell that injects a
different, ephemeral sidecar port.

### Regenerating the API contract

Whenever a backend route or Pydantic schema changes:

```bash
cd backend && uv run python scripts/export_openapi.py   # writes lab/contract/openapi.json
cd frontend && bun run generate:api                       # writes src/api/generated.ts
```

Both outputs are committed. Review the diff — it *is* the contract changing. `backend.ts`
should need no changes for a pure schema/route addition; only new operations that the UI
needs to call require a new `LabBackend` method.

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

- **The origin-correction gotcha is fixed.** `MetrologyModel.apply` used to build its
  fixture as `scale·R(angle)·point + (x, y)`, skipping the `origin` subtraction that
  `ShapeMatch::pose` applies (`position + scale·R(angle)·(point − origin)`), so the
  backend had to pre-correct the translation itself (`correct_translation` in
  `backend/src/vm_lab/geometry.py`). `apply` now takes an `origin` keyword and builds the
  same fixture `ShapeMatch::pose` does (`crates/vm-python/src/measure_py.rs`); the backend
  passes the model's `origin` straight through and `correct_translation` is gone.
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
- **Align's crop cache is in-memory and single-slot per `(image_id, model_id)`**: each
  `POST /rectify` overwrites the previous crops cached for that pair, so a `GET` against a
  stale `index` from an older call — or against a `crop` spec that changed between calls —
  is undefined until the reader re-runs `/rectify`. Fine for one interactive workbench user
  looking at one crop grid at a time; would need real keys (a spec hash, or persistence) to
  serve concurrent readers or a "compare two crop specs side by side" UI.
