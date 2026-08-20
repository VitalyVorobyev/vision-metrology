# Visual Metrology Lab

A local interactive workbench over `vision_metrology` — **one frontend, two shells**: a
browser build over the FastAPI/Python bindings (below), and a Tauri desktop build that
calls the Rust crates directly over native commands/events, no HTTP at all (see
[Desktop (Tauri)](#desktop-tauri)). Where `examples/python/` shows the library's chain in
a script, the lab makes it a workbench:
upload an image, drag a box to **teach** a shape model, **find** it elsewhere in the
image, **measure** circles/lines against the found pose, and **judge** the fit — rms,
max deviation, per-caliper hit/reject, the raw intensity profile behind each hit. The
**Align** tab closes the last seam: rectify every found instance into a canonical,
model-frame crop (`ShapeMatch.model_frame_map`), the fixed-size tensor an anomaly model
needs across frames regardless of the pose a part was found at. Uploading a
**calibration** (a calibration-rs `RigExtrinsicsExport` or a `table_calibration`
`calibration.json`, either format, detected by shape) turns the Measure tab's pixel
results into millimetres too — `vision_metrology.metric.pixel_to_plane` over the
calibration's own camera/plane. The **Bird's-eye** tab composites two or more calibrated
cameras' rectified views of that same shared plane into one grid — no-blending,
nearest-camera-centre priority, so every mosaic pixel still traces to one camera's own
calibration (`vision_metrology.metric.plane_grid_map` per camera + `Map.apply_with_mask`,
composited server-side). It is the library's own end-to-end chain (`ShapeModel` →
`ShapeMatcher` → `MetrologyModel` / `Map` / `metric`), made visible.

## Architecture

```
lab/backend/    FastAPI, sync REST, in-memory registries + files under backend/data/
                (gitignored). No jobs, no websockets, no database — see "Out of scope".
lab/contract/   openapi.json — the committed wire contract, dumped from the FastAPI app
                by `backend/scripts/export_openapi.py`. Source of truth for the
                frontend's typed client; regenerate it whenever a route or schema
                changes and commit the diff (the diff *is* the contract changing).
                fixtures/ — golden request/response JSON + synthetic PNGs for the core
                operations, the anti-drift gate between the browser and desktop shells
                (see "Desktop (Tauri)" and fixtures/README below it in this file).
lab/frontend/   Vite + React 19 + TypeScript strict + Tailwind v4, built on the shared
                @vitavision/lab-ui design system (ZoomPanCanvas, MeasureOverlay,
                LineProfile, SchemaForm, ...). `src/api/generated.ts` (openapi-typescript
                output, also committed) supplies every request/response type; `backend.ts`
                is the one `LabBackend` interface every tab/component calls through —
                `httpBackend` (browser, over `openapi-fetch`) or `tauriBackend` (desktop,
                over `@tauri-apps/api` invoke/events) — no raw `fetch` or Tauri import
                outside those two files. `shell.ts`'s `isTauri()` check picks which one
                `getBackend()` returns; `baseUrl.ts` (browser-only) resolves where the
                HTTP backend lives.
lab/frontend/   Tauri v2 desktop shell, crate `vm-lab-desktop` — a standalone Cargo
  src-tauri/    workspace (`[workspace]` in its own Cargo.toml), never a member of the
                repo-root one. Calls `vision-metrology`/`vm-primitives` directly via path
                dependencies; commands mirror the FastAPI routers, `src/state.rs` is a
                thin Rust port of `store.py`. See "Desktop (Tauri)" below.
```

Both shells do all the geometry server/command-side: every response carries
**source-image pixel coordinates** — caliper boxes, measured points, fitted
circles/lines, per-caliper profiles — so the frontend only draws what it is told and
never reconstructs a pose or a fixture transform itself. Caliper *placement* used to be
a second, hand-written copy of that geometry (`vm_lab/geometry.py`, deleted); it now
lives once, in Rust, as `measure::diagnostics::layout` — both the Python router
(`routers/measure.py`) and the Tauri command (`src-tauri/src/commands/measure.rs`) call
the *same* function `MetrologyModel::apply` calls internally, so an overlay can never
draw a caliper the actual measurement did not look at. See system-design.md's `measure`
entry for the API.

`lab/backend/` and `lab/frontend/` (excluding `src-tauri/`) are **not** part of the
Cargo workspace; they contain no Rust and depend on `vision_metrology`, the PyO3 bindings
published from `crates/vm-python`, as an editable local dependency.
`lab/frontend/src-tauri/` **is** Rust, but is its own standalone Cargo workspace,
detached from the repo root's (see "Desktop (Tauri)").

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
| `POST` | `/calibration` | Upload a calibration JSON — a calibration-rs `RigExtrinsicsExport` or a `table_calibration` `calibration.json`, format detected by shape (`kind == "rig_extrinsics"` vs. top-level `intrinsic`/`extrinsic`). Parsed eagerly (bad JSON is a 400), saved to disk. Returns `calibration_id`, detected `format`, `n_cameras`. |
| `GET` | `/calibration` | List uploaded calibrations. |
| `POST` | `/displacement` | `corr.displacement` pairwise over an ordered `image_ids` sequence, tracking a `window` rect through it. Returns per-pair `dx`/`dy`/`score` plus the running `cumulative_x`/`cumulative_y` trajectory. |
| `POST` | `/mosaic` | Composite `>= 2` calibrated cameras' rectified views of the calibration's `z = 0` plane into one grid — no-blending, nearest-camera-centre priority (each destination pixel keeps whichever valid camera's reprojection lands closest to its own principal point). `grid` auto-fits from the cameras' own image-border footprints on the plane when omitted. Returns `image_url`/`source_id_url`, per-camera coverage, union/overlap fractions, seam disparity p50/p95. |
| `GET` | `/mosaic/{id}/image` | The composited mosaic PNG for a `POST /mosaic` result — `?feather=true` switches to the opt-in display-only linear feather, never the default. Cached in memory, `images/{id}/{tier}`-style ETag. |
| `GET` | `/mosaic/{id}/source_id` | The `source_id` map, palette-colored per camera (tinted by the mosaic's own intensity), uncovered pixels dark gray. |
| `GET` | `/health` | Liveness. |

`POST /measure` additionally accepts `calibration_id` + `camera_index` (which camera in
the calibration's list) + `plane` (`{nx, ny, nz, d}`, defaulting to the reference frame's
own `z = 0`). When set, the response's fitted circle center/radius and every hit
caliper's edge position are augmented with millimetre fields (`circle_cx_mm`/
`circle_cy_mm`/`circle_r_mm`, `x_mm`/`y_mm` on each caliper's edge mark) alongside the
existing pixel ones — the pixel fields never disappear, mm is additive. Radius is
measured via two diametral points converted through `pixel_to_plane` independently (exact
per point; a fronto-parallel-view approximation of the radius itself under a tilted
camera — see `vm_lab/routers/measure.py`). A pixel whose ray misses the plane (behind the
camera, or parallel to it) leaves the corresponding mm field `null` rather than failing
the whole request.

## Run it (browser)

Two terminals, from `lab/`:

```bash
cd backend && uv sync && uv run uvicorn vm_lab.app:app --reload   # :8000
cd frontend && bun install && bun run dev                         # :5174, calls :8000 directly
```

Open http://localhost:5174. Data lands under `lab/backend/data/` (images, tier cache,
saved models) — gitignored, safe to delete to reset the workbench.

The frontend talks to the backend over CORS'd HTTP (`Settings.cors_origins` in
`config.py` allows `:5174`), not a dev-server proxy — `src/api/baseUrl.ts` resolves the
backend's base URL itself (`VITE_API_BASE_URL` env override, else the `:8000` default).
This path is browser-only; see below for the desktop build, which never starts this
backend at all.

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

### Contract fixtures — the anti-drift gate

`lab/contract/fixtures/` (golden request/response JSON + small synthetic PNGs for teach/
find/measure/measure-with-mm/rectify/displacement) is what keeps the two shells honest:
`lab/backend/scripts/export_contract_fixtures.py` generates them from the FastAPI
backend, `lab/backend/tests/test_contract_fixtures.py` replays them against it, and
`lab/frontend/src-tauri/tests/contract_parity.rs` replays the *same* fixtures through the
Tauri command layer — one JSON file, two replay tests, in two languages. Regenerate with
`uv run --directory lab/backend python scripts/export_contract_fixtures.py` whenever the
operation sequence or its synthetic inputs change, and commit the diff. Full rules
(normalization, tolerance) are in `lab/contract/README.md`.

## Desktop (Tauri)

One frontend bundle, two transports, chosen at runtime by `getBackend()`
(`src/api/backend.ts`) via `isTauriShell()` (`src/api/shell.ts`, `@tauri-apps/api/core`'s
`isTauri()`):

```
Browser build                              Desktop build (Tauri)
┌──────────────┐   HTTP (openapi-fetch)     ┌──────────────┐   invoke()/emit()
│  React UI    │ ─────────────────────────▶ │  React UI    │ ─────────────────▶
│ (httpBackend)│ ◀───────────────────────── │(tauriBackend)│ ◀─────────────────
└──────────────┘   JSON, PNG/WebP by URL    └──────────────┘   JSON, PNG as raw
       │                                            │           IPC bytes (no
       ▼                                            ▼           base64)
┌──────────────┐                            ┌──────────────────────────────┐
│   FastAPI    │                            │  vm-lab-desktop (Tauri, Rust)│
│  (vm_lab)    │                            │  commands/* -> vision-       │
└──────┬───────┘                            │  metrology directly, no HTTP │
       │ PyO3                               └──────────────┬───────────────┘
       ▼                                                    │ path deps
┌──────────────┐                                            ▼
│vision_metrology (Python bindings, crates/vm-python)  vision-metrology + vm-primitives
└──────────────┘                                       (same Rust crates, called directly)
```

Both shells call the *same* Rust code (`crates/vision-metrology`,
`crates/vm-primitives`) — the browser build through `vm-python`'s PyO3 bindings, the
desktop build directly. `lab/contract/fixtures/` (above) is what turns "should agree" into
"verified to agree".

```bash
cd lab/frontend
bun install
bun run tauri dev       # Vite dev server + a native window, hot-reloads the frontend
bun run tauri build     # release .app/.dmg (macOS) under src-tauri/target/release/bundle/
```

No backend process, no `VITE_API_BASE_URL`, no CORS config — the desktop build never
opens a socket to reach `vision_metrology`.

**Where state lives.** `lab/frontend/src-tauri/src/state.rs` is a thin Rust port of
`vm_lab/store.py`'s essentials: in-memory registries (`images`/`models`/`calibrations`,
behind `std::sync::Mutex`) rebuilt on startup from files under the Tauri app-data
directory (`AppState::rehydrated`, called from `lib.rs`'s `setup` hook) —
`~/Library/Application Support/dev.vitavision.metrology-lab/{images,models,
calibrations}/` on macOS. Same spirit as the backend's `data/` directory, different path
and no shared code (a ~450-line, deliberately small module — not a byte-for-byte port of
`store.py`, which also has thumbnail-tier caching this crate does not).

**Progress events.** Long operations can emit `lab://progress` events
(`{op, stage: "started"|"finished", elapsed_ms}`) over `tauri::Emitter`; `find` is wired
today (the plan's "wire one real progress case") since its cost is the one that scales
with scene size in a way a user notices. `teach`/`images_*` are effectively instant on
lab-sized images and are not wired.

**Deliberately not ported this wave: mosaic.** `routers/mosaic.py` (~315 lines: grid
auto-fit, nearest-camera-centre compositing, source_id map, feather display mode) has no
Tauri command. `tauriBackend.ts`'s `mosaic`/`mosaicImageUrl`/`mosaicSourceIdUrl` throw a
clear "not available in the desktop build" error rather than silently returning nothing —
the Bird's-eye tab is browser-only until this is ported (tracked in `docs/backlog.md`).

**`imageUrl`/`rectifyCropUrl` are synchronous** (every `LabBackend` call site drops the
result straight into `<img src>`), but the underlying data arrives over an async
`invoke()` returning raw bytes (`tauri::ipc::Response`, no base64). `tauriBackend.ts`
bridges this with a blob-URL cache: the first call for a given id/tier kicks off the
fetch in the background and returns a 1x1 placeholder immediately; the *next* call for
that key (the app's own re-renders — TanStack Query updates, selection changes — happen
often enough in practice) returns the real `blob:` URL. A component that renders an image
exactly once and never again would keep showing the placeholder; no call site in this app
does that today.

**Tiers are PNG, not WebP.** The browser backend's `thumb`/`preview` tiers are WebP
(`media.py`); the desktop command (`image_data`) resizes and PNG-encodes all three tiers
instead — the plan's own instruction ("PNG bytes, no base64") plus one fewer codec
dependency. No on-disk tier cache either: a resize is a few milliseconds at lab image
sizes, and a cache-invalidation story is a second thing to keep correct for a
single-user workbench.

## Tests

```bash
cd backend && uv run pytest       # smoke: upload -> teach -> find -> measure a synthetic disc -> rectify ->
                                   # upload a real table_calibration.json -> measure with mm; plus displacement
                                   # and mosaic (synthetic 2-camera calibration, both built in their own tests);
                                   # plus the contract-fixture replay (test_contract_fixtures.py)
cd frontend && bun run typecheck && bun run test && bun run build
cd frontend/src-tauri && cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test
                                   # cargo test includes tests/contract_parity.rs, the desktop-side replay
```

## Out of MVP scope

- **No jobs, no websockets.** Every endpoint is synchronous REST; teach/find/measure on
  lab-sized images complete well within a request. If frame-scale batch work ever
  becomes a real need, that is a new track, not a retrofit onto these endpoints.
- **No database.** In-memory registries (`store.py`), rebuilt from JSON sidecars on disk
  at startup (`Store.rehydrate`). Fine for a single local user; not a design to scale
  past that without real work.
- **Millimetres cover fitted circle/caliper positions only, not every number.** `rms` and
  `max_dev` stay in pixels always — they are residuals against a caliper-axis profile,
  not a single point `pixel_to_plane` can convert, and no per-object local scale factor
  is computed to fake one. The frontend's px/mm toggle (Measure tab, shown once a
  calibration is selected) only affects the fields that actually have an mm
  counterpart.
- **No plane editor in the UI.** The backend accepts an arbitrary `plane` per request;
  the frontend always sends the default (`z = 0` of the calibration's reference frame).
  A plane-picking control is a natural follow-up once a second calibration with a
  genuinely offset plane shows up in practice.
- **No auth, no multi-user.** A local workbench for one person at a time.

## Notes for the next session

- **The origin-correction gotcha is fixed** (historical: `MetrologyModel.apply` used to
  build its fixture as `scale·R(angle)·point + (x, y)`, skipping the `origin` subtraction
  `ShapeMatch::pose` applies, so the backend pre-corrected the translation itself in a
  now-deleted `geometry.py` helper). `apply` takes an `origin` keyword and builds the same
  fixture `ShapeMatch::pose` does (`crates/vm-python/src/measure_py.rs`); the backend
  passes the model's `origin` straight through.
- **Per-caliper hit/reject and profiles are recomputed by both shells** (`measure.py`'s
  `_measure_calipers`, `src-tauri`'s `commands::measure::measure_calipers`), not read off
  `MetrologyModel.apply`'s result — that call only returns the fit and its used `hits`,
  not what every caliper individually found. This used to mean *placement* was
  duplicated too (`measure.py`'s old `_place_calipers`, deleted in W6): both shells now
  get placement from `measure::diagnostics::layout`, the same function
  `MetrologyModel::apply` calls internally, so a caliper can never be drawn somewhere the
  actual measurement did not look. What's still duplicated is only the *re-measurement*
  itself (calling `Caliper::measure` a second time per caliper to recover the rejection
  reason and raw profile `apply` doesn't surface) — both call sites start from identical
  geometry, so they can disagree on *why* a caliper failed but never on *where* it was.
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
