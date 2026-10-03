# Lab architecture

How the lab is built, for people changing it. Why it is built this way is
[ADR-0015](../docs/dev/adr/0015-the-lab.md).

## Layout

```
lab/backend/            FastAPI app `vm_lab` over the Python bindings. Synchronous REST,
                        in-memory registries rebuilt from files under backend/data/.
lab/contract/           openapi.json (the HTTP contract) + fixtures/ (the anti-drift gate).
lab/frontend/           Vite + React 19 + TypeScript strict + Tailwind v4.
lab/frontend/src-tauri/ Desktop shell, crate `vm-lab-desktop`: its own Cargo workspace,
                        path-dependent on vision-metrology and vm-primitives.
```

`lab/` is outside the root Cargo workspace. The backend depends on `crates/vm-python` as
an editable path dependency (`[tool.uv.sources]`), which `uv sync` builds through
maturin. If that ever fights the resolver, build a wheel with `maturin build -m
crates/vm-python/Cargo.toml --release` and install it.

## One frontend, two transports

```
React UI ── LabBackend ─┬─ httpBackend  ── openapi-fetch ──► FastAPI (vm_lab) ── PyO3 ──┐
                        └─ tauriBackend ── invoke/listen ──► Tauri commands ────────────┴─► vision-metrology
```

- **`LabBackend` is the only door.** `src/api/backend.ts` defines it, and `getBackend()`
  picks the implementation through `isTauri()` (`src/api/shell.ts`). Components never call
  `fetch` or Tauri directly.
- **The browser contract is typed.** HTTP types come from `src/api/generated.ts`, which
  `openapi-typescript` generates from `lab/contract/openapi.json`.
- **All geometry is computed in the backend or command layer.** Responses carry
  source-image pixel coordinates (caliper boxes, edge points, fitted primitives, profiles),
  and the UI only draws them.
- **Placement is not duplicated.** Caliper placement comes from
  `measure::diagnostics::layout`, the same code `MetrologyModel::apply` uses.
- **The overlay type is mirrored.** The backend's `OverlayPrimitiveOut` mirrors stage2d's
  `MeasurePrimitive` field for field (`src/overlay/toMeasurePrimitive.ts`, Tauri
  `types.rs`). Changes to it must be additive.

### Desktop specifics

- **Commands are async.** Every heavy command wraps its work in `spawn_blocking`, so the
  window keeps painting.
- **Images are registered by path.** `images_scan_dir` reads headers only;
  `images_open_paths` registers files without copying, and `AppState` decodes on demand
  behind a small LRU. `images_upload` (bytes over IPC) remains for drag-and-drop.
- **Tiers are cached on disk.** The tiers are `thumb` 256, `preview` 1024 and `full`. Each
  is PNG-encoded once into `{app_cache}/tiers/{sha256}/{tier}.png` and served to the
  webview as an `asset:` URL (`convertFileSrc`, scoped to `$APPCACHE`). Keying on the pixel
  hash makes the cache survive reopening a file. In-memory crops (rectify, model crop) come
  back as bytes.
- **State.** `src-tauri/src/state.rs` holds the registries (images, models,
  calibrations), rebuilt at startup from the app-data directory. A file it cannot read is
  skipped, not fatal.
- **Events.** `lab://progress` (`{op, stage, elapsed_ms}`) feeds the status bar, and
  `lab://batch` reports per-frame batch-find results.
- **Desktop-only commands** have no HTTP route because a browser cannot read local paths:
  `images_scan_dir`, `images_open_paths`, `teach_preview` (+ `keep_contours` on
  `models_create`), `model_geometry`, `model_crop`, `batch_find`.
  `LabBackend.canOpenFiles()` gates them in the UI.
- **No mosaic command yet.** The desktop backend reports mosaic as unavailable (backlog).
- **Crash screen.** `shell/CrashScreen.tsx` is an error boundary plus global error
  handlers. It renders with inline styles and no package imports, so it still works when
  the stylesheet or a UI package is what failed.

## The canvas

`src/canvas/CanvasStage.tsx` mounts one `ImageStage` (`@vitavision/stage2d`) for every
workspace, so switching screens keeps the view. The stage element is laid out at
the source image's pixel size and carries the whole transform, so every layer is a child
`<svg>` in image coordinates and stays registered at any viewport size.

Layers, bottom to top:

```
ImageLayer            the photograph; pixelated past 4×; preview → full tier on zoom
MeasureOverlay        results, pointer-events: none
interaction surface   the one full-frame pointer target (useCanvasInteraction)
ContourLayer          candidate contours with wide transparent hit strokes
RoiLayer              region outline and eight handles
DatumLayer            model origin and its 0° arm
```

- **Who handles a press.** `useCanvasInteraction` decides what a press means (move the
  region, draw a box, sweep-select, or decline so the stage pans). Drags listen on
  `window`.
- **Screen-space sizes.** Handle and stroke sizes go through `stage.imageLength`, so they
  stay a constant number of screen pixels.
- **The half-pixel convention.** Layers use `imageViewBox(stage.image)` rather than
  `0 0 W H`, because image coordinates name pixel centres and SVG names pixel edges.

## Contract and fixtures

- **`lab/contract/openapi.json`** is generated from the FastAPI app. After a route or
  schema change:

  ```bash
  uv run --directory lab/backend python scripts/export_openapi.py
  cd lab/frontend && bun run generate:api
  ```

- **`lab/contract/fixtures/`** holds golden request/response JSON for teach, find,
  measure, measure in mm, rectify and displacement, over small synthetic images. Two tests
  replay them:
  - `lab/backend/tests/test_contract_fixtures.py` against FastAPI;
  - `lab/frontend/src-tauri/tests/contract_parity.rs` against the Tauri commands.

  Regenerate them with
  `uv run --directory lab/backend python scripts/export_contract_fixtures.py`. The
  normalization and tolerance rules are in [contract/README.md](contract/README.md).

Commit the generated files: their diff is the contract changing.

## Tests

```bash
cd lab/backend && uv run pytest                     # API smoke chain + fixture replay
cd lab/frontend && bun run typecheck && bun run test && bun run build
cd lab/frontend/src-tauri && cargo clippy --all-targets -- -D warnings && cargo test
cd lab/frontend/src-tauri && cargo run --release --example find_probe   # find timing by setting
```

`src-tauri/tests/folder_flow.rs` exercises the desktop-only path:
- scan a folder, open by path, the tier cache;
- preview, curated teach, model geometry;
- find and batch find.

It runs over a real capture and skips, with a message, when the dataset is absent.

## Known limits

- **Align's crop cache** holds one set of crops per `(image, model)`, and each rectify call
  replaces it.
- **The Measure screen auto-finds the fixture.** The API also accepts an explicit
  `fixture`.
- **Line overlays** draw the fitted segment's extent from the measured hit points, not from
  the fit's inlier set (cosmetic only).
