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
- **Measure is one pass.** `measure::diagnostics::explain_model` (Python:
  `MetrologyModel.explain`) measures each caliper once. Per object it returns what
  `MetrologyModel::apply` returns (the fit, its residuals and hits), with every caliper's
  placement and trace. `routers/measure.py` and `commands/measure.rs` build the caliper
  list (hit or rejection reason, edge and its amplitude, residual against the fit, profile
  with the span its samples cover) and the overlay from those traces, so there is no
  second, per-caliper measurement to drift from the fit. The placements are `apply`'s own:
  the same code as `measure::diagnostics::layout`. A radial placement's `center` is its
  circle's, so its box is drawn `radius` out along its axis. Each caliper's box and edge
  mark carry the id `caliper-<object>-<index>`.
- **The overlay type is mirrored.** The backend's `OverlayPrimitiveOut` mirrors stage2d's
  `MeasurePrimitive` field for field (`src/overlay/toMeasurePrimitive.ts`, Tauri
  `types.rs`), `id` included. Changes to it must be additive. `state` is the UI's: the
  backend never sets it.

### Desktop specifics

- **Commands are async.** Every heavy command wraps its work in `spawn_blocking`, so the
  window keeps painting.
- **Images are registered by path.** `images_scan_dir` reads headers only;
  `images_open_paths` registers files without copying, and `AppState` decodes on demand
  behind a small LRU. Dropped files arrive as paths too: the Library's `FileDrop`
  (`@vitavision/workbench`) takes a `PathSource` built on `pickImages` and
  `LabBackend.onFileDrop`, which listens to the webview's native drag and drop.
  `images_upload` (bytes over IPC) remains for a bare `File`.
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
`<svg>` in image coordinates and stays registered at any viewport size. A `null` view
opens at fit (`initialView="fit"`), small frames included.

Layers, bottom to top:

```
ImageLayer       the photograph (stage2d): preview tier, full tier past it, pixelated past 4×
MeasureOverlay   results, pointer-events: none
StageSurface     the bare-image target (stage2d): starts a sweep, declines everything else
RectRoiEditor    the region (stage2d): interior, eight handles, and the box tool's draw surface
PolylineSet      candidate contours (stage2d), drawn batched and picked through a spatial index
vertices         the hovered and selected contours' samples, from 3×
sweep band       the rubber band while a sweep is in flight
DatumLayer       model origin and its 0° arm
```

- **Who handles a press.** The topmost element under the pointer gets it:
  - a datum handle, then a contour (click selects, ⌘/Ctrl toggles);
  - then the region: a handle resizes it, the interior moves it, and with the box tool (or
    no region yet) a drag elsewhere draws a new one;
  - then `StageSurface`, which sweeps on shift or with the marquee tool and otherwise
    declines, so the stage pans.

  The region sits under the contours so a contour inside it stays clickable. One wrapper
  around both offers each press to the sweep first, in the capture phase, so shift or the
  marquee tool selects over the region and over a contour rather than moving one or
  picking the other; a region handle is exempt. `PolylineSet`'s own band starts only from
  its lines, so the lab never lets it start one and draws a single band for all three
  starting points. Every drag (sweep, datum) is stage2d's `useStageDrag`, which listens on
  `window`.
- **Hover is shared.** `PolylineSet`'s hover is controlled by the Teach inventory's
  `hovered`, so a row hovered in the list and a contour hovered on the image are the same
  state.
- **Results are linked by id.** `MeasureOverlay` takes no pointer events, so a route whose
  list is linked to its overlay (Find's matches, Measure's calipers) gives each primitive
  an `id` (`match-<index>`, or the backend's `caliper-<object>-<index>`) and a `state`
  (`hover`, `selected`, `dimmed` outside the list's filter), and pushes an `OverlayPicker`
  into `LabContext`. The picker resolves an image point to an id (`state/rotatedBox.ts`: each
  item is a rotated box, the nearest centre wins). `CanvasStage` asks it on the stage's
  `onHover` and on `onBackgroundClick`, a press that did not pan, so a drag over a result
  still pans. The route's `hovered` is the picker's, as Teach's is `PolylineSet`'s.
- **Batch results are per frame.** A batch find (Find's "In all frames", or the Library's
  run) is kept in `LabContext.batch`. Selecting a frame it covered loads that frame's
  matches and request as `matches` / `lastFind`, so Find, Verify and Measure see the frame's
  own result; the frame strip and menu mark the frames with no match.
- **The region commits on release.** The editor's moves go to a local draft, and the shared
  `roi` changes once per gesture, which is what the Teach panel re-extracts from.
  `canvas/roi.ts` converts between the backend's `Roi` tuple and stage2d's `Rect`, so a
  typed region and a dragged one go through the same `clampRect`.
- **Panels drive the view** through the stage's own handle: `ImageStage`'s `ref` is
  `LabContext`'s `canvas` (`frame`, `fit`, `zoomTo`), `null` while no canvas is mounted.
- **The full tier is resolved lazily.** stage2d's `ImageLayer` wants the full tier's URL up
  front, but on the desktop asking for a tier renders it. `CanvasStage` asks only once the
  preview would be magnified, `ImageLayer`'s own rule, and keeps it for that frame.
- **Screen-space sizes.** Handle and stroke sizes go through `useScreenPx`
  (`stage.imageLength`), so they stay a constant number of screen pixels. No layer uses
  `vector-effect: non-scaling-stroke`, which is unreliable under a CSS transform.
- **Colours are overlay roles.** Overlays take stage2d's role tokens (`overlayRole()`,
  `--stage-*`), one set for both themes, with a halo under the lab's own strokes:
  - kept contours are `feature`, dropped ones dashed `structure`, and a selection
    `selection`;
  - the datum and the model's points are `model`. Find draws the selected match in the
    `selected` state and the hovered one in `hover`, each with its extent outlined;
  - vertices are `label` dots. `PolylineSet` can draw them, but in the selection colour,
    so on a selected line they vanish.

  Measurement verdicts (caliper hit or miss) keep their verdict tones. ESLint's
  `tokens-only` rule covers every overlay file; only `CrashScreen.tsx` is exempt.
- **Model points stay ticks.** A model's points are drawn as one `segment` each, not as a
  `polyline`: the model stores them stratified (a golden-ratio permutation), not in contour
  order.
- **The half-pixel convention.** Layers use `imageViewBox(stage.image)` rather than
  `0 0 W H`, because image coordinates name pixel centres and SVG names pixel edges.
- **Frames.** The header's frame switcher is workbench's `SequenceNavigator` (a lazy
  thumbnail strip with previous, next and `[` / `]`) plus a menu of every frame. The Library
  keeps its own grid: browsing wants each frame's name and size and a double-click to open
  it, not a strip. Thumbnails everywhere are fetched once near the viewport
  (`hooks/useNearViewport.ts`), because on the desktop asking for a tier renders it.
- **The layers menu stays local** (`CanvasControls.tsx`). stage2d's `StageLayersMenu` names a
  layer with a plain string, so it cannot show the colour swatches, and its one label would
  put the hidden count in the menu's heading too.

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
