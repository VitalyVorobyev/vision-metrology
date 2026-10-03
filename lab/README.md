# Visual Metrology Lab

An interactive workbench over `vision-metrology`. Open captures, teach a shape model, find
it across a set, rectify, measure with calipers in pixels or millimetres, and see the
evidence behind every number: per-caliper hits and rejection reasons, intensity profiles,
residuals.

It runs two ways with the same interface:
- **Desktop app (Tauri).** Calls the Rust library directly. It opens folders by path and
  supports contour curation and batch runs.
- **Browser.** Talks to a local FastAPI server over the Python bindings.

## Run it

Desktop, with no server needed:

```bash
cd lab/frontend
bun install
bun run tauri dev       # development window with hot reload
bun run tauri build     # release .app / .dmg under src-tauri/target/release/bundle/
```

Browser, in two terminals:

```bash
cd lab/backend && uv sync && uv run uvicorn vm_lab.app:app --reload   # API on :8000
cd lab/frontend && bun install && bun run dev                         # UI on :5174
```

Then open http://localhost:5174. Browser data (images, saved models, calibrations) is kept
under `lab/backend/data/`; delete it to reset. The desktop app keeps its data in the
platform's app-data directory.

## Workspaces

| Workspace | Screen | What you do |
|---|---|---|
| Library | — | Open images or a whole folder (desktop), browse thumbnails, pick the current frame |
| Recognize | Teach | Draw a region, inspect the extracted contours, keep or drop them, set the datum (origin and 0° direction), build a shape model |
| | Find | Search the current frame (or, on desktop, every frame) for the model |
| | Verify | Compare the model with a found instance, both rectified into the same frame (checker, wipe, difference) |
| Gauge | Measure | Calipers and fits at the found pose: per-caliper hits, rejection reasons and profiles, in pixels or, with a calibration loaded, millimetres |
| | Align | Rectify each found part into a fixed-size, canonically oriented crop |
| Camera | Motion | Track a window between consecutive frames (subpixel displacement) |
| | Mosaic | Composite calibrated cameras onto their shared plane (browser only for now) |

A frame switcher in the header changes the current frame on every screen (`[` / `]`).

## On the canvas

- **Zoom and pan.** The wheel zooms about the cursor and dragging pans. Double-click
  toggles between fit and the previous view.
- **Keys.** `+` / `-` zoom, `0` fits, `1` is 100%. Space or middle-drag pans from any
  tool.
- **Selecting contours.** Click selects a contour; ⌘/Ctrl-click adds or removes one;
  shift-drag sweeps a selection.
- **Inventory keys (Teach).** `↑` / `↓` step through the inventory, `Space` toggles keep,
  `Delete` drops, `Enter` keeps only the selection, `F` frames it, `Esc` clears it.
- **Coordinates.** Image coordinates name pixel centres, as everywhere in the library.

**Contour picks belong to one extraction.** Contours are numbered by their position in an
extraction. If you move the region or change the contrast after curating, the panel marks
the preview stale and asks you to re-extract before building.

## Calibration and millimetres

Load a calibration-rs rig export or a `table_calibration` JSON file. Measure then adds
millimetre values for fitted circle centres and radii and for caliper edge positions,
alongside the pixel values. `rms` and `max_dev` stay in pixels: they are residuals along a
caliper's axis, not points that can be projected.

## Limits

- **Single user.** It is a local workbench with no accounts and no database.
- **Desktop-only features.** Folder opening, contour curation and batch find exist only in
  the desktop app; a browser page cannot read local paths.
- **Mosaic is browser-only for now.** The browser composites on the calibration's
  reference plane (`z = 0`), which is meaningful only when that plane lies on the target.

Developer documentation (architecture, contract, tests) is in
[ARCHITECTURE.md](ARCHITECTURE.md).
