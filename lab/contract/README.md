# `lab/contract/`

The contract between the lab's two transports
([ADR-0015](../../docs/dev/adr/0015-the-lab.md)). Both artifacts here are generated and
committed, and their diff is the contract changing.

- `openapi.json`: the FastAPI backend's OpenAPI schema, the source of the frontend's typed
  HTTP client (`lab/frontend/src/api/generated.ts`).
- `fixtures/`: golden request/response pairs for the core operations, over small
  deterministic synthetic images. The browser transport (FastAPI) and the desktop transport
  (native Rust commands over `vision-metrology`) must agree on what teach, find, measure,
  rectify and displacement report for the same input, although neither talks to the other
  and the desktop never goes over HTTP.

## Regenerating

After a route or schema change, regenerate the schema and the typed client:

```bash
uv run --directory lab/backend python scripts/export_openapi.py
cd lab/frontend && bun run generate:api
```

After a change to the synthetic inputs, the operation sequence or what an operation
reports, regenerate the fixtures with
[`lab/backend/scripts/export_contract_fixtures.py`](../backend/scripts/export_contract_fixtures.py):

```bash
uv run --directory lab/backend python scripts/export_contract_fixtures.py
```

Commit the regenerated files with the change that caused them.

## What's in `fixtures/`

| File | What it is |
|---|---|
| `disc.png` | 128x128, anti-aliased bright disc (centre `(64, 64)`, radius `24`) — the teach/find/measure/rectify target. |
| `frame_a.png`, `frame_b.png` | 112x112 deterministic value-noise textures; `frame_b`'s content is `frame_a`'s shifted by exactly `(4.0, 3.0)` px — the displacement pair. |
| `calibration.json` | A copy of `crates/vision-metrology/tests/fixtures/table_calibration.json` (2 real cameras) — the mm-path calibration upload. |
| `rectify_crop.png` | The rectified crop PNG from the `rectify` fixture's first match, for a pixel-tolerance image comparison (not just the JSON metadata). |
| `teach.json`, `find.json`, `measure.json`, `measure_mm.json`, `rectify.json`, `displacement.json` | `{operation, description, request, response}` — one per core operation. `measure.json` has no calibration (pixel units only); `measure_mm.json` sets `calibration_id`/`camera_index`/`plane` and asserts the mm fields populate. |

## Normalization

Everything in a fixture's `request`/`response` is captured verbatim from the live API
**except** the ids the store assigns at runtime (`img-N`, `model-N`, `cal-N` on the
FastAPI side; the Tauri store has its own, unrelated counters). Reproducing those
exactly across two independent backends is neither possible nor meaningful, so every
id-shaped string is replaced, everywhere it appears — including embedded in a URL like
`crop_url` — by a fixed placeholder:

| Real id (FastAPI's own counters) | Placeholder |
|---|---|
| the disc image's id | `$IMAGE_ID` |
| `frame_a.png`'s id | `$FRAME_A_ID` |
| `frame_b.png`'s id | `$FRAME_B_ID` |
| the taught model's id | `$MODEL_ID` |
| the calibration's id | `$CALIBRATION_ID` |

Nothing else is normalized: this API has no timestamps, and every other field
(geometry, scores, pixel values, mm conversions) is deterministic given the same
synthetic input and the same library version. Invariant 12 (no RNG in library code) is
what makes that reproducible.

A replay test performs the identical substitution on its own run's freshly-assigned ids
before comparing against the committed golden, so the assertion is
`normalize(actual) == golden` up to a float tolerance: `1e-3` relative or absolute in the
Python replay, `1e-2` in the Rust replay.

## Replays

- **Browser path:** `lab/backend/tests/test_contract_fixtures.py`, run with
  `cd lab/backend && uv run pytest`. CI does not run it. It runs
  `export_contract_fixtures.Run`, the *same* operation-sequence code the generator uses,
  not a hand-maintained parallel copy, against a fresh FastAPI backend and asserts
  agreement with the committed JSON. It also checks that the committed PNGs are
  byte-identical to what the generator renders, and that `rectify_crop.png`'s *decoded
  pixels* (not its compressed bytes, which are not stable across PIL/zlib versions) are
  within tolerance of a freshly rectified crop.
- **Desktop path:** `lab/frontend/src-tauri/tests/contract_parity.rs`, run with
  `cd lab/frontend/src-tauri && cargo test`. CI's Lab job runs it. It runs the same
  synthetic images and request parameters through the Tauri command handlers (factored
  as plain functions over a `State`, no GUI needed) and asserts the same numeric
  agreement, using `serde_json::Value` field lookups rather than a full response-schema
  match. The two backends' JSON *shapes* differ in places (this is a numeric-agreement
  contract, not a wire format between them), but the fields both report must agree: pose,
  score, measured radius and rms, each caliper's verdict, edge and residual, the measure
  overlay, mm values, and displacement dx/dy/score.
