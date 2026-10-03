# ADR-0015: The lab: one frontend, two transports, shared UI packages

- Status: Accepted
- Date: 2026-08-19 (desktop transport 2026-08-20; shared UI packages 2026-10-03)

## Context

The library needs a workbench where a person can open real captures, teach a model, see what
it learned, run it over a set, and measure. It also needs a regression surface that exercises
the Python bindings and the Rust API end to end. Several sibling applications by the same
maintainer need the same UI building blocks: an image stage, overlays, inputs, charts and an
app shell.

## Decision

- **The lab lives in this repository** under `lab/`, outside the Cargo workspace. It has a
  FastAPI backend over `vision_metrology` (the Python bindings), a React frontend, and a Tauri
  desktop shell (`lab/frontend/src-tauri`).
- **One frontend, two transports.** `LabBackend` is one interface with an HTTP implementation
  (browser → FastAPI) and a Tauri implementation (desktop → native commands and events). The
  desktop never runs a local HTTP server. Neither transport re-implements an algorithm: each
  command builds a config, calls the library, and translates the result.
- **Contract fixtures are the anti-drift gate.** `lab/contract/fixtures/` holds golden
  request/response JSON over small synthetic images. Both
  `lab/backend/tests/test_contract_fixtures.py` and
  `lab/frontend/src-tauri/tests/contract_parity.rs` replay them, so a change on either side
  fails the replay it broke.
- **The desktop crate is its own Cargo workspace.** It declares an empty `[workspace]`, so the
  root workspace never sweeps in Tauri.
- **Shared UI comes from the `@vitavision/*` npm packages** (`ui`, `stage2d`, `charts`,
  `workbench`, …), developed in the `lab-ui` monorepo. The lab keeps only lab-specific
  components. A component that gains a second consumer moves upstream rather than being
  copied (the packages' promotion rule).

## Alternatives

- **A separate lab repository.** It loses the end-to-end check against unreleased library
  changes.
- **A sidecar HTTP server inside the desktop app.** A port to discover, CORS to keep in sync,
  and a second process, for no gain over native commands.
- **Local copies of shared components.** Every copy drifts. Six frontends already carried
  independent image stages.

## Consequences

- An API change that the lab uses touches three places (library, Python, Tauri command); the
  contract fixtures enforce agreement.
- Lab UI work that would benefit another app is done in `lab-ui` first, released, then
  consumed here.
- The developer-facing structure of the lab is described in `lab/ARCHITECTURE.md`.
