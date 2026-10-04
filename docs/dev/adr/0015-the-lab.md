# ADR-0015: The lab: one frontend, two transports, shared UI packages

- Status: Accepted
- Date: 2026-10-03

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
  request/response JSON over small synthetic images, and each transport replays them, so a
  change on either side fails the replay it broke. CI gates only the Tauri replay; the
  FastAPI replay runs with the backend's pytest, which CI does not run. Regeneration and
  replay are described in
  [`lab/contract/README.md`](../../../lab/contract/README.md).
- **The desktop crate is its own Cargo workspace.** It declares an empty `[workspace]`, so the
  root workspace never sweeps in Tauri.
- **Shared UI goes upstream.** It comes from the `@vitavision/ui`, `stage2d`, `charts` and
  `workbench` npm packages, developed in the `lab-ui` repository. The lab keeps only
  lab-specific components. A component that a second app needs moves to `lab-ui`, is
  released, and is consumed here from npm, rather than being copied.

## Alternatives

- **A separate lab repository.** It loses the end-to-end check against unreleased library
  changes.
- **A sidecar HTTP server inside the desktop app.** A port to discover, CORS to keep in sync,
  and a second process, for no gain over native commands.
- **Local copies of shared components.** Every copy drifts.

## Consequences

- An API change that the lab uses touches three places (library, Python, Tauri command); the
  contract fixtures enforce agreement.
- UI work that another app needs waits for a `lab-ui` release before the lab uses it.
- The developer-facing structure of the lab is described in `lab/ARCHITECTURE.md`.
