# AGENTS.md

Guidance for coding agents in `vision-metrology`, a Rust library (with Python bindings and
a lab app) for high-precision industrial metrology.

## Read first

1. [`docs/dev/system-design.md`](docs/dev/system-design.md): layering, the numbered
   invariants, and the ADR index ([`docs/dev/adr/`](docs/dev/adr)).
2. [`docs/dev/roadmap.md`](docs/dev/roadmap.md): the open tracks and their acceptance
   criteria.
3. [`docs/dev/backlog.md`](docs/dev/backlog.md): known debt.
4. [`CONTRIBUTING.md`](CONTRIBUTING.md): the gates, tests, benchmarks, Python workflow and
   documentation rules.

Trust these over reconstructing state from git history.

## Rules

- **Run the CONTRIBUTING gates before every commit.** If a hot path changed, also run the
  affected benches.
- **Invariants are design constraints.** Breaking one needs an ADR change first.
- **Parity in the same PR.** New public Rust API ships its vm-python binding, `.pyi` stub
  and a Python test together (invariant 15). Lab-facing API changes also update the Tauri
  command and the contract fixtures.
- **Docs move with the code.** A change to scope or decisions updates `docs/dev/` in the
  same PR, rewriting the affected entry rather than appending. Finished roadmap items move
  to `CHANGELOG.md` `[Unreleased]`.
- **Keep the audiences apart.** User-facing docs and rustdoc carry no plan labels, PR
  numbers or history.
- **Scoped commits; never revert unrelated changes.**
- **Shared UI goes upstream.** It lives in the `@vitavision/*` packages (the `lab-ui`
  repository). A component that a second app needs goes there, not into `lab/`.

## Skills

Task-specific guidance is in `.claude/skills/`. `.agents/skills` links to it.
- `metrology-invariants`: coordinate and subpixel conventions.
- `tests-synthetic-fixtures`: fixtures with known ground truth.
- `hotpath-rust` and `criterion-bench`: performance work.
- `api-shaping`: public API changes.
- `laser-extract`: the `laser` module.
