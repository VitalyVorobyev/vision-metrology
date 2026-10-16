# ADR-0020: The shared min-path and smoother cores stay inside `measure`

- Status: Accepted
- Date: 2026-10-16

## Context

Three planned `measure` features choose a sequence of states under a continuity penalty:
- the coupled two-boundary extraction (ADR-0019), over (centre, width) bins per
  station;
- the bead-end rim, over a radius per ray, with its end rays pinned to the walls;
- the fusion of boundary evidence.

Each needs an exact layered min-path and its max-marginals. ADR-0018's regularised offset
solve is a robust Whittaker smoother, which also removes the bin quantisation a path
leaves. ADR-0002 put that banded solve beside its only caller, and it is about to have
more than one caller.

## Decision

- **One module, `measure::numeric`, holds both cores, crate-private.**
  - `minpath::LayeredPath` is the exact min-path.
  - `whittaker::Smoother` is the robust smoother, with its banded `LDLᵀ`.

  Every caller is in `measure`, and no module outside it needs either core now. A public
  operator would carry a semver promise, Python parity and accuracy rows for a generic
  optimiser whose output is not a measurement. The public API is also under revision.
- **The files depend on nothing else in the crate.** The bench and the allocation gate
  compile them directly through `#[path]`, so the cores are measured and gated without
  being exposed.
- **The min-path.**
  - Every layer has the same grid: planes, each a row-major grid of one or two
    dimensions. A layer with fewer states marks the missing ones with `+∞` unaries.
  - A step costs a switch cost between planes plus a quadratic cost per dimension,
    bounded by a band. The cost is separable, so the minimum over predecessors is one
    banded pass per dimension. ADR-0019's (centre, width) state fits this, since
    `Δl² + Δr² = 2Δc² + Δw²/2`.
  - Forward and backward passes give the path and the max-marginals; a caller that needs
    only the path runs the forward pass alone.
  - A plane with no finite unary in a layer is skipped in both passes. It holds `+∞`
    there, which is exact, so a plane that exists only at a few layers costs nothing
    elsewhere.
  - Soft pins are strong finite unaries.
  - Ties go to the lower state index.
- **Precision.** Costs are stored in `f32`, each layer relative to its own minimum, and
  those minima accumulate in `f64`. The stored values stay small near the optimum
  whatever the path's length. This halves the tables against `f64`, and on random
  instances the max-marginals still match an `f64` brute force.
- **The smoother.** It has ADR-0018's penalty (damping, tension and bending lengths over
  the spacing) and per-sample base weights: 0 is a gap and a heavy weight is a pin. It
  runs IRLS under a schedule from the caller, so `fit::RobustLoss` and its annealing stay
  in the bead adaptor. ADR-0018's solve now runs on it, and its results are unchanged to
  the bit.
- **Invariant 13 is amended.** A structured solve that nalgebra lacks lives beside its
  callers, in the module that owns them, rather than beside its only caller.

## Alternatives

- **Public operators, for example a `numeric` module at the crate root.** That means a
  generic optimiser with a semver promise, bindings and no inspection case of its own.
  Reopen it when a module outside `measure` needs a core, by moving the core to a
  crate-private module first.
- **Keeping the banded solve inside `bead`.** The coupled extraction would then
  duplicate the solve or reach into another tool's internals.
- **`f64` tables.** They are simpler, but they double the tables' memory and halve the
  vector width of the band passes, which take most of the min-path's time, for no
  accuracy that the tests can see.
- **A different state count per layer.** A banded step needs grids that line up, so the
  layers share one grid and `+∞` marks the states a layer lacks.
- **A dynamic-programming crate.** None offers banded separable transitions with
  max-marginals, and ADR-0002 keeps the dependency list short.

## Consequences

- ADR-0019's coupled pairing is the min-path's first caller; the bead-end rim will be the
  second, and the first to pin.
- `benches/measure_numeric.rs` tracks both cores. Its numbers are in
  `docs/performance.md`.
- A caller's costs and units, such as ADR-0019's per-unit-length data and `1/h`
  smoothness, belong to the caller. The core only sees grid steps.
