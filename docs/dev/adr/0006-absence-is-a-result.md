# ADR-0006: A measurement that found nothing is a result

- Status: Accepted
- Date: 2026-08-19

## Context

A metrology call that finds nothing has a reason: the part is missing, the window is too
short, the only edge has the wrong polarity, the scan left the image. Returning an empty
slice throws that diagnosis away. Keeping a second "checked" twin of each call doubles the
API and invites the lossy one. A batch call that skips failed items also renumbers its
results, so the caller cannot tell which item failed.

## Decision

- `Caliper::measure` returns `Result<&[MeasureEdge], RejectReason>`. `Ok(&[])` cannot occur:
  an extraction that found nothing always carries a typed `RejectReason`.
- `MetrologyModel::apply` returns one `Result` per object, in `objects()` order. The caliper
  hits travel inside each result rather than in a parallel array.
- Cheap borrowing accessors (`profile()`, …) stay on the detector. Diagnostic
  **computation** (layout without an image, per-caliper explanation) lives in a
  `diagnostics` module, off the hot path.
- Python raises `MeasureRejected` with the reason as a string.

## Alternatives

- **An empty slice or `Option`.** It loses the reason.
- **`measure` plus `measure_checked`.** The same computation twice, with one variant
  discarding the diagnosis.

## Consequences

- Callers handle `Err` explicitly. A fixture reporting the dominant rejection reason across
  its calipers separates "part missing" from "caliper misplaced".
- New rejection modes are new `RejectReason` variants, which is a public API change.
