# ADR-0005: The stored shape model is opaque, read-only and versioned

- Status: Accepted
- Date: 2026-08-19 (format 5: 2026-08-21)

## Context

A saved `ShapeModel` is the one artefact of this crate that outlives a build. Every change
to its contents can invalidate files on disk. The search also trusts several of its derived
quantities.

## Decision

- **Opaque.** The API is `save`/`load` and `to_bytes`/`from_bytes`. The encoding (JSON today)
  is not a promise, and the format version is `pub(crate)`. `load` refuses a foreign or
  too-old document with an error instead of mis-reading it.
- **Readable, not writable.** `ModelPoint` and `ShapeModelLevel` can be read for overlays but
  not constructed by callers. Point order is load-bearing (greedy termination evaluates a
  prefix, which must sample the whole contour), and `radius`, `angle_step` and `scale_step`
  are derived.
- **The model carries its own `PreSmooth`.** `ShapeMatcher::find` reads it off the model,
  which makes invariant 3 impossible to break from the search config.
- **Breaking bumps are batched; additive bumps are backward-loading.**
  - Format 3 is the minimum that `load` accepts, because below it the document shape differs.
  - Every later format only adds `#[serde(default)]` fields, and an older document reads the
    value the field always implicitly meant. Format 4 added `teach_points` (an older model
    reads none, and resampling refuses cleanly). Format 5 added `reference_angle` (an older
    model reads 0).
  - The current format is 5. The per-version table is the doc comment on
    `matching::model::persist::FORMAT_VERSION`.

## Alternatives

- **A documented wire format with a public version constant.** It makes every internal field
  a compatibility obligation, and the constant only helps callers hand-assemble documents
  that this crate should be the only writer of.
- **A breaking bump per change.** It invalidates users' models for additions that have a
  natural default.

## Consequences

- Removing or re-meaning a field is a breaking bump, batched with others and recorded in the
  changelog.
- Models are portable across releases at and above the minimum format.
