# ADR-0016: Scope: what this library deliberately does not build

- Status: Accepted
- Date: 2026-08-19

## Context

A metrology library is judged on the accuracy and determinism of a small set of operators,
not on breadth. Each feature carries maintenance, an accuracy obligation and Python parity
(invariant 15).

## Decision

These are out of scope until a concrete inspection case needs them:

- **Watershed variants.** `segment::watershed` stays a single tool; there are no
  marker-controlled or hierarchical variants.
- **Rank filters beyond median.** The planned `filter` module covers Gaussian, box mean,
  median and grayscale morphology.
- **FFT-based methods.** No periodic-pattern removal and no frequency-domain correlation;
  `corr` covers correlation in the spatial domain.
- **Timeouts or anytime search.** They make results depend on machine load, which breaks the
  determinism contract (invariant 12). A deterministic budget, such as a maximum number of
  poses evaluated, would be acceptable.
- **A multi-scale edge detector without a real scale space.** A box-mean pyramid with a
  fixed-σ derivative of Gaussian is not a scale space. Scale selection, if needed, gets
  designed as one.

## Alternatives

Building them speculatively. Each would need its own accuracy rows and bindings before
anyone used it.

## Consequences

Requests in these areas start as a backlog item with the use case that motivates them,
followed by an update to this ADR.
