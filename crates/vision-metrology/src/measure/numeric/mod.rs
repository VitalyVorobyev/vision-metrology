//! Numerical cores shared inside `measure`: an exact layered min-path with max-marginals,
//! and a robust Whittaker smoother (ADR-0020).
//!
//! Both are crate-private. The bead tracker's regularised solve (ADR-0018) runs on the
//! smoother. The coupled two-boundary extraction and the bead-end rim are the min-path's
//! callers; until they land it is exercised by its tests, its bench and its allocation
//! gate only.

// The callers of the min-path are not merged yet; its tests use all of it.
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "its callers, the coupled extraction and the bead-end rim, are not merged yet"
    )
)]
pub(crate) mod minpath;
pub(crate) mod whittaker;

#[cfg(test)]
mod tests;
