//! Numerical cores shared inside `measure`: an exact layered min-path with max-marginals,
//! and a robust Whittaker smoother (ADR-0020).
//!
//! Both are crate-private. The bead tracker's regularised solve (ADR-0018) runs on the
//! smoother, and its coupled pairing (ADR-0019) on the min-path. The bead-end rim will
//! be the min-path's second caller, and the first to pin.

// Soft pins have no caller until the bead-end rim lands; the tests use them.
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "soft pins' caller, the bead-end rim, is not merged yet"
    )
)]
pub(crate) mod minpath;
pub(crate) mod whittaker;

#[cfg(test)]
mod tests;
