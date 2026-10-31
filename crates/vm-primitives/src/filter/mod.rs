//! Linear filters (ADR-0023).
//!
//! Phase 1: the folded symmetric [`Kernel1D`] and the exact separable FIR on `f32`.
//! Phase 2: exact integer box mean and binomial smoothing on `u8`, with runtime CPU dispatch.
//! Everything is crate-internal until the public API lands in Phase 4; until then nothing
//! outside the crate reaches these items, hence the allow.

#![allow(dead_code, unused_imports)]

mod fir;
mod integer;
mod kernel;

pub(crate) use fir::{FirScratch, separable_fir_f32};
pub(crate) use integer::{IntScratch, binomial_u8, box_mean_u8};
pub(crate) use kernel::Kernel1D;
