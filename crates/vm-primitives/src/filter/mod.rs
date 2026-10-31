//! Linear filters (roadmap B3, ADR-0023).
//!
//! Phase 1: the folded symmetric [`Kernel1D`] and the exact separable FIR, crate-internal.
//! The public API, its Python parity and the box/binomial/large-sigma methods arrive in
//! later phases; until then nothing outside the crate reaches these items, hence the allow.

#![allow(dead_code, unused_imports)]

mod fir;
mod kernel;

pub(crate) use fir::{FirScratch, separable_fir_f32};
pub(crate) use kernel::Kernel1D;
