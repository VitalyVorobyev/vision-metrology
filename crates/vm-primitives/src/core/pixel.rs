//! The scalar pixel types this library reads.
//!
//! Every detector used to expose one entry point per pixel type — `detect_u8`,
//! `detect_u16`, `detect_f32` — which triplicated the public surface, the docs
//! and the tests, and still did not extend to a fourth type. [`Pixel`] collapses
//! those into a single generic entry point per algorithm. Monomorphisation emits
//! the same specialised code the hand-written triplets did.
//!
//! The trait is **sealed**: only `u8`, `u16` and `f32` implement it, and only
//! this crate can add more. That keeps adding a pixel type a non-breaking
//! change for downstream crates.
//!
//! ## What is deliberately *not* here
//!
//! There is no `FULL_SCALE` constant. `u8` and `u16` have an obvious full scale,
//! but an `f32` image in this workspace carries whatever numeric range its
//! producer chose — [`to_f32`](Pixel::to_f32) on a `u8` yields `0.0..=255.0`,
//! while a caller's own `f32` buffer may be `0.0..=1.0`. Thresholds therefore
//! stay absolute and documented per config field rather than being silently
//! rescaled.

use core::ops::Add;

mod sealed {
    /// Prevents downstream implementations of [`Pixel`](super::Pixel).
    pub trait Sealed {}
    impl Sealed for u8 {}
    impl Sealed for u16 {}
    impl Sealed for f32 {}
}

/// A scalar pixel type: `u8`, `u16` or `f32`.
///
/// Implemented for exactly those three and sealed against others. Generic
/// algorithms take `&ImageView<'_, P>` where `P: Pixel`.
///
/// # Example
/// ```
/// use vm_primitives::Pixel;
///
/// fn mean<P: Pixel>(px: &[P]) -> f32 {
///     px.iter().map(|p| p.to_f32()).sum::<f32>() / px.len() as f32
/// }
///
/// assert_eq!(mean(&[0u8, 255]), 127.5);
/// assert_eq!(mean(&[0u16, 100]), 50.0);
/// assert_eq!(mean(&[1.0f32, 2.0]), 1.5);
/// ```
pub trait Pixel: sealed::Sealed + Copy + Default + PartialOrd + core::fmt::Debug + 'static {
    /// Accumulator wide enough to sum a 2×2 block of this type without loss.
    ///
    /// `u32` for both integer types (`4 × u16::MAX` fits comfortably), `f32`
    /// for `f32` — which keeps float summation order and rounding identical to
    /// a hand-written `a + b + c + d`.
    type Acc: Copy + Add<Output = Self::Acc>;

    /// The additive identity, for clearing scratch buffers generically.
    const ZERO: Self;

    /// Value as `f32`. Exact for all three types.
    fn to_f32(self) -> f32;

    /// Nearest representable value of `v`, saturating at the type's range.
    ///
    /// Integer types round half-away-from-zero and clamp; `f32` is the
    /// identity. `NaN` maps to [`ZERO`](Pixel::ZERO) for the integer types,
    /// which is what Rust's `as` conversion already does.
    fn from_f32_sat(v: f32) -> Self;

    /// Widen for accumulation.
    fn to_acc(self) -> Self::Acc;

    /// Exact `f32` value of an accumulator.
    fn acc_to_f32(a: Self::Acc) -> f32;

    /// Mean of four accumulated samples, back in this pixel type.
    ///
    /// Integer types round half-up (`(sum + 2) / 4`); `f32` scales by `0.25`.
    /// This is the 2×2 box-mean kernel the pyramid is built from, so its exact
    /// arithmetic is load-bearing — see system-design invariants 2 and 3.
    fn acc_mean4(a: Self::Acc) -> Self;
}

impl Pixel for u8 {
    type Acc = u32;
    const ZERO: Self = 0;

    #[inline]
    fn to_f32(self) -> f32 {
        f32::from(self)
    }

    #[inline]
    fn from_f32_sat(v: f32) -> Self {
        // `as` saturates at the integer bounds and maps NaN to 0 since Rust 1.45.
        v.round() as u8
    }

    #[inline]
    fn to_acc(self) -> u32 {
        u32::from(self)
    }

    #[inline]
    fn acc_to_f32(a: u32) -> f32 {
        a as f32
    }

    #[inline]
    fn acc_mean4(a: u32) -> Self {
        ((a + 2) / 4) as u8
    }
}

impl Pixel for u16 {
    type Acc = u32;
    const ZERO: Self = 0;

    #[inline]
    fn to_f32(self) -> f32 {
        f32::from(self)
    }

    #[inline]
    fn from_f32_sat(v: f32) -> Self {
        v.round() as u16
    }

    #[inline]
    fn to_acc(self) -> u32 {
        u32::from(self)
    }

    #[inline]
    fn acc_to_f32(a: u32) -> f32 {
        a as f32
    }

    #[inline]
    fn acc_mean4(a: u32) -> Self {
        ((a + 2) / 4) as u16
    }
}

impl Pixel for f32 {
    type Acc = f32;
    const ZERO: Self = 0.0;

    #[inline]
    fn to_f32(self) -> f32 {
        self
    }

    #[inline]
    fn from_f32_sat(v: f32) -> Self {
        v
    }

    #[inline]
    fn to_acc(self) -> f32 {
        self
    }

    #[inline]
    fn acc_to_f32(a: f32) -> f32 {
        a
    }

    #[inline]
    fn acc_mean4(a: f32) -> Self {
        a * 0.25
    }
}

#[cfg(test)]
mod tests {
    use super::Pixel;

    /// The rounding rule the integer pyramid levels depend on: half rounds up.
    #[test]
    fn integer_box_mean_rounds_half_up() {
        // 1+1+1+2 = 5 -> 5/4 = 1.25 -> 1
        assert_eq!(u8::acc_mean4(5), 1);
        // 1+1+2+2 = 6 -> 1.5 -> 2 (half up, not banker's)
        assert_eq!(u8::acc_mean4(6), 2);
        // 2+2+2+1 = 7 -> 1.75 -> 2
        assert_eq!(u8::acc_mean4(7), 2);
        assert_eq!(u16::acc_mean4(6), 2);
        // Saturating input: four maxima average back to the maximum.
        assert_eq!(u8::acc_mean4(4 * 255), 255);
        assert_eq!(u16::acc_mean4(4 * 65535), 65535);
    }

    #[test]
    fn float_box_mean_is_a_plain_quarter() {
        assert_eq!(f32::acc_mean4(6.0), 1.5);
        assert_eq!(f32::acc_mean4(-2.0), -0.5);
    }

    /// `Acc` must hold the largest 2x2 sum each type can produce.
    #[test]
    fn accumulator_holds_a_full_2x2_block() {
        let s = (0..4).fold(0u32, |a, _| a + u8::MAX.to_acc());
        assert_eq!(s, 1020);
        let s = (0..4).fold(0u32, |a, _| a + u16::MAX.to_acc());
        assert_eq!(s, 262_140);
        assert_eq!(u16::acc_to_f32(s), 262_140.0);
    }

    #[test]
    fn from_f32_saturates_and_rounds() {
        assert_eq!(u8::from_f32_sat(-5.0), 0);
        assert_eq!(u8::from_f32_sat(300.0), 255);
        assert_eq!(u8::from_f32_sat(2.5), 3);
        assert_eq!(u8::from_f32_sat(f32::NAN), 0);
        assert_eq!(u16::from_f32_sat(70_000.0), 65535);
        assert_eq!(f32::from_f32_sat(1.25), 1.25);
    }

    #[test]
    fn round_trips_through_f32() {
        for v in [0u8, 1, 127, 128, 254, 255] {
            assert_eq!(u8::from_f32_sat(v.to_f32()), v);
        }
        for v in [0u16, 1, 32767, 65535] {
            assert_eq!(u16::from_f32_sat(v.to_f32()), v);
        }
    }
}
