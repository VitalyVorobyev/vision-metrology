//! Structuring element definitions for morphological operations.

/// Flat structuring element shape for binary morphology.
///
/// Pixel values are treated as binary: `> 0` is foreground.
/// The structuring element is always centered on the pixel being processed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StructuringElement {
    /// Square of side `2*radius + 1` (e.g., `radius = 1` gives the classic 3×3 square).
    Square(usize),
    /// Disk (filled circle) with the given pixel radius.
    ///
    /// A pixel `(dx, dy)` is inside the disk when `dx² + dy² ≤ radius²`.
    Disk(usize),
}

impl StructuringElement {
    /// Radius in pixels (half the kernel width, rounded down).
    pub fn radius(&self) -> usize {
        match self {
            Self::Square(r) | Self::Disk(r) => *r,
        }
    }

    /// Return `true` if the offset `(dx, dy)` from the center belongs to this SE.
    #[inline]
    pub fn contains(&self, dx: isize, dy: isize) -> bool {
        match self {
            Self::Square(r) => dx.unsigned_abs() <= *r && dy.unsigned_abs() <= *r,
            Self::Disk(r) => {
                let r2 = (*r * *r) as isize;
                dx * dx + dy * dy <= r2
            }
        }
    }
}
