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

#[cfg(test)]
mod tests {
    use super::StructuringElement;

    #[test]
    fn radius_is_reported_for_both_shapes() {
        assert_eq!(StructuringElement::Square(0).radius(), 0);
        assert_eq!(StructuringElement::Square(3).radius(), 3);
        assert_eq!(StructuringElement::Disk(0).radius(), 0);
        assert_eq!(StructuringElement::Disk(5).radius(), 5);
    }

    #[test]
    fn radius_zero_contains_only_the_centre() {
        for se in [StructuringElement::Square(0), StructuringElement::Disk(0)] {
            assert!(se.contains(0, 0), "{se:?} must contain its own centre");
            for &(dx, dy) in &[(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1)] {
                assert!(!se.contains(dx, dy), "{se:?} must not contain ({dx}, {dy})");
            }
        }
    }

    #[test]
    fn square_is_the_full_chebyshev_ball() {
        // Square(1) is the classic 3x3: every offset with |dx| <= 1 and |dy| <= 1.
        let se = StructuringElement::Square(1);
        let mut count = 0;
        for dy in -2..=2_isize {
            for dx in -2..=2_isize {
                let want = dx.abs() <= 1 && dy.abs() <= 1;
                assert_eq!(se.contains(dx, dy), want, "at ({dx}, {dy})");
                count += usize::from(want);
            }
        }
        assert_eq!(count, 9, "Square(1) covers 3x3 = 9 offsets");
    }

    #[test]
    fn disk_uses_an_inclusive_radius() {
        // Documented rule: inside when dx^2 + dy^2 <= r^2, so the axis extremes
        // are included and the diagonal at the same distance is not.
        let se = StructuringElement::Disk(2);
        assert!(se.contains(2, 0), "(2,0) is exactly at radius 2");
        assert!(se.contains(0, -2));
        assert!(se.contains(1, 1), "1+1=2 <= 4");
        assert!(!se.contains(2, 2), "4+4=8 > 4");
        assert!(!se.contains(3, 0));
    }

    #[test]
    fn disk_is_contained_in_the_square_of_equal_radius() {
        // A disk never reaches beyond the Chebyshev ball of the same radius.
        for r in 0..6usize {
            let disk = StructuringElement::Disk(r);
            let square = StructuringElement::Square(r);
            let bound = r as isize + 2;
            for dy in -bound..=bound {
                for dx in -bound..=bound {
                    if disk.contains(dx, dy) {
                        assert!(
                            square.contains(dx, dy),
                            "Disk({r}) contains ({dx}, {dy}) but Square({r}) does not"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn membership_is_symmetric_under_reflection() {
        // Both shapes are centrally symmetric, which morphological duality
        // (erosion/dilation adjointness) relies on.
        for se in [StructuringElement::Square(3), StructuringElement::Disk(3)] {
            for dy in -4..=4_isize {
                for dx in -4..=4_isize {
                    assert_eq!(
                        se.contains(dx, dy),
                        se.contains(-dx, -dy),
                        "{se:?} asymmetric at ({dx}, {dy})"
                    );
                    assert_eq!(se.contains(dx, dy), se.contains(dy, dx));
                }
            }
        }
    }
}
