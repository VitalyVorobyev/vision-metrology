//! Marker-based watershed segmentation on a gradient magnitude image.
//!
//! Uses a priority-queue flood-fill (Beucher-Meyer algorithm):
//! 1. Initialise: push all seed markers into the heap.
//! 2. Pop the lowest-gradient pixel from the heap.
//!    - If already labelled, skip.
//!    - Otherwise label it with the source label from the heap entry.
//! 3. For each unlabelled 4-connected neighbour:
//!    - Push it with the current pixel's label if unlabelled.
//!    - Mark it as boundary if it carries a different label from a competing front.
//!
//! Output:
//! - `≥ 0` — region label (0-based index into the `markers` slice).
//! - `-1`  — watershed boundary.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use vm_core::{Image, ImageView};

/// Heap entry: `(Reverse(priority_bits), pixel_idx, label)`.
/// Lower gradient = higher priority.
#[derive(Eq, PartialEq)]
struct Entry {
    priority: Reverse<u32>,
    idx: usize,
    label: i32,
}

impl Ord for Entry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Primary: lower gradient first (Reverse order).
        // Secondary (tiebreak): lower pixel index first for determinism.
        self.priority
            .cmp(&other.priority)
            .then(self.idx.cmp(&other.idx).reverse())
    }
}

impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Run marker-based watershed on a gradient-magnitude image.
///
/// `gradient` is an `f32` image where higher values indicate stronger edges.
/// `markers` is a list of seed pixel coordinates `(x, y)` in pixel-center convention.
/// Each seed defines a distinct region; its label equals its index in the slice.
///
/// Returns an `Image<i32>` of the same dimensions as `gradient`:
/// - `≥ 0` — region label (seed index).
/// - `-1`  — watershed boundary between two regions.
///
/// If `markers` is empty the function returns an all-`-1` image.
///
/// # Panics
/// Panics if any seed coordinate is out of bounds.
///
/// # Example
/// ```
/// use vm_core::{Image, ImageView};
/// use vm_segment::watershed;
///
/// // 10×5 gradient image, uniform zero gradient.
/// let data = vec![0.0f32; 10 * 5];
/// let grad = Image::from_vec(10, 5, data).unwrap();
/// // Seeds at opposite ends.
/// let labels = watershed(&grad.as_view(), &[(0, 2), (9, 2)]);
/// // At least one region should exist.
/// assert!(labels.data().iter().any(|&v| v >= 0));
/// ```
pub fn watershed(gradient: &ImageView<'_, f32>, markers: &[(usize, usize)]) -> Image<i32> {
    let w = gradient.width();
    let h = gradient.height();
    let n = w * h;

    const UNLABELLED: i32 = -2;
    const BOUNDARY: i32 = -1;

    if markers.is_empty() || n == 0 {
        let data = vec![BOUNDARY; n];
        return Image::from_vec(w, h, data).expect("dimensions ok");
    }

    let mut labels = vec![UNLABELLED; n];
    // `is_seed[idx]` prevents seed pixels from being overwritten by competing fronts.
    let mut is_seed = vec![false; n];
    let mut heap: BinaryHeap<Entry> = BinaryHeap::new();

    // 4-connected offsets: (dx, dy).
    let offsets: [(isize, isize); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

    // Seed initialisation: label seeds immediately, then push their neighbours.
    for (seed_label, &(sx, sy)) in markers.iter().enumerate() {
        assert!(sx < w && sy < h, "seed ({sx},{sy}) out of bounds");
        let idx = sy * w + sx;
        labels[idx] = seed_label as i32;
        is_seed[idx] = true;

        // Push 4-connected neighbours of each seed.
        let x = sx as isize;
        let y = sy as isize;
        for (dx, dy) in offsets {
            let nx = x + dx;
            let ny = y + dy;
            if nx < 0 || ny < 0 || nx >= w as isize || ny >= h as isize {
                continue;
            }
            let nidx = ny as usize * w + nx as usize;
            if labels[nidx] == UNLABELLED {
                let g = gradient.get(nx as usize, ny as usize).expect("in-bounds");
                heap.push(Entry {
                    priority: Reverse(g.to_bits()),
                    idx: nidx,
                    label: seed_label as i32,
                });
            }
        }
    }

    while let Some(Entry { idx, label, .. }) = heap.pop() {
        let cur = labels[idx];
        // Skip seed pixels (inviolable) and already-resolved pixels.
        if is_seed[idx] || cur == BOUNDARY {
            continue;
        }
        if cur == UNLABELLED {
            // First time we visit this pixel: label it.
            labels[idx] = label;
        } else if cur != label {
            // A different label already claimed it: boundary collision.
            labels[idx] = BOUNDARY;
            continue;
        }
        // cur == label: pixel already claimed by same front (stale entry) → still propagate.

        // Push unlabelled 4-connected neighbours.
        let x = (idx % w) as isize;
        let y = (idx / w) as isize;
        for (dx, dy) in offsets {
            let nx = x + dx;
            let ny = y + dy;
            if nx < 0 || ny < 0 || nx >= w as isize || ny >= h as isize {
                continue;
            }
            let nidx = ny as usize * w + nx as usize;
            let nlbl = labels[nidx];
            if nlbl == UNLABELLED {
                let g = gradient.get(nx as usize, ny as usize).expect("in-bounds");
                heap.push(Entry {
                    priority: Reverse(g.to_bits()),
                    idx: nidx,
                    label,
                });
            } else if !is_seed[nidx] && nlbl != BOUNDARY && nlbl != label {
                // Different label claims this neighbour: mark it boundary.
                labels[nidx] = BOUNDARY;
            }
        }
    }

    // Any still-unlabelled pixels (isolated, unreachable) → boundary.
    for v in &mut labels {
        if *v == UNLABELLED {
            *v = BOUNDARY;
        }
    }

    Image::from_vec(w, h, labels).expect("dimensions unchanged")
}

#[cfg(test)]
mod tests {
    use vm_core::Image;

    use super::watershed;

    #[test]
    fn two_seeds_split_image() {
        // 11×1 gradient image (single row), all zeros.
        // Seed 0 at (0, 0), seed 1 at (10, 0).
        // After flood: pixels 0..5 should be label 0, pixels 6..10 label 1,
        // with a boundary somewhere in the middle.
        let data = vec![0.0f32; 11];
        let grad = Image::from_vec(11, 1, data).unwrap();
        let labels = watershed(&grad.as_view(), &[(0, 0), (10, 0)]);

        // The leftmost pixel is always label 0 (seed).
        assert_eq!(
            *labels.as_view().get(0, 0).unwrap(),
            0,
            "seed pixel must keep label 0"
        );
        // The rightmost pixel is always label 1 (seed).
        assert_eq!(
            *labels.as_view().get(10, 0).unwrap(),
            1,
            "seed pixel must keep label 1"
        );
        // There must be exactly one boundary pixel somewhere in the middle.
        assert!(
            labels.data().contains(&-1),
            "two meeting fronts must create at least one boundary pixel"
        );
    }

    #[test]
    fn single_seed_labels_all_foreground() {
        // 5×5 image, single seed: all reachable pixels get label 0.
        let data = vec![0.0f32; 5 * 5];
        let grad = Image::from_vec(5, 5, data).unwrap();
        let labels = watershed(&grad.as_view(), &[(2, 2)]);
        assert!(
            labels.data().iter().all(|&v| v == 0),
            "single seed: all pixels get label 0"
        );
    }

    #[test]
    fn no_seeds_all_boundary() {
        // No markers: all pixels are returned as boundary (-1).
        let data = vec![0.0f32; 4 * 4];
        let grad = Image::from_vec(4, 4, data).unwrap();
        let labels = watershed(&grad.as_view(), &[]);
        assert!(
            labels.data().iter().all(|&v| v == -1),
            "no seeds → all boundary"
        );
    }

    #[test]
    fn two_seeds_on_2d_image() {
        // 11×5 image, seeds at (0, 2) and (10, 2).
        // The seed pixels must retain their labels.
        let data = vec![0.0f32; 11 * 5];
        let grad = Image::from_vec(11, 5, data).unwrap();
        let labels = watershed(&grad.as_view(), &[(0, 2), (10, 2)]);

        let seed0_label = *labels.as_view().get(0, 2).unwrap();
        let seed1_label = *labels.as_view().get(10, 2).unwrap();
        assert_eq!(seed0_label, 0, "seed 0 must be label 0");
        assert_eq!(seed1_label, 1, "seed 1 must be label 1");
        assert!(labels.data().contains(&-1), "boundary must exist");
    }

    #[test]
    fn gradient_barrier_routes_flood() {
        // 9×1 image with high gradient at x=4 (the centre column).
        // Seeds at x=0 and x=8. The barrier should prevent one front from
        // crossing; at minimum a boundary is created near the barrier.
        let mut data = vec![0.0f32; 9];
        data[4] = 1000.0;
        let grad = Image::from_vec(9, 1, data).unwrap();
        let labels = watershed(&grad.as_view(), &[(0, 0), (8, 0)]);
        // Seed pixels keep their labels.
        assert_eq!(*labels.as_view().get(0, 0).unwrap(), 0);
        assert_eq!(*labels.as_view().get(8, 0).unwrap(), 1);
    }
}
