//! Marker-based watershed segmentation on a gradient magnitude image.
//!
//! Uses a priority-queue flood-fill (Beucher-Meyer algorithm):
//! 1. Initialise: label each seed and push its unlabelled 4-neighbours.
//! 2. Pop the lowest-gradient pixel from the heap.
//!    - If it is a seed or already a boundary, skip it.
//!    - If a *different* label already claimed it, it is where two fronts met:
//!      mark it a boundary and stop.
//!    - If the *same* label already claimed it, the entry is a stale duplicate:
//!      drop it.
//!    - Otherwise label it and push its unlabelled 4-neighbours.
//!
//! Two properties of the ordering matter and are easy to get wrong:
//!
//! - **Ties break FIFO, not by pixel index.** Within a plateau every pixel has
//!   the same priority, so the tiebreak alone decides how the fronts advance.
//!   FIFO advances them in lock-step and partitions the plateau by geodesic
//!   distance from the seeds. Ordering by raster index instead lets the
//!   lowest-index seed run away and swallow the plateau.
//! - **A pixel propagates exactly once**, when it leaves `UNLABELLED`.
//!   Re-propagating on stale pops compounds heap entries along the front and
//!   makes the runtime exponential in the image size.
//!
//! Output:
//! - `≥ 0` — region label (0-based index into the `markers` slice).
//! - `-1`  — watershed boundary.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use vm_primitives::{Image, ImageView};

/// Heap entry: `(Reverse(priority_bits), insertion_seq, pixel_idx, label)`.
/// Lower gradient = higher priority.
#[derive(Eq, PartialEq)]
struct Entry {
    priority: Reverse<u32>,
    /// Monotonic insertion counter, used to break priority ties in FIFO order.
    seq: u64,
    idx: usize,
    label: i32,
}

impl Ord for Entry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // `BinaryHeap` is a max-heap, so "greater" means "popped sooner".
        //
        // Primary: lower gradient first (hence `Reverse`).
        //
        // Secondary: **FIFO** among equal gradients, i.e. smaller `seq` is
        // greater. This is what makes plateaus come out right. On a flat
        // region every pixel has the same priority, so the tiebreak alone
        // decides the order the fronts advance in. FIFO advances every front
        // in lock-step, which partitions a plateau by geodesic distance from
        // the seeds -- the defining property of the Beucher-Meyer flood.
        // Breaking ties on `idx` instead would order the frontier by raster
        // position, letting the lowest-index seed run away and swallow the
        // whole plateau.
        //
        // `seq` is unique per push, so this is a total order and the output is
        // deterministic.
        self.priority
            .cmp(&other.priority)
            .then_with(|| other.seq.cmp(&self.seq))
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
/// use vm_primitives::{Image, ImageView};
/// use vision_metrology::segment::watershed;
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
    // Monotonic push counter backing the FIFO tiebreak in `Entry::cmp`.
    let mut seq: u64 = 0;

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
                    seq,
                    idx: nidx,
                    label: seed_label as i32,
                });
                seq += 1;
            }
            // A neighbour that is already labelled is deliberately left alone.
            // It becomes a watershed pixel only if a competing front pops it
            // with a different label (handled at the top of this loop), which
            // is the Beucher-Meyer rule. Retroactively converting an already
            // claimed neighbour to BOUNDARY here instead would steal a pixel
            // from the front that legitimately reached it first, widening every
            // boundary to 2 px and biasing it toward the lower-numbered seed.
        }
    }

    while let Some(Entry { idx, label, .. }) = heap.pop() {
        let cur = labels[idx];
        // Skip seed pixels (inviolable) and already-resolved pixels.
        if is_seed[idx] || cur == BOUNDARY {
            continue;
        }
        if cur != UNLABELLED {
            // Already claimed. A competing front turns it into a boundary; a
            // stale duplicate entry from our own front is simply dropped.
            //
            // Dropping it is what keeps the algorithm near-linear. A pixel's
            // neighbours are pushed exactly once, at the moment it transitions
            // out of UNLABELLED. Labels only ever move
            // UNLABELLED -> label -> BOUNDARY and never back, so re-scanning on
            // a stale pop can never find a neighbour the first pop missed --
            // it only re-pushes entries, which compounds along the front and
            // makes the runtime exponential on plateaus (see the flat-image
            // regression test below).
            if cur != label {
                labels[idx] = BOUNDARY;
            }
            continue;
        }

        // First time we visit this pixel: label it, then propagate.
        labels[idx] = label;

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
                    seq,
                    idx: nidx,
                    label,
                });
                seq += 1;
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
    use vm_primitives::Image;

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
        // On a flat gradient the fronts advance at equal speed, so the split
        // must land exactly on the geodesic midline: 0..=4 | boundary | 6..=10.
        assert_eq!(
            labels.data(),
            &[0, 0, 0, 0, 0, -1, 1, 1, 1, 1, 1],
            "flat 11x1 with seeds at both ends must split at the midline"
        );
    }

    /// Renders a label image as one char per pixel: `#` for boundary, else the
    /// label digit. Keeps the plateau expectations below readable.
    fn render(labels: &Image<i32>, w: usize, h: usize) -> Vec<String> {
        (0..h)
            .map(|y| {
                (0..w)
                    .map(|x| match labels.data()[y * w + x] {
                        -1 => '#',
                        v => char::from(b'0' + v as u8),
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn flat_plateau_splits_at_geodesic_midline() {
        // Every pixel of a flat image has the same priority, so the tiebreak
        // alone decides how the fronts advance. FIFO order makes them advance
        // in lock-step, putting the boundary exactly halfway between the seeds.
        let grad = Image::from_vec(7, 7, vec![0.0f32; 7 * 7]).unwrap();

        let left_right = watershed(&grad.as_view(), &[(0, 3), (6, 3)]);
        assert_eq!(
            render(&left_right, 7, 7),
            vec!["000#111"; 7],
            "seeds left and right of a plateau must give a vertical midline"
        );

        let top_bottom = watershed(&grad.as_view(), &[(3, 0), (3, 6)]);
        assert_eq!(
            render(&top_bottom, 7, 7),
            vec![
                "0000000", "0000000", "0000000", "#######", "1111111", "1111111", "1111111"
            ],
            "seeds above and below a plateau must give a horizontal midline"
        );
    }

    #[test]
    fn flat_plateau_four_seeds_form_a_cross() {
        // Four corner seeds on a plateau partition it into four equal quadrants
        // separated by a one-pixel cross.
        let grad = Image::from_vec(9, 9, vec![0.0f32; 9 * 9]).unwrap();
        let labels = watershed(&grad.as_view(), &[(0, 0), (8, 0), (0, 8), (8, 8)]);
        assert_eq!(
            render(&labels, 9, 9),
            vec![
                "0000#1111",
                "0000#1111",
                "0000#1111",
                "0000#1111",
                "#########",
                "2222#3333",
                "2222#3333",
                "2222#3333",
                "2222#3333",
            ],
            "four corner seeds must produce four equal quadrants and a 1-px cross"
        );
    }

    #[test]
    fn large_flat_image_is_not_exponential() {
        // Regression guard. A stale heap entry must not re-propagate: a pixel's
        // neighbours are pushed exactly once, when it leaves UNLABELLED.
        // Re-pushing on every stale pop compounds along the front and made this
        // exponential in the image size -- a 14x14 plateau took 1.6 s and
        // 1280x1024 never finished. A regression here manifests as a hang, so
        // this test is deliberately large enough that the old code could not
        // complete it.
        let (w, h) = (128, 128);
        let grad = Image::from_vec(w, h, vec![0.0f32; w * h]).unwrap();
        let labels = watershed(&grad.as_view(), &[(0, 64), (127, 64)]);

        // Same midline property as the 7x7 case, at a size the old code could
        // not reach. With an even width the exact midpoint (63.5) falls between
        // pixels, so the front that arrives first claims through column 63 and
        // the collision lands on the single column 64.
        for y in 0..h {
            for x in 0..w {
                let got = labels.data()[y * w + x];
                let want = match x {
                    0..=63 => 0,
                    64 => -1,
                    _ => 1,
                };
                assert_eq!(got, want, "pixel ({x},{y}) on a 128x128 plateau");
            }
        }
    }

    #[test]
    fn output_is_deterministic() {
        // `seq` is unique per push, so `Entry::cmp` is a total order and repeated
        // runs must agree bit for bit.
        let w = 24;
        let grad = Image::from_vec(w, w, vec![0.0f32; w * w]).unwrap();
        let seeds = [(0, 0), (w - 1, 0), (w / 2, w - 1)];
        let a = watershed(&grad.as_view(), &seeds);
        let b = watershed(&grad.as_view(), &seeds);
        assert_eq!(a.data(), b.data(), "watershed output must be deterministic");
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
