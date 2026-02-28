//! Connected-component labeling (CCL) using two-pass union-find.
//!
//! Implements Rosenfeld-Pfaltz two-pass algorithm with union-find (path compression
//! + union-by-rank) for O(N α(N)) total complexity.
//!
//! Label 0 is always background. Labels 1..=num_labels are foreground components.

use vm_contour::Connectivity;
use vm_core::{Image, ImageView, Point2f, Rect2f};

// ---------------------------------------------------------------------------
// Union-Find
// ---------------------------------------------------------------------------

struct UnionFind {
    parent: Vec<u32>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        let parent = (0..n as u32).collect();
        let rank = vec![0u8; n];
        Self { parent, rank }
    }

    fn find(&mut self, mut x: u32) -> u32 {
        // Path compression (iterative).
        while self.parent[x as usize] != x {
            // Path halving: point each node to its grandparent.
            let gp = self.parent[self.parent[x as usize] as usize];
            self.parent[x as usize] = gp;
            x = gp;
        }
        x
    }

    fn union(&mut self, a: u32, b: u32) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        // Union by rank.
        match self.rank[ra as usize].cmp(&self.rank[rb as usize]) {
            std::cmp::Ordering::Less => self.parent[ra as usize] = rb,
            std::cmp::Ordering::Greater => self.parent[rb as usize] = ra,
            std::cmp::Ordering::Equal => {
                self.parent[rb as usize] = ra;
                self.rank[ra as usize] += 1;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CcLabel output type
// ---------------------------------------------------------------------------

/// Result of connected-component labeling.
///
/// `label_map` has the same dimensions as the input binary image.
/// Pixel value `0` means background; values `1..=num_labels` are foreground
/// component labels, assigned in raster order of first encounter.
pub struct CcLabel {
    /// Label image: 0 = background, 1..=num_labels = foreground components.
    pub label_map: Image<u32>,
    /// Number of distinct foreground components found.
    pub num_labels: u32,
}

// ---------------------------------------------------------------------------
// Two-pass CCL
// ---------------------------------------------------------------------------

/// Label connected foreground components in a binary `u8` image.
///
/// Foreground pixels satisfy `pixel > 0`. The output `label_map` assigns each
/// foreground pixel a label in `1..=num_labels`; background pixels get `0`.
///
/// `connectivity` controls which neighbors are considered connected:
/// - [`Connectivity::C4`]: only axis-aligned (N, S, E, W).
/// - [`Connectivity::C8`]: includes diagonals.
///
/// The algorithm is the classic two-pass Rosenfeld-Pfaltz method with
/// union-find (path compression + union-by-rank) for near-linear performance.
///
/// # Example
/// ```
/// use vm_core::{Image, ImageView};
/// use vm_contour::Connectivity;
/// use vm_segment::label_connected_components_u8;
///
/// // 5×5 image: single 3×3 white rectangle in the center.
/// let mut data = vec![0u8; 5 * 5];
/// for dy in 0..3 { for dx in 0..3 { data[(1 + dy) * 5 + (1 + dx)] = 255; } }
/// let img = Image::from_vec(5, 5, data).unwrap();
/// let ccl = label_connected_components_u8(&img.as_view(), Connectivity::C4);
/// assert_eq!(ccl.num_labels, 1, "single rectangle → one component");
/// ```
pub fn label_connected_components_u8(
    binary: &ImageView<'_, u8>,
    connectivity: Connectivity,
) -> CcLabel {
    let w = binary.width();
    let h = binary.height();
    let n = w * h;

    if n == 0 {
        return CcLabel {
            label_map: Image::from_vec(w, h, Vec::new()).expect("0×0 image"),
            num_labels: 0,
        };
    }

    // Provisional label map (0 = background, label = provisional root).
    let mut prov = vec![0u32; n];
    // Union-find with at most n/2 + 1 labels possible.
    let max_labels = n / 2 + 2;
    let mut uf = UnionFind::new(max_labels);
    let mut next_label = 1u32;

    // --- Pass 1: assign provisional labels and record equivalences ---
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if *binary.get(x, y).expect("in-bounds") == 0 {
                continue;
            }

            // Collect neighbor labels already assigned (above / left in raster order).
            let mut neighbors = [0u32; 4];
            let mut nn = 0usize;

            // West (C4 + C8)
            if x > 0 && prov[idx - 1] != 0 {
                neighbors[nn] = prov[idx - 1];
                nn += 1;
            }
            // North (C4 + C8)
            if y > 0 && prov[idx - w] != 0 {
                neighbors[nn] = prov[idx - w];
                nn += 1;
            }
            if connectivity == Connectivity::C8 {
                // NW
                if x > 0 && y > 0 && prov[idx - w - 1] != 0 {
                    neighbors[nn] = prov[idx - w - 1];
                    nn += 1;
                }
                // NE
                if x + 1 < w && y > 0 && prov[idx - w + 1] != 0 {
                    neighbors[nn] = prov[idx - w + 1];
                    nn += 1;
                }
            }

            if nn == 0 {
                // New label.
                debug_assert!(next_label < max_labels as u32);
                prov[idx] = next_label;
                next_label += 1;
            } else {
                // Minimum root among neighbors.
                let mut root = uf.find(neighbors[0]);
                for &nb in &neighbors[1..nn] {
                    let r = uf.find(nb);
                    if r != root {
                        uf.union(root, r);
                        root = uf.find(root);
                    }
                }
                prov[idx] = root;
            }
        }
    }

    // --- Relabeling: compress roots to 1..=num_labels ---
    let mut relabel = vec![0u32; max_labels];
    let mut num_labels = 0u32;

    for label in prov.iter_mut().take(n) {
        if *label == 0 {
            continue;
        }
        let root = uf.find(*label);
        if relabel[root as usize] == 0 {
            num_labels += 1;
            relabel[root as usize] = num_labels;
        }
        *label = relabel[root as usize];
    }

    let label_map = Image::from_vec(w, h, prov).expect("dimensions unchanged");
    CcLabel {
        label_map,
        num_labels,
    }
}

// ---------------------------------------------------------------------------
// Component statistics
// ---------------------------------------------------------------------------

/// Per-component statistics derived from a [`CcLabel`] result.
#[derive(Debug, Clone)]
pub struct ComponentStats {
    /// Component label (1-based index into `CcLabel.label_map`).
    pub label: u32,
    /// Number of foreground pixels in this component.
    pub pixel_count: u32,
    /// Axis-aligned bounding box of the component (in pixel-center coordinates).
    pub bbox: Rect2f,
    /// Centroid of the component (average pixel-center position).
    pub centroid: Point2f,
}

/// Compute per-component statistics from a [`CcLabel`] result.
///
/// Components with fewer than `min_area` pixels are excluded from the output.
/// The returned `Vec` is sorted by `label` in ascending order.
///
/// Bounding box coordinates and centroid use pixel-center convention:
/// pixel `(x, y)` is at floating-point position `(x as f32, y as f32)`.
pub fn component_stats(cl: &CcLabel, min_area: u32) -> Vec<ComponentStats> {
    let w = cl.label_map.width();
    let h = cl.label_map.height();
    let n = cl.num_labels as usize;
    if n == 0 {
        return Vec::new();
    }

    // Accumulators per label (index 0 unused; labels are 1-based).
    let mut counts = vec![0u32; n + 1];
    let mut sum_x = vec![0.0f64; n + 1];
    let mut sum_y = vec![0.0f64; n + 1];
    let mut min_x = vec![u32::MAX; n + 1];
    let mut min_y = vec![u32::MAX; n + 1];
    let mut max_x = vec![0u32; n + 1];
    let mut max_y = vec![0u32; n + 1];

    let data = cl.label_map.data();
    for y in 0..h {
        for x in 0..w {
            let lbl = data[y * w + x] as usize;
            if lbl == 0 {
                continue;
            }
            counts[lbl] += 1;
            sum_x[lbl] += x as f64;
            sum_y[lbl] += y as f64;
            if (x as u32) < min_x[lbl] {
                min_x[lbl] = x as u32;
            }
            if (x as u32) > max_x[lbl] {
                max_x[lbl] = x as u32;
            }
            if (y as u32) < min_y[lbl] {
                min_y[lbl] = y as u32;
            }
            if (y as u32) > max_y[lbl] {
                max_y[lbl] = y as u32;
            }
        }
    }

    let mut result = Vec::new();
    for lbl in 1..=n {
        let cnt = counts[lbl];
        if cnt < min_area {
            continue;
        }
        let centroid = Point2f {
            x: (sum_x[lbl] / cnt as f64) as f32,
            y: (sum_y[lbl] / cnt as f64) as f32,
        };
        // bbox: [min_x..=max_x] × [min_y..=max_y] in pixel-center coords.
        let x0 = min_x[lbl] as f32;
        let y0 = min_y[lbl] as f32;
        let x1 = max_x[lbl] as f32;
        let y1 = max_y[lbl] as f32;
        let bbox = Rect2f {
            x: x0,
            y: y0,
            width: x1 - x0,
            height: y1 - y0,
        };
        result.push(ComponentStats {
            label: lbl as u32,
            pixel_count: cnt,
            bbox,
            centroid,
        });
    }

    result
}

#[cfg(test)]
mod tests {
    use vm_contour::Connectivity;
    use vm_core::Image;

    use super::{CcLabel, component_stats, label_connected_components_u8};

    fn rect_image(w: usize, h: usize, rx: usize, ry: usize, rw: usize, rh: usize) -> Image<u8> {
        let mut data = vec![0u8; w * h];
        for dy in 0..rh {
            for dx in 0..rw {
                data[(ry + dy) * w + (rx + dx)] = 255;
            }
        }
        Image::from_vec(w, h, data).unwrap()
    }

    #[test]
    fn single_rectangle_one_component() {
        // 10×10 image with a 4×4 white rectangle at (3, 3).
        // Expected: exactly 1 foreground component.
        let img = rect_image(10, 10, 3, 3, 4, 4);
        let ccl = label_connected_components_u8(&img.as_view(), Connectivity::C4);
        assert_eq!(ccl.num_labels, 1, "single rectangle → 1 component");
        let positives = ccl.label_map.data().iter().filter(|&&v| v == 1).count();
        assert_eq!(positives, 16, "4×4 rectangle has 16 foreground pixels");
    }

    #[test]
    fn two_separated_squares_two_components() {
        // 12×6 image: two 2×2 white squares far apart.
        // Square A at (1, 1), Square B at (9, 1).
        let mut data = vec![0u8; 12 * 6];
        for dy in 0..2 {
            for dx in 0..2 {
                data[(1 + dy) * 12 + (1 + dx)] = 255; // A
                data[(1 + dy) * 12 + (9 + dx)] = 255; // B
            }
        }
        let img = Image::from_vec(12, 6, data).unwrap();
        let ccl = label_connected_components_u8(&img.as_view(), Connectivity::C4);
        assert_eq!(ccl.num_labels, 2, "two separated squares → 2 components");
    }

    #[test]
    fn diagonal_touch_c4_vs_c8() {
        // Two pixels touching diagonally: (0,0) and (1,1).
        // Under C4: no shared axis-aligned neighbor → 2 components.
        // Under C8: diagonal is a connection → 1 component.
        let mut data = vec![0u8; 3 * 3];
        data[0] = 255; // (0,0)
        data[4] = 255; // (1,1)
        let img = Image::from_vec(3, 3, data).unwrap();

        let ccl_c4 = label_connected_components_u8(&img.as_view(), Connectivity::C4);
        let ccl_c8 = label_connected_components_u8(&img.as_view(), Connectivity::C8);

        assert_eq!(
            ccl_c4.num_labels, 2,
            "C4: diagonal touch → 2 separate components"
        );
        assert_eq!(
            ccl_c8.num_labels, 1,
            "C8: diagonal touch → 1 connected component"
        );
    }

    #[test]
    fn all_background_zero_components() {
        // All-zero image: no foreground → num_labels = 0.
        let img = Image::from_vec(5, 5, vec![0u8; 25]).unwrap();
        let ccl = label_connected_components_u8(&img.as_view(), Connectivity::C4);
        assert_eq!(ccl.num_labels, 0);
    }

    #[test]
    fn empty_image_no_panic() {
        let img = Image::from_vec(0, 0, vec![]).unwrap();
        let ccl = label_connected_components_u8(&img.as_view(), Connectivity::C4);
        assert_eq!(ccl.num_labels, 0);
    }

    // ---------------------------------------------------------------------------
    // ComponentStats tests
    // ---------------------------------------------------------------------------

    #[test]
    fn component_stats_rectangle_correct() {
        // 10×8 image: 4×4 rectangle at (2, 2).
        // Expected: pixel_count = 16, centroid ≈ (3.5, 3.5), bbox width = height = 3.
        let img = rect_image(10, 8, 2, 2, 4, 4);
        let ccl = label_connected_components_u8(&img.as_view(), Connectivity::C4);
        assert_eq!(ccl.num_labels, 1);

        let stats = component_stats(&ccl, 1);
        assert_eq!(stats.len(), 1);
        let s = &stats[0];
        assert_eq!(s.pixel_count, 16);
        // Centroid of pixels at x=2,3,4,5 and y=2,3,4,5 is (3.5, 3.5).
        assert!(
            (s.centroid.x - 3.5).abs() < 1e-4,
            "centroid.x expected 3.5, got {}",
            s.centroid.x
        );
        assert!(
            (s.centroid.y - 3.5).abs() < 1e-4,
            "centroid.y expected 3.5, got {}",
            s.centroid.y
        );
        // bbox: x=2, y=2, width=3 (from x=2 to x=5), height=3.
        assert!((s.bbox.x - 2.0).abs() < 1e-4);
        assert!((s.bbox.y - 2.0).abs() < 1e-4);
        assert!((s.bbox.width - 3.0).abs() < 1e-4);
        assert!((s.bbox.height - 3.0).abs() < 1e-4);
    }

    #[test]
    fn component_stats_min_area_filter() {
        // Two components: a 4×4 block (16 px) and a single pixel.
        // min_area = 10 → only the block survives.
        let mut data = vec![0u8; 12 * 6];
        // 4×4 block at (1, 1)
        for dy in 0..4 {
            for dx in 0..4 {
                data[(1 + dy) * 12 + (1 + dx)] = 255;
            }
        }
        // Single pixel at (10, 1)
        data[12 + 10] = 255;
        let img = Image::from_vec(12, 6, data).unwrap();
        let ccl = label_connected_components_u8(&img.as_view(), Connectivity::C4);
        assert_eq!(ccl.num_labels, 2);

        let stats_all = component_stats(&ccl, 1);
        let stats_filtered = component_stats(&ccl, 10);
        assert_eq!(stats_all.len(), 2);
        assert_eq!(
            stats_filtered.len(),
            1,
            "only the 16-px block passes min_area=10"
        );
        assert_eq!(stats_filtered[0].pixel_count, 16);
    }

    #[test]
    fn fake_ccl_test() {
        // Exercise CcLabel constructor directly with known data.
        let data: Vec<u32> = vec![0, 1, 1, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let label_map = Image::from_vec(4, 4, data).unwrap();
        let cl = CcLabel {
            label_map,
            num_labels: 2,
        };
        let stats = component_stats(&cl, 1);
        assert_eq!(stats.len(), 2);
        // Label 1 occupies (1,0) and (2,0).
        let s1 = stats.iter().find(|s| s.label == 1).unwrap();
        assert_eq!(s1.pixel_count, 2);
        assert!((s1.centroid.x - 1.5).abs() < 1e-4);
        assert!((s1.centroid.y - 0.0).abs() < 1e-4);
    }
}
