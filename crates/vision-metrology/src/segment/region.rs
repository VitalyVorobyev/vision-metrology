//! Edgel-driven region growing using a chamfer distance map.
//!
//! Pixels farther than `max_gap` from any contour edgel form the seeds of
//! flood-filled regions. Pixels within `max_gap` are treated as potential
//! boundaries and are not initially seeded. The result is a label image
//! where `≥ 0` is a region label and `-1` is an edge-adjacent boundary.

use crate::contour::ContourGraph;
use vm_primitives::chamfer_distance_u8;
use vm_primitives::{Image, ImageView};

/// Configuration for edgel-based region growing.
#[derive(Debug, Clone, PartialEq)]
pub struct RegionGrowConfig {
    /// Maximum chamfer distance (pixels) from any edgel for a pixel to be
    /// considered an interior (non-boundary) seed. Pixels closer than or equal
    /// to this distance are marked as edge-adjacent boundaries.
    /// Default: 2.0 px.
    pub max_gap: f32,
    /// Minimum region area in pixels. Regions with fewer pixels than this
    /// threshold are relabelled as boundary (`-1`) after growing.
    /// Default: 50.
    pub min_area: u32,
}

impl Default for RegionGrowConfig {
    fn default() -> Self {
        Self {
            max_gap: 2.0,
            min_area: 50,
        }
    }
}

/// Grow regions from interior pixels using the contour graph as the boundary.
///
/// The algorithm:
/// 1. Rasterise `graph` edgels into a binary mask.
/// 2. Compute the chamfer distance map from those edgels.
/// 3. Pixels with `distance > max_gap` are potential region interiors.
/// 4. Flood-fill (BFS) those pixels, assigning a unique label to each
///    connected component in 4-connectivity.
/// 5. Relabel components with fewer than `min_area` pixels to `-1`.
///
/// Returns an `Image<i32>` of the same dimensions as `img`:
/// - `≥ 0` — region label (0-based).
/// - `-1`  — boundary or edge-adjacent pixel.
///
/// `img` is used only for its dimensions; intensity values are not read.
pub fn grow_regions(
    graph: &ContourGraph,
    img: &ImageView<'_, u8>,
    cfg: &RegionGrowConfig,
) -> Image<i32> {
    let w = img.width();
    let h = img.height();
    let n = w * h;

    if n == 0 {
        return Image::from_vec(w, h, Vec::new()).expect("0-size");
    }

    // --- Step 1: rasterise edgels into a binary mask ---
    let mut edgel_mask = vec![0u8; n];
    for edge in graph.iter_edges() {
        for pt in &edge.points {
            let x = pt.x.round() as isize;
            let y = pt.y.round() as isize;
            if x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h {
                edgel_mask[y as usize * w + x as usize] = 255;
            }
        }
    }
    // Also rasterise node positions.
    for node in &graph.nodes {
        let x = node.idx.0;
        let y = node.idx.1;
        if x < w && y < h {
            edgel_mask[y * w + x] = 255;
        }
    }

    // --- Step 2: chamfer distance map ---
    let mask_img = Image::from_vec(w, h, edgel_mask).expect("valid dims");
    let dist = chamfer_distance_u8(&mask_img.as_view());
    let dist_data = dist.data();

    // --- Step 3: mark interior pixels ---
    // labels: -2 = unvisited interior, -1 = boundary, ≥0 = region label.
    const UNVISITED: i32 = -2;
    const BOUNDARY: i32 = -1;
    let mut labels = vec![BOUNDARY; n];
    for i in 0..n {
        if dist_data[i] > cfg.max_gap {
            labels[i] = UNVISITED;
        }
    }

    // --- Step 4: BFS flood fill ---
    let mut next_label = 0i32;
    let mut queue = Vec::new(); // reused scratch
    let mut region_sizes: Vec<u32> = Vec::new();

    for start in 0..n {
        if labels[start] != UNVISITED {
            continue;
        }
        // New region.
        let lbl = next_label;
        next_label += 1;
        region_sizes.push(0);

        labels[start] = lbl;
        region_sizes[lbl as usize] += 1;
        queue.clear();
        queue.push(start);

        let mut qi = 0;
        while qi < queue.len() {
            let idx = queue[qi];
            qi += 1;

            let x = (idx % w) as isize;
            let y = (idx / w) as isize;

            // 4-connected neighbours.
            for (dx, dy) in [(-1isize, 0isize), (1, 0), (0, -1), (0, 1)] {
                let nx = x + dx;
                let ny = y + dy;
                if nx < 0 || ny < 0 || nx >= w as isize || ny >= h as isize {
                    continue;
                }
                let nidx = ny as usize * w + nx as usize;
                if labels[nidx] == UNVISITED {
                    labels[nidx] = lbl;
                    region_sizes[lbl as usize] += 1;
                    queue.push(nidx);
                }
            }
        }
    }

    // --- Step 5: discard small regions ---
    for v in &mut labels {
        if *v >= 0 {
            let sz = region_sizes[*v as usize];
            if sz < cfg.min_area {
                *v = BOUNDARY;
            }
        }
    }

    Image::from_vec(w, h, labels).expect("dimensions unchanged")
}

#[cfg(test)]
mod tests {
    use crate::contour::{Connectivity, ContourBuildConfig, build_graph_from_edgels};
    use vm_primitives::edge::Edgel;
    use vm_primitives::{Image, Point2f, Vec2f};

    use super::{RegionGrowConfig, grow_regions};

    fn make_edgel(x: usize, y: usize) -> Edgel {
        Edgel {
            p: Point2f::new(x as f32, y as f32),
            n: Vec2f::new(1.0, 0.0),
            strength: 1.0,
            idx: (x, y),
        }
    }

    #[test]
    fn vertical_edge_creates_two_regions() {
        // 21×11 image with a vertical line of edgels at x=10.
        // Interior pixels at x=0..5 and x=15..20 are far from the edge.
        let w = 21usize;
        let h = 11usize;
        let edgels: Vec<Edgel> = (0..h).map(|y| make_edgel(10, y)).collect();
        let cfg_build = ContourBuildConfig {
            connectivity: Connectivity::C4,
            min_component_size: 2,
            record_geometry: false,
            record_strengths: false,
            ..Default::default()
        };
        let graph = build_graph_from_edgels(w, h, &edgels, &cfg_build);

        let img_data = vec![128u8; w * h];
        let img = Image::from_vec(w, h, img_data).unwrap();
        let cfg = RegionGrowConfig {
            max_gap: 2.0,
            min_area: 1,
        };
        let result = grow_regions(&graph, &img.as_view(), &cfg);

        // Pixels far to the left and right should have different (or the same) labels but
        // must both be ≥ 0 (interior).
        let left = *result.as_view().get(2, 5).unwrap();
        let right = *result.as_view().get(18, 5).unwrap();
        assert!(left >= 0, "left interior pixel should be in a region");
        assert!(right >= 0, "right interior pixel should be in a region");
        // The two sides should be in different regions because the vertical edge separates them.
        assert_ne!(
            left, right,
            "left and right sides should be in different regions"
        );
    }

    #[test]
    fn no_edgels_entire_image_is_one_region() {
        // Empty graph: chamfer distance = INF everywhere, so all pixels are interior.
        let w = 10usize;
        let h = 10usize;
        let graph = build_graph_from_edgels(w, h, &[], &ContourBuildConfig::default());
        let img = Image::from_vec(w, h, vec![0u8; w * h]).unwrap();
        let cfg = RegionGrowConfig {
            max_gap: 2.0,
            min_area: 1,
        };
        let result = grow_regions(&graph, &img.as_view(), &cfg);
        // All pixels should be in a single region (label 0).
        assert!(
            result.data().iter().all(|&v| v == 0),
            "no edgels: all pixels in one region"
        );
    }

    #[test]
    fn empty_image_no_panic() {
        let graph = build_graph_from_edgels(0, 0, &[], &ContourBuildConfig::default());
        let img = Image::from_vec(0, 0, vec![]).unwrap();
        let result = grow_regions(&graph, &img.as_view(), &RegionGrowConfig::default());
        assert_eq!(result.width(), 0);
        assert_eq!(result.height(), 0);
    }
}
