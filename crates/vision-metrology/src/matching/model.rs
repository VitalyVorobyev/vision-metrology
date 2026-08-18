//! Edge model: a set of edgels in model-local coordinates with a pre-computed
//! chamfer distance map for fast scene lookup.
//!
//! ## Normal convention (OQ-5)
//! Model normals must follow the same dark-to-bright convention as `Edgel.n`
//! (pointing from dark toward bright). When building a model from a CAD boundary
//! (where photometric context is absent) the caller is responsible for assigning
//! outward normals and is warned that the normal-coherence scorer compares them
//! against the scene's dark-to-bright normals.

use vm_primitives::chamfer_distance_u8;
use vm_primitives::edge::edge2d::Edgel;
use vm_primitives::{Image, Point2f};

/// A set of model edgels in model-local coordinates with a pre-computed chamfer
/// distance map for fast scoring during rigid matching.
///
/// After construction:
/// - `edgels[i].p` is in model-local coordinates (centroid subtracted).
/// - `edgels[i].n` is unchanged (caller's normal direction).
/// - `chamfer_map` covers the tight bounding box of the translated edgels,
///   expanded by `map_margin` on each side.
///
/// ## Coordinate layout
/// The chamfer map origin is at `(-bbox_min_x + map_margin, -bbox_min_y + map_margin)`
/// relative to the model-local frame. Use [`EdgeModel::map_offset`] to convert a
/// model-local point `(px, py)` to a map pixel coordinate `(col, row)`.
pub struct EdgeModel {
    /// Model edgels in model-local coordinates (centroid at origin).
    ///
    /// Normal direction follows the dark-to-bright convention of
    /// [`vm_primitives::Edgel`].
    /// For CAD-derived models, outward normals are the caller's responsibility.
    pub edgels: Vec<Edgel>,
    /// Centroid of the original edgels (in scene/input coordinates).
    /// Stored for informational purposes; not used internally after construction.
    pub centroid: Point2f,
    /// Chamfer distance map computed over the model edgels.
    /// Each pixel value is the approximate Euclidean distance (in pixels) to
    /// the nearest model edgel.
    pub chamfer_map: Image<f32>,
    /// Width of the chamfer map in pixels.
    pub chamfer_w: usize,
    /// Height of the chamfer map in pixels.
    pub chamfer_h: usize,
    /// (x, y) offset to convert a model-local point to a chamfer-map pixel.
    /// `map_col = (local_x + offset_x).round()`, `map_row = (local_y + offset_y).round()`.
    pub map_offset: (f32, f32),
}

impl EdgeModel {
    /// Build an [`EdgeModel`] from a slice of edgels.
    ///
    /// Steps:
    /// 1. Compute the centroid of all edgel positions.
    /// 2. Translate each edgel so the centroid is at the origin.
    /// 3. Compute the tight bounding box of the translated positions.
    /// 4. Expand the bounding box by `map_margin` on each side.
    /// 5. Rasterise each edgel into a binary mask and run `chamfer_distance_u8`.
    ///
    /// # Panics
    /// Does not panic; returns an empty model with a 1×1 chamfer map if
    /// `edgels` is empty.
    pub fn from_edgels(edgels: Vec<Edgel>, map_margin: usize) -> Self {
        if edgels.is_empty() {
            let chamfer_map = Image::from_vec(1, 1, vec![0.0f32]).expect("1×1 image");
            return Self {
                edgels: Vec::new(),
                centroid: Point2f::default(),
                chamfer_map,
                chamfer_w: 1,
                chamfer_h: 1,
                map_offset: (0.0, 0.0),
            };
        }

        // --- Step 1: centroid ---
        let n = edgels.len() as f32;
        let cx = edgels.iter().map(|e| e.p.x).sum::<f32>() / n;
        let cy = edgels.iter().map(|e| e.p.y).sum::<f32>() / n;
        let centroid = Point2f { x: cx, y: cy };

        // --- Step 2: translate edgels to centroid-origin ---
        let mut translated: Vec<Edgel> = edgels
            .iter()
            .map(|e| Edgel {
                p: Point2f {
                    x: e.p.x - cx,
                    y: e.p.y - cy,
                },
                n: e.n,
                strength: e.strength,
                idx: e.idx,
            })
            .collect();

        // --- Step 3: tight bounding box ---
        let mut min_x = translated[0].p.x;
        let mut max_x = min_x;
        let mut min_y = translated[0].p.y;
        let mut max_y = min_y;
        for e in &translated {
            min_x = min_x.min(e.p.x);
            max_x = max_x.max(e.p.x);
            min_y = min_y.min(e.p.y);
            max_y = max_y.max(e.p.y);
        }

        // --- Step 4: expand by margin ---
        let m = map_margin as f32;
        let origin_x = min_x - m; // model-local coordinate of map col 0
        let origin_y = min_y - m; // model-local coordinate of map row 0
        let map_w = ((max_x - min_x).ceil() as usize + 2 * map_margin).max(1);
        let map_h = ((max_y - min_y).ceil() as usize + 2 * map_margin).max(1);

        // map_offset converts model-local → map pixel:
        //   col = local_x - origin_x  ⟹  offset_x = -origin_x
        let offset_x = -origin_x;
        let offset_y = -origin_y;

        // --- Step 5: rasterise and compute chamfer map ---
        let mut mask = vec![0u8; map_w * map_h];
        for e in &translated {
            let col = (e.p.x + offset_x).round() as isize;
            let row = (e.p.y + offset_y).round() as isize;
            if col >= 0 && row >= 0 && (col as usize) < map_w && (row as usize) < map_h {
                mask[row as usize * map_w + col as usize] = 255;
            }
        }
        let mask_img = Image::from_vec(map_w, map_h, mask).expect("valid dims");
        let chamfer_map = chamfer_distance_u8(&mask_img.as_view());

        // Normalise idx fields so they reflect the translated (model-local) integer position.
        for e in &mut translated {
            let col = (e.p.x + offset_x).round() as isize;
            let row = (e.p.y + offset_y).round() as isize;
            let col = col.max(0) as usize;
            let row = row.max(0) as usize;
            e.idx = (col, row);
        }

        Self {
            edgels: translated,
            centroid,
            chamfer_map,
            chamfer_w: map_w,
            chamfer_h: map_h,
            map_offset: (offset_x, offset_y),
        }
    }
}

#[cfg(test)]
mod tests {
    use vm_primitives::edge::edge2d::Edgel;
    use vm_primitives::{Point2f, Vec2f};

    use super::EdgeModel;

    fn make_edgel(x: f32, y: f32) -> Edgel {
        Edgel {
            p: Point2f { x, y },
            n: Vec2f { x: 1.0, y: 0.0 },
            strength: 1.0,
            idx: (x as usize, y as usize),
        }
    }

    #[test]
    fn centroid_is_at_origin_after_construction() {
        // Four corners of a 10×10 square centred at (5, 5).
        let edgels = vec![
            make_edgel(0.0, 0.0),
            make_edgel(10.0, 0.0),
            make_edgel(10.0, 10.0),
            make_edgel(0.0, 10.0),
        ];
        let model = EdgeModel::from_edgels(edgels, 2);

        // Centroid of translated edgels should be ≈ (0, 0).
        let n = model.edgels.len() as f32;
        let mx = model.edgels.iter().map(|e| e.p.x).sum::<f32>() / n;
        let my = model.edgels.iter().map(|e| e.p.y).sum::<f32>() / n;
        assert!(mx.abs() < 1e-4, "model centroid.x should be ~0, got {mx}");
        assert!(my.abs() < 1e-4, "model centroid.y should be ~0, got {my}");
    }

    #[test]
    fn chamfer_score_zero_on_model_own_map() {
        // An edgel exactly on a rasterised position should have chamfer distance ≈ 0.
        let edgels = vec![
            make_edgel(0.0, 0.0),
            make_edgel(5.0, 0.0),
            make_edgel(5.0, 5.0),
            make_edgel(0.0, 5.0),
        ];
        let model = EdgeModel::from_edgels(edgels, 3);

        // The chamfer map distance at each model edgel's rasterised position should be 0.
        for e in &model.edgels {
            let col = (e.p.x + model.map_offset.0).round() as usize;
            let row = (e.p.y + model.map_offset.1).round() as usize;
            if col < model.chamfer_w && row < model.chamfer_h {
                let d = *model.chamfer_map.as_view().get(col, row).unwrap();
                assert!(
                    d < 0.1,
                    "edgel at model-local ({:.1},{:.1}) → map ({col},{row}) has chamfer dist {d}, expected ~0",
                    e.p.x,
                    e.p.y
                );
            }
        }
    }

    #[test]
    fn empty_edgels_no_panic() {
        let model = EdgeModel::from_edgels(vec![], 5);
        assert_eq!(model.edgels.len(), 0);
        assert_eq!(model.chamfer_w, 1);
        assert_eq!(model.chamfer_h, 1);
    }
}
