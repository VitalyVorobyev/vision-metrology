//! Planar rig substrate pose: per-step increment error on the three-view fixture
//! (`rig/fixture.rs`), 300 steps of known `(tx, ty, θ)` with a slow rotation, under
//! 0.05 px of Gaussian shift noise per axis, over eight seeds. Each seed is one cell; the
//! worst cell's bias and σ are reported per component.
use super::{Measured, mean_std};
use vision_metrology::metric::rig::{RigPoseConfig, RigPoseEstimator};

#[path = "../rig/fixture.rs"]
mod fixture;

const STEPS: usize = 300;
const SIGMA_PX: f64 = 0.05;

/// Per seed: the errors of tx, ty (mm) and θ (degrees).
fn cells() -> Vec<[(f32, f32); 3]> {
    let truth = fixture::increments(STEPS);
    let views = fixture::views(3);
    let mut est = RigPoseEstimator::new();
    (1..=8)
        .map(|seed| {
            let sh = fixture::shifts(3, &truth, SIGMA_PX, seed);
            let t = est
                .estimate(&views, &sh, &RigPoseConfig::default())
                .unwrap();
            let mut err = [Vec::new(), Vec::new(), Vec::new()];
            for (k, inc) in truth.iter().enumerate() {
                let f = &t.frames[k + 1].increment;
                err[0].push(f.translation.x - inc.tx as f32);
                err[1].push(f.translation.y - inc.ty as f32);
                err[2].push((f.rotation.angle() as f64 - inc.theta).to_degrees() as f32);
            }
            [mean_std(&err[0]), mean_std(&err[1]), mean_std(&err[2])]
        })
        .collect()
}

/// Translation (mm), the worse of tx and ty.
pub(super) fn translation() -> Measured {
    Measured::worst_of(cells().into_iter().flat_map(|c| [c[0], c[1]]))
}

/// Rotation (degrees).
pub(super) fn rotation() -> Measured {
    Measured::worst_of(cells().into_iter().map(|c| c[2]))
}
