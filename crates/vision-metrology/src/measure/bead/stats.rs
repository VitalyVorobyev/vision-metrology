//! Tallies and statistics over a stage's stations (invariants 20 and 21).

use super::result::{BeadHit, BeadReject, BeadStats};

/// Stations rejected, counted by reason, indexed by [`BeadReject::index`].
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct Tally([usize; BeadReject::ALL.len()]);

impl Tally {
    pub(super) fn add(&mut self, r: BeadReject) {
        self.0[r.index()] += 1;
    }

    /// The non-zero counts, in [`BeadReject::ALL`] order.
    pub(super) fn to_vec(self) -> Vec<(BeadReject, usize)> {
        BeadReject::ALL
            .iter()
            .zip(self.0)
            .filter(|&(_, n)| n > 0)
            .map(|(&r, n)| (r, n))
            .collect()
    }
}

/// The longest run of `false` in `found`, in stations.
pub(super) fn longest_run_missing(found: impl IntoIterator<Item = bool>) -> usize {
    let (mut run, mut longest) = (0usize, 0usize);
    for f in found {
        run = if f { 0 } else { run + 1 };
        longest = longest.max(run);
    }
    longest
}

/// Position and width statistics over `hits`; `None` when there are none.
pub(super) fn bead_stats<'a>(hits: impl Iterator<Item = &'a BeadHit>) -> Option<BeadStats> {
    let (mut n, mut off2, mut off_max) = (0usize, 0.0f64, 0.0f64);
    let (mut w_sum, mut w2, mut w_min, mut w_max) =
        (0.0f64, 0.0f64, f64::INFINITY, f64::NEG_INFINITY);
    for h in hits {
        let (o, w) = (f64::from(h.offset), f64::from(h.pair.width));
        n += 1;
        off2 += o * o;
        off_max = off_max.max(o.abs());
        w_sum += w;
        w2 += w * w;
        w_min = w_min.min(w);
        w_max = w_max.max(w);
    }
    if n == 0 {
        return None;
    }
    let nf = n as f64;
    let mean = w_sum / nf;
    Some(BeadStats {
        n_used: n,
        center_rms: (off2 / nf).sqrt() as f32,
        center_max_dev: off_max as f32,
        width_mean: mean as f32,
        width_std: (w2 / nf - mean * mean).max(0.0).sqrt() as f32,
        width_min: w_min as f32,
        width_max: w_max as f32,
    })
}
