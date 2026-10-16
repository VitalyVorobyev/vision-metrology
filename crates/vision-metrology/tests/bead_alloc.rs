//! Allocation gate for `BeadTracker` (invariant 6): once warmed up, a `track` call allocates
//! only its result, with per-station and with coupled pairing, even when the bead leaves
//! the image and stations go one-sided or absent.
//!
//! A result holds the centreline, the samples, the passes, and each pass's and the final
//! stage's rejection counts when there are any: at most `4 + passes` allocations.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use vision_metrology::measure::{
    BeadConfig, BeadCoupling, BeadPairing, BeadReject, BeadTracker, TrackedBead,
};
use vision_metrology::{Image, Point2f};

thread_local! {
    static ACTIVE: Cell<bool> = const { Cell::new(false) };
    static COUNT: Cell<usize> = const { Cell::new(0) };
}

struct CountAllocator;
// SAFETY: every operation delegates the identical layout/pointer to System.
unsafe impl GlobalAlloc for CountAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if ACTIVE.try_with(Cell::get).unwrap_or(false) {
            let _ = COUNT.try_with(|n| n.set(n.get() + 1));
        }
        // SAFETY: caller guarantees the allocation layout is valid.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: caller guarantees pointer and layout came from this allocator.
        unsafe { System.dealloc(pointer, layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        if ACTIVE.try_with(Cell::get).unwrap_or(false) {
            let _ = COUNT.try_with(|n| n.set(n.get() + 1));
        }
        // SAFETY: caller guarantees the pointer, layout and requested size are valid.
        unsafe { System.realloc(pointer, layout, size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountAllocator = CountAllocator;

/// A light bead 40 px wide along a line that runs off the image's bottom edge, so that its
/// lower wall leaves the image before its upper one.
fn scene() -> (Image<u8>, Vec<Point2f>) {
    let (w, h) = (320usize, 200usize);
    let (a, t) = ((20.0f64, 150.0f64), (0.96f64, 0.28f64));
    let n = (-t.1, t.0);
    let data = (0..w * h)
        .map(|i| {
            let (x, y) = ((i % w) as f64, (i / w) as f64);
            let d = (x - a.0) * n.0 + (y - a.1) * n.1;
            let cover = (d + 20.5).clamp(0.0, 1.0) - (d - 19.5).clamp(0.0, 1.0);
            (40.0 + 150.0 * cover).round() as u8
        })
        .collect();
    let prior = (0..=60)
        .map(|k| {
            let s = 5.0 * f64::from(k);
            Point2f::new(
                (a.0 + s * t.0 + 3.0 * n.0) as f32,
                (a.1 + s * t.1 + 3.0 * n.1) as f32,
            )
        })
        .collect();
    (Image::from_vec(w, h, data).expect("sized"), prior)
}

/// The allocations of one warmed-up `track` call, and its result.
fn second_call(cfg: BeadConfig) -> (usize, TrackedBead) {
    let (img, prior) = scene();
    let mut tracker = BeadTracker::new(cfg).expect("valid");
    tracker.track(&img.as_view(), &prior).expect("tracked");
    COUNT.with(|n| n.set(0));
    ACTIVE.with(|a| a.set(true));
    let bead = tracker.track(&img.as_view(), &prior).expect("tracked");
    ACTIVE.with(|a| a.set(false));
    (COUNT.with(Cell::get), bead)
}

#[test]
fn a_warmed_up_tracker_allocates_only_its_result() {
    let base = BeadConfig {
        min_width: 30.0,
        max_width: 50.0,
        ..BeadConfig::default()
    };
    for pairing in [
        BeadPairing::PerStation,
        BeadPairing::Coupled(BeadCoupling::default()),
    ] {
        let (count, bead) = second_call(BeadConfig { pairing, ..base });
        let bound = 4 + bead.track.passes.len();
        let mode = if pairing == BeadPairing::PerStation {
            "per station"
        } else {
            "coupled"
        };
        eprintln!("{mode}: {count} allocations (bound {bound})");
        assert!(
            count <= bound,
            "{mode}: {count} allocations, more than {bound}"
        );
        if matches!(pairing, BeadPairing::Coupled(_)) {
            let one_sided = bead
                .samples
                .iter()
                .filter(|s| s.hit == Err(BeadReject::OneSided))
                .count();
            assert!(one_sided > 0, "the scene has one-sided stations");
        }
    }
}
