//! Allocation gate for `metric::rig::RigPoseEstimator::step` (invariant 6): once `begin`
//! has sized the scratch, a step allocates nothing, whether it is measured, single-view,
//! bridged or has a gross view.
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use vision_metrology::Vec2f;
use vision_metrology::metric::rig::{RigPoseConfig, RigPoseEstimator, RigStep, ViewUse};

#[path = "rig/fixture.rs"]
mod fixture;

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

#[test]
fn stepping_allocates_nothing() {
    const N: usize = 3;
    let truth = fixture::increments(40);
    let mut sh = fixture::shifts(N, &truth, 0.05, 3);
    sh[10 * N + 1].as_mut().unwrap().d_px += Vec2f::new(3.0, -1.0);
    sh[20 * N] = None;
    sh[20 * N + 1] = None;
    for s in &mut sh[30 * N..31 * N] {
        *s = None;
    }
    let mut est = RigPoseEstimator::new();
    est.begin(&fixture::views(N), &RigPoseConfig::default())
        .unwrap();
    let mut uses = [ViewUse::default(); N];
    let mut steps = Vec::with_capacity(40);
    COUNT.with(|n| n.set(0));
    ACTIVE.with(|a| a.set(true));
    for chunk in sh.as_chunks::<N>().0 {
        let frame = est.step(chunk, &mut uses).unwrap();
        steps.push(frame.step);
    }
    ACTIVE.with(|a| a.set(false));
    let count = COUNT.with(Cell::get);
    assert_eq!(count, 0, "step allocated {count} times");
    assert_eq!(steps[20], RigStep::SingleView);
    assert_eq!(steps[30], RigStep::Bridged);
    assert_eq!(steps[10], RigStep::Measured);
}
