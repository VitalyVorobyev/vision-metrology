//! Allocation gate for `measure`'s crate-private numerical cores (ADR-0020): once sized, the
//! min-path and the robust smoother allocate nothing, even when a call's size shrinks.
//!
//! The cores are not public API, so this test compiles their source files directly; they
//! depend on nothing else in the crate.

#[allow(dead_code)]
#[path = "../src/measure/numeric/minpath.rs"]
mod minpath;
#[allow(dead_code)]
#[path = "../src/measure/numeric/whittaker.rs"]
mod whittaker;

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use minpath::{Grid, LayeredPath, Steps};
use whittaker::{Penalty, Smoother};

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

/// The allocations `f` makes on this thread.
fn allocations(f: impl FnOnce()) -> usize {
    COUNT.with(|n| n.set(0));
    ACTIVE.with(|a| a.set(true));
    f();
    ACTIVE.with(|a| a.set(false));
    COUNT.with(Cell::get)
}

fn fill(dp: &mut LayeredPath, seed: usize) {
    for k in 0..dp.layers() {
        for (s, u) in dp.unary_mut(k).iter_mut().enumerate() {
            *u = (((k * 31 + s * 17 + seed) % 23) as f32) * 0.1;
        }
    }
}

#[test]
fn a_sized_min_path_allocates_nothing() {
    let steps = Steps {
        coef: [0.2, 0.5],
        band: [3, 1],
        switch: vec![0.0, 1.0, 1.0, 0.0],
    };
    let big = Grid {
        planes: 2,
        dims: [20, 6],
    };
    let mut dp = LayeredPath::default();
    dp.reset(40, big, &steps);
    fill(&mut dp, 0);
    assert!(dp.solve().is_some());

    let n = allocations(|| {
        // The same size again, then a smaller one, with pins and max-marginals.
        dp.reset(40, big, &steps);
        fill(&mut dp, 1);
        dp.pin(0, 0, [3, 2], [5, 3], 100.0);
        assert!(dp.solve().is_some());
        let m: f64 = (0..big.states()).map(|s| dp.max_marginal(20, s)).sum();
        assert!(m.is_finite());
        let small = Grid {
            planes: 2,
            dims: [9, 4],
        };
        dp.reset(25, small, &steps);
        fill(&mut dp, 2);
        assert!(dp.solve().is_some());
        assert_eq!(dp.path().len(), 25);
    });
    assert_eq!(n, 0, "the min-path allocated {n} times");
}

#[test]
fn a_sized_smoother_allocates_nothing() {
    let obs: Vec<f64> = (0..500)
        .map(|i| (0.03 * i as f64).sin() + if i % 41 == 0 { 5.0 } else { 0.0 })
        .collect();
    let base: Vec<f64> = (0..500)
        .map(|i| if i % 13 == 0 { 0.0 } else { 1.0 })
        .collect();
    let p = Penalty::new(0.0, 1.0, 4.0, 1.0);
    let huber = |r: f64| if r.abs() <= 0.5 { 1.0 } else { 0.5 / r.abs() };
    let mut sm = Smoother::default();
    assert_eq!(
        sm.solve_robust(&obs, &base, p, 4, 0.0, |_, _| (huber, true)),
        Some(4)
    );

    let n = allocations(|| {
        assert_eq!(
            sm.solve_robust(&obs, &base, p, 4, 0.0, |_, _| (huber, true)),
            Some(4)
        );
        assert!(sm.solve(&obs[..300], &base[..300], p));
        assert_eq!(sm.u.len(), 300);
    });
    assert_eq!(n, 0, "the smoother allocated {n} times");
}
