//! Allocation gate for the continuous derivative-of-Gaussian profiles: a warmed-up
//! `ProfileSampler`, and a warmed-up caliper on `Derivative::Continuous`, allocate
//! nothing per call (invariant 6).
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::num::NonZeroUsize;

use vision_metrology::measure::{Caliper, Derivative, MeasureConfig, MeasureStrip, ProfileConfig};
use vision_metrology::vm_primitives::edge::{ProfileSampler, ProfileSamplerConfig, ProfileStation};
use vision_metrology::{Image, Point2f, Vec2f};

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

/// Allocations made by `f` on this thread.
fn allocations(f: impl FnOnce()) -> usize {
    COUNT.with(|n| n.set(0));
    ACTIVE.with(|a| a.set(true));
    f();
    ACTIVE.with(|a| a.set(false));
    COUNT.with(Cell::get)
}

/// A bright bar on columns 40..60 of a 96 × 96 image, with a gentle texture.
fn bar() -> Image<u8> {
    let data = (0..96 * 96)
        .map(|i| {
            let (x, y) = (i % 96, i / 96);
            let base = if (40..60).contains(&x) { 200 } else { 30 };
            (base + (x * 7 + y * 13) % 5) as u8
        })
        .collect();
    Image::from_vec(96, 96, data).expect("valid image")
}

#[test]
fn a_warmed_up_sampler_allocates_nothing() {
    let img = bar();
    let mask = Image::new_fill(96, 96, 255u8);
    let stations: Vec<ProfileStation> = (0..16)
        .map(|k| ProfileStation {
            p: Point2f::new(50.0, 20.0 + 3.7 * k as f32),
            n: Vec2f::new(1.0, 0.1 * k as f32),
        })
        .collect();
    let offsets: Vec<f32> = (0..21).map(|j| j as f32 - 10.0).collect();
    let mut out = vec![None; stations.len() * offsets.len()];
    let mut sampler = ProfileSampler::new();
    let wide = ProfileSamplerConfig { sigma: 2.5 };
    sampler
        .sample(&img.as_view(), None, &stations, &offsets, &wide, &mut out)
        .expect("valid input");
    let n = allocations(|| {
        // The same σ, a smaller one, and a mask: the scratch only shrinks.
        for sigma in [2.5f32, 1.0, 1.7] {
            let cfg = ProfileSamplerConfig { sigma };
            sampler
                .sample(
                    &img.as_view(),
                    Some(&mask.as_view()),
                    &stations,
                    &offsets,
                    &cfg,
                    &mut out,
                )
                .expect("valid input");
        }
    });
    assert_eq!(n, 0, "a warmed-up sampler must not allocate");
    assert!(out.iter().filter(|s| s.is_some()).count() > out.len() / 2);
}

#[test]
fn a_warmed_up_continuous_caliper_allocates_nothing() {
    let img = bar();
    let strip = |y: f32, end: f32| MeasureStrip {
        start: Point2f::new(28.0, y),
        end: Point2f::new(end, y + 0.4),
        half_width: 1.5,
        samples: None,
        across: NonZeroUsize::new(3),
    };
    let cfg = MeasureConfig {
        threshold: 10.0,
        profile: ProfileConfig {
            sigma: 1.5,
            derivative: Derivative::Continuous,
            ..ProfileConfig::default()
        },
        ..MeasureConfig::default()
    };
    let mut cal = Caliper::strip(strip(48.3, 68.0), cfg);
    assert_eq!(cal.measure(&img.as_view()).expect("two edges").len(), 2);
    let n = allocations(|| {
        // Moved along the bar, slightly shorter each time, as a tracker moves it.
        for k in 0..8 {
            cal.set_strip(strip(30.0 + 4.0 * k as f32, 68.0 - 0.03 * k as f32));
            let edges = cal.measure(&img.as_view()).expect("two edges");
            assert_eq!(edges.len(), 2);
        }
    });
    assert_eq!(n, 0, "a warmed-up continuous caliper must not allocate");
}
