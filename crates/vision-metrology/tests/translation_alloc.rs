//! Allocation gate for the prepared native registration path.
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use vision_metrology::{
    Image,
    corr::{TranslationConfig, TranslationEstimator, TranslationStatus},
};

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
fn prepared_estimation_allocates_nothing_even_when_masks_change() {
    let image = Image::from_vec(
        161,
        97,
        (0..161 * 97)
            .map(|i| {
                let (x, y) = ((i % 161) as f32, (i / 161) as f32);
                100.0 + 30.0 * (x * 0.17 + y * 0.13).sin() + 25.0 * (x * 0.08 - y * 0.21).cos()
            })
            .collect(),
    )
    .unwrap();
    let mut mask = Image::new_fill(161, 97, 0u8);
    let cfg = TranslationConfig::default();
    let mut estimator = TranslationEstimator::new();
    let initial = estimator
        .estimate(
            &image.as_view(),
            &image.as_view(),
            Some(&mask.as_view()),
            Some(&mask.as_view()),
            None,
            &cfg,
        )
        .unwrap();
    assert_eq!(initial.status, TranslationStatus::InsufficientSupport);
    mask.data_mut().fill(1);
    COUNT.with(|n| n.set(0));
    ACTIVE.with(|a| a.set(true));
    let result = estimator.estimate(
        &image.as_view(),
        &image.as_view(),
        Some(&mask.as_view()),
        Some(&mask.as_view()),
        None,
        &cfg,
    );
    ACTIVE.with(|a| a.set(false));
    let count = COUNT.with(Cell::get);
    assert_eq!(count, 0, "prepared registration must allocate nothing");
    assert_eq!(result.unwrap().status, TranslationStatus::Accepted);
}

#[cfg(feature = "corr-experimental")]
#[test]
fn prepared_similarity_allocates_nothing_after_empty_evidence() {
    use vision_metrology::corr::{SimilarityConfig, SimilarityEstimator};
    let image = Image::from_vec(
        161,
        97,
        (0..161 * 97)
            .map(|i| {
                let (x, y) = ((i % 161) as f32, (i / 161) as f32);
                100. + 25. * (x * 0.17 + y * 0.13).sin()
                    + 20. * (x * 0.08 - y * 0.21).cos()
                    + 15. * (x * 0.31 + y * 0.07).sin()
                    + 10. * (x * 0.03 + y * 0.39).cos()
            })
            .collect(),
    )
    .unwrap();
    let mut mask = Image::new_fill(161, 97, 0u8);
    let cfg = SimilarityConfig::default();
    let mut e = SimilarityEstimator::new();
    assert_eq!(
        e.estimate(
            &image.as_view(),
            &image.as_view(),
            Some(&mask.as_view()),
            Some(&mask.as_view()),
            None,
            &cfg
        )
        .unwrap()
        .status,
        TranslationStatus::InsufficientSupport
    );
    mask.data_mut().fill(1);
    COUNT.with(|n| n.set(0));
    ACTIVE.with(|a| a.set(true));
    let result = e.estimate(
        &image.as_view(),
        &image.as_view(),
        Some(&mask.as_view()),
        Some(&mask.as_view()),
        None,
        &cfg,
    );
    ACTIVE.with(|a| a.set(false));
    let count = COUNT.with(Cell::get);
    assert_eq!(count, 0, "prepared similarity must allocate nothing");
    assert_eq!(result.unwrap().status, TranslationStatus::Accepted);
}
