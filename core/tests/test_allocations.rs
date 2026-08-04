//! Verifies the allocation behavior of population density queries.

mod common;

use common::open_database;
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

const GENERATED_DATABASE_PATH: &str = "../population_density_database.bin";
const PACKAGED_DATABASE_PATH: &str =
    "../../../packages/apps/NetworkLocation/res/raw/population_density_database.bin";
const QUERY_COUNT: u64 = 100_000;
const S2_FACE_COUNT: u64 = 6;
const S2_FACE_SHIFT: u32 = 61;
const S2_POSITION_BITS: u32 = 60;
const POSITION_MULTIPLIER: u64 = 0x9e37_79b9_7f4a_7c15;

static TRACK_ALLOCATIONS: AtomicBool = AtomicBool::new(false);
static ALLOCATION_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Counts allocations while explicitly enabled.
struct CountingAllocator;

// SAFETY: Every operation delegates to the system allocator with the original layout and pointer.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        // SAFETY: The caller provides the layout required by GlobalAlloc.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        // SAFETY: The caller provides the layout required by GlobalAlloc.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record_allocation();
        // SAFETY: The caller provides the pointer and layout required by GlobalAlloc.
        unsafe { System.realloc(pointer, layout, new_size) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: The caller provides the pointer and layout required by GlobalAlloc.
        unsafe { System.dealloc(pointer, layout) }
    }
}

#[global_allocator]
static GLOBAL_ALLOCATOR: CountingAllocator = CountingAllocator;

/// Increments the allocation count when tracking is enabled.
fn record_allocation() {
    if TRACK_ALLOCATIONS.load(Ordering::Relaxed) {
        ALLOCATION_COUNT.fetch_add(1, Ordering::Relaxed);
    }
}

/// Disables allocation tracking when dropped.
struct AllocationTrackingGuard;

impl AllocationTrackingGuard {
    /// Starts a new allocation count.
    fn start() -> Self {
        ALLOCATION_COUNT.store(0, Ordering::Relaxed);
        TRACK_ALLOCATIONS.store(true, Ordering::Relaxed);
        Self
    }
}

impl Drop for AllocationTrackingGuard {
    fn drop(&mut self) {
        TRACK_ALLOCATIONS.store(false, Ordering::Relaxed);
    }
}

/// Returns an available S2PD fixture path.
fn database_path() -> PathBuf {
    let manifest_path = Path::new(env!("CARGO_MANIFEST_DIR"));
    let generated_path = manifest_path.join(GENERATED_DATABASE_PATH);
    if generated_path.exists() {
        return generated_path;
    }

    let packaged_path = manifest_path.join(PACKAGED_DATABASE_PATH);
    assert!(
        packaged_path.exists(),
        "S2PD database not found at '{}' or '{}'",
        generated_path.display(),
        packaged_path.display()
    );
    packaged_path
}

/// Returns a deterministic valid level-30 S2 cell ID.
fn query_cell_id(query_index: u64) -> u64 {
    let face = query_index % S2_FACE_COUNT;
    let position_mask = (1u64 << S2_POSITION_BITS) - 1;
    let position = query_index.wrapping_mul(POSITION_MULTIPLIER) & position_mask;
    (face << S2_FACE_SHIFT) | (position << 1) | 1
}

/// Verifies successful valid queries do not allocate.
#[test]
fn valid_queries_do_not_allocate() {
    let query_engine = open_database(database_path()).unwrap();

    let tracking_guard = AllocationTrackingGuard::start();
    for query_index in 0..QUERY_COUNT {
        black_box(query_engine.query(query_cell_id(query_index)).unwrap());
    }
    drop(tracking_guard);

    assert_eq!(ALLOCATION_COUNT.load(Ordering::Relaxed), 0);
}
