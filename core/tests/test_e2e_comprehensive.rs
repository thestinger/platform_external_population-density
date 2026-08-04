//! Exercises S2PD queries across geometry, geography, and concurrent workloads.

mod common;

use common::open_database;
use population_density::{
    BITS_PER_LEVEL, MAX_DB_LEVEL, MAX_S2_LEVEL, QueryEngine, S2_FACE_BITS, SHIFT_COMPACT,
    get_ancestor, get_children_ids, get_level, get_parent_id,
};
use s2::cellid::CellID;
use s2::latlng::LatLng;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const GENERATED_DATABASE_PATH: &str = "../population_density_database.bin";
const PACKAGED_DATABASE_PATH: &str =
    "../../../packages/apps/NetworkLocation/res/raw/population_density_database.bin";
const RANDOM_QUERY_COUNT: usize = 100_000;
const CONCURRENT_THREAD_COUNT: usize = 16;
const QUERIES_PER_THREAD: usize = 10_000;
const LINEAR_CONGRUENTIAL_MULTIPLIER: u64 = 6_364_136_223_846_793_005;
const LINEAR_CONGRUENTIAL_INCREMENT: u64 = 1_442_695_040_888_963_407;

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

/// Generates deterministic pseudo-random values.
struct DeterministicRandom {
    state: u64,
}

impl DeterministicRandom {
    /// Creates a generator with the supplied seed.
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Returns the next pseudo-random value.
    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(LINEAR_CONGRUENTIAL_MULTIPLIER)
            .wrapping_add(LINEAR_CONGRUENTIAL_INCREMENT);
        self.state
    }

    /// Returns a valid pseudo-random S2 cell at the requested level.
    fn next_cell_id(&mut self, level: u32) -> u64 {
        assert!(level <= MAX_S2_LEVEL);
        let face = self.next_u64() % 6;
        let position_bit_count = BITS_PER_LEVEL * level;
        let position_mask = if position_bit_count == 0 {
            0
        } else {
            (1u64 << position_bit_count) - 1
        };
        let position = self.next_u64() & position_mask;
        let sentinel_shift = BITS_PER_LEVEL * (MAX_S2_LEVEL - level);
        (face << 61) | (((position << 1) | 1) << sentinel_shift)
    }
}

/// Answers queries from a reconstructed sorted compact leaf set.
struct ReferenceQueryEngine {
    leaves: Vec<u32>,
}

impl ReferenceQueryEngine {
    /// Creates a reference engine from the decoded leaf set.
    fn new(query_engine: &QueryEngine) -> Self {
        Self {
            leaves: query_engine.reconstruct_all_leaves().unwrap(),
        }
    }

    /// Finds the deepest qualifying ancestor using neighboring leaves.
    fn query(&self, cell_id: u64) -> u64 {
        let query_level = get_level(cell_id);
        let normalized_cell_id = if query_level > MAX_DB_LEVEL {
            get_ancestor(cell_id, MAX_DB_LEVEL)
        } else {
            cell_id
        };
        let compact_cell_id = (normalized_cell_id >> SHIFT_COMPACT) as u32;
        let insertion_index = self.leaves.partition_point(|&leaf| leaf < compact_cell_id);

        let left_index = insertion_index.checked_sub(1);
        let right_index = (insertion_index < self.leaves.len()).then_some(insertion_index);
        [left_index, right_index]
            .into_iter()
            .flatten()
            .filter_map(|index| self.common_ancestor(cell_id, query_level, index))
            .max_by_key(|&ancestor| get_level(ancestor))
            .unwrap_or(0)
    }

    /// Returns the common database ancestor for one neighboring leaf.
    fn common_ancestor(&self, cell_id: u64, query_level: u32, leaf_index: usize) -> Option<u64> {
        let database_cell_id = u64::from(self.leaves[leaf_index]) << SHIFT_COMPACT;
        let leading_zeros = (cell_id ^ database_cell_id).leading_zeros();
        if leading_zeros < S2_FACE_BITS {
            return None;
        }

        let database_level = get_level(database_cell_id);
        let common_level = ((leading_zeros - S2_FACE_BITS) / BITS_PER_LEVEL)
            .min(query_level)
            .min(database_level)
            .min(MAX_DB_LEVEL);
        Some(get_ancestor(cell_id, common_level))
    }
}

/// Verifies dense, sparse, island, and ocean query levels.
#[test]
fn test_geographic_density_transitions() {
    let query_engine = open_database(database_path()).unwrap();
    let exact_levels = [
        ("Beijing", 39.9042, 116.4074, 12),
        ("London", 51.5074, -0.1278, 12),
        ("Kathmandu", 27.7172, 85.3240, 12),
        ("Mount Everest", 27.9881, 86.9250, 10),
    ];
    for (name, latitude, longitude, expected_level) in exact_levels {
        let cell_id = CellID::from(LatLng::from_degrees(latitude, longitude)).0;
        let result = query_engine.query(cell_id).unwrap();
        assert_eq!(
            get_level(result),
            expected_level,
            "unexpected level for {name}"
        );
    }

    let coarse_locations = [
        ("Pacific Ocean", 0.0, -140.0, 3),
        ("Antarctica", -80.0, 0.0, 3),
        ("Henderson Island", -24.37, -128.3, 6),
    ];
    for (name, latitude, longitude, maximum_level) in coarse_locations {
        let cell_id = CellID::from(LatLng::from_degrees(latitude, longitude)).0;
        let result = query_engine.query(cell_id).unwrap();
        assert_ne!(result, 0, "query returned no cell for {name}");
        assert!(
            get_level(result) <= maximum_level,
            "query returned an unexpectedly fine cell for {name}"
        );
    }
}

/// Verifies that inputs finer than level 12 resolve identically to level 12.
#[test]
fn test_finer_queries_normalize_to_level_12() {
    let query_engine = open_database(database_path()).unwrap();
    let points = [
        LatLng::from_degrees(39.9042, 116.4074),
        LatLng::from_degrees(27.9881, 86.9250),
        LatLng::from_degrees(0.0, -140.0),
    ];

    for point in points {
        let leaf_cell_id = CellID::from(point);
        let expected = query_engine
            .query(leaf_cell_id.parent(u64::from(MAX_DB_LEVEL)).0)
            .unwrap();
        for level in MAX_DB_LEVEL + 1..=MAX_S2_LEVEL {
            let result = query_engine
                .query(leaf_cell_id.parent(u64::from(level)).0)
                .unwrap();
            assert_eq!(result, expected, "query differs at level {level}");
        }
    }
}

/// Verifies that coarse queries return the query cell or one of its ancestors.
#[test]
fn test_coarse_queries_return_ancestors() {
    let query_engine = open_database(database_path()).unwrap();
    let points = [
        LatLng::from_degrees(39.9042, 116.4074),
        LatLng::from_degrees(27.9881, 86.9250),
        LatLng::from_degrees(0.0, -140.0),
    ];

    for point in points {
        let leaf_cell_id = CellID::from(point);
        for level in 0..=MAX_DB_LEVEL {
            let query_cell_id = leaf_cell_id.parent(u64::from(level)).0;
            let result = query_engine.query(query_cell_id).unwrap();
            let result_level = get_level(result);
            assert!(result_level <= level);
            assert_eq!(get_ancestor(query_cell_id, result_level), result);
        }
    }
}

/// Verifies S2 levels, ancestors, parents, and children against the S2 library.
#[test]
fn test_s2_geometry_utilities() {
    let points = [
        LatLng::from_degrees(39.9042, 116.4074),
        LatLng::from_degrees(51.5074, -0.1278),
        LatLng::from_degrees(40.7128, -74.0060),
    ];

    for point in points {
        let leaf_cell_id = CellID::from(point);
        for level in 0..=MAX_S2_LEVEL {
            let expected = leaf_cell_id.parent(u64::from(level)).0;
            assert_eq!(get_level(expected), level);
            assert_eq!(get_ancestor(leaf_cell_id.0, level), expected);
            assert_eq!(get_parent_id(leaf_cell_id.0, level), expected);

            if level < MAX_S2_LEVEL {
                let generated_children = get_children_ids(expected, level);
                let expected_children = CellID(expected).children().map(|child| child.0);
                assert_eq!(generated_children, expected_children);
                for (child_index, &child) in generated_children.iter().enumerate() {
                    assert!(!generated_children[..child_index].contains(&child));
                    assert_eq!(get_level(child), level + 1);
                    assert_eq!(get_ancestor(child, level), expected);
                }
            }
        }
    }
}

/// Verifies deterministic random queries against a leaf-vector oracle.
#[test]
fn test_random_query_parity() {
    let query_engine = open_database(database_path()).unwrap();
    let reference_engine = ReferenceQueryEngine::new(&query_engine);
    let mut random = DeterministicRandom::new(0);

    for query_index in 0..RANDOM_QUERY_COUNT {
        let level = query_index as u32 % (MAX_S2_LEVEL + 1);
        let cell_id = random.next_cell_id(level);
        assert_eq!(
            query_engine.query(cell_id).unwrap(),
            reference_engine.query(cell_id),
            "query mismatch for cell {cell_id:016x}"
        );
    }
}

/// Verifies that one mapped engine serves concurrent deterministic queries.
#[test]
fn test_concurrent_query_parity() {
    let query_engine = Arc::new(open_database(database_path()).unwrap());
    let reference_engine = Arc::new(ReferenceQueryEngine::new(&query_engine));
    let mut threads = Vec::with_capacity(CONCURRENT_THREAD_COUNT);

    for thread_index in 0..CONCURRENT_THREAD_COUNT {
        let query_engine = Arc::clone(&query_engine);
        let reference_engine = Arc::clone(&reference_engine);
        threads.push(std::thread::spawn(move || {
            let mut random = DeterministicRandom::new(thread_index as u64);
            for query_index in 0..QUERIES_PER_THREAD {
                let level = (query_index + thread_index) as u32 % (MAX_S2_LEVEL + 1);
                let cell_id = random.next_cell_id(level);
                assert_eq!(
                    query_engine.query(cell_id).unwrap(),
                    reference_engine.query(cell_id)
                );
            }
        }));
    }

    for thread in threads {
        thread.join().unwrap();
    }
}

/// Verifies that reconstructed S2PD leaves are sorted, valid, and count-consistent.
#[test]
fn test_reconstructed_leaf_invariants() {
    let query_engine = open_database(database_path()).unwrap();
    let leaves = query_engine.reconstruct_all_leaves().unwrap();
    assert_eq!(leaves.len(), query_engine.count());
    assert!(leaves.windows(2).all(|window| window[0] < window[1]));

    for compact_cell_id in leaves {
        let cell_id = u64::from(compact_cell_id) << SHIFT_COMPACT;
        assert!(get_level(cell_id) <= MAX_DB_LEVEL);
    }
}
