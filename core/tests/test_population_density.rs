//! Provides integration tests for the S2 population density database engine.
//!
//! Verifies correct S2 cell level calculations, ancestor mapping correctness,
//! boundary/error handling, and query lookups for predefined geographic targets.

mod common;

use anyhow::{Result, anyhow, ensure};
use common::open_database;
use population_density::{
    BITS_PER_LEVEL, MAX_DB_LEVEL, MAX_S2_LEVEL, NUM_ROOT_FACES, POPULATION_FIXED_POINT_SCALE,
    POPULATION_THRESHOLD_FIXED, QueryEngine, S2_FACE_BITS, S2_FACE_SHIFT, SHIFT_COMPACT,
    get_ancestor, get_children_ids, get_level, get_parent_id,
};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use s2::cellid::CellID;
use s2::latlng::LatLng;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::LazyLock;
use tempfile::NamedTempFile;
use tiff::decoder::{Decoder, DecodingResult, Limits};

const TIFF_COMPRESSION_LZW: u16 = 5;

const LEVEL_0_SENTINEL_SHIFT: u32 = 60;

const GEOTIFF_MAX_LATITUDE: f64 = 84.0;
const GEOTIFF_MIN_LONGITUDE: f64 = -180.0;
const GEOTIFF_PIXEL_SCALE: f64 = 0.008333333333333333;
const PIXEL_CENTER_OFFSET: f64 = 0.5;
const GEOTIFF_NODATA_VALUE: f32 = -99999.0;

const SHARED_TIFF_PATH: &str = "../data/global_pop_2026_CN_1km_R2025A_UA_v1.tif";
const GENERATED_DATABASE_PATH: &str = "../population_density_database.bin";
const PACKAGED_DATABASE_PATH: &str =
    "../../../packages/apps/NetworkLocation/res/raw/population_density_database.bin";

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

/// Stores the deterministic GeoTIFF-derived oracle shared by the parity tests.
static SHARED_TIFF_ORACLE: LazyLock<NaiveOracle> = LazyLock::new(|| {
    assert!(
        Path::new(SHARED_TIFF_PATH).exists(),
        "GeoTIFF source '{}' not found",
        SHARED_TIFF_PATH
    );
    NaiveOracle::from_tiff(SHARED_TIFF_PATH).unwrap()
});

/// Returns whether a TIFF file uses LZW compression.
fn is_lzw_compressed<P: AsRef<Path>>(path: P) -> Result<bool> {
    let file = File::open(path)?;
    let mut decoder = Decoder::new(BufReader::new(file))?.with_limits(Limits::unlimited());
    let compression = decoder.get_tag_unsigned::<u16>(tiff::tags::Tag::Compression)?;
    Ok(compression == TIFF_COMPRESSION_LZW)
}

/// Converts a finite population value to fixed-point units.
fn convert_population_to_fixed(population: f32) -> i64 {
    let fixed_population = (f64::from(population) * POPULATION_FIXED_POINT_SCALE as f64).round();
    assert!(
        fixed_population < i64::MAX as f64,
        "population value exceeds the fixed-point range"
    );
    fixed_population as i64
}

/// Adds population to a cell with checked fixed-point accumulation.
fn add_population(cell_populations: &mut FxHashMap<u64, i64>, cell_id: u64, population: i64) {
    let accumulated_population = cell_populations.entry(cell_id).or_insert(0);
    *accumulated_population = accumulated_population
        .checked_add(population)
        .unwrap_or_else(|| panic!("fixed-point population overflow for S2 cell {cell_id:016x}"));
}

/// Recursively finds the leaves of the pruned S2 cell quadtree.
fn find_leaves_recursive(
    cell_id: u64,
    level: u32,
    cell_populations: &FxHashMap<u64, i64>,
    leaves: &mut Vec<u64>,
) {
    let population = *cell_populations.get(&cell_id).unwrap_or(&0);
    if population < POPULATION_THRESHOLD_FIXED {
        return;
    }

    if level == MAX_DB_LEVEL {
        leaves.push(cell_id);
        return;
    }

    let children = get_children_ids(cell_id, level);
    let mut children_in_tree = false;
    for &child in &children {
        if *cell_populations.get(&child).unwrap_or(&0) >= POPULATION_THRESHOLD_FIXED {
            children_in_tree = true;
            break;
        }
    }

    if !children_in_tree {
        leaves.push(cell_id);
        return;
    }

    for &child in &children {
        find_leaves_recursive(child, level + 1, cell_populations, leaves);
    }
}

/// Returns several geographic test points.
fn get_test_points() -> Vec<LatLng> {
    vec![
        LatLng::from_degrees(39.9042, 116.4074), // Beijing
        LatLng::from_degrees(51.5074, -0.1278),  // London
        LatLng::from_degrees(40.7128, -74.0060), // New York
    ]
}

/// Verifies that S2 cell level calculation returns the correct level.
#[test]
fn test_bitwise_levels() {
    for point in get_test_points() {
        let cell_id = CellID::from(point);
        for level in 0..=MAX_S2_LEVEL {
            let parent = cell_id.parent(level as u64);
            assert_eq!(get_level(parent.0), level);
        }
    }
}

/// Verifies that get_ancestor returns the correct ancestor ID at any level.
#[test]
fn test_bitwise_ancestors() {
    for point in get_test_points() {
        let cell_id = CellID::from(point);
        for level in 0..=MAX_S2_LEVEL {
            let expected = cell_id.parent(level as u64).0;
            let actual = get_ancestor(cell_id.0, level);
            assert_eq!(actual, expected);
        }
    }
}

/// Verifies that opening a nonexistent database file fails gracefully.
#[test]
fn test_missing_database() {
    let temporary_directory = tempfile::tempdir().unwrap();
    let missing_database = temporary_directory.path().join("missing.bin");

    // SAFETY: The path does not exist, so no file-backed mapping can be created.
    let result = unsafe { QueryEngine::open(&missing_database) };
    assert!(result.is_err());
}

/// Runs geographic query lookups and asserts accuracy against expected levels.
#[test]
fn test_actual_queries() {
    let database_path = database_path();
    let query_engine = open_database(&database_path).unwrap();

    // Query Beijing and verify it has a populated ancestor.
    let beijing_point = LatLng::from_degrees(39.9042, 116.4074);
    let beijing_cell_id = CellID::from(beijing_point);
    let beijing_result = query_engine.query(beijing_cell_id.0).unwrap();
    assert_ne!(
        beijing_result, 0,
        "Beijing should have a populated ancestor"
    );
    assert!(get_level(beijing_result) <= MAX_DB_LEVEL);

    // Query an ocean cell and verify it has a coarse ancestor.
    let ocean_point = LatLng::from_degrees(0.0, -140.0);
    let ocean_cell_id = CellID::from(ocean_point);
    let ocean_result = query_engine.query(ocean_cell_id.0).unwrap();
    assert_ne!(ocean_result, 0, "ocean should have a coarse ancestor");
    assert!(get_level(ocean_result) <= 3);

    // Query Mount Everest and verify its ancestor level.
    let everest_point = LatLng::from_degrees(27.9881, 86.9250);
    let everest_cell_id = CellID::from(everest_point);
    let everest_result = query_engine.query(everest_cell_id.0).unwrap();
    assert_ne!(
        everest_result, 0,
        "Mount Everest should have a populated ancestor"
    );
    assert_eq!(
        get_level(everest_result),
        10,
        "Mount Everest ancestor should resolve to level 10"
    );
}

/// Implements reference queries over a flat leaf cell ID array.
struct NaiveOracle {
    leaves: Vec<u32>,
    /// Stores propagated fixed-point population per S2 cell.
    populations: FxHashMap<u64, i64>,
}

impl NaiveOracle {
    /// Creates a naive oracle by parsing the raw GeoTIFF directly.
    fn from_tiff<P: AsRef<Path>>(tiff_path: P) -> Result<Self> {
        let is_lzw = is_lzw_compressed(tiff_path.as_ref())?;

        let mut current_tiff_path = tiff_path.as_ref().to_path_buf();
        let mut _temporary_directory = None;

        if is_lzw {
            let temporary_directory = tempfile::tempdir()?;
            let temporary_path = temporary_directory.path().join("prepared.tif");
            let status = Command::new("tiffcp")
                .arg("-m")
                .arg("0")
                .arg("-c")
                .arg("zip")
                .arg(tiff_path.as_ref())
                .arg(&temporary_path)
                .status()
                .map_err(|error| anyhow!("failed to run tiffcp: {error}"))?;
            if !status.success() {
                return Err(anyhow!("tiffcp failed with status {status}"));
            }

            current_tiff_path = temporary_path;
            _temporary_directory = Some(temporary_directory);
        }

        let file = File::open(&current_tiff_path)?;
        let mut decoder = Decoder::new(BufReader::new(file))?.with_limits(Limits::unlimited());

        let (width, height) = decoder.dimensions()?;
        let image_result = decoder.read_image()?;
        let width_usize = usize::try_from(width)?;
        let height_usize = usize::try_from(height)?;
        let expected_pixel_count = width_usize
            .checked_mul(height_usize)
            .ok_or_else(|| anyhow!("GeoTIFF dimensions exceed the addressable range"))?;

        let data = match image_result {
            DecodingResult::F32(data) => {
                ensure!(
                    data.len() == expected_pixel_count,
                    "decoded image length {} does not match {width} x {height} = {expected_pixel_count}",
                    data.len()
                );
                data
            }
            _ => return Err(anyhow!("unexpected image data type (expected Float32)")),
        };

        let level_12_populations = (0..height_usize)
            .into_par_iter()
            .fold(FxHashMap::default, |mut local_map, pixel_y| {
                let row_offset = pixel_y * width_usize;
                let pixel_y_float = pixel_y as f64;
                for pixel_x in 0..width_usize {
                    let value = data[row_offset + pixel_x];
                    if value == 0.0 || value == GEOTIFF_NODATA_VALUE {
                        continue;
                    }
                    assert!(
                        value.is_finite() && value > 0.0,
                        "source GeoTIFF pixel ({pixel_x}, {pixel_y}) contains invalid population {value}"
                    );
                    let latitude = GEOTIFF_MAX_LATITUDE
                        - (pixel_y_float + PIXEL_CENTER_OFFSET) * GEOTIFF_PIXEL_SCALE;
                    let longitude = GEOTIFF_MIN_LONGITUDE
                        + ((pixel_x as f64) + PIXEL_CENTER_OFFSET) * GEOTIFF_PIXEL_SCALE;
                    let latitude_longitude = LatLng::from_degrees(latitude, longitude);
                    let cell_id = CellID::from(latitude_longitude).parent(MAX_DB_LEVEL as u64);
                    let population = convert_population_to_fixed(value);
                    add_population(&mut local_map, cell_id.0, population);
                }
                local_map
            })
            .reduce(FxHashMap::default, |mut accumulator_map, local_map| {
                for (cell_id, population) in local_map {
                    add_population(&mut accumulator_map, cell_id, population);
                }
                accumulator_map
            });

        let mut cell_populations = level_12_populations;
        cell_populations.reserve(cell_populations.len() / 3);

        let mut current_level_cells: Vec<u64> = cell_populations.keys().cloned().collect();
        let mut parent_level_cells = Vec::with_capacity(current_level_cells.len());

        for level in (0..MAX_DB_LEVEL).rev() {
            parent_level_cells.clear();
            for &cell_id in &current_level_cells {
                let parent_id = get_parent_id(cell_id, level);
                let population = *cell_populations.get(&cell_id).unwrap_or(&0);
                add_population(&mut cell_populations, parent_id, population);
                parent_level_cells.push(parent_id);
            }
            parent_level_cells.sort_unstable();
            parent_level_cells.dedup();
            std::mem::swap(&mut current_level_cells, &mut parent_level_cells);
        }

        let face_ids: Vec<u64> = (0..NUM_ROOT_FACES)
            .map(|face| ((face as u64) << S2_FACE_SHIFT) | (1u64 << LEVEL_0_SENTINEL_SHIFT))
            .collect();
        let mut leaves = Vec::new();

        for &face_id in &face_ids {
            find_leaves_recursive(face_id, 0, &cell_populations, &mut leaves);
        }

        leaves.sort_unstable();

        let compact_leaves: Vec<u32> = leaves
            .iter()
            .map(|&cell_id| (cell_id >> SHIFT_COMPACT) as u32)
            .collect();

        Ok(Self {
            leaves: compact_leaves,
            populations: cell_populations,
        })
    }

    /// Performs the query using flat binary search and ancestor matching.
    fn query(&self, s2_cell_id: u64) -> u64 {
        debug_assert!(s2_cell_id != 0, "S2 cell ID cannot be 0");
        let query_level = get_level(s2_cell_id);
        let query_cell_id = if query_level > MAX_DB_LEVEL {
            get_ancestor(s2_cell_id, MAX_DB_LEVEL)
        } else {
            s2_cell_id
        };
        let compact_query_id = (query_cell_id >> SHIFT_COMPACT) as u32;

        let search_result = self.leaves.binary_search(&compact_query_id);

        let (left_index_option, right_index_option) = match search_result {
            Ok(index) => (
                Some(index),
                if index + 1 < self.leaves.len() {
                    Some(index + 1)
                } else {
                    None
                },
            ),
            Err(index) => (
                if index > 0 { Some(index - 1) } else { None },
                if index < self.leaves.len() {
                    Some(index)
                } else {
                    None
                },
            ),
        };

        let mut best_ancestor = 0u64;
        let mut best_level = -1i32;

        if let Some(left_index) = left_index_option {
            let compact_value = self.leaves[left_index];
            let database_cell_id = (compact_value as u64) << SHIFT_COMPACT;
            let xor_value = s2_cell_id ^ database_cell_id;
            let leading_zeros = xor_value.leading_zeros();
            if leading_zeros >= S2_FACE_BITS {
                let database_level = get_level(database_cell_id);
                let common_level = std::cmp::min(
                    (leading_zeros - S2_FACE_BITS) / BITS_PER_LEVEL,
                    std::cmp::min(query_level, database_level),
                );
                let final_level = std::cmp::min(MAX_DB_LEVEL, common_level);
                best_ancestor = get_ancestor(s2_cell_id, final_level);
                best_level = final_level as i32;
            }
        }

        if let Some(right_index) = right_index_option {
            let compact_value = self.leaves[right_index];
            let database_cell_id = (compact_value as u64) << SHIFT_COMPACT;
            let xor_value = s2_cell_id ^ database_cell_id;
            let leading_zeros = xor_value.leading_zeros();
            if leading_zeros >= S2_FACE_BITS {
                let database_level = get_level(database_cell_id);
                let common_level = std::cmp::min(
                    (leading_zeros - S2_FACE_BITS) / BITS_PER_LEVEL,
                    std::cmp::min(query_level, database_level),
                );
                let final_level = std::cmp::min(MAX_DB_LEVEL, common_level);
                if (final_level as i32) > best_level {
                    best_ancestor = get_ancestor(s2_cell_id, final_level);
                }
            }
        }

        best_ancestor
    }
}

/// Verifies exhaustive parity for all level-12 query cells.
///
/// This test runs in parallel across all six S2 faces, verifying that
/// the optimized QueryEngine query matches the NaiveOracle query perfectly.
#[test]
fn test_exhaustive_global_l12_parity() {
    const LEVEL_12_CELL_STEP_SHIFT: u32 = 37;

    let database_path = database_path();
    let query_engine = open_database(&database_path).unwrap();
    let naive_oracle = &*SHARED_TIFF_ORACLE;

    println!("checking exhaustive level-12 parity in parallel");
    let start_time = std::time::Instant::now();

    let mismatches: usize = (0..NUM_ROOT_FACES)
        .into_par_iter()
        .map(|face| {
            let mut face_mismatches = 0;
            let start_cell = (face as u64) << S2_FACE_SHIFT | (1u64 << SHIFT_COMPACT);
            let end_cell = ((face + 1) as u64) << S2_FACE_SHIFT;
            let step = 1u64 << LEVEL_12_CELL_STEP_SHIFT;

            let mut s2_cell_id = start_cell;
            while s2_cell_id < end_cell {
                let optimized_result = query_engine.query(s2_cell_id).unwrap();
                let naive_result = naive_oracle.query(s2_cell_id);

                if optimized_result != naive_result {
                    face_mismatches += 1;
                    if face_mismatches <= 5 {
                        println!(
                            "mismatch on face {} at cell {:016x}: optimized={:016x}, naive={:016x}",
                            face, s2_cell_id, optimized_result, naive_result
                        );
                    }
                }
                s2_cell_id += step;
            }
            face_mismatches
        })
        .sum();

    let elapsed = start_time.elapsed();
    println!("exhaustive parity completed in {elapsed:.2?}");
    assert_eq!(
        mismatches, 0,
        "exhaustive parity failed with {} mismatches",
        mismatches
    );
}

/// Verifies that public helper preconditions panic on invalid inputs.
#[test]
fn test_assert_preconditions_panic() {
    // Require a nonzero cell ID.
    let result = std::panic::catch_unwind(|| {
        get_level(0);
    });
    assert!(result.is_err());

    // Require a supported ancestor level.
    let result = std::panic::catch_unwind(|| {
        get_ancestor(1, 31);
    });
    assert!(result.is_err());

    // Require the supplied child-parent level to match the cell ID.
    let level_0_cell_id = 1u64 << 60;
    let result = std::panic::catch_unwind(|| {
        get_children_ids(level_0_cell_id, 1);
    });
    assert!(result.is_err());
}

/// Verifies query parity for finer-level S2 cell ID queries where bit 36 is 0.
#[test]
fn test_query_finer_levels_regression() {
    let database_path = database_path();
    let query_engine = open_database(&database_path).unwrap();
    let naive_oracle = &*SHARED_TIFF_ORACLE;

    // Choose a specific Level 17 cell ID where bit 36 is 0.
    // face 0, level 17, trailing zeros = 26.
    let cell_id_level_17 = 0x1000000004000000u64;
    let optimized_result = query_engine.query(cell_id_level_17).unwrap();
    let naive_result = naive_oracle.query(cell_id_level_17);
    assert_eq!(optimized_result, naive_result);
}

/// Verifies that calling get_level with an invalid S2 cell ID triggers a panic safely.
#[test]
fn test_adversarial_get_level_underflow() {
    let cell_id_too_high = 1u64 << 61;
    let result = std::panic::catch_unwind(|| {
        get_level(cell_id_too_high);
    });
    assert!(
        result.is_err(),
        "get_level must panic on an invalid cell ID"
    );
}

/// Verifies that querying with a cell ID of 0 returns an error instead of panicking.
#[test]
fn test_query_zero_cell_id() {
    let database_path = database_path();
    let query_engine = open_database(&database_path).unwrap();
    let result = query_engine.query(0);
    assert!(result.is_err());
    let error_message = result.err().unwrap().to_string();
    assert!(error_message.contains("S2 cell ID cannot be 0"));
}

/// Verifies resolved S2 levels across geographic transition scenarios.
#[test]
fn test_geographic_transition_scenarios() {
    let database_path = database_path();
    let query_engine = open_database(&database_path).unwrap();
    let naive_oracle = &*SHARED_TIFF_ORACLE;

    let verify_query = |latitude: f64,
                        longitude: f64,
                        description: &str,
                        level_matches: fn(u32) -> bool| {
        let point = LatLng::from_degrees(latitude, longitude);
        let query_cell_id = CellID::from(point).0;

        let optimized_result = query_engine.query(query_cell_id).unwrap();
        let naive_result = naive_oracle.query(query_cell_id);

        assert_eq!(
            optimized_result, naive_result,
            "query mismatch at {} ({}, {}): optimized = {:016x}, naive = {:016x}",
            description, latitude, longitude, optimized_result, naive_result
        );

        let level = get_level(optimized_result);
        assert!(
            level_matches(level),
            "level expectation failed at {} ({}, {}): resolved to level {}, which did not satisfy the constraint",
            description,
            latitude,
            longitude,
            level
        );
    };

    // 1. Beijing radius.
    verify_query(39.9042, 116.4074, "Beijing Center", |level| level == 12);
    verify_query(39.9132, 116.4074, "Beijing 1km N", |level| level == 12);
    verify_query(39.9042, 116.4191, "Beijing 1km E", |level| level == 12);
    verify_query(39.8592, 116.4074, "Beijing 5km S", |level| level == 12);
    verify_query(39.9042, 116.2904, "Beijing 10km W", |level| level == 12);
    verify_query(
        40.3542,
        116.4074,
        "Huairou Mountains (Beijing 50km N)",
        |level| level <= 10,
    );

    // 2. Everest peaks and valleys.
    verify_query(27.7172, 85.3240, "Kathmandu Valley", |level| level == 12);
    verify_query(29.6524, 91.1172, "Lhasa", |level| level == 12);
    verify_query(27.9881, 86.9250, "Mount Everest Peak", |level| level == 10);

    // 3. Shanghai coastline.
    verify_query(31.19, 121.40, "Shanghai Inland", |level| level == 12);
    verify_query(31.19, 121.70, "Pudong Coastal Land", |level| level == 12);
    verify_query(31.19, 121.80, "Shanghai Coastline Water Edge", |level| {
        level == 11
    });
    verify_query(31.19, 122.20, "Near-Shore Ocean", |level| {
        level == 7 || level == 6
    });
    verify_query(31.19, 123.50, "Open Sea", |level| {
        level == 5 || level == 3 || level == 0
    });

    // 4. Unpopulated islands.
    verify_query(-24.3797, -128.3242, "Henderson Island", |level| {
        level == 3 || level == 0
    });
    verify_query(10.2983, -109.2192, "Clipperton Island", |level| {
        level == 3 || level == 0
    });
    verify_query(-49.3500, 69.3500, "Kerguelen Islands", |level| {
        level == 3 || level == 0
    });
    verify_query(-54.4208, 3.3464, "Bouvet Island", |level| {
        level == 3 || level == 0
    });
}

/// Verifies that ancestor helpers reject levels deeper than the source cell.
#[test]
fn test_ancestor_level_boundaries() {
    use population_density::try_get_ancestor;

    // Use a known Level 17 S2 cell ID.
    let cell_id = 0x1000000004000000u64;
    assert_eq!(get_level(cell_id), 17);

    assert!(try_get_ancestor(cell_id, 17).is_ok());
    assert!(try_get_ancestor(cell_id, 10).is_ok());

    assert!(try_get_ancestor(cell_id, 18).is_err());
    assert!(try_get_ancestor(cell_id, 30).is_err());

    let result = std::panic::catch_unwind(|| {
        get_ancestor(cell_id, 18);
    });
    assert!(result.is_err());
}

/// Verifies returned cells meet the threshold in the GeoTIFF-derived population map.
///
/// It uses propagated GeoTIFF populations rather than reconstructed leaves, so a
/// pruning regression cannot be mirrored by the oracle.
#[test]
fn test_returned_cells_meet_population_threshold() {
    let database_path = database_path();
    let query_engine = open_database(&database_path).unwrap();
    let oracle = &*SHARED_TIFF_ORACLE;

    // Cover dense, sparse, and ocean locations with a deterministic global grid.
    let mut query_points: Vec<(f64, f64)> = vec![
        (39.9042, 116.4074), // Beijing
        (51.5074, -0.1278),  // London
        (40.7128, -74.0060), // New York
    ];
    let mut latitude_degrees = -80i32;
    while latitude_degrees <= 80 {
        let mut longitude_degrees = -180i32;
        while longitude_degrees < 180 {
            query_points.push((latitude_degrees as f64, longitude_degrees as f64));
            longitude_degrees += 11;
        }
        latitude_degrees += 7;
    }

    for (latitude, longitude) in query_points {
        let leaf_cell_id = CellID::from(LatLng::from_degrees(latitude, longitude)).0;
        let returned_cell_id = query_engine.query(leaf_cell_id).unwrap();
        assert_ne!(
            returned_cell_id, 0,
            "query returned no qualifying cell for ({}, {})",
            latitude, longitude
        );
        let returned_population = *oracle.populations.get(&returned_cell_id).unwrap_or(&0);
        assert!(
            returned_population >= POPULATION_THRESHOLD_FIXED,
            "under-coarsening at ({}, {}): returned cell {:016x} (level {}) represents {} fixed-point population, below threshold {}",
            latitude,
            longitude,
            returned_cell_id,
            get_level(returned_cell_id),
            returned_population,
            POPULATION_THRESHOLD_FIXED
        );
    }
}

/// Verifies that S2PD handles an unaligned memory mapping correctly.
///
/// S2PD reads byte fields, so an unaligned base must return the same query result.
#[test]
fn test_from_mmap_unaligned_base_handling() {
    use population_density::memmap2::MmapOptions;

    let database_path = database_path();
    let database_bytes = std::fs::read(&database_path).unwrap();
    assert!(database_bytes.starts_with(b"S2PD"));

    let padded_file = NamedTempFile::new().unwrap();
    let mut padded = Vec::with_capacity(database_bytes.len() + 1);
    padded.push(0);
    padded.extend_from_slice(&database_bytes);
    std::fs::write(padded_file.path(), &padded).unwrap();

    let file = File::open(padded_file.path()).unwrap();
    // SAFETY: The private temporary file remains unmodified while this mapping exists.
    let mmap = unsafe {
        MmapOptions::new()
            .offset(1)
            .len(database_bytes.len())
            .map(&file)
            .unwrap()
    };
    assert_eq!(mmap.as_ptr() as usize % 4, 1);
    let unaligned_engine =
        QueryEngine::from_mmap(mmap).expect("S2PD must support an unaligned mapping base");

    let verified_file = File::open(padded_file.path()).unwrap();
    // SAFETY: The private temporary file remains unmodified while this mapping exists.
    let verified_mmap = unsafe {
        MmapOptions::new()
            .offset(1)
            .len(database_bytes.len())
            .map(&verified_file)
            .unwrap()
    };
    let verified_unaligned_engine = QueryEngine::from_verified_mmap(verified_mmap)
        .expect("verified S2PD must support an unaligned mapping base");
    let aligned_engine = open_database(&database_path).unwrap();
    let query_cell_id = CellID::from(LatLng::from_degrees(0.0, 0.0)).0;
    let expected = aligned_engine.query(query_cell_id).unwrap();
    assert_eq!(unaligned_engine.query(query_cell_id).unwrap(), expected);
    assert_eq!(
        verified_unaligned_engine.query(query_cell_id).unwrap(),
        expected
    );
}
