//! Provides integration tests for the S2 population density database engine.
//!
//! Verifies correct S2 cell level calculations, ancestor mapping correctness,
//! boundary/error handling, and query lookups for predefined geographic targets.

#![allow(clippy::needless_range_loop, clippy::collapsible_if)]

use population_density::{
    BITS_PER_LEVEL, BLOCK_HEADER_SIZE, MAX_DB_LEVEL, MAX_S2_LEVEL, NUM_ROOT_FACES,
    POPULATION_FIXED_POINT_SCALE, POPULATION_THRESHOLD_FIXED, QueryEngine, S2_FACE_BITS,
    SHIFT_COMPACT, get_ancestor, get_children_ids, get_level, get_parent_id,
};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use s2::cellid::CellID;
use s2::latlng::LatLng;
use std::fs::File;
use std::io::{BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicUsize, Ordering};
use tiff::decoder::{Decoder, DecodingResult, Limits};

const TIFF_COMPRESSION_NONE: u16 = 1;
const TIFF_COMPRESSION_LZW: u16 = 5;

static TEMP_FILE_COUNTER: AtomicUsize = AtomicUsize::new(0);

const S2_FACE_SHIFT: u32 = 61;
const LEVEL_0_SENTINEL_SHIFT: u32 = 60;

const GEOTIFF_MAX_LATITUDE: f64 = 84.0;
const GEOTIFF_MIN_LONGITUDE: f64 = -180.0;
const GEOTIFF_PIXEL_SCALE: f64 = 0.008333333333333333;
const PIXEL_CENTER_OFFSET: f64 = 0.5;
const GEOTIFF_NODATA_VALUE: f32 = -99999.0;

const SHARED_TIFF_PATH: &str = "../data/global_pop_2026_CN_1km_R2025A_UA_v1.tif";

/// The naive oracle is an expensive but deterministic, read-only product of the GeoTIFF: build it
/// once and share it across the parity tests instead of re-decoding the multi-hundred-megabyte
/// raster per test. This does not weaken coverage — every test still compares query() against this
/// same immutable oracle, and nothing asserts properties of the build itself.
static SHARED_TIFF_ORACLE: LazyLock<NaiveOracle> = LazyLock::new(|| {
    assert!(
        Path::new(SHARED_TIFF_PATH).exists(),
        "GeoTIFF file '{}' not found. Please ensure the raw data file is present at that path.",
        SHARED_TIFF_PATH
    );
    NaiveOracle::new_from_tiff(SHARED_TIFF_PATH).unwrap()
});

/// Manages automatic cleanup of a temporary file when dropped from scope.
struct CleanupGuard {
    path: Option<PathBuf>,
}

impl CleanupGuard {
    /// Creates a new guard for a temporary file path.
    fn new(path: PathBuf) -> Self {
        Self { path: Some(path) }
    }
}

impl Drop for CleanupGuard {
    fn drop(&mut self) {
        if let Some(ref path) = self.path {
            if path.exists() {
                if let Err(error) = std::fs::remove_file(path) {
                    eprintln!(
                        "Warning: Failed to remove temporary file '{}': {}",
                        path.display(),
                        error
                    );
                } else {
                    println!(
                        "Successfully cleaned up temporary file '{}'",
                        path.display()
                    );
                }
            }
        }
    }
}

/// Checks if a TIFF file is LZW compressed.
fn is_lzw_compressed<P: AsRef<Path>>(path: P) -> anyhow::Result<bool> {
    let file = File::open(path)?;
    let mut decoder = Decoder::new(BufReader::new(file))?.with_limits(Limits::unlimited());
    let compression = match decoder.get_tag_unsigned::<u16>(tiff::tags::Tag::Compression) {
        Ok(compression_tag) => compression_tag,
        Err(_) => TIFF_COMPRESSION_NONE, // Defaults to uncompressed if tag is missing.
    };
    Ok(compression == TIFF_COMPRESSION_LZW)
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

/// Generates a set of test points across representative locations.
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

/// Verifies that loading a nonexistent database file fails gracefully.
#[test]
fn test_missing_database() {
    let result = QueryEngine::new("nonexistent_db.db");
    assert!(result.is_err());
}

/// Verifies that a database with invalid magic bytes fails to load.
#[test]
fn test_invalid_magic_bytes() {
    let temporary_database = "scratch/temp_invalid.db";
    std::fs::create_dir_all("scratch").unwrap();
    {
        let mut file = File::create(temporary_database).unwrap();
        file.write_all(b"BAD_MAGIC").unwrap();
    }
    let result = QueryEngine::new(temporary_database);
    assert!(result.is_err());
    std::fs::remove_file(temporary_database).unwrap();
}

/// Runs geographic query lookups and asserts accuracy against expected levels.
#[test]
fn test_actual_queries() {
    let database_path = "../population_density_database.bin";
    assert!(
        std::path::Path::new(database_path).exists(),
        "Database file '{}' not found. Please build the database first using 'cargo run --release --bin build_database'.",
        database_path
    );

    let query_engine = QueryEngine::new(database_path).unwrap();

    // Query Beijing and verify it has a populated ancestor.
    let beijing_point = LatLng::from_degrees(39.9042, 116.4074);
    let beijing_cell_id = CellID::from(beijing_point);
    let beijing_result = query_engine.query(beijing_cell_id.0).unwrap();
    assert_ne!(
        beijing_result, 0,
        "Beijing should have a populated ancestor!"
    );
    assert!(get_level(beijing_result) <= MAX_DB_LEVEL);

    // Query an ocean cell and verify it has a coarse ancestor.
    let ocean_point = LatLng::from_degrees(0.0, -140.0);
    let ocean_cell_id = CellID::from(ocean_point);
    let ocean_result = query_engine.query(ocean_cell_id.0).unwrap();
    assert_ne!(ocean_result, 0, "Ocean should have a coarse ancestor!");
    assert!(get_level(ocean_result) <= 3);

    // Query Mount Everest and verify its ancestor level.
    let everest_point = LatLng::from_degrees(27.9881, 86.9250);
    let everest_cell_id = CellID::from(everest_point);
    let everest_result = query_engine.query(everest_cell_id.0).unwrap();
    assert_ne!(
        everest_result, 0,
        "Mount Everest should have a populated ancestor!"
    );
    assert_eq!(
        get_level(everest_result),
        10,
        "Mount Everest ancestor should resolve to level 10"
    );
}

/// Represents a naive baseline reference query oracle.
///
/// This implementation performs a standard binary search and direct bitwise
/// ancestor matching on a fully reconstructed flat leaf cell ID array to serve
/// as the 100% correct baseline for correctness verification.
pub struct NaiveOracle {
    leaves: Vec<u32>,
    /// Propagated population per S2 cell in fixed-point units, used by the population-invariant test.
    populations: FxHashMap<u64, i64>,
}

impl NaiveOracle {
    /// Creates a new naive oracle by reconstructing all leaf cells from a QueryEngine.
    pub fn new(query_engine: &QueryEngine) -> Self {
        Self {
            leaves: query_engine.reconstruct_all_leaves().unwrap(),
            populations: FxHashMap::default(),
        }
    }

    /// Creates a new naive oracle by parsing the raw GeoTIFF directly.
    pub fn new_from_tiff<P: AsRef<Path>>(tiff_path: P) -> anyhow::Result<Self> {
        let is_lzw = is_lzw_compressed(tiff_path.as_ref())?;

        let mut current_tiff_path = tiff_path.as_ref().to_path_buf();
        let _guard;

        if is_lzw {
            // Check if tiffcp is installed on the system.
            let check_command = Command::new("tiffcp").arg("-i").output();
            if check_command.is_err() {
                return Err(anyhow::anyhow!(
                    "System utility 'tiffcp' is not installed or not found on your PATH.\n\
                     This tool is required to convert LZW-compressed GeoTIFFs to Deflate compression on-the-fly."
                ));
            }

            let counter = TEMP_FILE_COUNTER.fetch_add(1, Ordering::SeqCst);
            let temporary_path = tiff_path
                .as_ref()
                .with_extension(format!("tmp_{}.tif", counter));
            _guard = CleanupGuard::new(temporary_path.clone());

            let mut child = Command::new("tiffcp")
                .arg("-m")
                .arg("0")
                .arg("-c")
                .arg("zip")
                .arg(tiff_path.as_ref())
                .arg(&temporary_path)
                .spawn()
                .map_err(|error| anyhow::anyhow!("Failed to spawn tiffcp subprocess: {}", error))?;

            let status = child.wait().map_err(|error| {
                anyhow::anyhow!("Failed to wait on tiffcp subprocess: {}", error)
            })?;
            if !status.success() {
                return Err(anyhow::anyhow!(
                    "tiffcp failed with exit code: {:?}",
                    status.code().unwrap_or(-1)
                ));
            }

            current_tiff_path = temporary_path;
        } else {
            _guard = CleanupGuard { path: None };
        }

        let file = File::open(&current_tiff_path)?;
        let mut decoder = Decoder::new(BufReader::new(file))?.with_limits(Limits::unlimited());

        let (width, height) = decoder.dimensions()?;
        let image_result = decoder.read_image()?;

        let data = match image_result {
            DecodingResult::F32(vec) => {
                if vec.len() != (width as usize) * (height as usize) {
                    return Err(anyhow::anyhow!(
                        "Decoded image data vector length {} does not match expected size {} x {} = {}",
                        vec.len(),
                        width,
                        height,
                        (width as usize) * (height as usize)
                    ));
                }
                vec
            }
            _ => {
                return Err(anyhow::anyhow!(
                    "Unexpected image data type (expected Float32)"
                ));
            }
        };

        let width_usize = width as usize;
        let level_12_populations = (0..height as usize)
            .into_par_iter()
            .fold(FxHashMap::default, |mut local_map, pixel_y| {
                let row_offset = pixel_y * width_usize;
                let pixel_y_float = pixel_y as f64;
                for pixel_x in 0..width_usize {
                    let value = data[row_offset + pixel_x];
                    if value.is_finite() && value > 0.0 && value != GEOTIFF_NODATA_VALUE {
                        let latitude = GEOTIFF_MAX_LATITUDE
                            - (pixel_y_float + PIXEL_CENTER_OFFSET) * GEOTIFF_PIXEL_SCALE;
                        let longitude = GEOTIFF_MIN_LONGITUDE
                            + ((pixel_x as f64) + PIXEL_CENTER_OFFSET) * GEOTIFF_PIXEL_SCALE;
                        let latitude_longitude = LatLng::from_degrees(latitude, longitude);
                        let cell_id = CellID::from(latitude_longitude).parent(MAX_DB_LEVEL as u64);
                        *local_map.entry(cell_id.0).or_insert(0i64) +=
                            (value as f64 * POPULATION_FIXED_POINT_SCALE as f64).round() as i64;
                    }
                }
                local_map
            })
            .reduce(FxHashMap::default, |mut accumulator_map, local_map| {
                for (cell_id, population) in local_map {
                    *accumulator_map.entry(cell_id).or_insert(0) += population;
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
                *cell_populations.entry(parent_id).or_insert(0) += population;
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
    pub fn query(&self, s2_cell_id: u64) -> u64 {
        debug_assert!(s2_cell_id != 0, "S2 cell ID cannot be 0");
        let query_level = get_level(s2_cell_id);
        let query_cell_id = if query_level > MAX_DB_LEVEL {
            get_ancestor(s2_cell_id, MAX_DB_LEVEL)
        } else {
            s2_cell_id
        };
        let compact_query_id = ((query_cell_id >> SHIFT_COMPACT) & 0xFFFFFFFF) as u32;

        // Perform standard binary search on the flat sorted leaves array.
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

/// Verifies 100% query parity for all 100,663,296 Level 12 cell IDs globally.
///
/// This test runs in parallel across all 6 S2 faces using Rayon, verifying that
/// the optimized QueryEngine query matches the NaiveOracle query perfectly.
#[test]
fn test_exhaustive_global_l12_parity() {
    const S2_FACE_SHIFT: u32 = 61;
    const STEP_SHIFT: u32 = 37;

    let database_path = "../population_density_database.bin";
    assert!(
        std::path::Path::new(database_path).exists(),
        "Database file '{}' not found. Please build the database first using 'cargo run --release --bin build_database'.",
        database_path
    );

    let query_engine = QueryEngine::new(database_path).unwrap();
    let naive_oracle = &*SHARED_TIFF_ORACLE;

    println!(
        "Starting exhaustive global Level 12 parity test over 100,663,296 cells in parallel..."
    );
    let start_time = std::time::Instant::now();

    let faces: Vec<usize> = (0..6).collect();
    let mismatches: usize = faces
        .into_par_iter()
        .map(|face| {
            let mut face_mismatches = 0;
            let start_cell = (face as u64) << S2_FACE_SHIFT | (1u64 << SHIFT_COMPACT);
            let end_cell = ((face + 1) as u64) << S2_FACE_SHIFT;
            let step = 1u64 << STEP_SHIFT;

            let mut s2_cell_id = start_cell;
            while s2_cell_id < end_cell {
                let optimized_result = query_engine.query(s2_cell_id).unwrap();
                let naive_result = naive_oracle.query(s2_cell_id);

                if optimized_result != naive_result {
                    face_mismatches += 1;
                    if face_mismatches <= 5 {
                        println!(
                            "    [Mismatch] Face {}: Cell {:016x}: Optimized={:016x}, Naive={:016x}",
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
    println!("Exhaustive parity test completed in {:.2?}.", elapsed);
    assert_eq!(
        mismatches, 0,
        "Exhaustive parity test failed with {} mismatches!",
        mismatches
    );
    println!("SUCCESS: Verified all 100,663,296 Level 12 cells with exactly 0 mismatches.");
}

/// Verifies that a database with an invalid/oversized block_size fails to load.
#[test]
fn test_invalid_block_size() {
    let temporary_database = "scratch/temp_invalid_block_size.db";
    std::fs::create_dir_all("scratch").unwrap();
    {
        let mut file = File::create(temporary_database).unwrap();
        // S2PP magic bytes
        file.write_all(b"S2PP").unwrap();
        // count = 0 (4 bytes)
        file.write_all(&0u32.to_le_bytes()).unwrap();
        // block_size = 257 (4 bytes), which is > MAX_DELTAS_CAPACITY + 1
        file.write_all(&257u32.to_le_bytes()).unwrap();
        // block_count = 0 (4 bytes)
        file.write_all(&0u32.to_le_bytes()).unwrap();
    }
    let result = QueryEngine::new(temporary_database);
    assert!(result.is_err());
    let error_message = result.err().unwrap().to_string();
    assert!(
        error_message.contains("invalid block size in database: 257"),
        "Unexpected error message: {}",
        error_message
    );
    std::fs::remove_file(temporary_database).unwrap();
}

/// Verifies that standard `assert!` statements inside public helpers panic under invalid levels or cell IDs.
#[test]
fn test_assert_preconditions_panic() {
    // Assert 0 cell ID panics in get_level.
    let result = std::panic::catch_unwind(|| {
        get_level(0);
    });
    assert!(result.is_err());

    // Assert level > 30 panics in get_ancestor.
    let result = std::panic::catch_unwind(|| {
        get_ancestor(1, 31);
    });
    assert!(result.is_err());
}

/// Verifies that a database block with bit_width > 28 is detected as corrupt during query and reconstruction.
#[test]
fn test_corrupt_database_bit_width_limit() {
    let temporary_database = "scratch/temp_corrupt_bit_width.db";
    std::fs::create_dir_all("scratch").unwrap();
    {
        let mut file = File::create(temporary_database).unwrap();
        // S2PP magic bytes
        file.write_all(b"S2PP").unwrap();
        // count = 2 (4 bytes)
        file.write_all(&2u32.to_le_bytes()).unwrap();
        // block_size = 256 (4 bytes)
        file.write_all(&256u32.to_le_bytes()).unwrap();
        // block_count = 1 (4 bytes)
        file.write_all(&1u32.to_le_bytes()).unwrap();
        // headers = [0u32] (4 bytes)
        file.write_all(&0u32.to_le_bytes()).unwrap();
        // absolute_offsets = [0u32] (4 bytes)
        file.write_all(&0u32.to_le_bytes()).unwrap();
        // relative_offsets = [0u16] (2 bytes)
        file.write_all(&0u16.to_le_bytes()).unwrap();
        // block bytes: bit_width of subblock 0 is 29, exception_count = 0.
        // 12-byte header with bit_widths[0] = 29.
        let mut header = [0u8; BLOCK_HEADER_SIZE];
        header[0] = 29;
        file.write_all(&header).unwrap();
    }

    let engine = QueryEngine::new(temporary_database).unwrap();

    // Querying should fail.
    let query_result = engine.query(1);
    assert!(query_result.is_err());
    assert!(
        query_result
            .err()
            .unwrap()
            .to_string()
            .contains("bit_width 29 exceeds maximum limit 28")
    );

    // Reconstruction should fail.
    let reconstruct_result = engine.reconstruct_all_leaves();
    assert!(reconstruct_result.is_err());
    assert!(
        reconstruct_result
            .err()
            .unwrap()
            .to_string()
            .contains("bit_width 29 exceeds maximum limit 28")
    );

    std::fs::remove_file(temporary_database).unwrap();
}

/// Verifies that a database block with corrupt/out-of-bounds exception indexing is caught during query and reconstruction.
#[test]
fn test_corrupt_database_exception_bounds() {
    let temporary_database = "scratch/temp_corrupt_exception.db";
    std::fs::create_dir_all("scratch").unwrap();
    {
        let mut file = File::create(temporary_database).unwrap();
        // S2PP magic bytes
        file.write_all(b"S2PP").unwrap();
        // count = 2 (4 bytes)
        file.write_all(&2u32.to_le_bytes()).unwrap();
        // block_size = 256 (4 bytes)
        file.write_all(&256u32.to_le_bytes()).unwrap();
        // block_count = 1 (4 bytes)
        file.write_all(&1u32.to_le_bytes()).unwrap();
        // headers = [0u32] (4 bytes)
        file.write_all(&0u32.to_le_bytes()).unwrap();
        // absolute_offsets = [0u32] (4 bytes)
        file.write_all(&0u32.to_le_bytes()).unwrap();
        // relative_offsets = [0u16] (2 bytes)
        file.write_all(&0u16.to_le_bytes()).unwrap();
        // block bytes: bit_width = 8, exception_count = 1, mode = U8 (1).
        // 12-byte header: bit_widths[0] = 8, exception_count = 1, flags = 1 (EXCEPTION_MODE_U8).
        let mut header = [0u8; BLOCK_HEADER_SIZE];
        header[0] = 8;
        header[BLOCK_HEADER_SIZE - 2] = 1;
        header[BLOCK_HEADER_SIZE - 1] = 1;
        file.write_all(&header).unwrap();
        // Also write 1 dummy byte representing the packed delta.
        // We omit the actual exception index and exception value bytes so the bounds check fails.
        file.write_all(&[0]).unwrap();
    }

    let engine = QueryEngine::new(temporary_database).unwrap();

    // Querying should fail.
    let query_result = engine.query(1);
    assert!(query_result.is_err());
    assert!(
        query_result
            .err()
            .unwrap()
            .to_string()
            .contains("truncated exception indices")
    );

    // Reconstruction should fail.
    let reconstruct_result = engine.reconstruct_all_leaves();
    assert!(reconstruct_result.is_err());
    assert!(
        reconstruct_result
            .err()
            .unwrap()
            .to_string()
            .contains("truncated exception indices")
    );

    std::fs::remove_file(temporary_database).unwrap();
}

/// Verifies that a database block with a corrupt/out-of-range exception index is caught during query and reconstruction.
#[test]
fn test_corrupt_database_exception_index_out_of_range() {
    let temporary_database = "scratch/temp_corrupt_exception_out_of_range.db";
    std::fs::create_dir_all("scratch").unwrap();
    {
        let mut file = File::create(temporary_database).unwrap();
        // S2PP magic bytes
        file.write_all(b"S2PP").unwrap();
        // count = 2 (4 bytes)
        file.write_all(&2u32.to_le_bytes()).unwrap();
        // block_size = 256 (4 bytes)
        file.write_all(&256u32.to_le_bytes()).unwrap();
        // block_count = 1 (4 bytes)
        file.write_all(&1u32.to_le_bytes()).unwrap();
        // headers = [0u32] (4 bytes)
        file.write_all(&0u32.to_le_bytes()).unwrap();
        // absolute_offsets = [0u32] (4 bytes)
        file.write_all(&0u32.to_le_bytes()).unwrap();
        // relative_offsets = [0u16] (2 bytes)
        file.write_all(&0u16.to_le_bytes()).unwrap();
        // block bytes.
        // Header (12 bytes): bit_widths[0] = 8, exception_count = 1, flags = 1 (EXCEPTION_MODE_U8).
        let mut header = [0u8; BLOCK_HEADER_SIZE];
        header[0] = 8;
        header[BLOCK_HEADER_SIZE - 2] = 1;
        header[BLOCK_HEADER_SIZE - 1] = 1;
        file.write_all(&header).unwrap();
        // 0: 1 byte of packed deltas.
        // 1: 1 byte of exception index (since delta_count = 1, index 1 is out of range 0..1).
        // 100: 1 byte exception value.
        file.write_all(&[0, 1, 100]).unwrap();
    }

    let engine = QueryEngine::new(temporary_database).unwrap();

    // Querying should fail.
    let query_result = engine.query(1);
    assert!(query_result.is_err());
    let error_message = query_result.err().unwrap().to_string();
    assert!(
        error_message.contains("corrupt database block exception index 1 out of range"),
        "Unexpected error message: {}",
        error_message
    );

    // Reconstruction should fail.
    let reconstruct_result = engine.reconstruct_all_leaves();
    assert!(reconstruct_result.is_err());
    let error_message = reconstruct_result.err().unwrap().to_string();
    assert!(
        error_message.contains("corrupt database block exception index 1 out of range"),
        "Unexpected error message: {}",
        error_message
    );

    std::fs::remove_file(temporary_database).unwrap();
}

/// Verifies that loading a database with a block size of zero returns a validation error.
#[test]
fn test_zero_block_size() {
    let temporary_database = "scratch/temp_zero_block_size.db";
    std::fs::create_dir_all("scratch").unwrap();
    {
        let mut file = File::create(temporary_database).unwrap();
        // S2PP magic bytes
        file.write_all(b"S2PP").unwrap();
        // count = 0 (4 bytes)
        file.write_all(&0u32.to_le_bytes()).unwrap();
        // block_size = 0 (4 bytes)
        file.write_all(&0u32.to_le_bytes()).unwrap();
        // block_count = 0 (4 bytes)
        file.write_all(&0u32.to_le_bytes()).unwrap();
    }

    let query_engine_result = QueryEngine::new(temporary_database);
    assert!(query_engine_result.is_err());
    let error_message = query_engine_result.err().unwrap().to_string();
    assert!(
        error_message.contains("invalid database: block_size must be greater than 0"),
        "Unexpected error message: {}",
        error_message
    );
    std::fs::remove_file(temporary_database).unwrap();
}

/// Verifies that loading a database with zero blocks returns a validation error.
#[test]
fn test_zero_block_count() {
    let temporary_database = "scratch/temp_zero_block_count.db";
    std::fs::create_dir_all("scratch").unwrap();
    {
        let mut file = File::create(temporary_database).unwrap();
        // S2PP magic bytes
        file.write_all(b"S2PP").unwrap();
        // count = 0 (4 bytes)
        file.write_all(&0u32.to_le_bytes()).unwrap();
        // block_size = 256 (4 bytes)
        file.write_all(&256u32.to_le_bytes()).unwrap();
        // block_count = 0 (4 bytes)
        file.write_all(&0u32.to_le_bytes()).unwrap();
    }

    let query_engine_result = QueryEngine::new(temporary_database);
    assert!(query_engine_result.is_err());
    let error_message = query_engine_result.err().unwrap().to_string();
    assert!(
        error_message.contains("invalid database: block_count must be greater than 0"),
        "Unexpected error message: {}",
        error_message
    );
    std::fs::remove_file(temporary_database).unwrap();
}

/// Verifies query parity for finer-level S2 cell ID queries where bit 36 is 0.
#[test]
fn test_query_finer_levels_regression() {
    let database_path = "../population_density_database.bin";
    assert!(
        std::path::Path::new(database_path).exists(),
        "Database file '{}' not found. Please build the database first using 'cargo run --release --bin build_database'.",
        database_path
    );
    let query_engine = QueryEngine::new(database_path).unwrap();
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
        "Vulnerability 1: get_level must panic on invalid cell ID"
    );
}

/// Verifies that loading a database with mathematically inconsistent headers returns an error.
#[test]
fn test_adversarial_inconsistent_header_underflow() {
    let temporary_database = "scratch/temp_inconsistent_header.db";
    std::fs::create_dir_all("scratch").unwrap();
    {
        let mut file = std::fs::File::create(temporary_database).unwrap();
        file.write_all(b"S2PP").unwrap();
        // Write count of 1.
        file.write_all(&1u32.to_le_bytes()).unwrap();
        // Write block_size of 256.
        file.write_all(&256u32.to_le_bytes()).unwrap();
        // Write block_count of 2.
        file.write_all(&2u32.to_le_bytes()).unwrap();
        // Write headers.
        file.write_all(&[0u32.to_le_bytes(), 0u32.to_le_bytes()].concat())
            .unwrap();
        // Write absolute offsets.
        file.write_all(&[0u32.to_le_bytes()].concat()).unwrap();
        // Write relative offsets.
        file.write_all(&[0u16.to_le_bytes(), 0u16.to_le_bytes()].concat())
            .unwrap();
        // Write block data.
        file.write_all(&[0u8, 0u8]).unwrap();
    }

    let result = QueryEngine::new(temporary_database);
    assert!(
        result.is_err(),
        "Vulnerability 2: QueryEngine::new must reject mathematically inconsistent headers"
    );
    let _ = std::fs::remove_file(temporary_database);
}

/// Verifies that querying with a cell ID of 0 returns a graceful error instead of panicking.
#[test]
fn test_query_zero_cell_id() {
    let database_path = "../population_density_database.bin";
    assert!(
        std::path::Path::new(database_path).exists(),
        "Database file '{}' not found. Please build the database first using 'cargo run --release --bin build_database'.",
        database_path
    );

    let query_engine = QueryEngine::new(database_path).unwrap();
    let result = query_engine.query(0);
    assert!(result.is_err());
    let error_message = result.err().unwrap().to_string();
    assert!(error_message.contains("S2 cell ID cannot be 0"));
}

/// Verifies that compact S2 cell ID boundary checks fail if database contains out-of-bounds compact IDs.
#[test]
fn test_corrupt_database_max_compact_limit() {
    let temporary_database = "scratch/temp_max_compact_limit.db";
    std::fs::create_dir_all("scratch").unwrap();
    {
        let mut file = std::fs::File::create(temporary_database).unwrap();
        // Write standard S2PP magic bytes.
        file.write_all(b"S2PP").unwrap();
        // Write count of 2.
        file.write_all(&2u32.to_le_bytes()).unwrap();
        // Write block_size of 256.
        file.write_all(&256u32.to_le_bytes()).unwrap();
        // Write block_count of 1.
        file.write_all(&1u32.to_le_bytes()).unwrap();
        // Write an out-of-bounds header ID of 0x10000000.
        file.write_all(&0x10000000u32.to_le_bytes()).unwrap();
        // Write absolute offset of 0.
        file.write_all(&0u32.to_le_bytes()).unwrap();
        // Write relative offset of 0.
        file.write_all(&0u16.to_le_bytes()).unwrap();
        // Write block bytes: bit_width = 8, exception_count = 0.
        file.write_all(&[8, 0]).unwrap();
    }

    let result = QueryEngine::new(temporary_database);
    assert!(result.is_err());
    let error_message = result.err().unwrap().to_string();
    assert!(
        error_message.contains("header cell ID 10000000 exceeds maximum compact S2 cell ID limit")
    );
    let _ = std::fs::remove_file(temporary_database);
}

/// Verifies resolved S2 levels across high-fidelity geographic transition scenarios.
#[test]
fn test_geographic_transition_scenarios() {
    let database_path = "../population_density_database.bin";
    assert!(
        std::path::Path::new(database_path).exists(),
        "Database file '{}' not found. Please build the database first using 'cargo run --release --bin build_database'.",
        database_path
    );

    let query_engine = QueryEngine::new(database_path).unwrap();
    let naive_oracle = &*SHARED_TIFF_ORACLE;

    let verify_query = |lat: f64, lng: f64, desc: &str, expected_level_check: fn(u32) -> bool| {
        let point = LatLng::from_degrees(lat, lng);
        let query_cell_id = CellID::from(point).0;

        let optimized_result = query_engine.query(query_cell_id).unwrap();
        let naive_result = naive_oracle.query(query_cell_id);

        assert_eq!(
            optimized_result, naive_result,
            "Query mismatch at {} ({}, {}): optimized = {:016x}, naive = {:016x}",
            desc, lat, lng, optimized_result, naive_result
        );

        let level = get_level(optimized_result);
        assert!(
            expected_level_check(level),
            "Level expectation failed at {} ({}, {}): resolved to level {}, which did not satisfy the constraint",
            desc,
            lat,
            lng,
            level
        );
    };

    // 1. Beijing Radius Scenario
    verify_query(39.9042, 116.4074, "Beijing Center", |lvl| lvl == 12);
    verify_query(39.9132, 116.4074, "Beijing 1km N", |lvl| lvl == 12);
    verify_query(39.9042, 116.4191, "Beijing 1km E", |lvl| lvl == 12);
    verify_query(39.8592, 116.4074, "Beijing 5km S", |lvl| lvl == 12);
    verify_query(39.9042, 116.2904, "Beijing 10km W", |lvl| lvl == 12);
    verify_query(
        40.3542,
        116.4074,
        "Huairou Mountains (Beijing 50km N)",
        |lvl| lvl <= 10,
    );

    // 2. Everest Peaks-vs-Valleys Scenario
    verify_query(27.7172, 85.3240, "Kathmandu Valley", |lvl| lvl == 12);
    verify_query(29.6524, 91.1172, "Lhasa", |lvl| lvl == 12);
    verify_query(27.9881, 86.9250, "Mount Everest Peak", |lvl| lvl == 10);

    // 3. Shanghai Coastline Scenario
    verify_query(31.19, 121.40, "Shanghai Inland", |lvl| lvl == 12);
    verify_query(31.19, 121.70, "Pudong Coastal Land", |lvl| lvl == 12);
    verify_query(31.19, 121.80, "Shanghai Coastline Water Edge", |lvl| {
        lvl == 11
    });
    verify_query(31.19, 122.20, "Near-Shore Ocean", |lvl| {
        lvl == 7 || lvl == 6
    });
    verify_query(31.19, 123.50, "Open Sea", |lvl| {
        lvl == 5 || lvl == 3 || lvl == 0
    });

    // 4. Unpopulated Islands Scenario
    verify_query(-24.3797, -128.3242, "Henderson Island", |lvl| {
        lvl == 3 || lvl == 0
    });
    verify_query(10.2983, -109.2192, "Clipperton Island", |lvl| {
        lvl == 3 || lvl == 0
    });
    verify_query(-49.3500, 69.3500, "Kerguelen Islands", |lvl| {
        lvl == 3 || lvl == 0
    });
    verify_query(-54.4208, 3.3464, "Bouvet Island", |lvl| {
        lvl == 3 || lvl == 0
    });
}

/// Verifies that loading a database with a non-zero first block offset fails.
#[test]
fn test_corrupt_database_non_zero_first_offset() {
    let temporary_database = "scratch/temp_non_zero_first_offset.db";
    std::fs::create_dir_all("scratch").unwrap();
    {
        let mut file = std::fs::File::create(temporary_database).unwrap();
        file.write_all(b"S2PP").unwrap();
        // Write count of 2.
        file.write_all(&2u32.to_le_bytes()).unwrap();
        // Write block_size of 256.
        file.write_all(&256u32.to_le_bytes()).unwrap();
        // Write block_count of 1.
        file.write_all(&1u32.to_le_bytes()).unwrap();
        // Write header S2 cell ID.
        file.write_all(&0u32.to_le_bytes()).unwrap();
        // Write non-zero first absolute offset.
        file.write_all(&1u32.to_le_bytes()).unwrap();
        // Write relative offset.
        file.write_all(&0u16.to_le_bytes()).unwrap();
        // Write block bytes: bit_width = 8, exception_count = 0.
        file.write_all(&[8, 0]).unwrap();
    }

    let result = QueryEngine::new(temporary_database);
    assert!(result.is_err());
    let error_message = result.err().unwrap().to_string();
    assert!(
        error_message.contains("first block offset must be 0"),
        "Unexpected error: {}",
        error_message
    );
    let _ = std::fs::remove_file(temporary_database);
}

/// Verifies that loading a database with non-monotonic block offsets fails.
#[test]
fn test_corrupt_database_offsets_not_monotonic() {
    let temporary_database = "scratch/temp_offsets_not_monotonic.db";
    std::fs::create_dir_all("scratch").unwrap();
    {
        let mut file = std::fs::File::create(temporary_database).unwrap();
        file.write_all(b"S2PP").unwrap();
        // Write count of 257.
        file.write_all(&257u32.to_le_bytes()).unwrap();
        // Write block_size of 256.
        file.write_all(&256u32.to_le_bytes()).unwrap();
        // Write block_count of 2.
        file.write_all(&2u32.to_le_bytes()).unwrap();
        // Write header cell IDs.
        file.write_all(&0u32.to_le_bytes()).unwrap();
        file.write_all(&1u32.to_le_bytes()).unwrap();
        // Write absolute offsets.
        file.write_all(&0u32.to_le_bytes()).unwrap();
        // Write non-monotonic relative offsets.
        file.write_all(&0u16.to_le_bytes()).unwrap();
        file.write_all(&0u16.to_le_bytes()).unwrap();
        // Write block bytes.
        file.write_all(&[8, 0, 0, 0]).unwrap();
    }

    let result = QueryEngine::new(temporary_database);
    assert!(result.is_err());
    let error_message = result.err().unwrap().to_string();
    assert!(
        error_message.contains("block offsets must be strictly monotonic"),
        "Unexpected error: {}",
        error_message
    );
    let _ = std::fs::remove_file(temporary_database);
}

/// Verifies that loading a database with out-of-bounds block offsets fails.
#[test]
fn test_corrupt_database_offset_out_of_bounds() {
    let temporary_database = "scratch/temp_offset_out_of_bounds.db";
    std::fs::create_dir_all("scratch").unwrap();
    {
        let mut file = std::fs::File::create(temporary_database).unwrap();
        file.write_all(b"S2PP").unwrap();
        // Write count of 257.
        file.write_all(&257u32.to_le_bytes()).unwrap();
        // Write block_size of 256.
        file.write_all(&256u32.to_le_bytes()).unwrap();
        // Write block_count of 2.
        file.write_all(&2u32.to_le_bytes()).unwrap();
        // Write header cell IDs.
        file.write_all(&0u32.to_le_bytes()).unwrap();
        file.write_all(&1u32.to_le_bytes()).unwrap();
        // Write absolute offsets.
        file.write_all(&0u32.to_le_bytes()).unwrap();
        // Write out-of-bounds relative offset.
        file.write_all(&0u16.to_le_bytes()).unwrap();
        file.write_all(&9999u16.to_le_bytes()).unwrap();
        // Write block bytes.
        file.write_all(&[8, 0]).unwrap();
    }

    let result = QueryEngine::new(temporary_database);
    assert!(result.is_err());
    let error_message = result.err().unwrap().to_string();
    assert!(
        error_message.contains("exceeds physical file bounds"),
        "Unexpected error: {}",
        error_message
    );
    let _ = std::fs::remove_file(temporary_database);
}

/// Verifies that terminal block boundary checks detect a size mismatch when the last block has exactly 1 cell.
#[test]
fn test_corrupt_database_terminal_single_mismatch() {
    let temporary_database = "scratch/temp_terminal_single_mismatch.db";
    std::fs::create_dir_all("scratch").unwrap();
    {
        let mut file = std::fs::File::create(temporary_database).unwrap();
        file.write_all(b"S2PP").unwrap();
        // Write count of 257.
        file.write_all(&257u32.to_le_bytes()).unwrap();
        // Write block_size of 256.
        file.write_all(&256u32.to_le_bytes()).unwrap();
        // Write block_count of 2.
        file.write_all(&2u32.to_le_bytes()).unwrap();
        // Write header cell IDs.
        file.write_all(&0u32.to_le_bytes()).unwrap();
        file.write_all(&1u32.to_le_bytes()).unwrap();
        // Write absolute offsets.
        file.write_all(&0u32.to_le_bytes()).unwrap();
        // Write relative offsets.
        file.write_all(&0u16.to_le_bytes()).unwrap();
        file.write_all(&1u16.to_le_bytes()).unwrap();
        // Write block bytes.
        file.write_all(&[8, 0]).unwrap();
    }

    let result = QueryEngine::new(temporary_database);
    assert!(result.is_err());
    let error_message = result.err().unwrap().to_string();
    assert!(
        error_message.contains("terminal block size mismatch"),
        "Unexpected error: {}",
        error_message
    );
    let _ = std::fs::remove_file(temporary_database);
}

/// Verifies that terminal block boundary checks detect a size mismatch when the last block has more than 1 cell.
#[test]
fn test_corrupt_database_terminal_multi_mismatch() {
    let temporary_database = "scratch/temp_terminal_multi_mismatch.db";
    std::fs::create_dir_all("scratch").unwrap();
    {
        let mut file = std::fs::File::create(temporary_database).unwrap();
        file.write_all(b"S2PP").unwrap();
        // Write count of 258.
        file.write_all(&258u32.to_le_bytes()).unwrap();
        // Write block_size of 256.
        file.write_all(&256u32.to_le_bytes()).unwrap();
        // Write block_count of 2.
        file.write_all(&2u32.to_le_bytes()).unwrap();
        // Write header cell IDs.
        file.write_all(&0u32.to_le_bytes()).unwrap();
        file.write_all(&1u32.to_le_bytes()).unwrap();
        // Write absolute offsets.
        file.write_all(&0u32.to_le_bytes()).unwrap();
        // Write relative offsets.
        file.write_all(&0u16.to_le_bytes()).unwrap();
        file.write_all(&2u16.to_le_bytes()).unwrap();
        // Write block bytes.
        file.write_all(&[8, 0]).unwrap();
    }

    let result = QueryEngine::new(temporary_database);
    assert!(result.is_err());
    let error_message = result.err().unwrap().to_string();
    assert!(
        error_message.contains("terminal block offset")
            && error_message.contains("must be less than physical data length"),
        "Unexpected error: {}",
        error_message
    );
    let _ = std::fs::remove_file(temporary_database);
}

/// Verifies that get_ancestor and try_get_ancestor correctly reject levels deeper than the cell's own level.
#[test]
fn test_ancestor_level_boundaries() {
    use population_density::try_get_ancestor;

    // Use a known Level 17 S2 cell ID.
    let cell_id = 0x1000000004000000u64;
    assert_eq!(get_level(cell_id), 17);

    // level <= 17 should succeed.
    assert!(try_get_ancestor(cell_id, 17).is_ok());
    assert!(try_get_ancestor(cell_id, 10).is_ok());

    // level > 17 should fail/panic.
    assert!(try_get_ancestor(cell_id, 18).is_err());
    assert!(try_get_ancestor(cell_id, 30).is_err());

    let result = std::panic::catch_unwind(|| {
        get_ancestor(cell_id, 18);
    });
    assert!(result.is_err());
}

/// Independently verifies the core privacy invariant: for every populated query location, the cell
/// the engine returns represents at least POPULATION_THRESHOLD people.
///
/// This guard sums population from the propagated TIFF-derived cell map (NaiveOracle::populations),
/// NOT from the quadtree leaf set, so a pruning/threshold regression in find_leaves that returned a
/// too-fine cell would be caught here even though it would be mirrored in the parity oracles.
#[test]
fn test_returned_cells_meet_population_threshold() {
    let database_path = "../population_density_database.bin";
    assert!(
        std::path::Path::new(database_path).exists(),
        "Database file '{}' not found. Please build the database first using 'cargo run --release --bin build_database'.",
        database_path
    );
    let query_engine = QueryEngine::new(database_path).unwrap();
    let oracle = &*SHARED_TIFF_ORACLE;

    // Representative cities (resolve fine) plus a deterministic global grid covering dense, sparse,
    // and ocean locations (resolve coarse). Every case must still represent >= the threshold.
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

/// Verifies that a memory mapping whose base is not 4-byte aligned (the case a non-4-aligned APK
/// asset offset would produce) is rejected by from_mmap, rather than causing UB or a wrong result.
/// The u32/u16 table casts require a 4-aligned base, so this must fail closed.
#[test]
fn test_from_mmap_unaligned_base_rejected() {
    use population_density::memmap2::MmapOptions;

    let database_path = "../population_density_database.bin";
    assert!(
        std::path::Path::new(database_path).exists(),
        "Database file '{}' not found.",
        database_path
    );
    let database_bytes = std::fs::read(database_path).unwrap();

    // Prepend one pad byte so the database content starts at file offset 1.
    std::fs::create_dir_all("scratch").unwrap();
    let padded_path = "scratch/unaligned_padded.bin";
    let mut padded = Vec::with_capacity(database_bytes.len() + 1);
    padded.push(0u8);
    padded.extend_from_slice(&database_bytes);
    std::fs::write(padded_path, &padded).unwrap();

    let file = File::open(padded_path).unwrap();
    // memmap2 anchors the mapping at the requested offset, so the base mirrors offset % 4 = 1.
    let mmap = unsafe {
        MmapOptions::new()
            .offset(1)
            .len(database_bytes.len())
            .map(&file)
            .unwrap()
    };
    assert_eq!(
        mmap.as_ptr() as usize % 4,
        1,
        "test setup expects an unaligned mapping base"
    );

    let result = QueryEngine::from_mmap(mmap);
    assert!(
        result.is_err(),
        "from_mmap must reject a non-4-aligned mapping base (fail-closed), got Ok"
    );

    let _ = std::fs::remove_file(padded_path);
}

/// Verifies from_mmap rejects a corrupt in-memory mapping directly. QueryEngine::new routes through
/// from_mmap from files; this exercises the from_mmap buffer entry point on its own.
#[test]
fn test_from_mmap_bad_magic_rejected() {
    use population_density::memmap2::MmapOptions;

    std::fs::create_dir_all("scratch").unwrap();
    let corrupt_path = "scratch/from_mmap_bad_magic.bin";
    std::fs::write(corrupt_path, vec![0xFFu8; 64]).unwrap();

    let file = File::open(corrupt_path).unwrap();
    let mmap = unsafe { MmapOptions::new().map(&file).unwrap() };
    let result = QueryEngine::from_mmap(mmap);
    assert!(
        result.is_err(),
        "from_mmap must reject a buffer with invalid magic bytes"
    );

    let _ = std::fs::remove_file(corrupt_path);
}
