//! Provides comprehensive adversarial tests for the S2 population density database engine.
//!
//! Verifies robust error handling and corruption resistance under extremely malformed database structures,
//! boundary-condition S2 cell IDs, invalid float values inside TIFF inputs, and builder fast-fail constraints.

#![allow(clippy::needless_range_loop)]

use population_density::{
    MAX_COMPACT_CELL_ID, MAX_S2_LEVEL, QueryEngine, SHIFT_COMPACT, get_children_ids,
    try_get_ancestor, try_get_level,
};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use tiff::encoder::{TiffEncoder, colortype};

/// Manages automatic cleanup of temporary test databases.
struct TempDatabase {
    path: PathBuf,
}

impl TempDatabase {
    /// Creates a new temporary database at the specified path.
    fn new(name: &str) -> Self {
        std::fs::create_dir_all("scratch/adversarial").unwrap();
        let path = PathBuf::from(format!("scratch/adversarial/{}", name));
        Self { path }
    }

    /// Writes custom database contents to disk for corruption tests.
    #[allow(clippy::too_many_arguments)]
    fn write(
        &self,
        magic: &[u8; 4],
        count: u32,
        block_size: u32,
        block_count: u32,
        headers: &[u32],
        absolute_offsets: &[u32],
        relative_offsets: &[u16],
        block_data: &[u8],
    ) {
        let mut file = File::create(&self.path).unwrap();
        file.write_all(magic).unwrap();
        file.write_all(&count.to_le_bytes()).unwrap();
        file.write_all(&block_size.to_le_bytes()).unwrap();
        file.write_all(&block_count.to_le_bytes()).unwrap();
        for &header in headers {
            file.write_all(&header.to_le_bytes()).unwrap();
        }
        for &absolute_offset in absolute_offsets {
            file.write_all(&absolute_offset.to_le_bytes()).unwrap();
        }
        for &relative_offset in relative_offsets {
            file.write_all(&relative_offset.to_le_bytes()).unwrap();
        }
        file.write_all(block_data).unwrap();
    }
}

impl Drop for TempDatabase {
    /// Deletes the temporary file upon drop.
    fn drop(&mut self) {
        if self.path.exists() {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// Writes a mock GeoTIFF image with georeferencing tags matching the builder's assumed grid.
fn write_mock_tiff(path: &Path, width: u32, height: u32, data: &[f32]) {
    let file = File::create(path).unwrap();
    let mut encoder = TiffEncoder::new(file).unwrap();
    let mut image = encoder
        .new_image::<colortype::Gray32Float>(width, height)
        .unwrap();
    // Origin (longitude -180, latitude 84) at 1/120 degrees per pixel, matching the builder's
    // GEOTIFF_PIXEL_SCALE / GEOTIFF_MIN_LONGITUDE / GEOTIFF_MAX_LATITUDE constants so the build
    // georeferencing validation accepts these synthetic files.
    image
        .encoder()
        .write_tag(
            tiff::tags::Tag::ModelPixelScaleTag,
            &[0.008333333333333333f64, 0.008333333333333333, 0.0][..],
        )
        .unwrap();
    image
        .encoder()
        .write_tag(
            tiff::tags::Tag::ModelTiepointTag,
            &[0.0f64, 0.0, 0.0, -180.0, 84.0, 0.0][..],
        )
        .unwrap();
    image.write_data(data).unwrap();
}

/// Verifies that the query engine detects invalid/decreasing offset tables.
#[test]
fn test_corrupt_db_invalid_offset_tables() {
    let temp = TempDatabase::new("invalid_offsets.db");
    // Write a database with decreasing relative offsets: block 0 relative offset is 10, block 1 relative offset is 5.
    // Total block data size is 4 bytes.
    temp.write(
        b"S2PP",
        512,
        256,
        2,
        &[1, 100],
        &[0],
        &[10, 5],
        &[8, 0, 8, 0],
    );

    let engine_result = QueryEngine::new(&temp.path);
    assert!(engine_result.is_err());
    let error_message = engine_result.unwrap_err().to_string();
    assert!(
        error_message.contains("monotonic") || error_message.contains("bounds"),
        "Unexpected error: {}",
        error_message
    );
}

/// Verifies that the query engine detects corrupted delta blocks causing delta value overflow.
#[test]
fn test_corrupt_db_corrupted_delta_blocks() {
    let temp = TempDatabase::new("corrupted_delta.db");
    // Write a database block where the exception value is extremely large, causing delta_value.checked_add(1) to overflow.
    // count = 2, block_size = 256, block_count = 1.
    // Under the sub-block PFOR layout:
    // Header (12 bytes): bit_widths[0]=8 (first 5 bits of header are 8), exception_count=1, exception_mode=3, index_bitmask_flag=0.
    // Primary bitstream (1 byte) = 0xFF.
    // Exception index (1 byte) = 0.
    // Exception value (4 bytes) = 0x00FFFFFE.
    // delta_value = (((0x00FFFFFE + 1) << 8) as u32) | 0xFF = 0xFFFFFFFF.
    // delta = delta_value + 1 = 0x100000000 (overflows u32).
    let block_bytes = [
        8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3,    // Header
        0xFF, // Primary bitstream
        0,    // Exception index
        0xFE, 0xFF, 0xFF, 0x00, // Exception value
    ];
    temp.write(b"S2PP", 2, 256, 1, &[1], &[0], &[0], &block_bytes);

    let engine = QueryEngine::new(&temp.path).unwrap();
    // Querying should fail during reconstruction due to delta value overflow.
    let result = engine.query(1u64 << SHIFT_COMPACT);
    assert!(result.is_err());
    let error_message = result.unwrap_err().to_string();
    assert!(
        error_message.contains("delta value overflow"),
        "Unexpected error: {}",
        error_message
    );
}

/// Verifies that compact S2 cell ID reconstruction overflow is caught safely.
#[test]
fn test_corrupt_db_compact_s2_cell_id_overflow() {
    let temp = TempDatabase::new("compact_id_overflow.db");
    // Write a database block where reconstruction causes the compact S2 cell ID to overflow u32.
    // header = MAX_COMPACT_CELL_ID (0x0FFFFFFF).
    // Under the sub-block PFOR layout:
    // Header (12 bytes): bit_widths[0]=8 (first 5 bits are 8), exception_count=1, exception_mode=3, index_bitmask_flag=0.
    // Primary bitstream (1 byte) = 0x00.
    // Exception index (1 byte) = 0.
    // Exception value (4 bytes) = 0x00EFFFFF.
    // delta_value = (((0x00EFFFFF + 1) << 8) as u32) | 0x00 = 0xF0000000.
    // delta = delta_value + 1 = 0xF0000001.
    // current_value = 0x0FFFFFFF + 0xF0000001 = 0x100000000 (overflows u32).
    let block_bytes = [
        8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3,    // Header
        0x00, // Primary bitstream
        0,    // Exception index
        0xFF, 0xFF, 0xEF, 0x00, // Exception value
    ];
    temp.write(
        b"S2PP",
        2,
        256,
        1,
        &[MAX_COMPACT_CELL_ID],
        &[0],
        &[0],
        &block_bytes,
    );

    let engine = QueryEngine::new(&temp.path).unwrap();
    // A compact header of MAX_COMPACT_CELL_ID maps to S2 face 7, which query() now correctly
    // rejects as an invalid cell before decoding, so the overflow math is exercised through full
    // reconstruction instead.
    let result = engine.reconstruct_all_leaves();
    assert!(result.is_err());
    let error_message = result.unwrap_err().to_string();
    assert!(
        error_message.contains("compact S2 cell ID overflow"),
        "Unexpected error: {}",
        error_message
    );
}

/// Verifies that structurally truncated checkpoint absolute offsets are handled safely.
#[test]
fn test_corrupt_db_mismatched_checkpoints() {
    let temp = TempDatabase::new("mismatched_checkpoints.db");
    // Structurally truncate the file so the absolute offset table is not fully populated.
    let mut file = File::create(&temp.path).unwrap();
    file.write_all(b"S2PP").unwrap();
    file.write_all(&512u32.to_le_bytes()).unwrap();
    file.write_all(&256u32.to_le_bytes()).unwrap();
    file.write_all(&2u32.to_le_bytes()).unwrap();
    // Only write headers and partial absolute offsets, then stop.
    file.write_all(&[1u32.to_le_bytes(), 10u32.to_le_bytes()].concat())
        .unwrap();

    let result = QueryEngine::new(&temp.path);
    assert!(result.is_err());
    let error_message = match result {
        Err(err) => err.to_string(),
        Ok(_) => panic!("Expected database load to fail"),
    };
    assert!(
        error_message.contains("truncated") || error_message.contains("cast"),
        "Unexpected error: {}",
        error_message
    );
}

/// Verifies that the query engine safely handles exception mode 0 by processing it as EXCEPTION_MODE_U4.
#[test]
fn test_corrupt_db_invalid_exception_mode() {
    let temp = TempDatabase::new("invalid_mode.db");
    // Under the sub-block PFOR layout:
    // Header (12 bytes): bit_widths[0]=8, exception_count=1, exception_mode=0 (EXCEPTION_MODE_U4), index_bitmask_flag=0.
    // Primary bitstream (1 byte) = 0.
    // Exception index (1 byte) = 0.
    // Exception value (1 byte) = 10.
    let block_bytes = [
        8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,    // Header
        0x00, // Primary bitstream
        0,    // Exception index
        10,   // Exception value
    ];
    temp.write(b"S2PP", 2, 256, 1, &[1], &[0], &[0], &block_bytes);

    let engine = QueryEngine::new(&temp.path).unwrap();
    // The query should process successfully.
    let result = engine.query(1u64 << SHIFT_COMPACT);
    assert!(result.is_ok());
}

/// Verifies that u32 checked_add overflow of exception values in EXCEPTION_MODE_U32 is caught.
#[test]
fn test_corrupt_db_exception_value_overflow() {
    let temp = TempDatabase::new("exception_value_overflow.db");
    // Write a block with bit_width = 0, exception_count = 1, exception_mode = EXCEPTION_MODE_U32 (3).
    // exception value = u32::MAX (0xFFFFFFFF).
    // Under the sub-block PFOR layout:
    // Header (12 bytes): bit_widths all 0, exception_count=1, exception_mode=3, index_bitmask_flag=0.
    // Primary bitstream (0 bytes).
    // Exception index (1 byte) = 0.
    // Exception value (4 bytes) = 0xFFFFFFFF.
    let block_bytes = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, // Header
        0, // Exception index
        0xFF, 0xFF, 0xFF, 0xFF, // Exception value
    ];
    temp.write(b"S2PP", 2, 256, 1, &[1], &[0], &[0], &block_bytes);

    let engine = QueryEngine::new(&temp.path).unwrap();
    let result = engine.query(1u64 << SHIFT_COMPACT);
    assert!(result.is_err());
    let error_message = result.unwrap_err().to_string();
    assert!(
        error_message.contains("exception value overflow"),
        "Unexpected error: {}",
        error_message
    );
}

/// Verifies that out-of-order exception indices are rejected by both query and reconstruction.
#[test]
fn test_corrupt_db_out_of_order_exception_indices() {
    let temp = TempDatabase::new("out_of_order_exceptions.db");
    // count = 5 (delta_count = 4), single block, all bit-widths 0, two U8 exceptions whose
    // indices [3, 1] are descending. Both decoders must reject the non-increasing order rather
    // than silently dropping the second exception (which would diverge query from reconstruct).
    let block_bytes = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, // header: exception_count=2, mode U8, no bitmask
        3, 1, // exception indices (out of order)
        10, 20, // exception values (U8)
    ];
    temp.write(b"S2PP", 5, 256, 1, &[1], &[0], &[0], &block_bytes);

    let engine = QueryEngine::new(&temp.path).unwrap();

    let query_result = engine.query(1u64 << SHIFT_COMPACT);
    assert!(query_result.is_err());
    assert!(
        query_result
            .unwrap_err()
            .to_string()
            .contains("strictly increasing"),
        "query must reject out-of-order exception indices"
    );

    let reconstruct_result = engine.reconstruct_all_leaves();
    assert!(reconstruct_result.is_err());
    assert!(
        reconstruct_result
            .unwrap_err()
            .to_string()
            .contains("strictly increasing"),
        "reconstruct must reject out-of-order exception indices"
    );
}

/// Verifies that invalid S2 faces 6 and 7 are rejected rather than accepted at level 12.
#[test]
fn test_boundary_inputs_invalid_faces_rejected() {
    // face 6 and face 7 at level 12 (trailing zeros = 36) were previously wrongly accepted.
    let face_6_level_12 = (6u64 << 61) | (1u64 << 36);
    let face_7_level_12 = (7u64 << 61) | (1u64 << 36);

    assert!(try_get_level(face_6_level_12).is_err());
    assert!(try_get_level(face_7_level_12).is_err());

    for invalid_cell_id in [face_6_level_12, face_7_level_12] {
        let result = std::panic::catch_unwind(|| {
            population_density::get_level(invalid_cell_id);
        });
        assert!(
            result.is_err(),
            "get_level must panic on an invalid S2 face"
        );
    }
}

/// Verifies S2 cell ID limits and utility bounds on try_get_level and try_get_ancestor.
#[test]
fn test_boundary_inputs_s2_cell_id_limits() {
    // Test 0 S2 cell ID returns error.
    assert!(try_get_level(0).is_err());
    assert!(try_get_ancestor(0, 5).is_err());

    // Test Level 30 sentinel S2 cell ID.
    let level_30_sentinel = 1u64;
    let level = try_get_level(level_30_sentinel).unwrap();
    assert_eq!(level, 30);

    // Test Level 0 sentinel S2 cell ID.
    let level_0_sentinel = 1u64 << 60;
    let level = try_get_level(level_0_sentinel).unwrap();
    assert_eq!(level, 0);

    // Test try_get_ancestor with invalid level.
    assert!(try_get_ancestor(level_0_sentinel, 31).is_err());
}

/// Verifies that get_children_ids panics safely under invalid level bounds or parent sentinels.
#[test]
fn test_boundary_inputs_parent_children_bounds() {
    // Assert level 30 has no children (panics).
    let result = std::panic::catch_unwind(|| {
        get_children_ids(1u64, MAX_S2_LEVEL);
    });
    assert!(result.is_err());

    // Assert 0 cell ID has no children (panics).
    let result = std::panic::catch_unwind(|| {
        get_children_ids(0u64, 5);
    });
    assert!(result.is_err());

    // Assert cell ID less than sentinel position value panics.
    let result = std::panic::catch_unwind(|| {
        get_children_ids(1u64, 5);
    });
    assert!(result.is_err());
}

/// Verifies that try_get_level fails on invalid trailing zero counts.
#[test]
fn test_boundary_inputs_invalid_trailing_zeros() {
    // S2 cell lowest bit must be at an even index. Odd index (e.g. 1 trailing zero) must fail.
    let odd_trailing_zeros = 1u64 << 1;
    let result = try_get_level(odd_trailing_zeros);
    assert!(result.is_err());
    let error_message = result.unwrap_err().to_string();
    assert!(error_message.contains("must be at even index"));

    // Trailing zeros greater than 60 must fail.
    let too_many_trailing_zeros = 1u64 << 61;
    let result = try_get_level(too_many_trailing_zeros);
    assert!(result.is_err());
    let error_message = result.unwrap_err().to_string();
    assert!(error_message.contains("invalid S2 cell ID trailing zeros"));
}

/// Verifies that the query engine rejects database headers that are duplicate or out of order.
#[test]
fn test_engine_rejects_duplicate_or_out_of_order_headers() {
    let temp_duplicate = TempDatabase::new("duplicate_headers.db");
    // Write duplicate headers: 10u32 and 10u32.
    // count = 512, block_size = 256, block_count = 2.
    temp_duplicate.write(
        b"S2PP",
        512,
        256,
        2,
        &[10, 10],
        &[0],
        &[0, 0],
        &[8, 0, 8, 0],
    );

    let result = QueryEngine::new(&temp_duplicate.path);
    assert!(result.is_err());
    let error_message = match result {
        Err(err) => err.to_string(),
        Ok(_) => panic!("Expected database load to fail"),
    };
    assert!(error_message.contains("monotonically increasing"));

    let temp_order = TempDatabase::new("out_of_order_headers.db");
    // Write out-of-order headers: 10u32 and 5u32.
    temp_order.write(b"S2PP", 512, 256, 2, &[10, 5], &[0], &[0, 0], &[8, 0, 8, 0]);

    let result = QueryEngine::new(&temp_order.path);
    assert!(result.is_err());
    let error_message = match result {
        Err(err) => err.to_string(),
        Ok(_) => panic!("Expected database load to fail"),
    };
    assert!(error_message.contains("monotonically increasing"));
}

/// Verifies that the database builder fails fast when empty/zero-element builds are attempted.
#[test]
fn test_builder_empty_build_fails_fast() {
    let tiff_path = PathBuf::from("scratch/adversarial/empty_mock.tif");
    let database_path = PathBuf::from("scratch/adversarial/empty_mock.bin");
    std::fs::create_dir_all("scratch/adversarial").unwrap();

    // Create a 2x2 mock TIFF where all population density pixel values are 0.0f32.
    // They will not cross POPULATION_THRESHOLD, resulting in zero database leaves.
    let pixel_data = [0.0f32, 0.0f32, 0.0f32, 0.0f32];
    write_mock_tiff(&tiff_path, 2, 2, &pixel_data);

    // Run the build_database binary.
    let output = Command::new("cargo")
        .args([
            "run",
            "--package",
            "population-density-cli",
            "--bin",
            "build_database",
            "--",
            "--tiff-path",
            tiff_path.to_str().unwrap(),
            "--database-path",
            database_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to execute cargo run");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("no valid population data found to build a database"));

    // Cleanup files.
    let _ = std::fs::remove_file(tiff_path);
    let _ = std::fs::remove_file(database_path);
}

/// Verifies that non-finite float inputs (NaN, Infinity) are skipped and valid ones propagate successfully.
#[test]
fn test_builder_non_finite_float_inputs() {
    let tiff_path = PathBuf::from("scratch/adversarial/non_finite_mock.tif");
    let database_path = PathBuf::from("scratch/adversarial/non_finite_mock.bin");
    std::fs::create_dir_all("scratch/adversarial").unwrap();

    // Create a 2x2 mock TIFF with NaN, positive Infinity, a negative value, and a valid population density pixel.
    let pixel_data = [f32::NAN, f32::INFINITY, -100.0f32, 1500.0f32];
    write_mock_tiff(&tiff_path, 2, 2, &pixel_data);

    // Run the build_database binary.
    let output = Command::new("cargo")
        .args([
            "run",
            "--package",
            "population-density-cli",
            "--bin",
            "build_database",
            "--",
            "--tiff-path",
            tiff_path.to_str().unwrap(),
            "--database-path",
            database_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to execute cargo run");

    assert!(output.status.success());

    // Verify that the generated database successfully contains exactly the 1 valid leaf cell,
    // demonstrating that the other 3 non-finite/invalid float inputs were completely skipped.
    let engine = QueryEngine::new(&database_path).unwrap();
    assert_eq!(engine.count(), 1);

    // Cleanup files.
    let _ = std::fs::remove_file(tiff_path);
    let _ = std::fs::remove_file(database_path);
}

/// Executes the compiled unit tests inside the database builder binary to verify its internal invariants.
#[test]
fn test_builder_unit_tests() {
    let output = Command::new("cargo")
        .args([
            "test",
            "--package",
            "population-density-cli",
            "--bin",
            "build_database",
        ])
        .output()
        .expect("Failed to execute cargo test");

    assert!(output.status.success());
}
