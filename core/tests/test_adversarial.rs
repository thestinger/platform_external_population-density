//! Exercises adversarial S2PD inputs, S2 boundaries, and builder failures.

mod common;

use common::open_database;
use population_density::topology_format::{
    FILE_SIZE_OFFSET, FIXED_HEADER_SIZE, LEVEL_ENCODING_OFFSET, MAGIC, MAGIC_OFFSET, VERSION,
    VERSION_OFFSET,
};
use population_density::{MAX_S2_LEVEL, get_children_ids, try_get_ancestor, try_get_level};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::Command;
use tempfile::{Builder, NamedTempFile};
use tiff::encoder::{TiffEncoder, colortype};

const GENERATED_DATABASE_PATH: &str = "../population_density_database.bin";
const PACKAGED_DATABASE_PATH: &str =
    "../../../packages/apps/NetworkLocation/res/raw/population_density_database.bin";
const GEOTIFF_PIXEL_SCALE: f64 = 1.0 / 120.0;
const GEOTIFF_MIN_LONGITUDE: f64 = -180.0;
const GEOTIFF_MAX_LATITUDE: f64 = 84.0;
const DECLARED_SIZE_INCREMENT: u32 = 1;

/// Describes one named database-byte mutation.
type DatabaseCorruption = (&'static str, fn(&mut Vec<u8>));

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

/// Manages one temporary file.
struct TemporaryFile {
    file: NamedTempFile,
}

impl TemporaryFile {
    /// Creates a unique temporary file.
    fn new(name: &str, extension: &str) -> Self {
        Self {
            file: Builder::new()
                .prefix(name)
                .suffix(&format!(".{extension}"))
                .tempfile()
                .unwrap(),
        }
    }

    /// Returns the temporary path.
    fn path(&self) -> &Path {
        self.file.path()
    }

    /// Writes all bytes to the temporary file.
    fn write(&self, bytes: &[u8]) {
        std::fs::write(self.file.path(), bytes).unwrap();
    }
}

/// Manages a builder test's temporary input and output files.
struct BuilderTestFiles {
    tiff: TemporaryFile,
    database: TemporaryFile,
}

impl BuilderTestFiles {
    /// Creates unique temporary paths for a builder test.
    fn new(name: &str) -> Self {
        Self {
            tiff: TemporaryFile::new(name, "tif"),
            database: TemporaryFile::new(name, "bin"),
        }
    }

    /// Returns the temporary GeoTIFF path.
    fn tiff_path(&self) -> &Path {
        self.tiff.path()
    }

    /// Returns the temporary database path.
    fn database_path(&self) -> &Path {
        self.database.path()
    }
}

/// Writes a mock GeoTIFF matching the builder's aggregation grid.
fn write_mock_tiff(path: &Path, width: u32, height: u32, data: &[f32]) {
    let file = File::create(path).unwrap();
    let mut encoder = TiffEncoder::new(file).unwrap();
    let mut image = encoder
        .new_image::<colortype::Gray32Float>(width, height)
        .unwrap();
    image
        .encoder()
        .write_tag(
            tiff::tags::Tag::ModelPixelScaleTag,
            &[GEOTIFF_PIXEL_SCALE, GEOTIFF_PIXEL_SCALE, 0.0][..],
        )
        .unwrap();
    image
        .encoder()
        .write_tag(
            tiff::tags::Tag::ModelTiepointTag,
            &[
                0.0,
                0.0,
                0.0,
                GEOTIFF_MIN_LONGITUDE,
                GEOTIFF_MAX_LATITUDE,
                0.0,
            ][..],
        )
        .unwrap();
    image.write_data(data).unwrap();
}

/// Runs the database builder for a synthetic GeoTIFF.
fn run_builder(files: &BuilderTestFiles) -> std::process::Output {
    Command::new("cargo")
        .args([
            "run",
            "--package",
            "population-density-cli",
            "--bin",
            "build_database",
            "--",
            "--tiff-path",
            files.tiff_path().to_str().unwrap(),
            "--database-path",
            files.database_path().to_str().unwrap(),
        ])
        .output()
        .expect("failed to execute database builder")
}

/// Verifies that structural S2PD header corruption is rejected.
#[test]
fn test_s2pd_header_corruption_is_rejected() {
    let valid_database = std::fs::read(database_path()).unwrap();
    assert_eq!(
        &valid_database[MAGIC_OFFSET..MAGIC_OFFSET + MAGIC.len()],
        MAGIC
    );

    let corruptions: [DatabaseCorruption; 4] = [
        ("magic", |bytes| {
            bytes[MAGIC_OFFSET..MAGIC_OFFSET + MAGIC.len()].copy_from_slice(b"BAD!")
        }),
        ("version", |bytes| {
            bytes[VERSION_OFFSET..VERSION_OFFSET + std::mem::size_of::<u16>()]
                .copy_from_slice(&(VERSION + 1).to_le_bytes())
        }),
        ("declared-size", |bytes| {
            let declared_size = u32::from_le_bytes(
                bytes[FILE_SIZE_OFFSET..FILE_SIZE_OFFSET + 4]
                    .try_into()
                    .unwrap(),
            );
            bytes[FILE_SIZE_OFFSET..FILE_SIZE_OFFSET + 4].copy_from_slice(
                &declared_size
                    .checked_add(DECLARED_SIZE_INCREMENT)
                    .unwrap()
                    .to_le_bytes(),
            );
        }),
        ("level-encoding", |bytes| {
            bytes[FIXED_HEADER_SIZE + LEVEL_ENCODING_OFFSET] = u8::MAX;
        }),
    ];

    for (name, mutate) in corruptions {
        let temporary_database = TemporaryFile::new(name, "bin");
        let mut bytes = valid_database.clone();
        mutate(&mut bytes);
        temporary_database.write(&bytes);
        assert!(
            open_database(temporary_database.path()).is_err(),
            "S2PD must reject {name} corruption"
        );
    }
}

/// Verifies that truncated S2PD files are rejected at multiple boundaries.
#[test]
fn test_s2pd_truncation_is_rejected() {
    let valid_database = std::fs::read(database_path()).unwrap();
    for truncated_length in [
        0,
        MAGIC.len() - 1,
        FIXED_HEADER_SIZE,
        valid_database.len() - 1,
    ] {
        let temporary_database = TemporaryFile::new("truncated", "bin");
        temporary_database.write(&valid_database[..truncated_length]);
        assert!(
            open_database(temporary_database.path()).is_err(),
            "S2PD must reject a file truncated to {truncated_length} bytes"
        );
    }
}

/// Verifies that corrupt raw topology masks fail full validation.
#[test]
fn test_s2pd_raw_topology_corruption_is_rejected() {
    use population_density::topology_format::RAW_MASKS_OFFSET;

    let mut database = std::fs::read(database_path()).unwrap();
    let raw_masks_offset = u32::from_le_bytes(
        database[RAW_MASKS_OFFSET..RAW_MASKS_OFFSET + 4]
            .try_into()
            .unwrap(),
    ) as usize;
    database[raw_masks_offset] = 0;

    let temporary_database = TemporaryFile::new("raw-mask", "bin");
    temporary_database.write(&database);
    assert!(open_database(temporary_database.path()).is_err());
}

/// Verifies that invalid faces and malformed sentinels are rejected.
#[test]
fn test_invalid_s2_cell_ids_are_rejected() {
    const LEVEL_12_SENTINEL_SHIFT: u32 = 36;

    for face in 6..=7 {
        let cell_id = (face << 61) | (1u64 << LEVEL_12_SENTINEL_SHIFT);
        assert!(try_get_level(cell_id).is_err());
    }

    assert!(try_get_level(0).is_err());
    assert!(try_get_ancestor(0, 5).is_err());
    assert!(try_get_level(1u64 << 1).is_err());
    assert!(try_get_level(1u64 << 61).is_err());
}

/// Verifies S2 utility behavior at level boundaries.
#[test]
fn test_s2_level_boundaries() {
    let level_30_cell_id = 1u64;
    let level_0_cell_id = 1u64 << 60;
    assert_eq!(try_get_level(level_30_cell_id).unwrap(), MAX_S2_LEVEL);
    assert_eq!(try_get_level(level_0_cell_id).unwrap(), 0);
    assert!(try_get_ancestor(level_0_cell_id, MAX_S2_LEVEL + 1).is_err());

    assert!(std::panic::catch_unwind(|| get_children_ids(level_30_cell_id, MAX_S2_LEVEL)).is_err());
    assert!(std::panic::catch_unwind(|| get_children_ids(0, 5)).is_err());
    assert!(std::panic::catch_unwind(|| get_children_ids(level_30_cell_id, 5)).is_err());
}

/// Verifies that query rejects invalid cell IDs without panicking.
#[test]
fn test_query_rejects_invalid_cell_ids() {
    const LEVEL_12_SENTINEL_SHIFT: u32 = 36;

    let query_engine = open_database(database_path()).unwrap();
    for invalid_cell_id in [
        0,
        (6u64 << 61) | (1u64 << LEVEL_12_SENTINEL_SHIFT),
        1u64 << 1,
    ] {
        assert!(query_engine.query(invalid_cell_id).is_err());
    }
}

/// Verifies that the builder rejects a source outside the pinned dataset.
#[test]
fn test_builder_rejects_unpinned_source() {
    let files = BuilderTestFiles::new("unpinned");
    write_mock_tiff(files.tiff_path(), 2, 2, &[0.0; 4]);

    let output = run_builder(&files);
    assert!(!output.status.success());
    let standard_error = String::from_utf8_lossy(&output.stderr);
    assert!(
        standard_error.contains("source GeoTIFF SHA-256")
            && standard_error.contains("does not match pinned"),
        "builder must reject an unpinned source: {standard_error}"
    );
}
