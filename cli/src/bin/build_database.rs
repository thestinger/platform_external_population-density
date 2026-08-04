//! Builds the S2 population density database from a GeoTIFF file.
//!
//! Decodes the Float32 population density TIFF, aggregates pixel coordinates into
//! level 12 S2 cells in parallel, propagates the values up the quadtree, and prunes
//! cells below the population threshold to generate a compact database.

use anyhow::{Context, Result, anyhow, ensure};
use clap::Parser;
use population_density::{
    MAX_DB_LEVEL, NUM_ROOT_FACES, POPULATION_THRESHOLD, S2_FACE_SHIFT, SHIFT_COMPACT, get_parent_id,
};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use s2::cellid::CellID;
use s2::latlng::LatLng;
use sha2::{Digest, Sha256};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;
use tempfile::{Builder, TempDir};
use tiff::ColorType;
use tiff::decoder::{Decoder, DecodingResult, Limits};
use tiff::tags::Tag;

use population_density_cli::{
    add_population, convert_population_to_fixed, geotiff, open_database_snapshot,
    write_topology_database,
};

#[path = "../quadtree.rs"]
mod quadtree;

use quadtree::{LEVEL_0_SENTINEL_SHIFT, find_leaves};

const SOURCE_GEOTIFF_SHA256: [u8; 32] = [
    0xbd, 0xfe, 0x70, 0x81, 0x50, 0x6b, 0xd6, 0x12, 0x3d, 0x43, 0xa2, 0x9f, 0x22, 0xcf, 0x75, 0xcc,
    0xdd, 0x6e, 0xa5, 0xfe, 0xe3, 0xfd, 0xf7, 0xc3, 0xe9, 0x83, 0x5d, 0xeb, 0xf7, 0x1c, 0x6b, 0xc3,
];
const SHA256_BUFFER_SIZE: usize = 64 * 1024;
const SOURCE_SNAPSHOT_DIRECTORY_PREFIX: &str = "population-density-source-";
const SOURCE_SNAPSHOT_FILENAME: &str = "source.tif";
const GEOTIFF_WIDTH: u32 = 43_200;
const GEOTIFF_HEIGHT: u32 = 17_280;
const GEOTIFF_MAX_LATITUDE: f64 = 84.0;
const GEOTIFF_MIN_LONGITUDE: f64 = -180.0;
/// Defines the nominal 30-arc-second scale used for pixel-center coordinates.
const GEOTIFF_PIXEL_SCALE: f64 = 1.0 / 120.0;
/// Defines the decimal scale stored by the pinned GeoTIFF.
const GEOTIFF_METADATA_PIXEL_SCALE: f64 = 0.0083333333;
const PIXEL_CENTER_OFFSET: f64 = 0.5;
const GEOTIFF_NODATA_VALUE: f32 = -99999.0;
const GEOTIFF_NODATA_TEXT: &str = "-99999";
const GEOTIFF_BITS_PER_SAMPLE: [u16; 1] = [32];
const GEOTIFF_SAMPLES_PER_PIXEL: u16 = 1;
const GEOTIFF_SAMPLE_FORMAT: [u16; 1] = [3];
const GEOTIFF_PHOTOMETRIC_INTERPRETATION: u16 = 1;
const GEOTIFF_PLANAR_CONFIGURATION: u16 = 1;
const GEOTIFF_COMPRESSION: u16 = 5;
const GEOTIFF_PREDICTOR: u16 = 1;
const GEOTIFF_TILE_WIDTH: u32 = 512;
const GEOTIFF_TILE_HEIGHT: u32 = 512;
const GEOTIFF_PIXEL_SCALE_TAG: [f64; 3] = [
    GEOTIFF_METADATA_PIXEL_SCALE,
    GEOTIFF_METADATA_PIXEL_SCALE,
    0.0,
];
const GEOTIFF_TIEPOINT_TAG: [f64; 6] = [
    0.0,
    0.0,
    0.0,
    GEOTIFF_MIN_LONGITUDE,
    GEOTIFF_MAX_LATITUDE,
    0.0,
];
const GEOTIFF_KEY_DIRECTORY: [u16; 32] = [
    1, 1, 0, 7, 1024, 0, 1, 2, 1025, 0, 1, 1, 2048, 0, 1, 4326, 2049, 34737, 7, 0, 2054, 0, 1,
    9102, 2057, 34736, 1, 1, 2059, 34736, 1, 0,
];
const GEOTIFF_DOUBLE_PARAMETERS: [f64; 2] = [298.257223563, 6_378_137.0];
const GEOTIFF_ASCII_PARAMETERS: &str = "WGS 84|";

/// Holds the command-line arguments for building the S2 population density database.
#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Build the S2 population density database from GeoTIFF."
)]
struct Arguments {
    /// Specifies the input GeoTIFF path.
    #[arg(
        long = "tiff-path",
        default_value = "data/global_pop_2026_CN_1km_R2025A_UA_v1.tif"
    )]
    tiff_path: PathBuf,

    /// Specifies the path to the output database file.
    #[arg(
        long = "database-path",
        default_value = "population_density_database.bin"
    )]
    database_path: PathBuf,
}

/// Owns a private read-only snapshot of the source GeoTIFF.
struct SourceGeoTiffSnapshot {
    path: PathBuf,
    _temporary_directory: TempDir,
}

impl SourceGeoTiffSnapshot {
    /// Stages and validates the pinned source GeoTIFF.
    fn stage(source_path: &Path) -> Result<Self> {
        Self::stage_with_digest(source_path, &SOURCE_GEOTIFF_SHA256)
    }

    /// Stages a source GeoTIFF and requires the expected digest.
    fn stage_with_digest(source_path: &Path, expected_digest: &[u8; 32]) -> Result<Self> {
        let mut source_file = File::open(source_path).with_context(|| {
            format!("failed to open source GeoTIFF '{}'", source_path.display())
        })?;
        let temporary_directory = Builder::new()
            .prefix(SOURCE_SNAPSHOT_DIRECTORY_PREFIX)
            .tempdir()
            .context("failed to create a source GeoTIFF snapshot directory")?;
        let snapshot_path = temporary_directory.path().join(SOURCE_SNAPSHOT_FILENAME);
        let mut snapshot_file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&snapshot_path)
            .with_context(|| {
                format!(
                    "failed to create source GeoTIFF snapshot '{}'",
                    snapshot_path.display()
                )
            })?;

        let actual_digest = copy_and_calculate_sha256(&mut source_file, &mut snapshot_file)?;
        snapshot_file.flush()?;
        snapshot_file.sync_all()?;
        validate_source_digest(&actual_digest, expected_digest)?;

        let mut permissions = snapshot_file.metadata()?.permissions();
        permissions.set_readonly(true);
        snapshot_file.set_permissions(permissions)?;
        drop(snapshot_file);

        Ok(Self {
            path: snapshot_path,
            _temporary_directory: temporary_directory,
        })
    }

    /// Returns the immutable snapshot path.
    fn path(&self) -> &Path {
        &self.path
    }
}

/// Stores source metadata used by pixel-to-coordinate conversion.
#[derive(Clone, Debug, PartialEq)]
struct GeoTiffMetadata {
    dimensions: (u32, u32),
    color_type: ColorType,
    bits_per_sample: Vec<u16>,
    samples_per_pixel: u16,
    sample_format: Vec<u16>,
    photometric_interpretation: u16,
    planar_configuration: u16,
    compression: u16,
    predictor: u16,
    tile_width: u32,
    tile_height: u32,
    pixel_scale: Vec<f64>,
    tiepoint: Vec<f64>,
    key_directory: Vec<u16>,
    double_parameters: Vec<f64>,
    ascii_parameters: String,
    nodata: String,
    has_additional_images: bool,
}

impl GeoTiffMetadata {
    /// Reads the required metadata from a GeoTIFF decoder.
    fn read<R: Read + std::io::Seek>(decoder: &mut Decoder<R>) -> Result<Self> {
        Ok(Self {
            dimensions: decoder
                .dimensions()
                .context("failed to read TIFF dimensions")?,
            color_type: decoder
                .colortype()
                .context("failed to read TIFF color type")?,
            bits_per_sample: decoder
                .get_tag_u16_vec(Tag::BitsPerSample)
                .context("missing or invalid TIFF BitsPerSample tag")?,
            samples_per_pixel: decoder
                .get_tag_unsigned(Tag::SamplesPerPixel)
                .context("missing or invalid TIFF SamplesPerPixel tag")?,
            sample_format: decoder
                .get_tag_u16_vec(Tag::SampleFormat)
                .context("missing or invalid TIFF SampleFormat tag")?,
            photometric_interpretation: decoder
                .get_tag_unsigned(Tag::PhotometricInterpretation)
                .context("missing or invalid TIFF PhotometricInterpretation tag")?,
            planar_configuration: decoder
                .get_tag_unsigned(Tag::PlanarConfiguration)
                .context("missing or invalid TIFF PlanarConfiguration tag")?,
            compression: decoder
                .get_tag_unsigned(Tag::Compression)
                .context("missing or invalid TIFF Compression tag")?,
            predictor: decoder
                .get_tag_unsigned(Tag::Predictor)
                .context("missing or invalid TIFF Predictor tag")?,
            tile_width: decoder
                .get_tag_unsigned(Tag::TileWidth)
                .context("missing or invalid TIFF TileWidth tag")?,
            tile_height: decoder
                .get_tag_unsigned(Tag::TileLength)
                .context("missing or invalid TIFF TileLength tag")?,
            pixel_scale: decoder
                .get_tag_f64_vec(Tag::ModelPixelScaleTag)
                .context("missing or invalid GeoTIFF ModelPixelScaleTag")?,
            tiepoint: decoder
                .get_tag_f64_vec(Tag::ModelTiepointTag)
                .context("missing or invalid GeoTIFF ModelTiepointTag")?,
            key_directory: decoder
                .get_tag_u16_vec(Tag::GeoKeyDirectoryTag)
                .context("missing or invalid GeoTIFF GeoKeyDirectoryTag")?,
            double_parameters: decoder
                .get_tag_f64_vec(Tag::GeoDoubleParamsTag)
                .context("missing or invalid GeoTIFF GeoDoubleParamsTag")?,
            ascii_parameters: decoder
                .get_tag_ascii_string(Tag::GeoAsciiParamsTag)
                .context("missing or invalid GeoTIFF GeoAsciiParamsTag")?,
            nodata: decoder
                .get_tag_ascii_string(Tag::GdalNodata)
                .context("missing or invalid GeoTIFF GDAL_NODATA tag")?,
            has_additional_images: decoder.more_images(),
        })
    }

    /// Validates the exact metadata required by the pinned source dataset.
    fn validate(&self) -> Result<()> {
        require_metadata(
            "dimensions",
            &self.dimensions,
            &(GEOTIFF_WIDTH, GEOTIFF_HEIGHT),
        )?;
        require_metadata("color type", &self.color_type, &ColorType::Gray(32))?;
        require_metadata(
            "bits per sample",
            self.bits_per_sample.as_slice(),
            GEOTIFF_BITS_PER_SAMPLE.as_slice(),
        )?;
        require_metadata(
            "samples per pixel",
            &self.samples_per_pixel,
            &GEOTIFF_SAMPLES_PER_PIXEL,
        )?;
        require_metadata(
            "sample format",
            self.sample_format.as_slice(),
            GEOTIFF_SAMPLE_FORMAT.as_slice(),
        )?;
        require_metadata(
            "photometric interpretation",
            &self.photometric_interpretation,
            &GEOTIFF_PHOTOMETRIC_INTERPRETATION,
        )?;
        require_metadata(
            "planar configuration",
            &self.planar_configuration,
            &GEOTIFF_PLANAR_CONFIGURATION,
        )?;
        require_metadata("compression", &self.compression, &GEOTIFF_COMPRESSION)?;
        require_metadata("predictor", &self.predictor, &GEOTIFF_PREDICTOR)?;
        require_metadata("tile width", &self.tile_width, &GEOTIFF_TILE_WIDTH)?;
        require_metadata("tile height", &self.tile_height, &GEOTIFF_TILE_HEIGHT)?;
        require_metadata(
            "pixel scale",
            self.pixel_scale.as_slice(),
            GEOTIFF_PIXEL_SCALE_TAG.as_slice(),
        )?;
        require_metadata(
            "tiepoint",
            self.tiepoint.as_slice(),
            GEOTIFF_TIEPOINT_TAG.as_slice(),
        )?;
        require_metadata(
            "GeoKey directory",
            self.key_directory.as_slice(),
            GEOTIFF_KEY_DIRECTORY.as_slice(),
        )?;
        require_metadata(
            "double parameters",
            self.double_parameters.as_slice(),
            GEOTIFF_DOUBLE_PARAMETERS.as_slice(),
        )?;
        require_metadata(
            "ASCII parameters",
            self.ascii_parameters.as_str(),
            GEOTIFF_ASCII_PARAMETERS,
        )?;
        require_metadata("NoData value", self.nodata.as_str(), GEOTIFF_NODATA_TEXT)?;
        require_metadata(
            "additional image presence",
            &self.has_additional_images,
            &false,
        )
    }
}

/// Requires one metadata field to match its pinned value.
fn require_metadata<T: std::fmt::Debug + PartialEq + ?Sized>(
    name: &str,
    actual: &T,
    expected: &T,
) -> Result<()> {
    ensure!(
        actual == expected,
        "GeoTIFF {name} {actual:?} does not match expected {expected:?}"
    );
    Ok(())
}

/// Copies a reader while calculating its SHA-256 digest.
fn copy_and_calculate_sha256(mut reader: impl Read, mut writer: impl Write) -> Result<[u8; 32]> {
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; SHA256_BUFFER_SIZE];
    loop {
        let bytes_read = reader.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
        writer.write_all(&buffer[..bytes_read])?;
    }
    Ok(hasher.finalize().into())
}

/// Formats a SHA-256 digest as lowercase hexadecimal.
fn format_sha256(digest: &[u8; 32]) -> String {
    use std::fmt::Write;

    let mut output = String::with_capacity(digest.len() * 2);
    for byte in digest {
        write!(&mut output, "{byte:02x}").expect("writing to a String cannot fail");
    }
    output
}

/// Requires a source digest to match its expected value.
fn validate_source_digest(actual_digest: &[u8; 32], expected_digest: &[u8; 32]) -> Result<()> {
    ensure!(
        actual_digest == expected_digest,
        "source GeoTIFF SHA-256 {} does not match pinned {}",
        format_sha256(actual_digest),
        format_sha256(expected_digest)
    );
    Ok(())
}

/// Validates the pinned source metadata.
fn validate_source_metadata(path: &Path) -> Result<()> {
    let metadata_file = File::open(path)?;
    let mut decoder = Decoder::new(BufReader::new(metadata_file))?.with_limits(Limits::unlimited());
    GeoTiffMetadata::read(&mut decoder)?.validate()
}

/// Converts one valid source pixel to fixed-point population units.
fn convert_source_population(population: f32) -> Result<Option<i64>> {
    if population == GEOTIFF_NODATA_VALUE || population == 0.0 {
        return Ok(None);
    }
    ensure!(
        population.is_finite() && population > 0.0,
        "source population value must be finite, non-negative, or NoData: {}",
        population
    );
    Ok(Some(convert_population_to_fixed(population)?))
}

/// Extracts pruned quadtree leaves for every S2 face.
///
/// Returns an error if a face's total population is below the threshold because that region would
/// have no coarsening cell on-device.
fn extract_pruned_leaves(cell_populations: &FxHashMap<u64, i64>) -> Result<Vec<u64>> {
    let face_ids: Vec<u64> = (0..NUM_ROOT_FACES)
        .map(|face| ((face as u64) << S2_FACE_SHIFT) | (1u64 << LEVEL_0_SENTINEL_SHIFT))
        .collect();
    let mut leaves = Vec::new();

    for (face, &face_id) in face_ids.iter().enumerate() {
        let leaves_before = leaves.len();
        find_leaves(face_id, 0, cell_populations, &mut leaves);
        // Every S2 face must yield at least one leaf. Density-based coarse location relies on every
        // valid coordinate resolving to a cell (a face-level cell at worst); an empty face would
        // make the on-device query return no cell for that whole region, which the framework treats
        // as "no coarse location". Fail loudly so a future GeoTIFF or threshold change can't
        // silently empty a face and turn that suppression into a normal user-facing outcome.
        if leaves.len() == leaves_before {
            return Err(anyhow!(
                "S2 face {} produced no leaves because its population is below {}; verify the GeoTIFF and threshold",
                face,
                POPULATION_THRESHOLD
            ));
        }
    }

    Ok(leaves)
}

/// Runs the database build pipeline to compile S2 population density database from GeoTIFF.
fn main() -> Result<()> {
    let arguments = Arguments::parse();
    let start_time = Instant::now();

    // 1. GeoTIFF reading and pixel aggregation.
    println!("Step 1: Reading GeoTIFF and aggregating population into level 12 S2 cells...");

    let source_snapshot = SourceGeoTiffSnapshot::stage(&arguments.tiff_path)?;
    validate_source_metadata(source_snapshot.path())?;
    let prepared_tiff = geotiff::prepare_geotiff(source_snapshot.path())?;

    let file = File::open(prepared_tiff.path())?;
    let mut decoder = Decoder::new(BufReader::new(file))?.with_limits(Limits::unlimited());

    let (width, height) = decoder.dimensions()?;
    require_metadata(
        "prepared dimensions",
        &(width, height),
        &(GEOTIFF_WIDTH, GEOTIFF_HEIGHT),
    )?;
    println!("Image Dimensions: {} x {}", width, height);

    let color_type = decoder.colortype()?;
    require_metadata("prepared color type", &color_type, &ColorType::Gray(32))?;
    println!("Image Color Type: {:?}", color_type);

    let image_result = decoder.read_image()?;
    println!("GeoTIFF decoded in {:.2?}.", start_time.elapsed());
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

    let aggregation_start_time = Instant::now();
    println!("Aggregating pixels in parallel using rayon...");

    let level_12_populations = (0..height_usize)
        .into_par_iter()
        .try_fold(FxHashMap::default, |mut local_map, pixel_y| -> Result<_> {
            let row_offset = pixel_y * width_usize;
            let pixel_y_float = pixel_y as f64;
            for pixel_x in 0..width_usize {
                let value = data[row_offset + pixel_x];
                let Some(fixed_population) =
                    convert_source_population(value).with_context(|| {
                        format!("invalid source population at pixel ({pixel_x}, {pixel_y})")
                    })?
                else {
                    continue;
                };
                let latitude = GEOTIFF_MAX_LATITUDE
                    - (pixel_y_float + PIXEL_CENTER_OFFSET) * GEOTIFF_PIXEL_SCALE;
                let longitude = GEOTIFF_MIN_LONGITUDE
                    + ((pixel_x as f64) + PIXEL_CENTER_OFFSET) * GEOTIFF_PIXEL_SCALE;
                let latitude_longitude = LatLng::from_degrees(latitude, longitude);
                let cell_id = CellID::from(latitude_longitude).parent(MAX_DB_LEVEL as u64);
                // Accumulate fixed-point values so parallel reduction remains associative.
                add_population(&mut local_map, cell_id.0, fixed_population)?;
            }
            Ok(local_map)
        })
        .try_reduce(
            FxHashMap::default,
            |mut accumulator_map, local_map| -> Result<_> {
                for (cell_id, population) in local_map {
                    add_population(&mut accumulator_map, cell_id, population)?;
                }
                Ok(accumulator_map)
            },
        )?;

    println!(
        "Aggregation completed in {:.2?}.",
        aggregation_start_time.elapsed()
    );
    println!(
        "Total populated level 12 cells: {}",
        level_12_populations.len()
    );

    // 2. Upward population propagation from level 12 to 0.
    println!("\nStep 2: Propagating population upwards from level 12 to level 0...");
    let propagation_start_time = Instant::now();

    let mut cell_populations = level_12_populations;
    cell_populations.reserve(cell_populations.len() / 3);

    let mut current_level_cells: Vec<u64> = cell_populations.keys().cloned().collect();
    let mut parent_level_cells = Vec::with_capacity(current_level_cells.len());

    for level in (0..MAX_DB_LEVEL).rev() {
        parent_level_cells.clear();
        for &cell_id in &current_level_cells {
            let parent_id = get_parent_id(cell_id, level);
            let population = *cell_populations.get(&cell_id).unwrap_or(&0);
            add_population(&mut cell_populations, parent_id, population)?;
            parent_level_cells.push(parent_id);
        }
        parent_level_cells.sort_unstable();
        parent_level_cells.dedup();
        std::mem::swap(&mut current_level_cells, &mut parent_level_cells);
        println!(
            "  Level {}: {} populated cells",
            level,
            current_level_cells.len()
        );
    }

    println!(
        "Step 2 completed in {:.2?}.",
        propagation_start_time.elapsed()
    );

    // 3. Pruned quadtree leaf extraction.
    println!(
        "\nStep 3: Extracting leaves of the pruned quadtree T (where population >= {POPULATION_THRESHOLD})..."
    );
    let extraction_start_time = Instant::now();

    let mut leaves = extract_pruned_leaves(&cell_populations)?;

    println!(
        "Step 3 completed in {:.2?}.",
        extraction_start_time.elapsed()
    );
    println!("Extracted {} pruned leaves.", leaves.len());

    // Sort the extracted leaves.
    leaves.sort();

    // Compact leaves to 28-bit values.
    let compact_leaves: Vec<u32> = leaves
        .iter()
        .map(|&cell_id| (cell_id >> SHIFT_COMPACT) as u32)
        .collect();

    if compact_leaves.is_empty() {
        return Err(anyhow!(
            "no valid population data found to build a database"
        ));
    }

    println!("Writing mmap-only S2 quadtree topology database...");
    write_topology_database(&arguments.database_path, &compact_leaves)?;

    let query_engine = open_database_snapshot(&arguments.database_path)?;
    let reconstructed_leaves = query_engine.reconstruct_all_leaves()?;
    if reconstructed_leaves != compact_leaves {
        return Err(anyhow!(
            "database leaf reconstruction differs from the GeoTIFF-derived leaves"
        ));
    }

    println!(
        "Successfully generated and verified database file: '{}'",
        arguments.database_path.display()
    );
    println!(
        "File size: {:.3} MB",
        std::fs::metadata(&arguments.database_path)?.len() as f64 / (1024.0 * 1024.0)
    );
    println!("\nTotal Build Time: {:.2?}.", start_time.elapsed());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use population_density::POPULATION_FIXED_POINT_SCALE;
    use std::io::Cursor;

    /// Returns metadata matching the pinned GeoTIFF.
    fn pinned_metadata() -> GeoTiffMetadata {
        GeoTiffMetadata {
            dimensions: (GEOTIFF_WIDTH, GEOTIFF_HEIGHT),
            color_type: ColorType::Gray(32),
            bits_per_sample: GEOTIFF_BITS_PER_SAMPLE.to_vec(),
            samples_per_pixel: GEOTIFF_SAMPLES_PER_PIXEL,
            sample_format: GEOTIFF_SAMPLE_FORMAT.to_vec(),
            photometric_interpretation: GEOTIFF_PHOTOMETRIC_INTERPRETATION,
            planar_configuration: GEOTIFF_PLANAR_CONFIGURATION,
            compression: GEOTIFF_COMPRESSION,
            predictor: GEOTIFF_PREDICTOR,
            tile_width: GEOTIFF_TILE_WIDTH,
            tile_height: GEOTIFF_TILE_HEIGHT,
            pixel_scale: GEOTIFF_PIXEL_SCALE_TAG.to_vec(),
            tiepoint: GEOTIFF_TIEPOINT_TAG.to_vec(),
            key_directory: GEOTIFF_KEY_DIRECTORY.to_vec(),
            double_parameters: GEOTIFF_DOUBLE_PARAMETERS.to_vec(),
            ascii_parameters: GEOTIFF_ASCII_PARAMETERS.to_owned(),
            nodata: GEOTIFF_NODATA_TEXT.to_owned(),
            has_additional_images: false,
        }
    }

    /// Verifies SHA-256 calculation and pinned-digest enforcement.
    #[test]
    fn source_digest_validation_is_exact() {
        let mut copied_bytes = Vec::new();
        let abc_digest = copy_and_calculate_sha256(Cursor::new(b"abc"), &mut copied_bytes).unwrap();
        assert_eq!(copied_bytes, b"abc");
        assert_eq!(
            format_sha256(&abc_digest),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        validate_source_digest(&SOURCE_GEOTIFF_SHA256, &SOURCE_GEOTIFF_SHA256).unwrap();
        assert!(validate_source_digest(&[0u8; 32], &SOURCE_GEOTIFF_SHA256).is_err());
    }

    /// Verifies source mutation and replacement cannot change a staged snapshot.
    #[test]
    fn source_snapshot_isolated_from_source_path_changes() {
        let source_directory = tempfile::tempdir().unwrap();
        let source_path = source_directory.path().join("source.tif");
        let replacement_path = source_directory.path().join("replacement.tif");
        let reviewed_source = b"reviewed deterministic source";
        std::fs::write(&source_path, reviewed_source).unwrap();

        let mut digest_input = Vec::new();
        let expected_digest =
            copy_and_calculate_sha256(Cursor::new(reviewed_source), &mut digest_input).unwrap();
        assert_eq!(digest_input, reviewed_source);
        let snapshot =
            SourceGeoTiffSnapshot::stage_with_digest(&source_path, &expected_digest).unwrap();
        assert!(
            std::fs::metadata(snapshot.path())
                .unwrap()
                .permissions()
                .readonly()
        );

        std::fs::write(&source_path, b"mutated source").unwrap();
        assert_eq!(std::fs::read(snapshot.path()).unwrap(), reviewed_source);

        std::fs::write(&replacement_path, b"replacement source").unwrap();
        std::fs::rename(&replacement_path, &source_path).unwrap();
        assert_eq!(std::fs::read(snapshot.path()).unwrap(), reviewed_source);

        let snapshot_path = snapshot.path().to_path_buf();
        drop(snapshot);
        assert!(!snapshot_path.exists());
    }

    /// Verifies every pinned source metadata field is required exactly.
    #[test]
    fn source_metadata_validation_is_exact() {
        pinned_metadata().validate().unwrap();

        macro_rules! assert_rejected {
            ($field:ident, $value:expr) => {{
                let mut metadata = pinned_metadata();
                metadata.$field = $value;
                assert!(metadata.validate().is_err(), stringify!($field));
            }};
        }

        assert_rejected!(dimensions, (GEOTIFF_WIDTH - 1, GEOTIFF_HEIGHT));
        assert_rejected!(color_type, ColorType::Gray(16));
        assert_rejected!(bits_per_sample, vec![16]);
        assert_rejected!(samples_per_pixel, 2);
        assert_rejected!(sample_format, vec![1]);
        assert_rejected!(photometric_interpretation, 0);
        assert_rejected!(planar_configuration, 2);
        assert_rejected!(compression, 8);
        assert_rejected!(predictor, 2);
        assert_rejected!(tile_width, GEOTIFF_TILE_WIDTH / 2);
        assert_rejected!(tile_height, GEOTIFF_TILE_HEIGHT / 2);
        assert_rejected!(pixel_scale, vec![GEOTIFF_PIXEL_SCALE, 0.0, 0.0]);
        assert_rejected!(tiepoint, vec![0.0; GEOTIFF_TIEPOINT_TAG.len()]);
        assert_rejected!(key_directory, vec![1, 1, 0, 0]);
        assert_rejected!(
            double_parameters,
            vec![GEOTIFF_DOUBLE_PARAMETERS[1], GEOTIFF_DOUBLE_PARAMETERS[0]]
        );
        assert_rejected!(ascii_parameters, "WGS 72|".to_owned());
        assert_rejected!(nodata, "nan".to_owned());
        assert_rejected!(has_additional_images, true);
    }

    /// Verifies source population values reject undocumented sentinels.
    #[test]
    fn source_population_validation_is_exact() {
        assert_eq!(
            convert_source_population(GEOTIFF_NODATA_VALUE).unwrap(),
            None
        );
        assert_eq!(convert_source_population(0.0).unwrap(), None);
        assert_eq!(
            convert_source_population(1.5).unwrap(),
            Some(POPULATION_FIXED_POINT_SCALE + POPULATION_FIXED_POINT_SCALE / 2)
        );
        for invalid_population in [-1.0, f32::NEG_INFINITY, f32::INFINITY, f32::NAN] {
            assert!(convert_source_population(invalid_population).is_err());
        }
    }

    /// Verifies that population conversion preserves normal fixed-point values.
    #[test]
    fn test_convert_population_to_fixed() {
        assert_eq!(convert_population_to_fixed(0.0).unwrap(), 0);
        assert_eq!(
            convert_population_to_fixed(1.0).unwrap(),
            POPULATION_FIXED_POINT_SCALE
        );
        assert_eq!(
            convert_population_to_fixed(1.5).unwrap(),
            POPULATION_FIXED_POINT_SCALE + POPULATION_FIXED_POINT_SCALE / 2
        );
    }

    /// Verifies that population conversion rejects values outside the fixed-point range.
    #[test]
    fn test_convert_population_to_fixed_rejects_out_of_range_values() {
        const POPULATION_LIMIT_EXPONENT: u32 =
            i64::BITS - 1 - POPULATION_FIXED_POINT_SCALE.trailing_zeros();
        const FIRST_OUT_OF_RANGE_POPULATION: f32 = (1u64 << POPULATION_LIMIT_EXPONENT) as f32;

        let largest_in_range_population =
            f32::from_bits(FIRST_OUT_OF_RANGE_POPULATION.to_bits() - 1);
        assert!(convert_population_to_fixed(largest_in_range_population).is_ok());

        for out_of_range_population in [FIRST_OUT_OF_RANGE_POPULATION, f32::MAX] {
            let result = convert_population_to_fixed(out_of_range_population);
            assert!(result.is_err());
            assert!(
                result
                    .unwrap_err()
                    .to_string()
                    .contains("exceeds the fixed-point range")
            );
        }
    }

    /// Verifies that population accumulation accepts the maximum value and rejects overflow.
    #[test]
    fn test_add_population_boundaries() {
        const CELL_ID: u64 = 0x1000000000000000;

        let mut cell_populations = FxHashMap::default();
        cell_populations.insert(CELL_ID, i64::MAX - 1);

        add_population(&mut cell_populations, CELL_ID, 1).unwrap();
        assert_eq!(cell_populations[&CELL_ID], i64::MAX);

        let result = add_population(&mut cell_populations, CELL_ID, 1);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("fixed-point population overflow")
        );
        assert_eq!(cell_populations[&CELL_ID], i64::MAX);
    }

    /// Verifies that leaf extraction requires every S2 face to be populated.
    #[test]
    fn test_extract_pruned_leaves_requires_every_face() {
        use population_density::POPULATION_THRESHOLD_FIXED;

        // All six faces populated at the face level: one leaf each.
        let mut populations = FxHashMap::default();
        for face in 0..NUM_ROOT_FACES {
            let face_id = ((face as u64) << S2_FACE_SHIFT) | (1u64 << LEVEL_0_SENTINEL_SHIFT);
            populations.insert(face_id, POPULATION_THRESHOLD_FIXED);
        }
        let leaves =
            extract_pruned_leaves(&populations).expect("all faces populated should succeed");
        assert_eq!(leaves.len(), NUM_ROOT_FACES);

        // Dropping one face's population must make extraction fail loudly.
        let dropped_face_id = (0u64 << S2_FACE_SHIFT) | (1u64 << LEVEL_0_SENTINEL_SHIFT);
        populations.remove(&dropped_face_id);
        let result = extract_pruned_leaves(&populations);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("produced no leaves")
        );
    }
}
