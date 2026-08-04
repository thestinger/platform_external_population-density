//! Builds the S2 population density database from a GeoTIFF file.
//!
//! Decodes the Float32 population density TIFF, aggregates pixel coordinates into
//! level 12 S2 cells in parallel, propagates the values up the quadtree, and prunes
//! cells below the population threshold to generate a compact database.

use anyhow::{Result, anyhow};
use clap::Parser;
use population_density::{
    MAX_DB_LEVEL, NUM_ROOT_FACES, POPULATION_THRESHOLD, S2_FACE_SHIFT, SHIFT_COMPACT, get_parent_id,
};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use s2::cellid::CellID;
use s2::latlng::LatLng;
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use std::time::Instant;
use tiff::decoder::{Decoder, DecodingResult, Limits};

use population_density_cli::{
    add_population, convert_population_to_fixed, open_database_snapshot, write_topology_database,
};

#[path = "../geotiff.rs"]
mod geotiff;
#[path = "../quadtree.rs"]
mod quadtree;

use quadtree::{LEVEL_0_SENTINEL_SHIFT, find_leaves};

/// Configures GeoTIFF spatial parameters for coordinates and cell aggregation.
const GEOTIFF_MAX_LATITUDE: f64 = 84.0;
const GEOTIFF_MIN_LONGITUDE: f64 = -180.0;
const GEOTIFF_PIXEL_SCALE: f64 = 0.008333333333333333;
const PIXEL_CENTER_OFFSET: f64 = 0.5;
const GEOTIFF_NODATA_VALUE: f32 = -99999.0;
/// Bounds the allowed deviation of the GeoTIFF origin and pixel scale from the assumed grid.
const GEOREFERENCE_TOLERANCE: f64 = 1e-6;

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

/// Validates that the GeoTIFF origin and pixel scale match the assumed aggregation grid.
fn validate_georeferencing<R: std::io::Read + std::io::Seek>(
    decoder: &mut Decoder<R>,
) -> Result<()> {
    let pixel_scale = decoder
        .get_tag_f64_vec(tiff::tags::Tag::ModelPixelScaleTag)
        .map_err(|error| {
            anyhow!(
                "missing or unreadable GeoTIFF ModelPixelScaleTag (33550): {}",
                error
            )
        })?;
    let tiepoint = decoder
        .get_tag_f64_vec(tiff::tags::Tag::ModelTiepointTag)
        .map_err(|error| {
            anyhow!(
                "missing or unreadable GeoTIFF ModelTiepointTag (33922): {}",
                error
            )
        })?;

    if pixel_scale.len() < 2
        || (pixel_scale[0] - GEOTIFF_PIXEL_SCALE).abs() > GEOREFERENCE_TOLERANCE
        || (pixel_scale[1] - GEOTIFF_PIXEL_SCALE).abs() > GEOREFERENCE_TOLERANCE
    {
        return Err(anyhow!(
            "GeoTIFF pixel scale {:?} does not match the assumed {} degrees per pixel",
            pixel_scale,
            GEOTIFF_PIXEL_SCALE
        ));
    }

    // ModelTiepointTag stores a raster point (i, j, k) mapped to a model point (x, y, z); the
    // builder assumes raster origin (0, 0) maps to (GEOTIFF_MIN_LONGITUDE, GEOTIFF_MAX_LATITUDE).
    if tiepoint.len() < 6
        || tiepoint[0].abs() > GEOREFERENCE_TOLERANCE
        || tiepoint[1].abs() > GEOREFERENCE_TOLERANCE
        || (tiepoint[3] - GEOTIFF_MIN_LONGITUDE).abs() > GEOREFERENCE_TOLERANCE
        || (tiepoint[4] - GEOTIFF_MAX_LATITUDE).abs() > GEOREFERENCE_TOLERANCE
    {
        return Err(anyhow!(
            "GeoTIFF tiepoint {:?} does not match the assumed origin (longitude {}, latitude {})",
            tiepoint,
            GEOTIFF_MIN_LONGITUDE,
            GEOTIFF_MAX_LATITUDE
        ));
    }

    Ok(())
}

/// Extracts the leaves of the pruned quadtree for all S2 faces, requiring every face to yield at
/// least one leaf. Returns an error if any face's total population is below the threshold, since a
/// whole face with no cells would leave that region with no coarsening cell on-device.
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

    if !arguments.tiff_path.exists() {
        return Err(anyhow!(
            "input TIFF file '{}' not found; see the Data source section in README.md",
            arguments.tiff_path.display()
        ));
    }

    // Validate the GeoTIFF georeferencing against the assumed grid before the expensive
    // aggregation, so a file with a different origin or resolution fails fast instead of
    // silently mapping every pixel to the wrong S2 cell. This reads the original file because
    // the on-the-fly tiffcp recompression below drops the GeoTIFF geo tags.
    {
        let georeference_file = File::open(&arguments.tiff_path)?;
        let mut georeference_decoder =
            Decoder::new(BufReader::new(georeference_file))?.with_limits(Limits::unlimited());
        validate_georeferencing(&mut georeference_decoder)?;
    }

    let prepared_tiff = geotiff::prepare_geotiff(&arguments.tiff_path)?;

    let file = File::open(prepared_tiff.path())?;
    let mut decoder = Decoder::new(BufReader::new(file))?.with_limits(Limits::unlimited());

    let (width, height) = decoder.dimensions()?;
    println!("Image Dimensions: {} x {}", width, height);

    let color_type = decoder.colortype()?;
    println!("Image Color Type: {:?}", color_type);

    let image_result = decoder.read_image()?;
    println!("GeoTIFF decoded in {:.2?}.", start_time.elapsed());

    let data = match image_result {
        DecodingResult::F32(vec) => {
            if vec.len() != (width as usize) * (height as usize) {
                return Err(anyhow!(
                    "decoded image data vector length {} does not match expected size {} x {} = {}",
                    vec.len(),
                    width,
                    height,
                    (width as usize) * (height as usize)
                ));
            }
            vec
        }
        _ => return Err(anyhow!("unexpected image data type (expected Float32)")),
    };

    let aggregation_start_time = Instant::now();
    println!("Aggregating pixels in parallel using rayon...");

    let width_usize = width as usize;
    let level_12_populations = (0..height as usize)
        .into_par_iter()
        .try_fold(FxHashMap::default, |mut local_map, pixel_y| -> Result<_> {
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
                    // Accumulate in fixed-point integer units so the parallel reduction is
                    // associative and the build is bit-for-bit reproducible regardless of the
                    // rayon work-stealing split (f64 addition is not associative).
                    let fixed_population = convert_population_to_fixed(value)?;
                    add_population(&mut local_map, cell_id.0, fixed_population)?;
                }
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

    /// Verifies that leaf extraction requires every S2 face to be populated, failing loudly if any
    /// face is empty (which would leave a whole region with no coarsening cell on-device).
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
