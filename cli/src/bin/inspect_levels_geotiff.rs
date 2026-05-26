//! Analyzes and prints the distribution of S2 cell levels directly compiled from the GeoTIFF dataset.

use anyhow::{Result, anyhow};
use population_density::{
    MAX_DB_LEVEL, NUM_ROOT_FACES, POPULATION_FIXED_POINT_SCALE, POPULATION_THRESHOLD,
    S2_FACE_SHIFT, get_parent_id, try_get_level,
};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use s2::cellid::CellID;
use s2::latlng::LatLng;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::time::Instant;
use tiff::decoder::{Decoder, DecodingResult, Limits};

#[path = "../geotiff.rs"]
mod geotiff;
#[path = "../quadtree.rs"]
mod quadtree;

use quadtree::{LEVEL_0_SENTINEL_SHIFT, find_leaves};

/// Configures GeoTIFF spatial parameters.
const GEOTIFF_MAX_LATITUDE: f64 = 84.0;
const GEOTIFF_MIN_LONGITUDE: f64 = -180.0;
const GEOTIFF_PIXEL_SCALE: f64 = 0.008333333333333333;
const PIXEL_CENTER_OFFSET: f64 = 0.5;
const GEOTIFF_NODATA_VALUE: f32 = -99999.0;

fn main() -> Result<()> {
    let arguments_tiff_path = Path::new("data/global_pop_2026_CN_1km_R2025A_UA_v1.tif");
    if !arguments_tiff_path.exists() {
        return Err(anyhow!(
            "Input TIFF file '{}' not found.\n\
             Please refer to the Data source section in README.md for instructions on how to download the required GeoTIFF dataset.",
            arguments_tiff_path.display()
        ));
    }

    let (tiff_path, _guard) = geotiff::prepare_geotiff(arguments_tiff_path, "tmp_inspect.tif")?;

    println!("Loading and decoding GeoTIFF directly...");
    let start_time = Instant::now();
    let file = File::open(&tiff_path)?;
    let mut decoder = Decoder::new(BufReader::new(file))?.with_limits(Limits::unlimited());
    let (width, height) = decoder.dimensions()?;
    let image_result = decoder.read_image()?;
    let data = match image_result {
        DecodingResult::F32(vec) => vec,
        _ => return Err(anyhow!("Unexpected image data type (expected Float32)")),
    };

    println!("Aggregating pixels into Level 12 S2 cells in parallel...");
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

    println!("Propagating S2 quadtree populations up to Level 0...");
    let mut cell_populations = level_12_populations;
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

    println!(
        "Pruning S2 cell quadtree at threshold {}...",
        POPULATION_THRESHOLD
    );
    let face_ids: Vec<u64> = (0..NUM_ROOT_FACES)
        .map(|face| ((face as u64) << S2_FACE_SHIFT) | (1u64 << LEVEL_0_SENTINEL_SHIFT))
        .collect();
    let mut leaves = Vec::new();
    for &face_id in &face_ids {
        find_leaves(face_id, 0, &cell_populations, &mut leaves);
    }

    let total_leaves = leaves.len();
    println!("Total pruned leaves: {}", total_leaves);

    let mut level_counts = BTreeMap::new();
    for &cell_id in &leaves {
        let level = try_get_level(cell_id)?;
        *level_counts.entry(level).or_insert(0usize) += 1;
    }

    println!("\nGeoTIFF Pruned S2 Level Distribution:");
    println!("--------------------------------------------------");
    println!("| Level |      Count      |      Percentage      |");
    println!("--------------------------------------------------");
    for (level, count) in &level_counts {
        let percentage = (*count as f64 / total_leaves as f64) * 100.0;
        println!("| {:>5} | {:>15} | {:>18.4}% |", level, count, percentage);
    }
    println!("--------------------------------------------------");
    println!("Total analysis time: {:.2?}", start_time.elapsed());

    Ok(())
}
