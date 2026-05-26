//! Builds the S2 population density database from a GeoTIFF file.
//!
//! Decodes the Float32 population density TIFF, aggregates pixel coordinates into
//! level 12 S2 cells in parallel, propagates the values up the quadtree, and prunes
//! cells below the population threshold to generate a compact database.

#![allow(clippy::needless_range_loop, clippy::collapsible_if)]

use anyhow::{Result, anyhow};
use clap::Parser;
use population_density::{
    BLOCK_HEADER_SIZE, BLOCK_SIZE, EXCEPTION_INDEX_BITMASK_SIZE, EXCEPTION_MODE_U4,
    EXCEPTION_MODE_U8, EXCEPTION_MODE_U16, EXCEPTION_MODE_U32, LAST_SUB_BLOCK_SIZE, MAX_DB_LEVEL,
    MAX_PFOR_BIT_WIDTH, NUM_ROOT_FACES, POPULATION_FIXED_POINT_SCALE, POPULATION_THRESHOLD,
    S2_FACE_SHIFT, S2PP_CHECKPOINT_INTERVAL, SHIFT_COMPACT, SUB_BLOCK_BIT_WIDTH_BITS,
    SUB_BLOCK_COUNT, SUB_BLOCK_SIZE, get_parent_id,
};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use s2::cellid::CellID;
use s2::latlng::LatLng;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;
use tiff::decoder::{Decoder, DecodingResult, Limits};

#[path = "../geotiff.rs"]
mod geotiff;
#[path = "../quadtree.rs"]
mod quadtree;

use quadtree::{LEVEL_0_SENTINEL_SHIFT, find_leaves};

/// Configures S2PP database formatting constants.
const EXCEPTION_MODE_U4_MAX: u32 = 15;

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
    /// Specifies the path to the input GeoTIFF file. Download source: https://hub.worldpop.org/geodata/summary?id=80032
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

/// Writes the block-compressed S2PP PFOR database format to disk.
fn write_s2pp_database<P: AsRef<Path>>(database_path: P, compact_leaves: &[u32]) -> Result<()> {
    if compact_leaves.is_empty() {
        return Err(anyhow!(
            "no valid population data found to build a database"
        ));
    }
    let start_time = Instant::now();
    let count = compact_leaves.len();
    let block_count = count.div_ceil(BLOCK_SIZE);

    println!("Writing database in compact S2PP block-compressed S2 cell ID format...");
    let mut block_headers = Vec::with_capacity(block_count);
    let mut block_absolute_offsets = Vec::with_capacity(block_count);
    let mut block_data = Vec::new();

    let mut deltas_minus_one = Vec::with_capacity(BLOCK_SIZE);

    for block_index in 0..block_count {
        let start = block_index * BLOCK_SIZE;
        let end = std::cmp::min(start + BLOCK_SIZE, count);

        let header = compact_leaves[start];
        block_headers.push(header);
        block_absolute_offsets.push(block_data.len() as u32);

        if end - start <= 1 {
            // Skip storing deltas if block has only one element (its header).
            continue;
        }

        let delta_count = end - start - 1;

        // Compute deltas-minus-one.
        deltas_minus_one.clear();
        for element_index in start + 1..end {
            let delta = compact_leaves[element_index]
                .checked_sub(compact_leaves[element_index - 1])
                .ok_or_else(|| {
                    anyhow!(
                        "compact leaves are out of order or duplicate: cell at {} is less than or equal to preceding cell",
                        element_index
                    )
                })?;
            let delta_minus_one = delta
                .checked_sub(1)
                .ok_or_else(|| anyhow!("duplicate compact leaf ID found"))?;
            deltas_minus_one.push(delta_minus_one);
        }

        // Use dynamic programming (DP) to find the optimal sub-block bit-widths.
        let mut best_cost = usize::MAX;
        let mut best_mode = EXCEPTION_MODE_U32;
        let mut best_exception_count = 0;
        let mut best_bit_widths = [0u8; SUB_BLOCK_COUNT];

        for &mode in &[
            EXCEPTION_MODE_U4,
            EXCEPTION_MODE_U8,
            EXCEPTION_MODE_U16,
            EXCEPTION_MODE_U32,
        ] {
            const DP_ROWS: usize = SUB_BLOCK_COUNT + 1;
            let mut dp = [[usize::MAX; BLOCK_SIZE]; DP_ROWS];
            let mut parent_exception_count = [[0u8; BLOCK_SIZE]; DP_ROWS];
            let mut chosen_bit_width = [[0u8; BLOCK_SIZE]; DP_ROWS];

            dp[0][0] = 0;

            for sub_block_index in 0..SUB_BLOCK_COUNT {
                let sb_start = std::cmp::min(sub_block_index * SUB_BLOCK_SIZE, delta_count);
                let sb_end = std::cmp::min(
                    sb_start
                        + if sub_block_index == SUB_BLOCK_COUNT - 1 {
                            LAST_SUB_BLOCK_SIZE
                        } else {
                            SUB_BLOCK_SIZE
                        },
                    delta_count,
                );
                let sb_len = sb_end - sb_start;

                // Precompute each candidate bit-width's sub-block exception count and validity
                // once. Both depend only on (mode, sub-block, bit-width), not on exception_count,
                // so hoisting the scan out of the exception_count loop avoids rescanning the
                // sub-block deltas for every reachable exception_count state.
                let mut bit_width_exception_counts = [0usize; MAX_PFOR_BIT_WIDTH as usize + 1];
                let mut bit_width_valid = [false; MAX_PFOR_BIT_WIDTH as usize + 1];
                for bit_width in 0..=MAX_PFOR_BIT_WIDTH {
                    let mut sub_block_exception_count = 0;
                    let mut valid = true;
                    let limit = 1u32 << bit_width;

                    for &delta_minus_one in &deltas_minus_one[sb_start..sb_end] {
                        if delta_minus_one >= limit {
                            sub_block_exception_count += 1;
                            let val = (delta_minus_one >> bit_width) - 1;
                            match mode {
                                EXCEPTION_MODE_U4 => {
                                    if val > EXCEPTION_MODE_U4_MAX {
                                        valid = false;
                                        break;
                                    }
                                }
                                EXCEPTION_MODE_U8 => {
                                    if val > u8::MAX as u32 {
                                        valid = false;
                                        break;
                                    }
                                }
                                EXCEPTION_MODE_U16 => {
                                    if val > u16::MAX as u32 {
                                        valid = false;
                                        break;
                                    }
                                }
                                EXCEPTION_MODE_U32 => {}
                                _ => unreachable!(),
                            }
                        }
                    }

                    bit_width_valid[bit_width as usize] = valid;
                    bit_width_exception_counts[bit_width as usize] = sub_block_exception_count;
                }

                for exception_count in 0..=delta_count {
                    let current_cost = dp[sub_block_index][exception_count];
                    if current_cost == usize::MAX {
                        continue;
                    }

                    for bit_width in 0..=MAX_PFOR_BIT_WIDTH {
                        if !bit_width_valid[bit_width as usize] {
                            continue;
                        }

                        let next_exception_count =
                            exception_count + bit_width_exception_counts[bit_width as usize];
                        if next_exception_count <= delta_count {
                            let added_bits = sb_len * bit_width as usize;
                            let next_cost = current_cost + added_bits;
                            if next_cost < dp[sub_block_index + 1][next_exception_count] {
                                dp[sub_block_index + 1][next_exception_count] = next_cost;
                                parent_exception_count[sub_block_index + 1][next_exception_count] =
                                    exception_count as u8;
                                chosen_bit_width[sub_block_index + 1][next_exception_count] =
                                    bit_width;
                            }
                        }
                    }
                }
            }

            for exception_count in 0..=delta_count {
                let bit_cost = dp[SUB_BLOCK_COUNT][exception_count];
                if bit_cost == usize::MAX {
                    continue;
                }
                let primary_bytes = bit_cost.div_ceil(8);
                let index_cost_bytes = if exception_count >= EXCEPTION_INDEX_BITMASK_SIZE {
                    EXCEPTION_INDEX_BITMASK_SIZE
                } else {
                    exception_count
                };
                let value_cost_bytes = match mode {
                    EXCEPTION_MODE_U4 => exception_count.div_ceil(2),
                    EXCEPTION_MODE_U8 => exception_count * std::mem::size_of::<u8>(),
                    EXCEPTION_MODE_U16 => exception_count * std::mem::size_of::<u16>(),
                    EXCEPTION_MODE_U32 => exception_count * std::mem::size_of::<u32>(),
                    _ => unreachable!(),
                };
                let total_cost =
                    BLOCK_HEADER_SIZE + primary_bytes + index_cost_bytes + value_cost_bytes;
                if total_cost < best_cost {
                    best_cost = total_cost;
                    best_mode = mode;
                    best_exception_count = exception_count;
                    // Backtrack.
                    let mut current_exception_count = exception_count;
                    for sub_block_index in (0..SUB_BLOCK_COUNT).rev() {
                        best_bit_widths[sub_block_index] =
                            chosen_bit_width[sub_block_index + 1][current_exception_count];
                        current_exception_count = parent_exception_count[sub_block_index + 1]
                            [current_exception_count]
                            as usize;
                    }
                }
            }
        }

        // 1. Write the 12-byte block header metadata.
        let mut header_bytes = [0u8; BLOCK_HEADER_SIZE];
        let mut bit_offset = 0;
        for &bit_width in &best_bit_widths {
            let mut bits_left = SUB_BLOCK_BIT_WIDTH_BITS;
            let mut val = bit_width;
            while bits_left > 0 {
                let byte_idx = bit_offset / 8;
                let bit_idx = bit_offset % 8;
                let bits_to_write = std::cmp::min(8 - bit_idx, bits_left);
                let mask = (1 << bits_to_write) - 1;
                header_bytes[byte_idx] |= ((val & mask) << bit_idx) as u8;
                val >>= bits_to_write;
                bits_left -= bits_to_write;
                bit_offset += bits_to_write;
            }
        }
        header_bytes[BLOCK_HEADER_SIZE - 2] = best_exception_count as u8;
        let index_bitmask_flag = if best_exception_count >= EXCEPTION_INDEX_BITMASK_SIZE {
            1
        } else {
            0
        };
        header_bytes[BLOCK_HEADER_SIZE - 1] = best_mode | (index_bitmask_flag << 2);
        block_data.extend_from_slice(&header_bytes);

        // 2. Pack primary bitstream.
        let mut primary_bitstream = Vec::new();
        let mut current_byte = 0u8;
        let mut primary_bit_offset = 0;

        for sub_block_index in 0..SUB_BLOCK_COUNT {
            let bit_width = best_bit_widths[sub_block_index] as usize;
            if bit_width == 0 {
                continue;
            }
            let sb_start = std::cmp::min(sub_block_index * SUB_BLOCK_SIZE, delta_count);
            let sb_end = std::cmp::min(
                sb_start
                    + if sub_block_index == SUB_BLOCK_COUNT - 1 {
                        LAST_SUB_BLOCK_SIZE
                    } else {
                        SUB_BLOCK_SIZE
                    },
                delta_count,
            );
            for &delta_minus_one in &deltas_minus_one[sb_start..sb_end] {
                let mut val = delta_minus_one & ((1 << bit_width) - 1);
                let mut bits_left = bit_width;
                while bits_left > 0 {
                    let bits_to_write = std::cmp::min(8 - primary_bit_offset, bits_left);
                    let mask = (1 << bits_to_write) - 1;
                    current_byte |= ((val & mask) as u8) << primary_bit_offset;
                    val >>= bits_to_write;
                    bits_left -= bits_to_write;
                    primary_bit_offset += bits_to_write;
                    if primary_bit_offset == 8 {
                        primary_bitstream.push(current_byte);
                        current_byte = 0;
                        primary_bit_offset = 0;
                    }
                }
            }
        }
        if primary_bit_offset > 0 {
            primary_bitstream.push(current_byte);
        }
        block_data.extend_from_slice(&primary_bitstream);

        // Collect the exceptions list.
        let mut best_exceptions_list = Vec::with_capacity(best_exception_count);
        for sub_block_index in 0..SUB_BLOCK_COUNT {
            let bit_width = best_bit_widths[sub_block_index];
            let sb_start = std::cmp::min(sub_block_index * SUB_BLOCK_SIZE, delta_count);
            let sb_end = std::cmp::min(
                sb_start
                    + if sub_block_index == SUB_BLOCK_COUNT - 1 {
                        LAST_SUB_BLOCK_SIZE
                    } else {
                        SUB_BLOCK_SIZE
                    },
                delta_count,
            );
            let limit = 1u32 << bit_width;
            for delta_index in sb_start..sb_end {
                let delta_minus_one = deltas_minus_one[delta_index];
                if delta_minus_one >= limit {
                    let val = (delta_minus_one >> bit_width) - 1;
                    best_exceptions_list.push((delta_index as u8, val));
                }
            }
        }
        assert_eq!(best_exceptions_list.len(), best_exception_count);

        // 3. Pack exception indices.
        if index_bitmask_flag == 1 {
            let mut bitmask = [0u8; EXCEPTION_INDEX_BITMASK_SIZE];
            for &(delta_index, _) in &best_exceptions_list {
                let idx = delta_index as usize;
                bitmask[idx / 8] |= 1 << (idx % 8);
            }
            block_data.extend_from_slice(&bitmask);
        } else {
            for &(delta_index, _) in &best_exceptions_list {
                block_data.push(delta_index);
            }
        }

        // 4. Pack exception values.
        match best_mode {
            EXCEPTION_MODE_U4 => {
                let num_u4_bytes = best_exception_count.div_ceil(2);
                for byte_pair_index in 0..num_u4_bytes {
                    let low_nibble_value = best_exceptions_list[2 * byte_pair_index].1;
                    let high_nibble_value = if 2 * byte_pair_index + 1 < best_exception_count {
                        best_exceptions_list[2 * byte_pair_index + 1].1
                    } else {
                        0
                    };
                    block_data.push((low_nibble_value | (high_nibble_value << 4)) as u8);
                }
            }
            EXCEPTION_MODE_U8 => {
                for exception_index in 0..best_exception_count {
                    block_data.push(best_exceptions_list[exception_index].1 as u8);
                }
            }
            EXCEPTION_MODE_U16 => {
                for exception_index in 0..best_exception_count {
                    block_data.extend_from_slice(
                        &(best_exceptions_list[exception_index].1 as u16).to_le_bytes(),
                    );
                }
            }
            EXCEPTION_MODE_U32 => {
                for exception_index in 0..best_exception_count {
                    block_data
                        .extend_from_slice(&best_exceptions_list[exception_index].1.to_le_bytes());
                }
            }
            _ => unreachable!(),
        }
    }

    // Construct the two-level sparse offset tables.
    let mut absolute_offsets = Vec::new();
    let mut relative_offsets = Vec::with_capacity(block_count);
    for block_index in 0..block_count {
        if block_index % S2PP_CHECKPOINT_INTERVAL == 0 {
            absolute_offsets.push(block_absolute_offsets[block_index]);
        }
        let checkpoint_index = block_index / S2PP_CHECKPOINT_INTERVAL;
        let checkpoint_offset = block_absolute_offsets[checkpoint_index * S2PP_CHECKPOINT_INTERVAL];
        let relative_offset = block_absolute_offsets[block_index] - checkpoint_offset;
        if relative_offset > u16::MAX as u32 {
            return Err(anyhow!(
                "relative offset overflow: {} exceeds {}",
                relative_offset,
                u16::MAX
            ));
        }
        relative_offsets.push(relative_offset as u16);
    }

    let database_path_ref = database_path.as_ref();
    let temporary_path = database_path_ref.with_extension("tmp");
    {
        let file = File::create(&temporary_path)?;
        let mut writer = BufWriter::new(file);

        writer.write_all(b"S2PP")?;
        writer.write_all(&(count as u32).to_le_bytes())?;
        writer.write_all(&(BLOCK_SIZE as u32).to_le_bytes())?;
        writer.write_all(&(block_count as u32).to_le_bytes())?;

        for &header in &block_headers {
            writer.write_all(&header.to_le_bytes())?;
        }
        for &absolute_offset in &absolute_offsets {
            writer.write_all(&absolute_offset.to_le_bytes())?;
        }
        for &relative_offset in &relative_offsets {
            writer.write_all(&relative_offset.to_le_bytes())?;
        }
        writer.write_all(&block_data)?;
        writer.flush()?;
    }

    std::fs::rename(&temporary_path, database_path_ref)?;

    println!(
        "Compressed database written in {:.2?}.",
        start_time.elapsed()
    );
    println!(
        "Successfully generated database file: '{}'",
        database_path_ref.display()
    );
    println!(
        "File Size: {:.3} MB",
        std::fs::metadata(database_path_ref)?.len() as f64 / (1024.0 * 1024.0)
    );
    Ok(())
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

/// Runs the database build pipeline to compile S2 population density database from GeoTIFF.
fn main() -> Result<()> {
    let arguments = Arguments::parse();
    let start_time = Instant::now();

    // 1. GeoTIFF reading and pixel aggregation.
    println!("Step 1: Reading GeoTIFF and aggregating population into level 12 S2 cells...");

    if !arguments.tiff_path.exists() {
        return Err(anyhow!(
            "input TIFF file '{}' not found.\n\
             Please refer to the Data source section in README.md for instructions on how to download the required GeoTIFF dataset.",
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

    let (tiff_path, _guard) = geotiff::prepare_geotiff(&arguments.tiff_path, "tmp.tif")?;

    let file = File::open(&tiff_path)?;
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
                    // Accumulate in fixed-point integer units so the parallel reduction is
                    // associative and the build is bit-for-bit reproducible regardless of the
                    // rayon work-stealing split (f64 addition is not associative).
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
            *cell_populations.entry(parent_id).or_insert(0) += population;
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

    let face_ids: Vec<u64> = (0..NUM_ROOT_FACES)
        .map(|face| ((face as u64) << S2_FACE_SHIFT) | (1u64 << LEVEL_0_SENTINEL_SHIFT))
        .collect();
    let mut leaves = Vec::new();

    for &face_id in &face_ids {
        find_leaves(face_id, 0, &cell_populations, &mut leaves);
    }

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

    write_s2pp_database(&arguments.database_path, &compact_leaves)?;
    println!("\nTotal Build Time: {:.2?}.", start_time.elapsed());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies that writing an empty set of compact leaves fails fast.
    #[test]
    fn test_write_s2pp_database_empty_leaves_fails_fast() {
        let database_path = PathBuf::from("scratch/test_empty_leaves.db");
        let result = write_s2pp_database(&database_path, &[]);
        assert!(result.is_err());
        let error_message = result.unwrap_err().to_string();
        assert!(error_message.contains("no valid population data found to build a database"));
    }

    /// Verifies that duplicate compact leaf IDs are rejected with a clear error.
    #[test]
    fn test_write_s2pp_database_duplicate_leaves_rejected() {
        let database_path = PathBuf::from("scratch/test_duplicate_leaves.db");
        let duplicate_leaves = vec![100, 100];
        let result = write_s2pp_database(&database_path, &duplicate_leaves);
        assert!(result.is_err());
        let error_message = result.unwrap_err().to_string();
        assert!(error_message.contains("duplicate compact leaf ID found"));
    }

    /// Verifies that out-of-order compact leaf IDs are rejected with a clear error.
    #[test]
    fn test_write_s2pp_database_out_of_order_leaves_rejected() {
        let database_path = PathBuf::from("scratch/test_out_of_order_leaves.db");
        let out_of_order_leaves = vec![100, 99];
        let result = write_s2pp_database(&database_path, &out_of_order_leaves);
        assert!(result.is_err());
        let error_message = result.unwrap_err().to_string();
        assert!(error_message.contains("compact leaves are out of order or duplicate"));
    }

    /// Verifies that a valid sequence of compact leaf IDs builds a database successfully.
    #[test]
    fn test_write_s2pp_database_success() {
        let database_path = PathBuf::from("scratch/test_valid_leaves.db");
        std::fs::create_dir_all("scratch").unwrap();
        let valid_leaves = vec![100, 105, 110];
        let result = write_s2pp_database(&database_path, &valid_leaves);
        assert!(result.is_ok());
        if database_path.exists() {
            let _ = std::fs::remove_file(&database_path);
        }
    }
}
