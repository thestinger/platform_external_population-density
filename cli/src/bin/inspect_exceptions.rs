//! Analyzes and prints statistics about outliers (exceptions) in the Patched Frame of Reference blocks.

use anyhow::Result;
use memmap2::Mmap;
use population_density::{
    BLOCK_HEADER_SIZE, EXCEPTION_INDEX_BITMASK_SIZE, EXCEPTION_MODE_U4, EXCEPTION_MODE_U8,
    EXCEPTION_MODE_U16, EXCEPTION_MODE_U32, LAST_SUB_BLOCK_SIZE, MAGIC_HEADER_SIZE,
    S2PP_CHECKPOINT_INTERVAL, SUB_BLOCK_COUNT, SUB_BLOCK_SIZE, block_offset, unpack_bit_widths,
};
use std::fs::File;
use std::path::Path;

fn main() -> Result<()> {
    let database_path = Path::new("population_density_database.bin");
    if !database_path.exists() {
        println!("Database file not found: {:?}", database_path);
        return Ok(());
    }

    let file = File::open(database_path)?;
    let mmap = unsafe { Mmap::map(&file)? };

    let count = u32::from_le_bytes(mmap[4..8].try_into()?) as usize;
    let block_size = u32::from_le_bytes(mmap[8..12].try_into()?) as usize;
    let block_count = u32::from_le_bytes(mmap[12..16].try_into()?) as usize;

    let headers_start = MAGIC_HEADER_SIZE;
    let headers_end = headers_start + block_count * 4;

    let checkpoint_count = block_count.div_ceil(S2PP_CHECKPOINT_INTERVAL);
    let absolute_offsets_start = headers_end;
    let absolute_offsets_end = absolute_offsets_start + checkpoint_count * 4;

    let relative_offsets_start = absolute_offsets_end;
    let relative_offsets_end = relative_offsets_start + block_count * 2;

    let block_data_start = relative_offsets_end;
    let total_data_length = mmap.len() - block_data_start;

    let absolute_offsets: &[u32] =
        bytemuck::cast_slice(&mmap[absolute_offsets_start..absolute_offsets_end]);
    let relative_offsets: &[u16] =
        bytemuck::cast_slice(&mmap[relative_offsets_start..relative_offsets_end]);

    let get_block_offset = |index: usize| block_offset(absolute_offsets, relative_offsets, index);

    let mut total_exceptions = 0;
    let mut total_blocks = 0;
    let mut non_empty_blocks = 0;
    // Map counts by the EXCEPTION_MODE_* index directly.
    let mut exception_modes_count = [0; 4];
    let mut total_primary_bytes = 0;
    let mut total_index_bytes = 0;
    let mut total_val_bytes = 0;
    let mut sum_bit_widths = 0;

    for block_index in 0..block_count {
        let start_offset = get_block_offset(block_index)?;
        let end_offset = if block_index + 1 < block_count {
            get_block_offset(block_index + 1)?
        } else {
            total_data_length
        };
        let block_length = if block_index + 1 < block_count {
            block_size
        } else {
            count - block_index * block_size
        };

        total_blocks += 1;
        if block_length > 1 {
            non_empty_blocks += 1;
            let start = block_data_start + start_offset;
            let end = block_data_start + end_offset;
            let block_bytes = &mmap[start..end];
            if block_bytes.len() >= BLOCK_HEADER_SIZE {
                let exception_count = block_bytes[BLOCK_HEADER_SIZE - 2] as usize;
                let mode_and_flag = block_bytes[BLOCK_HEADER_SIZE - 1];
                let exception_mode = (mode_and_flag & 0x03) as usize;
                let index_bitmask_flag = (mode_and_flag & 0x04) != 0;
                total_exceptions += exception_count;

                exception_modes_count[exception_mode] += 1;

                // Decode bit widths to calculate primary bitstream size.
                let header_bytes = &block_bytes[0..BLOCK_HEADER_SIZE];
                let bit_widths = unpack_bit_widths(header_bytes);
                let mut total_bits = 0;
                let delta_count = block_length - 1;
                for (sub_block_index, &bit_width_value) in bit_widths.iter().enumerate() {
                    let bit_width = bit_width_value as usize;
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
                    total_bits += (sb_end - sb_start) * bit_width;
                }
                let primary_bytes = total_bits.div_ceil(8);
                let index_bytes = if index_bitmask_flag {
                    EXCEPTION_INDEX_BITMASK_SIZE
                } else {
                    exception_count
                };
                let val_bytes = match exception_mode as u8 {
                    EXCEPTION_MODE_U4 => exception_count.div_ceil(2),
                    EXCEPTION_MODE_U8 => exception_count * std::mem::size_of::<u8>(),
                    EXCEPTION_MODE_U16 => exception_count * std::mem::size_of::<u16>(),
                    EXCEPTION_MODE_U32 => exception_count * std::mem::size_of::<u32>(),
                    _ => unreachable!(),
                };
                total_primary_bytes += primary_bytes;
                total_index_bytes += index_bytes;
                total_val_bytes += val_bytes;
                for &bw in &bit_widths {
                    sum_bit_widths += bw as usize;
                }
            }
        }
    }

    println!("Database Statistics:");
    println!("--------------------------------------------------");
    println!("Total Leaf Cells (elements): {}", count);
    println!("Block Size: {}", block_size);
    println!("Total Blocks: {}", total_blocks);
    println!("Non-empty Blocks (length > 1): {}", non_empty_blocks);
    println!("Total Outliers (exceptions): {}", total_exceptions);
    println!(
        "Average outliers per non-empty block: {:.4}",
        total_exceptions as f64 / non_empty_blocks as f64
    );
    println!("Exception Modes Distribution (Blocks):");
    println!(
        "  U4 mode blocks:  {}",
        exception_modes_count[EXCEPTION_MODE_U4 as usize]
    );
    println!(
        "  U8 mode blocks:  {}",
        exception_modes_count[EXCEPTION_MODE_U8 as usize]
    );
    println!(
        "  U16 mode blocks: {}",
        exception_modes_count[EXCEPTION_MODE_U16 as usize]
    );
    println!(
        "  U32 mode blocks: {}",
        exception_modes_count[EXCEPTION_MODE_U32 as usize]
    );
    println!("Sizes of block components:");
    println!(
        "  Total Header Bytes:     {}",
        non_empty_blocks * BLOCK_HEADER_SIZE
    );
    println!("  Total Primary Bytes:    {}", total_primary_bytes);
    println!("  Total Index Bytes:      {}", total_index_bytes);
    println!("  Total Value Bytes:      {}", total_val_bytes);
    println!(
        "  Sum of all parts:       {}",
        non_empty_blocks * BLOCK_HEADER_SIZE
            + total_primary_bytes
            + total_index_bytes
            + total_val_bytes
    );
    println!(
        "  Average bit_width:      {:.4}",
        sum_bit_widths as f64 / (non_empty_blocks * SUB_BLOCK_COUNT) as f64
    );
    println!("--------------------------------------------------");

    Ok(())
}
