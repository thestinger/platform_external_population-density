//! Provides a highly optimized query engine and utility functions for the block-compressed S2 population density database.

use anyhow::{Result, anyhow};
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;

pub use memmap2;

// Define S2 geometry and database constants.
pub const MAX_S2_LEVEL: u32 = 30;
pub const BITS_PER_LEVEL: u32 = 2;
pub const MAX_S2_BITS: u32 = 60;
pub const MAX_DB_LEVEL: u32 = 12;
pub const BLOCK_SIZE: usize = 256;
/// Minimum number of people a returned S2 cell must represent for privacy coarsening.
pub const POPULATION_THRESHOLD: i64 = 1000;
/// Fixed-point scale (units per person) for deterministic integer population accumulation.
pub const POPULATION_FIXED_POINT_SCALE: i64 = 1 << 20;
/// Population threshold expressed in [`POPULATION_FIXED_POINT_SCALE`] fixed-point units.
pub const POPULATION_THRESHOLD_FIXED: i64 = POPULATION_THRESHOLD * POPULATION_FIXED_POINT_SCALE;
pub const NUM_ROOT_FACES: usize = 6;
pub const SHIFT_COMPACT: u32 = 36;
pub const MAX_DELTAS_CAPACITY: usize = 255;
pub const MAGIC_HEADER_SIZE: usize = 16;
/// Defines the maximum valid compact S2 cell ID limit.
pub const MAX_COMPACT_CELL_ID: u32 = 0x0FFFFFFF;

pub const BLOCK_HEADER_SIZE: usize = 12;
pub const SUB_BLOCK_COUNT: usize = 16;
pub const SUB_BLOCK_SIZE: usize = 16;
pub const LAST_SUB_BLOCK_SIZE: usize = 15;
pub const SUB_BLOCK_BIT_WIDTH_BITS: usize = 5;
pub const SUB_BLOCK_BIT_WIDTH_MASK: u8 = 0x1F;

pub const EXCEPTION_MODE_U4: u8 = 0;
pub const EXCEPTION_MODE_U8: u8 = 1;
pub const EXCEPTION_MODE_U16: u8 = 2;
pub const EXCEPTION_MODE_U32: u8 = 3;
pub const EXCEPTION_MODE_MASK: u8 = 0x03;

pub const INDEX_BITMASK_FLAG_MASK: u8 = 0x04;
pub const EXCEPTION_INDEX_BITMASK_SIZE: usize = 32;

pub const S2_FACE_BITS: u32 = 3;
/// Defines the bit position of the 3-bit S2 face field within a 64-bit cell ID.
pub const S2_FACE_SHIFT: u32 = 61;
pub const S2PP_CHECKPOINT_INTERVAL: usize = 64;
pub const MAX_PFOR_BIT_WIDTH: u8 = 28;

/// Calculates the S2 level of any S2 cell ID using trailing zeros.
#[allow(clippy::manual_is_multiple_of)]
pub fn get_level(s2_cell_id: u64) -> u32 {
    assert!(s2_cell_id != 0, "S2 cell ID cannot be 0");
    let face = s2_cell_id >> S2_FACE_SHIFT;
    assert!(face < NUM_ROOT_FACES as u64, "Invalid S2 face ID: {}", face);
    let lowest_bit = s2_cell_id & s2_cell_id.wrapping_neg();
    let trailing_zeros = lowest_bit.trailing_zeros();
    assert!(
        trailing_zeros <= MAX_S2_BITS,
        "Invalid S2 cell ID trailing zeros: {}",
        trailing_zeros
    );
    assert!(
        trailing_zeros % 2 == 0,
        "S2 cell ID lowest bit must be at even index: {}",
        trailing_zeros
    );
    (MAX_S2_BITS - trailing_zeros) / BITS_PER_LEVEL
}

/// Safely calculates the S2 level of any S2 cell ID using trailing zeros without panicking.
#[allow(clippy::manual_is_multiple_of)]
pub fn try_get_level(s2_cell_id: u64) -> Result<u32> {
    if s2_cell_id == 0 {
        return Err(anyhow!("S2 cell ID cannot be 0"));
    }
    let face = s2_cell_id >> S2_FACE_SHIFT;
    if face >= NUM_ROOT_FACES as u64 {
        return Err(anyhow!("invalid S2 face ID: {}", face));
    }
    let lowest_bit = s2_cell_id & s2_cell_id.wrapping_neg();
    let trailing_zeros = lowest_bit.trailing_zeros();
    if trailing_zeros > MAX_S2_BITS {
        return Err(anyhow!(
            "invalid S2 cell ID trailing zeros: {}",
            trailing_zeros
        ));
    }
    if trailing_zeros % 2 != 0 {
        return Err(anyhow!(
            "S2 cell ID lowest bit must be at even index: {}",
            trailing_zeros
        ));
    }
    Ok((MAX_S2_BITS - trailing_zeros) / BITS_PER_LEVEL)
}

/// Computes the parent/ancestor of an S2 cell ID at the specified level.
pub fn get_ancestor(s2_cell_id: u64, level: u32) -> u64 {
    let cell_level = get_level(s2_cell_id);
    assert!(
        level <= cell_level,
        "Level {} must be <= cell level {}",
        level,
        cell_level
    );
    assert!(level <= MAX_S2_LEVEL, "S2 level must be <= 30");
    let sentinel_position = MAX_S2_BITS - BITS_PER_LEVEL * level;
    let mask = !((1u64 << sentinel_position) - 1);
    (s2_cell_id & mask) | (1u64 << sentinel_position)
}

/// Safely computes the parent/ancestor of an S2 cell ID at the specified level without panicking.
pub fn try_get_ancestor(s2_cell_id: u64, level: u32) -> Result<u64> {
    let cell_level = try_get_level(s2_cell_id)?;
    if level > cell_level {
        return Err(anyhow!(
            "level {} is greater than cell level {}",
            level,
            cell_level
        ));
    }
    if level > MAX_S2_LEVEL {
        return Err(anyhow!("S2 level must be <= 30"));
    }
    let sentinel_position = MAX_S2_BITS - BITS_PER_LEVEL * level;
    let mask = !((1u64 << sentinel_position) - 1);
    Ok((s2_cell_id & mask) | (1u64 << sentinel_position))
}

/// Computes the bitwise S2 parent ID at the specified level (used by database builder).
pub fn get_parent_id(cell_id: u64, level: u32) -> u64 {
    get_ancestor(cell_id, level)
}

/// Computes the bitwise S2 children IDs at the specified level (used by database builder).
pub fn get_children_ids(cell_id: u64, level: u32) -> [u64; 4] {
    assert!(cell_id != 0, "S2 cell ID cannot be 0");
    let face = cell_id >> S2_FACE_SHIFT;
    assert!(face < NUM_ROOT_FACES as u64, "Invalid S2 face ID: {}", face);
    assert!(
        level < MAX_S2_LEVEL,
        "S2 level must be < 30 to have children"
    );
    let sentinel_position = MAX_S2_BITS - BITS_PER_LEVEL * level;
    // Prevent underflow by ensuring cell ID is at least the sentinel value.
    assert!(
        cell_id >= (1u64 << sentinel_position),
        "S2 cell ID must be greater than or equal to sentinel: cell_id = {}, sentinel = {}",
        cell_id,
        1u64 << sentinel_position
    );
    let prefix = cell_id & !(1u64 << sentinel_position);
    [
        prefix | (0u64 << (sentinel_position - 1)) | (1u64 << (sentinel_position - 2)),
        prefix | (1u64 << (sentinel_position - 1)) | (1u64 << (sentinel_position - 2)),
        prefix | (2u64 << (sentinel_position - 1)) | (1u64 << (sentinel_position - 2)),
        prefix | (3u64 << (sentinel_position - 1)) | (1u64 << (sentinel_position - 2)),
    ]
}

/// Implements a highly-efficient memory-mapped query engine for the block-compressed population density S2 quadtree database.
#[derive(Debug)]
pub struct QueryEngine {
    mmap: Mmap,
    count: usize,
    block_size: usize,
    block_count: usize,
    block_data_start: usize,
    total_data_length: usize,
    headers_range: std::ops::Range<usize>,
    absolute_offsets_range: std::ops::Range<usize>,
    relative_offsets_range: std::ops::Range<usize>,
}

/// Unpacks the per-sub-block PFOR bit-widths packed into a block header.
///
/// The caller must pass exactly the [`BLOCK_HEADER_SIZE`]-byte block header.
pub fn unpack_bit_widths(header_bytes: &[u8]) -> [u8; SUB_BLOCK_COUNT] {
    let mut bit_widths = [0u8; SUB_BLOCK_COUNT];
    for (sub_block_index, bit_width_slot) in bit_widths.iter_mut().enumerate() {
        let bit_offset = sub_block_index * SUB_BLOCK_BIT_WIDTH_BITS;
        let byte_idx = bit_offset / 8;
        let bit_idx = bit_offset % 8;
        let val = if byte_idx + 1 < BLOCK_HEADER_SIZE {
            u16::from_le_bytes([header_bytes[byte_idx], header_bytes[byte_idx + 1]])
        } else {
            header_bytes[byte_idx] as u16
        };
        *bit_width_slot = ((val >> bit_idx) & SUB_BLOCK_BIT_WIDTH_MASK as u16) as u8;
    }
    bit_widths
}

/// Computes a block's byte offset from the checkpointed absolute offset table and
/// the per-block relative offset table.
pub fn block_offset(
    absolute_offsets: &[u32],
    relative_offsets: &[u16],
    index: usize,
) -> Result<usize> {
    let checkpoint_index = index / S2PP_CHECKPOINT_INTERVAL;
    let absolute_value = absolute_offsets
        .get(checkpoint_index)
        .copied()
        .ok_or_else(|| {
            anyhow!(
                "checkpoint index {} out of bounds for absolute offsets",
                checkpoint_index
            )
        })? as usize;

    let relative_value = relative_offsets
        .get(index)
        .copied()
        .ok_or_else(|| anyhow!("block index {} out of bounds for relative offsets", index))?
        as usize;

    absolute_value.checked_add(relative_value).ok_or_else(|| {
        anyhow!(
            "block offset overflow: absolute {} + relative {}",
            absolute_value,
            relative_value
        )
    })
}

impl QueryEngine {
    /// Opens the database file and memory-maps it.
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let metadata = file.metadata()?;
        if (metadata.len() as usize) < MAGIC_HEADER_SIZE {
            return Err(anyhow!("database file too small to contain a valid header"));
        }
        // SAFETY: Memory mapping a file is safe here because the file is opened in read-only mode and is not mutated by other processes concurrently during the lifetime of this process.
        let mmap = unsafe { Mmap::map(&file)? };
        Self::from_mmap(mmap)
    }

    /// Initializes the query engine from an existing memory mapping.
    pub fn from_mmap(mmap: Mmap) -> Result<Self> {
        if mmap.len() < MAGIC_HEADER_SIZE {
            return Err(anyhow!("database file too small to contain a valid header"));
        }

        let magic = &mmap[0..4];
        if magic != b"S2PP" {
            return Err(anyhow!("invalid magic bytes in database: {:?}", magic));
        }

        let count = u32::from_le_bytes(mmap[4..8].try_into()?) as usize;
        let block_size = u32::from_le_bytes(mmap[8..12].try_into()?) as usize;
        if block_size == 0 {
            return Err(anyhow!(
                "invalid database: block_size must be greater than 0"
            ));
        }
        if block_size > MAX_DELTAS_CAPACITY + 1 {
            return Err(anyhow!("invalid block size in database: {}", block_size));
        }
        let block_count = u32::from_le_bytes(mmap[12..16].try_into()?) as usize;
        if block_count == 0 {
            return Err(anyhow!(
                "invalid database: block_count must be greater than 0"
            ));
        }

        if count == 0 {
            return Err(anyhow!("invalid database: count must be greater than 0"));
        }
        let expected_block_count = count.div_ceil(block_size);
        if block_count != expected_block_count {
            return Err(anyhow!(
                "invalid database: inconsistent block_count {} (expected {})",
                block_count,
                expected_block_count
            ));
        }

        let headers_start = MAGIC_HEADER_SIZE;
        let headers_end = headers_start
            .checked_add(
                block_count
                    .checked_mul(std::mem::size_of::<u32>())
                    .ok_or_else(|| anyhow!("overflow calculating headers size"))?,
            )
            .ok_or_else(|| anyhow!("overflow calculating headers end position"))?;

        let checkpoint_count = block_count.div_ceil(S2PP_CHECKPOINT_INTERVAL);
        let absolute_offsets_start = headers_end;
        let absolute_offsets_end = absolute_offsets_start
            .checked_add(
                checkpoint_count
                    .checked_mul(std::mem::size_of::<u32>())
                    .ok_or_else(|| anyhow!("overflow calculating absolute offsets size"))?,
            )
            .ok_or_else(|| anyhow!("overflow calculating absolute offsets end position"))?;

        let relative_offsets_start = absolute_offsets_end;
        let relative_offsets_end = relative_offsets_start
            .checked_add(
                block_count
                    .checked_mul(std::mem::size_of::<u16>())
                    .ok_or_else(|| anyhow!("overflow calculating relative offsets size"))?,
            )
            .ok_or_else(|| anyhow!("overflow calculating relative offsets end position"))?;

        let block_data_start = relative_offsets_end;

        if mmap.len() < relative_offsets_end {
            return Err(anyhow!("database file truncated"));
        }

        let total_data_length = mmap.len() - block_data_start;

        let headers_range = headers_start..headers_end;
        let absolute_offsets_range = absolute_offsets_start..absolute_offsets_end;
        let relative_offsets_range = relative_offsets_start..relative_offsets_end;

        // Validate slice alignment and structure using try_cast_slice. The u32/u16 offset and
        // header tables require the mapping base to be 4-byte aligned. For QueryEngine::new the
        // base is page-aligned (file offset 0). For the JNI path, memmap2 anchors the mapping at
        // the requested asset offset, so the base alignment mirrors that offset modulo 4: a
        // non-4-aligned asset offset makes these casts fail here, which is a fail-closed init
        // error (never UB or a wrong result). This relies on the packager 4-aligning the
        // uncompressed .bin asset (the -0 .bin aaptflag controls compression, not alignment).
        let headers_slice = bytemuck::try_cast_slice::<u8, u32>(&mmap[headers_range.clone()])
            .map_err(|error| anyhow!("failed to cast headers slice: {}", error))?;

        let mut last_header = None;
        for &header in headers_slice {
            if header > MAX_COMPACT_CELL_ID {
                return Err(anyhow!(
                    "invalid database: header cell ID {:X} exceeds maximum compact S2 cell ID limit",
                    header
                ));
            }
            // No try_get_level validation here to support arbitrary mock database headers.
            if last_header.is_some_and(|prev| header <= prev) {
                return Err(anyhow!(
                    "invalid database: headers must be strictly monotonically increasing (found {:X} after {:X})",
                    header,
                    last_header.unwrap()
                ));
            }
            last_header = Some(header);
        }
        let absolute_offsets =
            bytemuck::try_cast_slice::<u8, u32>(&mmap[absolute_offsets_range.clone()])
                .map_err(|error| anyhow!("failed to cast absolute offsets slice: {}", error))?;
        let relative_offsets =
            bytemuck::try_cast_slice::<u8, u16>(&mmap[relative_offsets_range.clone()])
                .map_err(|error| anyhow!("failed to cast relative offsets slice: {}", error))?;

        // Validate offset monotonicity, physical bounds, and terminal constraints.
        let mut last_offset = None;
        for block_index in 0..block_count {
            let checkpoint_index = block_index / S2PP_CHECKPOINT_INTERVAL;
            let absolute_base =
                absolute_offsets
                    .get(checkpoint_index)
                    .copied()
                    .ok_or_else(|| {
                        anyhow!(
                            "corrupt database: absolute offset checkpoint {} out of bounds",
                            checkpoint_index
                        )
                    })?;
            let relative_offset = relative_offsets.get(block_index).copied().ok_or_else(|| {
                anyhow!(
                    "corrupt database: relative offset index {} out of bounds",
                    block_index
                )
            })?;
            let absolute_offset = (absolute_base as usize)
                .checked_add(relative_offset as usize)
                .ok_or_else(|| {
                    anyhow!(
                        "corrupt database: block data offset overflow at block {}",
                        block_index
                    )
                })?;

            if absolute_offset > total_data_length {
                return Err(anyhow!(
                    "corrupt database: block {} offset {} exceeds physical file bounds {}",
                    block_index,
                    absolute_offset,
                    total_data_length
                ));
            }

            if block_index == 0 && absolute_offset != 0 {
                return Err(anyhow!(
                    "corrupt database: first block offset must be 0, found {}",
                    absolute_offset
                ));
            }

            if let Some(prev) = last_offset
                && absolute_offset <= prev
            {
                return Err(anyhow!(
                    "corrupt database: block offsets must be strictly monotonic (found {} after {})",
                    absolute_offset,
                    prev
                ));
            }
            last_offset = Some(absolute_offset);
        }

        if let Some(last) = last_offset {
            let last_block_is_single = (count % block_size) == 1;
            if last_block_is_single {
                if last != total_data_length {
                    return Err(anyhow!(
                        "corrupt database: terminal block size mismatch. Offset {} must equal physical data length {}",
                        last,
                        total_data_length
                    ));
                }
            } else {
                if last >= total_data_length {
                    return Err(anyhow!(
                        "corrupt database: terminal block offset {} must be less than physical data length {}",
                        last,
                        total_data_length
                    ));
                }
            }
        }

        Ok(Self {
            mmap,
            count,
            block_size,
            block_count,
            block_data_start,
            total_data_length,
            headers_range,
            absolute_offsets_range,
            relative_offsets_range,
        })
    }

    /// Returns a read-only slice of the database block headers.
    #[inline]
    fn headers(&self) -> &[u32] {
        bytemuck::cast_slice(&self.mmap[self.headers_range.clone()])
    }

    /// Returns a read-only slice of the absolute block offsets.
    #[inline]
    fn absolute_offsets(&self) -> &[u32] {
        bytemuck::cast_slice(&self.mmap[self.absolute_offsets_range.clone()])
    }

    /// Returns a read-only slice of the relative block offsets.
    #[inline]
    fn relative_offsets(&self) -> &[u16] {
        bytemuck::cast_slice(&self.mmap[self.relative_offsets_range.clone()])
    }

    /// Returns the total number of cell IDs stored in the database.
    #[inline]
    pub fn count(&self) -> usize {
        self.count
    }

    /// Returns the capacity size of each compressed block in the database.
    #[inline]
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Returns the total number of compressed blocks in the database.
    #[inline]
    pub fn block_count(&self) -> usize {
        self.block_count
    }

    #[inline]
    fn get_block_offset(&self, index: usize) -> Result<usize> {
        block_offset(self.absolute_offsets(), self.relative_offsets(), index)
    }

    #[inline]
    fn get_header_value(&self, index: usize) -> u32 {
        let slice = self.headers();
        slice[index]
    }

    /// Decodes the bit-widths and exception table shared by [`Self::query`] and
    /// [`Self::reconstruct_all_leaves`], guaranteeing both paths interpret a block
    /// identically. `delta_count` is the number of deltas in the block (block length
    /// minus its header element).
    ///
    /// The exception tables are written through `&mut` out-params instead of being
    /// returned by value: returning the ~1.5 KB `[u16; BLOCK_SIZE]`/`[u32; BLOCK_SIZE]`
    /// pair is not elided and regressed query throughput by ~20% in the hot path, so
    /// they are filled in place. Do not convert this back to a by-value return.
    fn decode_block_metadata(
        block_bytes: &[u8],
        delta_count: usize,
        exception_indices: &mut [u16; BLOCK_SIZE],
        exception_values: &mut [u32; BLOCK_SIZE],
    ) -> Result<([u8; SUB_BLOCK_COUNT], usize)> {
        if block_bytes.len() < BLOCK_HEADER_SIZE {
            return Err(anyhow!("corrupt database block: truncated block header"));
        }
        let header_bytes = &block_bytes[0..BLOCK_HEADER_SIZE];

        let bit_widths = unpack_bit_widths(header_bytes);

        let exception_count = header_bytes[BLOCK_HEADER_SIZE - 2] as usize;
        let mode_and_flag = header_bytes[BLOCK_HEADER_SIZE - 1];
        let exception_mode = mode_and_flag & EXCEPTION_MODE_MASK;
        let index_bitmask_flag = (mode_and_flag & INDEX_BITMASK_FLAG_MASK) != 0;

        // Validate bit-widths.
        for &bit_width in &bit_widths {
            if bit_width > MAX_PFOR_BIT_WIDTH {
                return Err(anyhow!(
                    "corrupt database block: bit_width {} exceeds maximum limit {}",
                    bit_width,
                    MAX_PFOR_BIT_WIDTH
                ));
            }
        }

        // Compute exact exception index position.
        let mut total_bits = 0;
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
        let packed_bytes = total_bits.div_ceil(8);
        let exception_index_position = BLOCK_HEADER_SIZE + packed_bytes;

        // Decode exception indices.
        let mut decoded_idx_count = 0;
        if index_bitmask_flag {
            let bitmask_end = exception_index_position + EXCEPTION_INDEX_BITMASK_SIZE;
            if bitmask_end > block_bytes.len() {
                return Err(anyhow!(
                    "corrupt database block: truncated exception bitmask"
                ));
            }
            let bitmask = &block_bytes[exception_index_position..bitmask_end];
            for delta_index in 0..delta_count {
                if (bitmask[delta_index / 8] & (1 << (delta_index % 8))) != 0 {
                    exception_indices[decoded_idx_count] = delta_index as u16;
                    decoded_idx_count += 1;
                }
            }
        } else {
            let index_end = exception_index_position + exception_count;
            if index_end > block_bytes.len() {
                return Err(anyhow!(
                    "corrupt database block: truncated exception indices"
                ));
            }
            let index_slice = &block_bytes[exception_index_position..index_end];
            // Exception indices must be strictly increasing so the single forward scan
            // in query() consumes every exception; reject out-of-order indices that
            // query() would otherwise silently drop, diverging from reconstruct.
            let mut previous_index: Option<u8> = None;
            for &idx in index_slice {
                if idx as usize >= delta_count {
                    return Err(anyhow!(
                        "corrupt database block exception index {} out of range",
                        idx
                    ));
                }
                if previous_index.is_some_and(|prev| idx <= prev) {
                    return Err(anyhow!(
                        "corrupt database block: exception indices must be strictly increasing (found {} after {})",
                        idx,
                        previous_index.unwrap()
                    ));
                }
                previous_index = Some(idx);
                exception_indices[decoded_idx_count] = idx as u16;
                decoded_idx_count += 1;
            }
        }

        if decoded_idx_count != exception_count {
            return Err(anyhow!(
                "corrupt database block: exception index count {} does not match expected {}",
                decoded_idx_count,
                exception_count
            ));
        }

        // Decode exception values.
        let exception_value_position = exception_index_position
            + if index_bitmask_flag {
                EXCEPTION_INDEX_BITMASK_SIZE
            } else {
                exception_count
            };
        let bytes_per_value = match exception_mode {
            EXCEPTION_MODE_U4 => 0,
            EXCEPTION_MODE_U8 => std::mem::size_of::<u8>(),
            EXCEPTION_MODE_U16 => std::mem::size_of::<u16>(),
            EXCEPTION_MODE_U32 => std::mem::size_of::<u32>(),
            _ => return Err(anyhow!("invalid exception mode")),
        };
        let exception_value_bytes = if exception_mode == EXCEPTION_MODE_U4 {
            exception_count.div_ceil(2)
        } else {
            exception_count * bytes_per_value
        };

        if exception_value_position + exception_value_bytes > block_bytes.len() {
            return Err(anyhow!(
                "corrupt database block: truncated exception values"
            ));
        }

        match exception_mode {
            EXCEPTION_MODE_U4 => {
                let val_slice = &block_bytes[exception_value_position
                    ..exception_value_position + exception_count.div_ceil(2)];
                for value_index in 0..exception_count {
                    let byte = val_slice[value_index / 2];
                    let val = if value_index % 2 == 0 {
                        byte & 0x0F
                    } else {
                        byte >> 4
                    };
                    exception_values[value_index] = val as u32;
                }
            }
            EXCEPTION_MODE_U8 => {
                let val_slice = &block_bytes
                    [exception_value_position..exception_value_position + exception_count];
                for (value_index, &val) in val_slice.iter().enumerate() {
                    exception_values[value_index] = val as u32;
                }
            }
            EXCEPTION_MODE_U16 => {
                let val_slice = &block_bytes
                    [exception_value_position..exception_value_position + exception_count * 2];
                for value_index in 0..exception_count {
                    let val = u16::from_le_bytes([
                        val_slice[value_index * 2],
                        val_slice[value_index * 2 + 1],
                    ]);
                    exception_values[value_index] = val as u32;
                }
            }
            EXCEPTION_MODE_U32 => {
                let val_slice = &block_bytes
                    [exception_value_position..exception_value_position + exception_count * 4];
                for value_index in 0..exception_count {
                    let val = u32::from_le_bytes([
                        val_slice[value_index * 4],
                        val_slice[value_index * 4 + 1],
                        val_slice[value_index * 4 + 2],
                        val_slice[value_index * 4 + 3],
                    ]);
                    exception_values[value_index] = val;
                }
            }
            _ => unreachable!(),
        }

        Ok((bit_widths, exception_count))
    }

    /// Finds the deepest ancestor S2 ID in levels 0-12 (inclusive) that contains at least [`POPULATION_THRESHOLD`] people.
    pub fn query(&self, s2_cell_id: u64) -> Result<u64> {
        if s2_cell_id == 0 {
            return Err(anyhow!("S2 cell ID cannot be 0"));
        }
        let query_level = try_get_level(s2_cell_id)
            .map_err(|error| anyhow!("invalid query S2 cell ID: {}", error))?;
        let query_cell_id = if query_level > MAX_DB_LEVEL {
            try_get_ancestor(s2_cell_id, MAX_DB_LEVEL)?
        } else {
            s2_cell_id
        };
        let compact_query_id = (query_cell_id >> SHIFT_COMPACT) as u32;

        // 1. Memory-based sparse search.
        let mut low = 0;
        let mut high = self.block_count;
        let mut block_index = None;
        let mut block_header = 0u32;

        let headers = self.headers();

        while low < high {
            let mid = low + (high - low) / 2;
            let header_value = headers[mid];
            if header_value <= compact_query_id {
                block_index = Some(mid);
                block_header = header_value;
                low = mid + 1;
            } else {
                high = mid;
            }
        }

        let mut compact_left = -1isize;
        let mut compact_right = -1isize;

        if let Some(index) = block_index {
            compact_left = block_header as isize;

            let start_offset = self.get_block_offset(index)?;
            let end_offset = if index + 1 < self.block_count {
                self.get_block_offset(index + 1)?
            } else {
                self.total_data_length
            };

            if start_offset > end_offset || end_offset > self.total_data_length {
                return Err(anyhow!("corrupt database block offsets out of bounds"));
            }

            let block_length = if index + 1 < self.block_count {
                self.block_size
            } else {
                self.count.checked_sub(index * self.block_size)
                    .ok_or_else(|| anyhow!("corrupt database: block count/size inconsistency during block length calculation"))?
            };

            if block_length == 0 {
                return Err(anyhow!("corrupt database: block length is 0"));
            }

            let start = self.block_data_start + start_offset;
            let end = self.block_data_start + end_offset;
            let block_bytes = &self.mmap[start..end];

            if block_length > 1 {
                let delta_count = block_length - 1;
                let mut current_value = block_header;
                if current_value > MAX_COMPACT_CELL_ID {
                    return Err(anyhow!(
                        "corrupt database block: cell ID {:X} exceeds maximum compact S2 cell ID limit",
                        current_value
                    ));
                }

                let mut exception_indices = [0u16; BLOCK_SIZE];
                let mut exception_values = [0u32; BLOCK_SIZE];
                let (bit_widths, exception_count) = Self::decode_block_metadata(
                    block_bytes,
                    delta_count,
                    &mut exception_indices,
                    &mut exception_values,
                )?;

                if current_value <= compact_query_id {
                    compact_left = current_value as isize;

                    let mut bit_buffer = 0u64;
                    let mut bit_count = 0;
                    let mut byte_index = BLOCK_HEADER_SIZE;
                    let mut exc_ptr = 0;
                    let mut early_break = false;

                    for (sub_block_index, &bit_width_value) in bit_widths.iter().enumerate() {
                        let bit_width = bit_width_value as u32;
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

                        if bit_width == 0 {
                            for delta_index in sb_start..sb_end {
                                let mut delta_value = 0u32;
                                if exc_ptr < exception_count
                                    && delta_index == exception_indices[exc_ptr] as usize
                                {
                                    let exc_val = exception_values[exc_ptr];
                                    exc_ptr += 1;
                                    if (exc_val as u64 + 1)
                                        .checked_shl(bit_width)
                                        .is_none_or(|val| val > u32::MAX as u64)
                                    {
                                        return Err(anyhow!(
                                            "corrupt database block: exception value overflow"
                                        ));
                                    }
                                    delta_value |= (exc_val + 1) << bit_width;
                                }
                                let delta = delta_value.checked_add(1).ok_or_else(|| {
                                    anyhow!("corrupt database block: delta value overflow")
                                })?;
                                current_value =
                                    current_value.checked_add(delta).ok_or_else(|| {
                                        anyhow!(
                                            "corrupt database block: compact S2 cell ID overflow"
                                        )
                                    })?;
                                if current_value > MAX_COMPACT_CELL_ID {
                                    return Err(anyhow!(
                                        "corrupt database block: cell ID {:X} exceeds maximum compact S2 cell ID limit",
                                        current_value
                                    ));
                                }
                                if current_value <= compact_query_id {
                                    compact_left = current_value as isize;
                                } else {
                                    compact_right = current_value as isize;
                                    early_break = true;
                                    break;
                                }
                            }
                            if early_break {
                                break;
                            }
                            continue;
                        }

                        for delta_index in sb_start..sb_end {
                            if bit_count < bit_width {
                                if byte_index + 4 <= block_bytes.len() {
                                    let bytes_u32 = u32::from_le_bytes(
                                        block_bytes[byte_index..byte_index + 4].try_into().unwrap(),
                                    );
                                    bit_buffer |= (bytes_u32 as u64) << bit_count;
                                    byte_index += 4;
                                    bit_count += 32;
                                } else {
                                    while bit_count < bit_width {
                                        if byte_index < block_bytes.len() {
                                            bit_buffer |=
                                                (block_bytes[byte_index] as u64) << bit_count;
                                            byte_index += 1;
                                            bit_count += 8;
                                        } else {
                                            break;
                                        }
                                    }
                                }
                            }

                            if bit_count < bit_width {
                                return Err(anyhow!("corrupt database block: truncated bitstream"));
                            }

                            let mut delta_value = (bit_buffer & ((1u64 << bit_width) - 1)) as u32;
                            bit_buffer >>= bit_width;
                            bit_count -= bit_width;

                            if exc_ptr < exception_count
                                && delta_index == exception_indices[exc_ptr] as usize
                            {
                                let exc_val = exception_values[exc_ptr];
                                exc_ptr += 1;
                                if (exc_val as u64 + 1)
                                    .checked_shl(bit_width)
                                    .is_none_or(|val| val > u32::MAX as u64)
                                {
                                    return Err(anyhow!(
                                        "corrupt database block: exception value overflow"
                                    ));
                                }
                                delta_value |= (exc_val + 1) << bit_width;
                            }

                            let delta = delta_value.checked_add(1).ok_or_else(|| {
                                anyhow!("corrupt database block: delta value overflow")
                            })?;
                            current_value = current_value.checked_add(delta).ok_or_else(|| {
                                anyhow!("corrupt database block: compact S2 cell ID overflow")
                            })?;
                            if current_value > MAX_COMPACT_CELL_ID {
                                return Err(anyhow!(
                                    "corrupt database block: cell ID {:X} exceeds maximum compact S2 cell ID limit",
                                    current_value
                                ));
                            }
                            if current_value <= compact_query_id {
                                compact_left = current_value as isize;
                            } else {
                                compact_right = current_value as isize;
                                early_break = true;
                                break;
                            }
                        }
                        if early_break {
                            break;
                        }
                    }
                } else {
                    compact_right = current_value as isize;
                }
            }

            if compact_right == -1 && index + 1 < self.block_count {
                compact_right = self.get_header_value(index + 1) as isize;
            }
        } else {
            compact_right = self.get_header_value(0) as isize;
        }

        // 2. Ancestor matching logic.
        let mut best_ancestor = 0u64;
        let mut best_level = -1i32;

        if compact_left > 0 {
            let database_cell_id = (compact_left as u64) << SHIFT_COMPACT;
            let xor_value = s2_cell_id ^ database_cell_id;
            let leading_zeros = xor_value.leading_zeros();
            if leading_zeros >= S2_FACE_BITS {
                let database_level = try_get_level(database_cell_id)?;
                let common_level = std::cmp::min(
                    MAX_DB_LEVEL,
                    std::cmp::min(
                        (leading_zeros - S2_FACE_BITS) / BITS_PER_LEVEL,
                        std::cmp::min(query_level, database_level),
                    ),
                );
                best_ancestor = try_get_ancestor(s2_cell_id, common_level)?;
                best_level = common_level as i32;
            }
        }

        if compact_right > 0 {
            let database_cell_id = (compact_right as u64) << SHIFT_COMPACT;
            let xor_value = s2_cell_id ^ database_cell_id;
            let leading_zeros = xor_value.leading_zeros();
            let required_leading_zeros = if best_level >= 0 {
                S2_FACE_BITS + BITS_PER_LEVEL * (best_level as u32 + 1)
            } else {
                S2_FACE_BITS
            };
            if leading_zeros >= required_leading_zeros {
                let database_level = try_get_level(database_cell_id)?;
                let common_level = std::cmp::min(
                    MAX_DB_LEVEL,
                    std::cmp::min(
                        (leading_zeros - S2_FACE_BITS) / BITS_PER_LEVEL,
                        std::cmp::min(query_level, database_level),
                    ),
                );
                if (common_level as i32) > best_level {
                    best_ancestor = try_get_ancestor(s2_cell_id, common_level)?;
                    best_level = common_level as i32;
                }
            }
        }

        if best_level >= 0 {
            Ok(best_ancestor)
        } else {
            Ok(0)
        }
    }

    /// Reconstructs all compact S2 leaf IDs stored in the database.
    pub fn reconstruct_all_leaves(&self) -> Result<Vec<u32>> {
        let mut leaves = Vec::with_capacity(self.count);
        for block_index in 0..self.block_count {
            let header_value = self.get_header_value(block_index);
            if header_value > MAX_COMPACT_CELL_ID {
                return Err(anyhow!(
                    "corrupt database block: cell ID {:X} exceeds maximum compact S2 cell ID limit",
                    header_value
                ));
            }
            leaves.push(header_value);

            let start_offset = self.get_block_offset(block_index)?;
            let end_offset = if block_index + 1 < self.block_count {
                self.get_block_offset(block_index + 1)?
            } else {
                self.total_data_length
            };

            if start_offset > end_offset || end_offset > self.total_data_length {
                return Err(anyhow!("corrupt database block offsets out of bounds"));
            }

            let block_length = if block_index + 1 < self.block_count {
                self.block_size
            } else {
                self.count.checked_sub(block_index * self.block_size)
                    .ok_or_else(|| anyhow!("corrupt database: block count/size inconsistency during block length calculation"))?
            };

            if block_length == 0 {
                return Err(anyhow!("corrupt database: block length is 0"));
            }

            let block_bytes = &self.mmap
                [self.block_data_start + start_offset..self.block_data_start + end_offset];

            if block_length > 1 {
                let delta_count = block_length - 1;
                let mut exception_indices = [0u16; BLOCK_SIZE];
                let mut exception_values = [0u32; BLOCK_SIZE];
                let (bit_widths, exception_count) = Self::decode_block_metadata(
                    block_bytes,
                    delta_count,
                    &mut exception_indices,
                    &mut exception_values,
                )?;

                // Decode block using Patched Frame of Reference (PFOR) on stack.
                let mut delta_values = [0u32; MAX_DELTAS_CAPACITY];

                let mut bit_buffer = 0u64;
                let mut bit_count = 0;
                let mut byte_index = BLOCK_HEADER_SIZE;

                for (sub_block_index, &bit_width_value) in bit_widths.iter().enumerate() {
                    let bit_width = bit_width_value as u32;
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

                    if bit_width == 0 {
                        delta_values[sb_start..sb_end].fill(0);
                        continue;
                    }

                    for delta_slot in &mut delta_values[sb_start..sb_end] {
                        if bit_count < bit_width {
                            if byte_index + 4 <= block_bytes.len() {
                                let bytes_u32 = u32::from_le_bytes(
                                    block_bytes[byte_index..byte_index + 4].try_into().unwrap(),
                                );
                                bit_buffer |= (bytes_u32 as u64) << bit_count;
                                byte_index += 4;
                                bit_count += 32;
                            } else {
                                while bit_count < bit_width {
                                    if byte_index < block_bytes.len() {
                                        bit_buffer |= (block_bytes[byte_index] as u64) << bit_count;
                                        byte_index += 1;
                                        bit_count += 8;
                                    } else {
                                        break;
                                    }
                                }
                            }
                        }

                        if bit_count < bit_width {
                            return Err(anyhow!("corrupt database block: truncated bitstream"));
                        }

                        *delta_slot = (bit_buffer & ((1u64 << bit_width) - 1)) as u32;
                        bit_buffer >>= bit_width;
                        bit_count -= bit_width;
                    }
                }

                // Patch exception values directly into delta_values.
                for exception_ordinal in 0..exception_count {
                    let idx = exception_indices[exception_ordinal] as usize;
                    if idx >= delta_count {
                        return Err(anyhow!(
                            "corrupt database block exception index {} out of range",
                            idx
                        ));
                    }
                    let sub_block_index = idx / SUB_BLOCK_SIZE;
                    let bit_width = bit_widths[sub_block_index];
                    let lower_bits = delta_values[idx];
                    let exc_val = exception_values[exception_ordinal];
                    if (exc_val as u64 + 1)
                        .checked_shl(bit_width as u32)
                        .is_none_or(|val| val > u32::MAX as u64)
                    {
                        return Err(anyhow!("corrupt database block: exception value overflow"));
                    }
                    delta_values[idx] = (((exc_val as u64 + 1) << bit_width) as u32) | lower_bits;
                }

                let mut current_value = header_value;
                for &delta_value in delta_values.iter().take(delta_count) {
                    let delta = delta_value
                        .checked_add(1)
                        .ok_or_else(|| anyhow!("corrupt database block: delta value overflow"))?;
                    current_value = current_value.checked_add(delta).ok_or_else(|| {
                        anyhow!("corrupt database block: compact S2 cell ID overflow")
                    })?;
                    if current_value > MAX_COMPACT_CELL_ID {
                        return Err(anyhow!(
                            "corrupt database block: cell ID {:X} exceeds maximum compact S2 cell ID limit",
                            current_value
                        ));
                    }
                    leaves.push(current_value);
                }
            }
        }
        Ok(leaves)
    }
}
