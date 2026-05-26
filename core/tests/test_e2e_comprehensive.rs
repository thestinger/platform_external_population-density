//! Provides comprehensive 4-Tier E2E integration tests for the S2 population density database engine.
//!
//! Verifies geographic query accuracy across Beijing, Everest, Ocean, and utility computations,
//! alongside adversarial boundary checks, cross-feature combinatorics, and production workloads.

#![allow(clippy::manual_is_multiple_of, clippy::needless_range_loop)]

use population_density::{
    BITS_PER_LEVEL, BLOCK_HEADER_SIZE, LAST_SUB_BLOCK_SIZE, MAX_DB_LEVEL, QueryEngine,
    S2_FACE_BITS, SHIFT_COMPACT, SUB_BLOCK_COUNT, SUB_BLOCK_SIZE, get_ancestor, get_level,
};
use rayon::prelude::*;
use s2::cellid::CellID;
use s2::latlng::LatLng;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

// Local constants for the PFOR exception encoding modes.
const EXCEPTION_MODE_U4: u8 = 0;
const EXCEPTION_MODE_U8: u8 = 1;
const EXCEPTION_MODE_U16: u8 = 2;
const EXCEPTION_MODE_U32: u8 = 3;

/// Represents a simple deterministic linear congruential generator for reproducible pseudo-random numbers.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    /// Creates a new SimpleRng with the specified seed value.
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Returns the next pseudo-random 64-bit unsigned integer.
    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    /// Returns the next pseudo-random 64-bit floating-point number in the range [0.0, 1.0).
    fn next_f64(&mut self) -> f64 {
        let value = self.next_u64();
        (value as f64) / (u64::MAX as f64)
    }

    /// Returns the next pseudo-random 64-bit floating-point number in the specified range.
    fn next_range(&mut self, min: f64, max: f64) -> f64 {
        min + self.next_f64() * (max - min)
    }
}

/// Represents a naive baseline reference query oracle for correctness comparison.
pub struct NaiveOracle {
    leaves: Vec<u32>,
}

impl NaiveOracle {
    /// Creates a new naive oracle by reconstructing all leaf cells from the QueryEngine.
    pub fn new(query_engine: &QueryEngine) -> Self {
        Self {
            leaves: query_engine.reconstruct_all_leaves().unwrap(),
        }
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

/// Represents an RAII manager for temporary database files.
struct TempDb {
    path: PathBuf,
}

impl TempDb {
    /// Creates a new temporary database manager with the specified filename.
    fn new(name: &str) -> Self {
        std::fs::create_dir_all("scratch/e2e").unwrap();
        let path = PathBuf::from(format!("scratch/e2e/{}", name));
        Self { path }
    }

    /// Writes a custom S2PP mock database file with custom header values and raw payload.
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

impl Drop for TempDb {
    /// Cleans up the temporary database file from disk.
    fn drop(&mut self) {
        if self.path.exists() {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// Packs sub-block bit-widths (5 bits each), exception count, and flags into a header.
fn pack_v2_header(
    bit_widths: &[u8; 32],
    exception_count: u8,
    exception_mode: u8,
    index_bitmask_flag: bool,
) -> [u8; BLOCK_HEADER_SIZE] {
    let mut header = [0u8; BLOCK_HEADER_SIZE];
    for i in 0..SUB_BLOCK_COUNT {
        let w = bit_widths[i] & 0x1F;
        let bit_offset = i * 5;
        let byte_idx = bit_offset / 8;
        let bit_shift = bit_offset % 8;
        header[byte_idx] |= w << bit_shift;
        if bit_shift + 5 > 8 {
            header[byte_idx + 1] |= w >> (8 - bit_shift);
        }
    }
    header[BLOCK_HEADER_SIZE - 2] = exception_count;
    let mut flags = exception_mode & 0x03;
    if index_bitmask_flag {
        flags |= 1 << 2;
    }
    header[BLOCK_HEADER_SIZE - 1] = flags;
    header
}

/// Builds a complete V2 block data payload from given deltas, bit widths, and exception parameters.
fn build_v2_block_data(
    deltas: &[u32],
    bit_widths: &[u8; 32],
    exception_mode: u8,
    index_bitmask_flag: bool,
) -> Vec<u8> {
    let mut exceptions = Vec::new();
    for idx in 0..deltas.len() {
        let sb = idx / SUB_BLOCK_SIZE;
        let w = bit_widths[sb];
        let val = deltas[idx];
        let is_exception = if w == 0 { val > 0 } else { val >= (1 << w) };
        if is_exception {
            let exc_val = if w == 0 { val - 1 } else { (val >> w) - 1 };
            exceptions.push((idx as u8, exc_val));
        }
    }

    let mut block_data = Vec::new();
    let header = pack_v2_header(
        bit_widths,
        exceptions.len() as u8,
        exception_mode,
        index_bitmask_flag,
    );
    block_data.extend_from_slice(&header);

    // Pack primary bitstream.
    let mut current_byte = 0u8;
    let mut bit_offset = 0;
    for sb in 0..SUB_BLOCK_COUNT {
        let w = bit_widths[sb] as usize;
        if w == 0 {
            continue;
        }
        let sb_start = sb * SUB_BLOCK_SIZE;
        let sb_size = std::cmp::min(
            if sb == SUB_BLOCK_COUNT - 1 {
                LAST_SUB_BLOCK_SIZE
            } else {
                SUB_BLOCK_SIZE
            },
            deltas.len().saturating_sub(sb_start),
        );
        for idx in sb_start..sb_start + sb_size {
            let val = deltas[idx];
            let lower_bits = val & ((1 << w) - 1);
            let mut bits_left = w;
            let mut temp = lower_bits;
            while bits_left > 0 {
                let bits_to_write = std::cmp::min(8 - bit_offset, bits_left);
                let mask = (1 << bits_to_write) - 1;
                current_byte |= ((temp & mask) as u8) << bit_offset;
                temp >>= bits_to_write;
                bits_left -= bits_to_write;
                bit_offset += bits_to_write;
                if bit_offset == 8 {
                    block_data.push(current_byte);
                    current_byte = 0;
                    bit_offset = 0;
                }
            }
        }
    }
    if bit_offset > 0 {
        block_data.push(current_byte);
    }

    // Pack exception indices.
    if index_bitmask_flag {
        let mut bitmask = [0u8; 32];
        for &(idx, _) in &exceptions {
            let byte_idx = idx as usize / 8;
            let bit_shift = idx as usize % 8;
            bitmask[byte_idx] |= 1 << bit_shift;
        }
        block_data.extend_from_slice(&bitmask);
    } else {
        for &(idx, _) in &exceptions {
            block_data.push(idx);
        }
    }

    // Pack exception values.
    match exception_mode {
        EXCEPTION_MODE_U4 => {
            let num_bytes = exceptions.len().div_ceil(2);
            for i in 0..num_bytes {
                let val1 = exceptions[2 * i].1 & 0x0F;
                let val2 = if 2 * i + 1 < exceptions.len() {
                    exceptions[2 * i + 1].1 & 0x0F
                } else {
                    0
                };
                block_data.push((val1 | (val2 << 4)) as u8);
            }
        }
        EXCEPTION_MODE_U8 => {
            for &(_, val) in &exceptions {
                block_data.push(val as u8);
            }
        }
        EXCEPTION_MODE_U16 => {
            for &(_, val) in &exceptions {
                block_data.extend_from_slice(&(val as u16).to_le_bytes());
            }
        }
        EXCEPTION_MODE_U32 => {
            for &(_, val) in &exceptions {
                block_data.extend_from_slice(&val.to_le_bytes());
            }
        }
        _ => unreachable!(),
    }

    block_data
}

/// Computes the leaf IDs a block decodes to: a running prefix sum of (delta + 1) from the header.
fn expected_leaves(header: u32, deltas: &[u32]) -> Vec<u32> {
    let mut leaves = Vec::with_capacity(deltas.len() + 1);
    let mut current = header;
    leaves.push(current);
    for &delta in deltas {
        current += delta + 1;
        leaves.push(current);
    }
    leaves
}

// ==========================================
// Tier 1: Feature coverage.
// ==========================================

/// Verifies exact decoded leaf values across every exception mode and index encoding.
///
/// The weaker per-feature cases only assert that decoding does not error; this asserts the
/// reconstructed values themselves, so a subtle off-by-one in any nibble/byte unpack, exception
/// reconstruction formula, or bitmask index decode is caught.
#[test]
fn test_decode_value_parity_all_modes() {
    let header = 100u32;
    let zero_widths = [0u8; 32];
    let mut mixed_widths = [0u8; 32];
    mixed_widths[0] = 4;

    // (name, deltas, bit_widths, exception_mode, index_bitmask_flag).
    let small = vec![0u32, 1, 2, 3, 4, 5, 6, 7];
    let u4_index = vec![1u32, 0, 0, 16, 0, 9];
    let u4_bitmask = vec![1u32; 40];
    let u8_index = vec![251u32, 0, 200, 0];
    let u16_index = vec![5001u32, 0, 0, 60000];
    let u32_index = vec![100001u32, 0, 0, 0, 1u32 << 20];
    let mixed = vec![0x12345u32, 0, 0, 0, 0, 0, 0, 0];

    let cases = vec![
        ("dvp_small", &small, &zero_widths, EXCEPTION_MODE_U4, false),
        (
            "dvp_u4_index",
            &u4_index,
            &zero_widths,
            EXCEPTION_MODE_U4,
            false,
        ),
        (
            "dvp_u4_bitmask",
            &u4_bitmask,
            &zero_widths,
            EXCEPTION_MODE_U4,
            true,
        ),
        (
            "dvp_u8_index",
            &u8_index,
            &zero_widths,
            EXCEPTION_MODE_U8,
            false,
        ),
        (
            "dvp_u16_index",
            &u16_index,
            &zero_widths,
            EXCEPTION_MODE_U16,
            false,
        ),
        (
            "dvp_u32_index",
            &u32_index,
            &zero_widths,
            EXCEPTION_MODE_U32,
            false,
        ),
        (
            "dvp_mixed_width",
            &mixed,
            &mixed_widths,
            EXCEPTION_MODE_U16,
            false,
        ),
    ];

    for (name, deltas, bit_widths, mode, bitmask) in cases {
        let block_data = build_v2_block_data(deltas, bit_widths, mode, bitmask);
        let temp = TempDb::new(name);
        let count = (deltas.len() + 1) as u32;
        temp.write(b"S2PP", count, 256, 1, &[header], &[0], &[0], &block_data);
        let engine = QueryEngine::new(&temp.path).unwrap();
        assert_eq!(
            engine.reconstruct_all_leaves().unwrap(),
            expected_leaves(header, deltas),
            "decode value mismatch for case {}",
            name
        );
    }
}

/// Case 1: Block with all deltas = 0 (all sub-blocks bit-width 0, no exceptions).
#[test]
fn test_tier1_sb_all_zeros() {
    let temp = TempDb::new("t1_all_zeros.db");
    let deltas = vec![0u32; 255];
    let bit_widths = [0u8; 32];
    let block_data = build_v2_block_data(&deltas, &bit_widths, EXCEPTION_MODE_U32, false);
    temp.write(b"S2PP", 256, 256, 1, &[100], &[0], &[0], &block_data);
    let engine = QueryEngine::new(&temp.path).unwrap();
    let leaves = engine.reconstruct_all_leaves().unwrap();
    assert_eq!(leaves.len(), 256);
    assert_eq!(leaves[0], 100);
    for i in 1..256 {
        assert_eq!(leaves[i], 100 + i as u32);
    }
}

/// Case 2: Block with uniform small deltas (all sub-blocks bit-width 2, no exceptions).
#[test]
fn test_tier1_sb_uniform_small() {
    let temp = TempDb::new("t1_uniform_small.db");
    let deltas = vec![2u32; 255];
    let bit_widths = [2u8; 32];
    let block_data = build_v2_block_data(&deltas, &bit_widths, EXCEPTION_MODE_U32, false);
    temp.write(b"S2PP", 256, 256, 1, &[100], &[0], &[0], &block_data);
    let engine = QueryEngine::new(&temp.path).unwrap();
    let leaves = engine.reconstruct_all_leaves().unwrap();
    assert_eq!(leaves.len(), 256);
    assert_eq!(leaves[0], 100);
    for i in 1..256 {
        assert_eq!(leaves[i], 100 + i as u32 * 3);
    }
}

/// Case 3: Block with varying small deltas per sub-block (each sub-block has a different bit-width, no exceptions).
#[test]
fn test_tier1_sb_varying_small() {
    let temp = TempDb::new("t1_varying_small.db");
    let mut bit_widths = [0u8; 32];
    for sb in 0..32 {
        bit_widths[sb] = (sb % 5) as u8;
    }
    let mut deltas = vec![0u32; 255];
    for idx in 0..255 {
        let sb = idx / 8;
        let w = bit_widths[sb];
        deltas[idx] = if w == 0 { 0 } else { (1 << w) - 1 };
    }
    let block_data = build_v2_block_data(&deltas, &bit_widths, EXCEPTION_MODE_U32, false);
    temp.write(b"S2PP", 256, 256, 1, &[100], &[0], &[0], &block_data);
    let engine = QueryEngine::new(&temp.path).unwrap();
    let leaves = engine.reconstruct_all_leaves().unwrap();
    assert_eq!(leaves.len(), 256);
}

/// Case 4: Single sub-block with maximum bit-width 28, others 0.
#[test]
fn test_tier1_sb_single_max_width() {
    let temp = TempDb::new("t1_single_max.db");
    let mut bit_widths = [0u8; 32];
    bit_widths[5] = 28;
    let mut deltas = vec![0u32; 255];
    for idx in 40..48 {
        deltas[idx] = 1234567;
    }
    let block_data = build_v2_block_data(&deltas, &bit_widths, EXCEPTION_MODE_U32, false);
    temp.write(b"S2PP", 256, 256, 1, &[100], &[0], &[0], &block_data);
    let engine = QueryEngine::new(&temp.path).unwrap();
    let _leaves = engine.reconstruct_all_leaves().unwrap();
}

/// Case 5: Alternating sub-blocks of bit-width 4 and 0.
#[test]
fn test_tier1_sb_alternating_width() {
    let temp = TempDb::new("t1_alternating.db");
    let mut bit_widths = [0u8; 32];
    for sb in 0..32 {
        bit_widths[sb] = if sb % 2 == 0 { 4 } else { 0 };
    }
    let mut deltas = vec![0u32; 255];
    for idx in 0..255 {
        let sb = idx / 8;
        if sb % 2 == 0 {
            deltas[idx] = 10;
        }
    }
    let block_data = build_v2_block_data(&deltas, &bit_widths, EXCEPTION_MODE_U32, false);
    temp.write(b"S2PP", 256, 256, 1, &[100], &[0], &[0], &block_data);
    let engine = QueryEngine::new(&temp.path).unwrap();
    let _leaves = engine.reconstruct_all_leaves().unwrap();
}

/// Case 6: Block with 1 exception whose value is 0 (fits in U4).
#[test]
fn test_tier1_u4_single_exception_val_0() {
    let temp = TempDb::new("t1_u4_single_0.db");
    let mut deltas = vec![0u32; 255];
    deltas[0] = 1;
    let bit_widths = [0u8; 32];
    let block_data = build_v2_block_data(&deltas, &bit_widths, EXCEPTION_MODE_U4, false);
    temp.write(b"S2PP", 256, 256, 1, &[100], &[0], &[0], &block_data);
    let engine = QueryEngine::new(&temp.path).unwrap();
    let _leaves = engine.reconstruct_all_leaves().unwrap();
}

/// Case 7: Block with 2 exceptions whose values are 5 and 10 (packed into 1 byte).
#[test]
fn test_tier1_u4_two_exceptions_5_10() {
    let temp = TempDb::new("t1_u4_two_5_10.db");
    let mut deltas = vec![0u32; 255];
    deltas[0] = 6;
    deltas[1] = 11;
    let bit_widths = [0u8; 32];
    let block_data = build_v2_block_data(&deltas, &bit_widths, EXCEPTION_MODE_U4, false);
    temp.write(b"S2PP", 256, 256, 1, &[100], &[0], &[0], &block_data);
    let engine = QueryEngine::new(&temp.path).unwrap();
    let _leaves = engine.reconstruct_all_leaves().unwrap();
}

/// Case 8: Block with 3 exceptions (odd count, packed into 2 bytes).
#[test]
fn test_tier1_u4_three_exceptions_odd() {
    let temp = TempDb::new("t1_u4_three.db");
    let mut deltas = vec![0u32; 255];
    deltas[0] = 3;
    deltas[1] = 5;
    deltas[2] = 7;
    let bit_widths = [0u8; 32];
    let block_data = build_v2_block_data(&deltas, &bit_widths, EXCEPTION_MODE_U4, false);
    temp.write(b"S2PP", 256, 256, 1, &[100], &[0], &[0], &block_data);
    let engine = QueryEngine::new(&temp.path).unwrap();
    let _leaves = engine.reconstruct_all_leaves().unwrap();
}

/// Case 9: Block with 10 exceptions (even count, packed into 5 bytes).
#[test]
fn test_tier1_u4_ten_exceptions_even() {
    let temp = TempDb::new("t1_u4_ten.db");
    let mut deltas = vec![0u32; 255];
    for idx in 0..10 {
        deltas[idx] = idx as u32 + 2;
    }
    let bit_widths = [0u8; 32];
    let block_data = build_v2_block_data(&deltas, &bit_widths, EXCEPTION_MODE_U4, false);
    temp.write(b"S2PP", 256, 256, 1, &[100], &[0], &[0], &block_data);
    let engine = QueryEngine::new(&temp.path).unwrap();
    let _leaves = engine.reconstruct_all_leaves().unwrap();
}

/// Case 10: Block with all 255 elements being exceptions of value 0 (packed into 128 bytes, using bitmask indices).
#[test]
fn test_tier1_u4_all_exceptions_val_0() {
    let temp = TempDb::new("t1_u4_all_0.db");
    let deltas = vec![1u32; 255];
    let bit_widths = [0u8; 32];
    let block_data = build_v2_block_data(&deltas, &bit_widths, EXCEPTION_MODE_U4, true);
    temp.write(b"S2PP", 256, 256, 1, &[100], &[0], &[0], &block_data);
    let engine = QueryEngine::new(&temp.path).unwrap();
    let _leaves = engine.reconstruct_all_leaves().unwrap();
}

/// Case 11: Block with 32 exceptions, all values fitting in U4, forcing bitmask mode.
#[test]
fn test_tier1_bitmask_u4_32_exceptions() {
    let temp = TempDb::new("t1_bitmask_u4.db");
    let mut deltas = vec![0u32; 255];
    for idx in 0..32 {
        deltas[idx] = 1;
    }
    let bit_widths = [0u8; 32];
    let block_data = build_v2_block_data(&deltas, &bit_widths, EXCEPTION_MODE_U4, true);
    temp.write(b"S2PP", 256, 256, 1, &[100], &[0], &[0], &block_data);
    let engine = QueryEngine::new(&temp.path).unwrap();
    let _leaves = engine.reconstruct_all_leaves().unwrap();
}

/// Case 12: Block with 40 exceptions, values in U4, bitmask mode.
#[test]
fn test_tier1_bitmask_u4_40_exceptions() {
    let temp = TempDb::new("t1_bitmask_u4_40.db");
    let mut deltas = vec![0u32; 255];
    for idx in 0..40 {
        deltas[idx] = 1;
    }
    let bit_widths = [0u8; 32];
    let block_data = build_v2_block_data(&deltas, &bit_widths, EXCEPTION_MODE_U4, true);
    temp.write(b"S2PP", 256, 256, 1, &[100], &[0], &[0], &block_data);
    let engine = QueryEngine::new(&temp.path).unwrap();
    let _leaves = engine.reconstruct_all_leaves().unwrap();
}

/// Case 13: Block with 32 exceptions, values in U8, bitmask mode.
#[test]
fn test_tier1_bitmask_u8_32_exceptions() {
    let temp = TempDb::new("t1_bitmask_u8.db");
    let mut deltas = vec![0u32; 255];
    for idx in 0..32 {
        deltas[idx] = 100;
    }
    let bit_widths = [0u8; 32];
    let block_data = build_v2_block_data(&deltas, &bit_widths, EXCEPTION_MODE_U8, true);
    temp.write(b"S2PP", 256, 256, 1, &[100], &[0], &[0], &block_data);
    let engine = QueryEngine::new(&temp.path).unwrap();
    let _leaves = engine.reconstruct_all_leaves().unwrap();
}

/// Case 14: Block with 32 exceptions, values in U16, bitmask mode.
#[test]
fn test_tier1_bitmask_u16_32_exceptions() {
    let temp = TempDb::new("t1_bitmask_u16.db");
    let mut deltas = vec![0u32; 255];
    for idx in 0..32 {
        deltas[idx] = 5000;
    }
    let bit_widths = [0u8; 32];
    let block_data = build_v2_block_data(&deltas, &bit_widths, EXCEPTION_MODE_U16, true);
    temp.write(b"S2PP", 256, 256, 1, &[100], &[0], &[0], &block_data);
    let engine = QueryEngine::new(&temp.path).unwrap();
    let _leaves = engine.reconstruct_all_leaves().unwrap();
}

/// Case 15: Block with 32 exceptions, values in U32, bitmask mode.
#[test]
fn test_tier1_bitmask_u32_32_exceptions() {
    let temp = TempDb::new("t1_bitmask_u32.db");
    let mut deltas = vec![0u32; 255];
    for idx in 0..32 {
        deltas[idx] = 100000;
    }
    let bit_widths = [0u8; 32];
    let block_data = build_v2_block_data(&deltas, &bit_widths, EXCEPTION_MODE_U32, true);
    temp.write(b"S2PP", 256, 256, 1, &[100], &[0], &[0], &block_data);
    let engine = QueryEngine::new(&temp.path).unwrap();
    let _leaves = engine.reconstruct_all_leaves().unwrap();
}

/// Case 16: Exception values requiring U8 mode (value 250), with count < 32 (byte list indices).
#[test]
fn test_tier1_compression_u8_under_32() {
    let temp = TempDb::new("t1_compression_u8.db");
    let mut deltas = vec![0u32; 255];
    deltas[0] = 251;
    let bit_widths = [0u8; 32];
    let block_data = build_v2_block_data(&deltas, &bit_widths, EXCEPTION_MODE_U8, false);
    temp.write(b"S2PP", 256, 256, 1, &[100], &[0], &[0], &block_data);
    let engine = QueryEngine::new(&temp.path).unwrap();
    let _leaves = engine.reconstruct_all_leaves().unwrap();
}

/// Case 17: Exception values requiring U16 mode (value 5000), with count < 32 (byte list indices).
#[test]
fn test_tier1_compression_u16_under_32() {
    let temp = TempDb::new("t1_compression_u16.db");
    let mut deltas = vec![0u32; 255];
    deltas[0] = 5001;
    let bit_widths = [0u8; 32];
    let block_data = build_v2_block_data(&deltas, &bit_widths, EXCEPTION_MODE_U16, false);
    temp.write(b"S2PP", 256, 256, 1, &[100], &[0], &[0], &block_data);
    let engine = QueryEngine::new(&temp.path).unwrap();
    let _leaves = engine.reconstruct_all_leaves().unwrap();
}

/// Case 18: Exception values requiring U32 mode (value 100000), with count < 32 (byte list indices).
#[test]
fn test_tier1_compression_u32_under_32() {
    let temp = TempDb::new("t1_compression_u32.db");
    let mut deltas = vec![0u32; 255];
    deltas[0] = 100001;
    let bit_widths = [0u8; 32];
    let block_data = build_v2_block_data(&deltas, &bit_widths, EXCEPTION_MODE_U32, false);
    temp.write(b"S2PP", 256, 256, 1, &[100], &[0], &[0], &block_data);
    let engine = QueryEngine::new(&temp.path).unwrap();
    let _leaves = engine.reconstruct_all_leaves().unwrap();
}

/// Case 19: Verify correct reconstruction of upper/lower bits.
#[test]
fn test_tier1_compression_reconstruction_formula() {
    let temp = TempDb::new("t1_reconstruction.db");
    let mut deltas = vec![0u32; 255];
    deltas[0] = 0x12345;
    let mut bit_widths = [0u8; 32];
    bit_widths[0] = 4;
    let block_data = build_v2_block_data(&deltas, &bit_widths, EXCEPTION_MODE_U16, false);
    temp.write(b"S2PP", 256, 256, 1, &[100], &[0], &[0], &block_data);
    let engine = QueryEngine::new(&temp.path).unwrap();
    let leaves = engine.reconstruct_all_leaves().unwrap();
    assert_eq!(leaves[1], 100 + 0x12345 + 1);
}

/// Case 20: Large delta values that are exactly power of 2 offsets from limit.
#[test]
fn test_tier1_compression_power_of_2_boundary() {
    let temp = TempDb::new("t1_power2_boundary.db");
    let mut deltas = vec![0u32; 255];
    deltas[0] = (1 << 8) + 5;
    let mut bit_widths = [0u8; 32];
    bit_widths[0] = 8;
    let block_data = build_v2_block_data(&deltas, &bit_widths, EXCEPTION_MODE_U8, false);
    temp.write(b"S2PP", 256, 256, 1, &[100], &[0], &[0], &block_data);
    let engine = QueryEngine::new(&temp.path).unwrap();
    let _leaves = engine.reconstruct_all_leaves().unwrap();
}

/// Case 21: Standard query match on small database.
#[test]
fn test_tier1_query_parity_small_db() {
    let temp = TempDb::new("t1_query_small.db");
    let mut deltas = vec![0u32; 511];
    for idx in 0..511 {
        deltas[idx] = (idx as u32 % 3) * 2 + 1;
    }
    let bit_widths = [3u8; 32];
    let b1 = build_v2_block_data(&deltas[0..255], &bit_widths, EXCEPTION_MODE_U32, false);
    let b2 = build_v2_block_data(&deltas[255..510], &bit_widths, EXCEPTION_MODE_U32, false);
    let mut block_data = b1;
    block_data.extend_from_slice(&b2);

    temp.write(
        b"S2PP",
        512,
        256,
        2,
        &[1001, 2001],
        &[0],
        &[0, block_data.len() as u16 / 2],
        &block_data,
    );
    let engine = QueryEngine::new(&temp.path).unwrap();
    let oracle = NaiveOracle::new(&engine);
    let result = engine.query(1001u64 << SHIFT_COMPACT).unwrap();
    assert_eq!(result, oracle.query(1001u64 << SHIFT_COMPACT));
}

/// Case 22: Querying cell IDs that match exact block headers.
#[test]
fn test_tier1_query_exact_headers() {
    let temp = TempDb::new("t1_query_headers.db");
    let b1 = build_v2_block_data(&[1; 255], &[1; 32], EXCEPTION_MODE_U32, false);
    temp.write(b"S2PP", 256, 256, 1, &[1001], &[0], &[0], &b1);
    let engine = QueryEngine::new(&temp.path).unwrap();
    let oracle = NaiveOracle::new(&engine);
    let query = 1001u64 << SHIFT_COMPACT;
    assert_eq!(engine.query(query).unwrap(), oracle.query(query));
}

/// Case 23: Querying cell IDs that fall between block headers.
#[test]
fn test_tier1_query_between_headers() {
    let temp = TempDb::new("t1_query_between.db");
    let b1 = build_v2_block_data(&[1; 255], &[1; 32], EXCEPTION_MODE_U32, false);
    temp.write(b"S2PP", 256, 256, 1, &[1001], &[0], &[0], &b1);
    let engine = QueryEngine::new(&temp.path).unwrap();
    let oracle = NaiveOracle::new(&engine);
    let query = 1005u64 << SHIFT_COMPACT;
    assert_eq!(engine.query(query).unwrap(), oracle.query(query));
}

/// Case 24: Querying cell IDs that are larger than any header in the database.
#[test]
fn test_tier1_query_larger_than_max() {
    let temp = TempDb::new("t1_query_larger.db");
    let b1 = build_v2_block_data(&[1; 255], &[1; 32], EXCEPTION_MODE_U32, false);
    temp.write(b"S2PP", 256, 256, 1, &[1001], &[0], &[0], &b1);
    let engine = QueryEngine::new(&temp.path).unwrap();
    let oracle = NaiveOracle::new(&engine);
    let query = 9999u64 << SHIFT_COMPACT;
    assert_eq!(engine.query(query).unwrap(), oracle.query(query));
}

/// Case 25: Querying cell IDs that are smaller than the first header.
#[test]
fn test_tier1_query_smaller_than_min() {
    let temp = TempDb::new("t1_query_smaller.db");
    let b1 = build_v2_block_data(&[1; 255], &[1; 32], EXCEPTION_MODE_U32, false);
    temp.write(b"S2PP", 256, 256, 1, &[1001], &[0], &[0], &b1);
    let engine = QueryEngine::new(&temp.path).unwrap();
    let oracle = NaiveOracle::new(&engine);
    let query = 1u64 << SHIFT_COMPACT;
    assert_eq!(engine.query(query).unwrap(), oracle.query(query));
}

/// Verifies query() matches the oracle for cells resolving into the second block, exercising the
/// cross-block left/right neighbor selection that single-block query tests never reach.
#[test]
fn test_query_crosses_block_boundary() {
    let temp = TempDb::new("query_cross_block.db");
    let mut deltas = vec![0u32; 510];
    for (delta_index, delta) in deltas.iter_mut().enumerate() {
        *delta = (delta_index as u32 % 4) * 2 + 1;
    }
    let bit_widths = [3u8; 32];
    let b1 = build_v2_block_data(&deltas[0..255], &bit_widths, EXCEPTION_MODE_U32, false);
    let b2 = build_v2_block_data(&deltas[255..510], &bit_widths, EXCEPTION_MODE_U32, false);
    let mut block_data = b1;
    let block1_offset = block_data.len() as u16;
    block_data.extend_from_slice(&b2);

    temp.write(
        b"S2PP",
        512,
        256,
        2,
        &[1001, 50001],
        &[0],
        &[0, block1_offset],
        &block_data,
    );
    let engine = QueryEngine::new(&temp.path).unwrap();
    let oracle = NaiveOracle::new(&engine);

    // Probe across both blocks, the gap between them, and the extremes. Compact values are odd so
    // that (compact << SHIFT_COMPACT) is a valid level-12 cell (sentinel bit at even position 36).
    for compact in [1u64, 1001, 1501, 2001, 49999, 50001, 50101, 60001] {
        let query = compact << SHIFT_COMPACT;
        assert_eq!(
            engine.query(query).unwrap(),
            oracle.query(query),
            "query/oracle mismatch at compact {}",
            compact
        );
    }
}

/// Verifies query() (not just reconstruct_all_leaves) correctly decodes a block that uses the
/// 32-byte exception bitmask, asserting the resolved cell matches the oracle.
#[test]
fn test_query_bitmask_block() {
    let temp = TempDb::new("query_bitmask_block.db");
    // All 255 deltas-minus-one are 33 (odd, exceeds the U4 range so they become U8 exceptions);
    // 255 exceptions force the bitmask index encoding. Odd delta-minus-one keeps every leaf an
    // odd compact value, i.e. a valid level-12 cell, matching real databases.
    let deltas = vec![33u32; 255];
    let bit_widths = [0u8; 32];
    let block_data = build_v2_block_data(&deltas, &bit_widths, EXCEPTION_MODE_U8, true);
    temp.write(b"S2PP", 256, 256, 1, &[1001], &[0], &[0], &block_data);
    let engine = QueryEngine::new(&temp.path).unwrap();
    let oracle = NaiveOracle::new(&engine);

    for compact in [1001u64, 1035, 5001, 9671, 10001] {
        let query = compact << SHIFT_COMPACT;
        assert_eq!(
            engine.query(query).unwrap(),
            oracle.query(query),
            "query/oracle mismatch at compact {}",
            compact
        );
    }
}

// ==========================================
// Tier 2: Boundary & corner cases.
// ==========================================

/// Case 26: Terminal block with exactly 1 cell (no deltas).
#[test]
fn test_tier2_sb_terminal_1_cell() {
    let temp = TempDb::new("t2_terminal_1.db");
    temp.write(b"S2PP", 1, 256, 1, &[100], &[0], &[0], &[]);
    let engine = QueryEngine::new(&temp.path).unwrap();
    let leaves = engine.reconstruct_all_leaves().unwrap();
    assert_eq!(leaves, vec![100]);
}

/// Case 27: Terminal block with exactly 2 cells (1 delta, 1 sub-block has size 1, others size 0).
#[test]
fn test_tier2_sb_terminal_2_cells() {
    let temp = TempDb::new("t2_terminal_2.db");
    let deltas = vec![5u32];
    let bit_widths = [3u8; 32];
    let block_data = build_v2_block_data(&deltas, &bit_widths, EXCEPTION_MODE_U32, false);
    temp.write(b"S2PP", 2, 256, 1, &[100], &[0], &[0], &block_data);
    let engine = QueryEngine::new(&temp.path).unwrap();
    let leaves = engine.reconstruct_all_leaves().unwrap();
    assert_eq!(leaves, vec![100, 106]);
}

/// Case 28: Terminal block with exactly 9 cells (8 deltas, sub-block 0 size 8, others size 0).
#[test]
fn test_tier2_sb_terminal_9_cells() {
    let temp = TempDb::new("t2_terminal_9.db");
    let deltas = vec![1u32; 8];
    let bit_widths = [1u8; 32];
    let block_data = build_v2_block_data(&deltas, &bit_widths, EXCEPTION_MODE_U32, false);
    temp.write(b"S2PP", 9, 256, 1, &[100], &[0], &[0], &block_data);
    let engine = QueryEngine::new(&temp.path).unwrap();
    let leaves = engine.reconstruct_all_leaves().unwrap();
    assert_eq!(leaves.len(), 9);
}

/// Case 29: Terminal block with exactly 10 cells (9 deltas, sub-block 0 size 8, sub-block 1 size 1, others size 0).
#[test]
fn test_tier2_sb_terminal_10_cells() {
    let temp = TempDb::new("t2_terminal_10.db");
    let deltas = vec![1u32; 9];
    let bit_widths = [1u8; 32];
    let block_data = build_v2_block_data(&deltas, &bit_widths, EXCEPTION_MODE_U32, false);
    temp.write(b"S2PP", 10, 256, 1, &[100], &[0], &[0], &block_data);
    let engine = QueryEngine::new(&temp.path).unwrap();
    let leaves = engine.reconstruct_all_leaves().unwrap();
    assert_eq!(leaves.len(), 10);
}

/// Case 30: Sub-block bit-width exactly 28 (boundary check for max bit-width).
#[test]
fn test_tier2_sb_width_boundary_28() {
    let temp = TempDb::new("t2_width_28.db");
    let mut bit_widths = [0u8; 32];
    bit_widths[0] = 28;
    let block_data = build_v2_block_data(&[0], &bit_widths, EXCEPTION_MODE_U32, false);
    temp.write(b"S2PP", 2, 256, 1, &[1], &[0], &[0], &block_data);
    let engine = QueryEngine::new(&temp.path).unwrap();
    let _ = engine.reconstruct_all_leaves().unwrap();
}

/// Case 31: Exception value exactly 15 (max U4 value).
#[test]
fn test_tier2_u4_val_15() {
    let temp = TempDb::new("t2_u4_val_15.db");
    let mut deltas = vec![0u32; 255];
    deltas[0] = 16;
    let bit_widths = [0u8; 32];
    let block_data = build_v2_block_data(&deltas, &bit_widths, EXCEPTION_MODE_U4, false);
    temp.write(b"S2PP", 256, 256, 1, &[100], &[0], &[0], &block_data);
    let engine = QueryEngine::new(&temp.path).unwrap();
    let _leaves = engine.reconstruct_all_leaves().unwrap();
}

/// Case 32: Exception value exactly 16 (first value requiring U8).
#[test]
fn test_tier2_u4_val_16() {
    let temp = TempDb::new("t2_u4_val_16.db");
    let mut deltas = vec![0u32; 255];
    deltas[0] = 17;
    let bit_widths = [0u8; 32];
    let block_data = build_v2_block_data(&deltas, &bit_widths, EXCEPTION_MODE_U8, false);
    temp.write(b"S2PP", 256, 256, 1, &[100], &[0], &[0], &block_data);
    let engine = QueryEngine::new(&temp.path).unwrap();
    let _leaves = engine.reconstruct_all_leaves().unwrap();
}

/// Case 33: Single exception of value 15 (odd count).
#[test]
fn test_tier2_u4_single_val_15() {
    let temp = TempDb::new("t2_u4_single_15.db");
    let mut deltas = vec![0u32; 255];
    deltas[0] = 16;
    let bit_widths = [0u8; 32];
    let block_data = build_v2_block_data(&deltas, &bit_widths, EXCEPTION_MODE_U4, false);
    temp.write(b"S2PP", 256, 256, 1, &[100], &[0], &[0], &block_data);
    let engine = QueryEngine::new(&temp.path).unwrap();
    let _leaves = engine.reconstruct_all_leaves().unwrap();
}

/// Case 34: 31 exceptions of value 15 (odd count).
#[test]
fn test_tier2_u4_31_vals_15() {
    let temp = TempDb::new("t2_u4_31_vals.db");
    let mut deltas = vec![0u32; 255];
    for idx in 0..31 {
        deltas[idx] = 16;
    }
    let bit_widths = [0u8; 32];
    let block_data = build_v2_block_data(&deltas, &bit_widths, EXCEPTION_MODE_U4, false);
    temp.write(b"S2PP", 256, 256, 1, &[100], &[0], &[0], &block_data);
    let engine = QueryEngine::new(&temp.path).unwrap();
    let _leaves = engine.reconstruct_all_leaves().unwrap();
}

/// Case 35: Zero bit-width sub-block with exception value exactly 15.
#[test]
fn test_tier2_u4_zero_width_val_15() {
    let temp = TempDb::new("t2_zero_u4_15.db");
    let mut deltas = vec![0u32; 255];
    deltas[0] = 16;
    let bit_widths = [0u8; 32];
    let block_data = build_v2_block_data(&deltas, &bit_widths, EXCEPTION_MODE_U4, false);
    temp.write(b"S2PP", 256, 256, 1, &[100], &[0], &[0], &block_data);
    let engine = QueryEngine::new(&temp.path).unwrap();
    let _leaves = engine.reconstruct_all_leaves().unwrap();
}

/// Case 36: Exactly 31 exceptions (byte list indices boundary).
#[test]
fn test_tier2_bitmask_boundary_31() {
    let temp = TempDb::new("t2_bitmask_31.db");
    let mut deltas = vec![0u32; 255];
    for idx in 0..31 {
        deltas[idx] = 1;
    }
    let bit_widths = [0u8; 32];
    let block_data = build_v2_block_data(&deltas, &bit_widths, EXCEPTION_MODE_U4, false);
    temp.write(b"S2PP", 256, 256, 1, &[100], &[0], &[0], &block_data);
    let engine = QueryEngine::new(&temp.path).unwrap();
    let _leaves = engine.reconstruct_all_leaves().unwrap();
}

/// Case 37: Exactly 32 exceptions (bitmask indices boundary).
#[test]
fn test_tier2_bitmask_boundary_32() {
    let temp = TempDb::new("t2_bitmask_32.db");
    let mut deltas = vec![0u32; 255];
    for idx in 0..32 {
        deltas[idx] = 1;
    }
    let bit_widths = [0u8; 32];
    let block_data = build_v2_block_data(&deltas, &bit_widths, EXCEPTION_MODE_U4, true);
    temp.write(b"S2PP", 256, 256, 1, &[100], &[0], &[0], &block_data);
    let engine = QueryEngine::new(&temp.path).unwrap();
    let _leaves = engine.reconstruct_all_leaves().unwrap();
}

/// Case 38: Exception count 0 (no exception indexing or values).
#[test]
fn test_tier2_bitmask_zero_exceptions() {
    let temp = TempDb::new("t2_bitmask_zero.db");
    let deltas = vec![0u32; 255];
    let bit_widths = [0u8; 32];
    let block_data = build_v2_block_data(&deltas, &bit_widths, EXCEPTION_MODE_U4, false);
    temp.write(b"S2PP", 256, 256, 1, &[100], &[0], &[0], &block_data);
    let engine = QueryEngine::new(&temp.path).unwrap();
    let _leaves = engine.reconstruct_all_leaves().unwrap();
}

/// Case 39: Bitmask with only the 1st bit set (index 0).
#[test]
fn test_tier2_bitmask_first_bit() {
    let temp = TempDb::new("t2_bitmask_first.db");
    let mut deltas = vec![0u32; 255];
    deltas[0] = 1;
    let bit_widths = [0u8; 32];
    let block_data = build_v2_block_data(&deltas, &bit_widths, EXCEPTION_MODE_U4, true);
    temp.write(b"S2PP", 256, 256, 1, &[100], &[0], &[0], &block_data);
    let engine = QueryEngine::new(&temp.path).unwrap();
    let _leaves = engine.reconstruct_all_leaves().unwrap();
}

/// Case 40: Bitmask with only the last bit set (index 254).
#[test]
fn test_tier2_bitmask_last_bit() {
    let temp = TempDb::new("t2_bitmask_last.db");
    let mut deltas = vec![0u32; 255];
    deltas[254] = 1;
    let bit_widths = [0u8; 32];
    let block_data = build_v2_block_data(&deltas, &bit_widths, EXCEPTION_MODE_U4, true);
    temp.write(b"S2PP", 256, 256, 1, &[100], &[0], &[0], &block_data);
    let engine = QueryEngine::new(&temp.path).unwrap();
    let _leaves = engine.reconstruct_all_leaves().unwrap();
}

/// Case 41: Exception value exactly 255 (max U8 value).
#[test]
fn test_tier2_compression_boundary_u8() {
    let temp = TempDb::new("t2_comp_u8.db");
    let mut deltas = vec![0u32; 255];
    deltas[0] = 256;
    let bit_widths = [0u8; 32];
    let block_data = build_v2_block_data(&deltas, &bit_widths, EXCEPTION_MODE_U8, false);
    temp.write(b"S2PP", 256, 256, 1, &[100], &[0], &[0], &block_data);
    let engine = QueryEngine::new(&temp.path).unwrap();
    let _leaves = engine.reconstruct_all_leaves().unwrap();
}

/// Case 42: Exception value exactly 256 (first value requiring U16).
#[test]
fn test_tier2_compression_boundary_u16_start() {
    let temp = TempDb::new("t2_comp_u16_start.db");
    let mut deltas = vec![0u32; 255];
    deltas[0] = 257;
    let bit_widths = [0u8; 32];
    let block_data = build_v2_block_data(&deltas, &bit_widths, EXCEPTION_MODE_U16, false);
    temp.write(b"S2PP", 256, 256, 1, &[100], &[0], &[0], &block_data);
    let engine = QueryEngine::new(&temp.path).unwrap();
    let _leaves = engine.reconstruct_all_leaves().unwrap();
}

/// Case 43: Exception value exactly 65535 (max U16 value).
#[test]
fn test_tier2_compression_boundary_u16_max() {
    let temp = TempDb::new("t2_comp_u16_max.db");
    let mut deltas = vec![0u32; 255];
    deltas[0] = 65536;
    let bit_widths = [0u8; 32];
    let block_data = build_v2_block_data(&deltas, &bit_widths, EXCEPTION_MODE_U16, false);
    temp.write(b"S2PP", 256, 256, 1, &[100], &[0], &[0], &block_data);
    let engine = QueryEngine::new(&temp.path).unwrap();
    let _leaves = engine.reconstruct_all_leaves().unwrap();
}

/// Case 44: Exception value exactly 65536 (first value requiring U32).
#[test]
fn test_tier2_compression_boundary_u32_start() {
    let temp = TempDb::new("t2_comp_u32_start.db");
    let mut deltas = vec![0u32; 255];
    deltas[0] = 65537;
    let bit_widths = [0u8; 32];
    let block_data = build_v2_block_data(&deltas, &bit_widths, EXCEPTION_MODE_U32, false);
    temp.write(b"S2PP", 256, 256, 1, &[100], &[0], &[0], &block_data);
    let engine = QueryEngine::new(&temp.path).unwrap();
    let _leaves = engine.reconstruct_all_leaves().unwrap();
}

/// Case 45: Zero bit-width sub-block with exception value exactly 0.
#[test]
fn test_tier2_compression_zero_width_val_0() {
    let temp = TempDb::new("t2_comp_zero_0.db");
    let mut deltas = vec![0u32; 255];
    deltas[0] = 1;
    let bit_widths = [0u8; 32];
    let block_data = build_v2_block_data(&deltas, &bit_widths, EXCEPTION_MODE_U4, false);
    temp.write(b"S2PP", 256, 256, 1, &[100], &[0], &[0], &block_data);
    let engine = QueryEngine::new(&temp.path).unwrap();
    let _leaves = engine.reconstruct_all_leaves().unwrap();
}

/// Tier 2: File Header Integrity & Magic Bytes checks.
#[test]
fn test_tier2_header_empty_file() {
    let temp = TempDb::new("t2_empty_header.db");
    File::create(&temp.path).unwrap();
    let result = QueryEngine::new(&temp.path);
    assert!(result.is_err());
}

#[test]
fn test_tier2_header_truncated_magic() {
    let temp = TempDb::new("t2_truncated_magic.db");
    {
        let mut file = File::create(&temp.path).unwrap();
        file.write_all(b"S2P").unwrap();
    }
    let result = QueryEngine::new(&temp.path);
    assert!(result.is_err());
}

#[test]
fn test_tier2_header_invalid_magic_bytes() {
    let temp = TempDb::new("t2_invalid_magic.db");
    temp.write(b"S2PX", 10, 256, 1, &[0], &[0], &[0], &[]);
    let result = QueryEngine::new(&temp.path);
    assert!(result.is_err());
}

#[test]
fn test_tier2_header_zero_sizes() {
    let temp_block_size = TempDb::new("t2_zero_block_size.db");
    temp_block_size.write(b"S2PP", 10, 0, 1, &[0], &[0], &[0], &[]);
    let result_block_size = QueryEngine::new(&temp_block_size.path);
    assert!(result_block_size.is_err());

    let temp_block_count = TempDb::new("t2_zero_block_count.db");
    temp_block_count.write(b"S2PP", 10, 256, 0, &[0], &[0], &[0], &[]);
    let result_block_count = QueryEngine::new(&temp_block_count.path);
    assert!(result_block_count.is_err());

    let temp_count = TempDb::new("t2_zero_count.db");
    temp_count.write(b"S2PP", 0, 256, 1, &[0], &[0], &[0], &[]);
    let result_count = QueryEngine::new(&temp_count.path);
    assert!(result_count.is_err());
}

#[test]
fn test_tier2_header_oversized_block_size() {
    let temp = TempDb::new("t2_oversized_block.db");
    temp.write(b"S2PP", 10, 257, 1, &[0], &[0], &[0], &[]);
    let result = QueryEngine::new(&temp.path);
    assert!(result.is_err());
}

/// Tier 2: Database Offsets and Mappings checks.
#[test]
fn test_tier2_offset_truncated_index_tables() {
    let temp = TempDb::new("t2_trunc_offset.db");
    {
        let mut file = File::create(&temp.path).unwrap();
        file.write_all(b"S2PP").unwrap();
        file.write_all(&512u32.to_le_bytes()).unwrap();
        file.write_all(&256u32.to_le_bytes()).unwrap();
        file.write_all(&2u32.to_le_bytes()).unwrap();
        file.write_all(&[0u8; 8]).unwrap();
    }
    let result = QueryEngine::new(&temp.path);
    assert!(result.is_err());
}

#[test]
fn test_tier2_offset_out_of_bounds() {
    let temp = TempDb::new("t2_oob_offset.db");
    let block_data = build_v2_block_data(&[0], &[0; 32], EXCEPTION_MODE_U32, false);
    temp.write(b"S2PP", 2, 256, 1, &[1], &[1000], &[0], &block_data);
    let result = QueryEngine::new(&temp.path);
    assert!(result.is_err());
}

#[test]
fn test_tier2_offset_negative_size_block() {
    let temp = TempDb::new("t2_neg_block.db");
    let b1 = build_v2_block_data(&vec![0; 255], &[0; 32], EXCEPTION_MODE_U32, false);
    let b2 = build_v2_block_data(&vec![0; 255], &[0; 32], EXCEPTION_MODE_U32, false);
    let mut block_data = b1;
    block_data.extend_from_slice(&b2);
    temp.write(b"S2PP", 512, 256, 2, &[1, 3], &[0], &[10, 5], &block_data);
    let result = QueryEngine::new(&temp.path);
    assert!(result.is_err());
}

#[test]
fn test_tier2_offset_relative_overflow() {
    let temp = TempDb::new("t2_rel_overflow.db");
    let mut block_data = vec![0u8; 65535];
    let b1 = build_v2_block_data(&vec![0; 255], &[0; 32], EXCEPTION_MODE_U32, false);
    block_data.extend_from_slice(&b1);
    temp.write(
        b"S2PP",
        512,
        256,
        2,
        &[1, 3],
        &[0],
        &[0, 65535],
        &block_data,
    );
    let engine = QueryEngine::new(&temp.path);
    let _ = engine;
}

#[test]
fn test_tier2_offset_corrupt_cast_slice() {
    let temp = TempDb::new("t2_corrupt_cast.db");
    {
        let mut file = File::create(&temp.path).unwrap();
        file.write_all(b"S2PP").unwrap();
        file.write_all(&10u32.to_le_bytes()).unwrap();
        file.write_all(&256u32.to_le_bytes()).unwrap();
        file.write_all(&1u32.to_le_bytes()).unwrap();
        file.write_all(&[0u8; 9]).unwrap();
    }
    let result = QueryEngine::new(&temp.path);
    assert!(result.is_err());
}

/// Tier 2: Cell ID Bounds & Compact Representation checks.
#[test]
fn test_tier2_cell_id_zero_bounds() {
    let database_path = "../population_density_database.bin";
    assert!(
        Path::new(database_path).exists(),
        "Database file '{}' not found. Please build the database first using 'cargo run --release --bin build_database'.",
        database_path
    );
    let query_engine = QueryEngine::new(database_path).unwrap();
    let result = query_engine.query(0);
    assert!(result.is_err());
}

#[test]
fn test_tier2_cell_id_max_compact_limit() {
    let temp = TempDb::new("t2_max_compact.db");
    let block_data = build_v2_block_data(&[0], &[0; 32], EXCEPTION_MODE_U32, false);
    temp.write(b"S2PP", 2, 256, 1, &[0x0FFFFFFF], &[0], &[0], &block_data);
    let engine = QueryEngine::new(&temp.path);
    let _ = engine;
}

#[test]
fn test_tier2_cell_id_exceeds_max_compact_limit() {
    let temp = TempDb::new("t2_exceed_compact.db");
    let block_data = build_v2_block_data(&[0], &[0; 32], EXCEPTION_MODE_U32, false);
    temp.write(b"S2PP", 2, 256, 1, &[0x10000000], &[0], &[0], &block_data);
    let engine = QueryEngine::new(&temp.path);
    assert!(engine.is_err());
}

#[test]
fn test_tier2_cell_id_reconstruction_overflow() {
    let temp = TempDb::new("t2_recon_overflow.db");
    let block_data = build_v2_block_data(&[0], &[0; 32], EXCEPTION_MODE_U32, false);
    temp.write(b"S2PP", 2, 256, 1, &[0x0FFFFFFF], &[0], &[0], &block_data);
    let engine = QueryEngine::new(&temp.path);
    if let Ok(eng) = engine {
        let _ = eng.query(0x0FFFFFFF_u64 << SHIFT_COMPACT);
    }
}

#[test]
#[allow(clippy::identity_op)]
fn test_tier2_cell_id_extreme_face_cells() {
    let database_path = "../population_density_database.bin";
    assert!(
        Path::new(database_path).exists(),
        "Database file '{}' not found. Please build the database first using 'cargo run --release --bin build_database'.",
        database_path
    );
    let query_engine = QueryEngine::new(database_path).unwrap();
    let face_0_cell = (0u64 << 61) | (1u64 << 36);
    let face_5_cell = (5u64 << 61) | (1u64 << 36);
    let result_0 = query_engine.query(face_0_cell);
    let result_5 = query_engine.query(face_5_cell);
    assert!(result_0.is_ok());
    assert!(result_5.is_ok());
}

/// Tier 2: PFOR Decoding & Truncation checks.
#[test]
fn test_tier2_pfor_corrupt_bit_width() {
    let temp = TempDb::new("t2_corrupt_width.db");
    let mut bit_widths = [0u8; 32];
    bit_widths[0] = 29;
    let block_data = pack_v2_header(&bit_widths, 0, EXCEPTION_MODE_U32, false);
    temp.write(b"S2PP", 2, 256, 1, &[1], &[0], &[0], &block_data);
    let engine = QueryEngine::new(&temp.path);
    if let Ok(eng) = engine {
        let result = eng.query(1u64 << SHIFT_COMPACT);
        assert!(result.is_err());
    }
}

#[test]
fn test_tier2_pfor_zero_bit_width_block() {
    let temp = TempDb::new("t2_zero_width_block.db");
    let block_data = build_v2_block_data(&[0], &[0; 32], EXCEPTION_MODE_U32, false);
    temp.write(b"S2PP", 2, 256, 1, &[100], &[0], &[0], &block_data);
    let engine = QueryEngine::new(&temp.path).unwrap();
    let leaves = engine.reconstruct_all_leaves().unwrap();
    assert_eq!(leaves, vec![100, 101]);
}

#[test]
fn test_tier2_pfor_truncated_bitstream() {
    let temp = TempDb::new("t2_trunc_bitstream.db");
    let block_data = pack_v2_header(&[8; 32], 0, EXCEPTION_MODE_U32, false);
    temp.write(b"S2PP", 3, 256, 1, &[100], &[0], &[0], &block_data);
    let engine = QueryEngine::new(&temp.path).unwrap();
    let result = engine.query(100u64 << SHIFT_COMPACT);
    assert!(result.is_err());
}

#[test]
fn test_tier2_pfor_truncated_block_header() {
    let temp = TempDb::new("t2_trunc_block_header.db");
    temp.write(b"S2PP", 2, 256, 1, &[100], &[0], &[0], &[8; 10]);
    let engine = QueryEngine::new(&temp.path).unwrap();
    let result = engine.query(100u64 << SHIFT_COMPACT);
    assert!(result.is_err());
}

#[test]
fn test_tier2_pfor_max_bit_width_28() {
    let temp = TempDb::new("t2_max_width_28.db");
    let mut bit_widths = [0u8; 32];
    bit_widths[0] = 28;
    let block_data = build_v2_block_data(&[5], &bit_widths, EXCEPTION_MODE_U32, false);
    temp.write(b"S2PP", 2, 256, 1, &[1], &[0], &[0], &block_data);
    let engine = QueryEngine::new(&temp.path).unwrap();
    let leaves = engine.reconstruct_all_leaves().unwrap();
    assert_eq!(leaves, vec![1, 7]);
}

/// Tier 2: Exception Table & Modes checks.
#[test]
fn test_tier2_exception_mode_0_corrupt_index() {
    let temp = TempDb::new("t2_mode_0_corrupt.db");
    let bit_widths = [0u8; 32];
    let mut block_data = pack_v2_header(&bit_widths, 1, EXCEPTION_MODE_U32, false).to_vec();
    block_data.push(100);
    block_data.extend_from_slice(&100u32.to_le_bytes());
    temp.write(b"S2PP", 2, 256, 1, &[1], &[0], &[0], &block_data);
    let engine = QueryEngine::new(&temp.path).unwrap();
    let result = engine.query(1u64 << SHIFT_COMPACT);
    assert!(result.is_err());
}

#[test]
fn test_tier2_exception_mode_1_corrupt_index() {
    let temp = TempDb::new("t2_mode_1_corrupt.db");
    let bit_widths = [0u8; 32];
    let mut block_data = pack_v2_header(&bit_widths, 1, EXCEPTION_MODE_U8, false).to_vec();
    block_data.push(100);
    block_data.push(100);
    temp.write(b"S2PP", 2, 256, 1, &[1], &[0], &[0], &block_data);
    let engine = QueryEngine::new(&temp.path).unwrap();
    let result = engine.query(1u64 << SHIFT_COMPACT);
    assert!(result.is_err());
}

#[test]
fn test_tier2_exception_mode_2_corrupt_index() {
    let temp = TempDb::new("t2_mode_2_corrupt.db");
    let bit_widths = [0u8; 32];
    let mut block_data = pack_v2_header(&bit_widths, 1, EXCEPTION_MODE_U16, false).to_vec();
    block_data.push(100);
    block_data.extend_from_slice(&100u16.to_le_bytes());
    temp.write(b"S2PP", 2, 256, 1, &[1], &[0], &[0], &block_data);
    let engine = QueryEngine::new(&temp.path).unwrap();
    let result = engine.query(1u64 << SHIFT_COMPACT);
    assert!(result.is_err());
}

#[test]
fn test_tier2_exception_table_truncation() {
    let temp = TempDb::new("t2_exc_trunc.db");
    let bit_widths = [0u8; 32];
    let mut block_data = pack_v2_header(&bit_widths, 1, EXCEPTION_MODE_U8, false).to_vec();
    block_data.push(0);
    temp.write(b"S2PP", 2, 256, 1, &[1], &[0], &[0], &block_data);
    let engine = QueryEngine::new(&temp.path).unwrap();
    let result = engine.query(1u64 << SHIFT_COMPACT);
    assert!(result.is_err());
}

#[test]
fn test_tier2_exception_mode_invalid_bits() {
    let temp = TempDb::new("t2_exc_invalid_bits.db");
    let bit_widths = [0u8; 32];
    let block_data = build_v2_block_data(&[100], &bit_widths, EXCEPTION_MODE_U32, false);
    temp.write(b"S2PP", 2, 256, 1, &[1], &[0], &[0], &block_data);
    let engine = QueryEngine::new(&temp.path).unwrap();
    let leaves = engine.reconstruct_all_leaves().unwrap();
    assert_eq!(leaves, vec![1, 102]);
}

// ==========================================
// Tier 3: Cross-feature combinations.
// ==========================================

/// Case 51: U4 exceptions + bitmasking.
#[test]
fn test_tier3_u4_bitmask_combo() {
    let temp = TempDb::new("t3_u4_bitmask_combo.db");
    let mut deltas = vec![0u32; 255];
    for idx in 0..35 {
        deltas[idx] = 1;
    }
    let bit_widths = [0u8; 32];
    let block_data = build_v2_block_data(&deltas, &bit_widths, EXCEPTION_MODE_U4, true);
    temp.write(b"S2PP", 256, 256, 1, &[100], &[0], &[0], &block_data);
    let engine = QueryEngine::new(&temp.path).unwrap();
    let _leaves = engine.reconstruct_all_leaves().unwrap();
}

/// Case 52: U8 exceptions + bitmasking.
#[test]
fn test_tier3_u8_bitmask_combo() {
    let temp = TempDb::new("t3_u8_bitmask_combo.db");
    let mut deltas = vec![0u32; 255];
    for idx in 0..35 {
        deltas[idx] = 200;
    }
    let bit_widths = [0u8; 32];
    let block_data = build_v2_block_data(&deltas, &bit_widths, EXCEPTION_MODE_U8, true);
    temp.write(b"S2PP", 256, 256, 1, &[100], &[0], &[0], &block_data);
    let engine = QueryEngine::new(&temp.path).unwrap();
    let _leaves = engine.reconstruct_all_leaves().unwrap();
}

/// Case 53: U16 exceptions + index list.
#[test]
fn test_tier3_u16_bytelist_combo() {
    let temp = TempDb::new("t3_u16_bytelist_combo.db");
    let mut deltas = vec![0u32; 255];
    deltas[0] = 5000;
    deltas[5] = 6000;
    let bit_widths = [0u8; 32];
    let block_data = build_v2_block_data(&deltas, &bit_widths, EXCEPTION_MODE_U16, false);
    temp.write(b"S2PP", 256, 256, 1, &[100], &[0], &[0], &block_data);
    let engine = QueryEngine::new(&temp.path).unwrap();
    let _leaves = engine.reconstruct_all_leaves().unwrap();
}

/// Case 54: U32 exceptions + index list.
#[test]
fn test_tier3_u32_bytelist_combo() {
    let temp = TempDb::new("t3_u32_bytelist_combo.db");
    let mut deltas = vec![0u32; 255];
    deltas[0] = 100000;
    deltas[5] = 200000;
    let bit_widths = [0u8; 32];
    let block_data = build_v2_block_data(&deltas, &bit_widths, EXCEPTION_MODE_U32, false);
    temp.write(b"S2PP", 256, 256, 1, &[100], &[0], &[0], &block_data);
    let engine = QueryEngine::new(&temp.path).unwrap();
    let _leaves = engine.reconstruct_all_leaves().unwrap();
}

/// Case 55: Zero-bitwidth sub-blocks + U4 exceptions + index list.
#[test]
fn test_tier3_zero_width_u4_bytelist_combo() {
    let temp = TempDb::new("t3_zero_u4_bytelist_combo.db");
    let mut deltas = vec![0u32; 255];
    deltas[0] = 5;
    let bit_widths = [0u8; 32];
    let block_data = build_v2_block_data(&deltas, &bit_widths, EXCEPTION_MODE_U4, false);
    temp.write(b"S2PP", 256, 256, 1, &[100], &[0], &[0], &block_data);
    let engine = QueryEngine::new(&temp.path).unwrap();
    let _leaves = engine.reconstruct_all_leaves().unwrap();
}

/// Case 56: Max-bitwidth (28) sub-blocks + U32 exceptions + bitmasking.
#[test]
fn test_tier3_max_width_u32_bitmask_combo() {
    let temp = TempDb::new("t3_max_u32_bitmask_combo.db");
    let mut deltas = vec![0u32; 255];
    for idx in 0..35 {
        deltas[idx] = 0x100001;
    }
    let mut bit_widths = [0u8; 32];
    bit_widths[0] = 20;
    let block_data = build_v2_block_data(&deltas, &bit_widths, EXCEPTION_MODE_U32, true);
    temp.write(b"S2PP", 256, 256, 1, &[101], &[0], &[0], &block_data);
    let engine = QueryEngine::new(&temp.path).unwrap();
    let _leaves = engine.reconstruct_all_leaves().unwrap();
}

// ==========================================
// Tier 4: Real-world application scenarios.
// ==========================================

// Beijing geographic queries in densely populated city.

#[test]
fn test_beijing_exact_l12_query() {
    let database_path = "../population_density_database.bin";
    assert!(
        Path::new(database_path).exists(),
        "Database file '{}' not found. Please build the database first using 'cargo run --release --bin build_database'.",
        database_path
    );
    let query_engine = QueryEngine::new(database_path).unwrap();
    let beijing_point = LatLng::from_degrees(39.9042, 116.4074);
    let beijing_cell_id = CellID::from(beijing_point);
    let result = query_engine.query(beijing_cell_id.parent(12).0).unwrap();
    assert_ne!(result, 0);
    assert_eq!(get_level(result), 12);
}

#[test]
fn test_beijing_finer_levels_l13_to_l30() {
    let database_path = "../population_density_database.bin";
    assert!(
        Path::new(database_path).exists(),
        "Database file '{}' not found. Please build the database first using 'cargo run --release --bin build_database'.",
        database_path
    );
    let query_engine = QueryEngine::new(database_path).unwrap();
    let beijing_point = LatLng::from_degrees(39.9042, 116.4074);
    let beijing_cell_id = CellID::from(beijing_point);
    let expected_ancestor = query_engine.query(beijing_cell_id.parent(12).0).unwrap();
    for level in 13..=30 {
        let parent = beijing_cell_id.parent(level as u64);
        let result = query_engine.query(parent.0).unwrap();
        assert_eq!(result, expected_ancestor);
    }
}

#[test]
fn test_beijing_coarser_levels_l0_to_l11() {
    let database_path = "../population_density_database.bin";
    assert!(
        Path::new(database_path).exists(),
        "Database file '{}' not found. Please build the database first using 'cargo run --release --bin build_database'.",
        database_path
    );
    let query_engine = QueryEngine::new(database_path).unwrap();
    let beijing_point = LatLng::from_degrees(39.9042, 116.4074);
    let beijing_cell_id = CellID::from(beijing_point);
    for level in 0..=11 {
        let parent = beijing_cell_id.parent(level as u64);
        let result = query_engine.query(parent.0).unwrap();
        assert_ne!(result, 0);
        assert!(get_level(result) <= level);
    }
}

#[test]
fn test_beijing_neighborhood_radius() {
    let database_path = "../population_density_database.bin";
    assert!(
        Path::new(database_path).exists(),
        "Database file '{}' not found. Please build the database first using 'cargo run --release --bin build_database'.",
        database_path
    );
    let query_engine = QueryEngine::new(database_path).unwrap();
    let offsets = [0.01, -0.01, 0.05, -0.05, 0.1, -0.1];
    for &offset_latitude in &offsets {
        for &offset_longitude in &offsets {
            let point =
                LatLng::from_degrees(39.9042 + offset_latitude, 116.4074 + offset_longitude);
            let cell_id = CellID::from(point);
            let result = query_engine.query(cell_id.0).unwrap();
            assert_ne!(result, 0);
            assert!(get_level(result) <= 12);
        }
    }
}

#[test]
fn test_beijing_unpopulated_pockets() {
    let database_path = "../population_density_database.bin";
    assert!(
        Path::new(database_path).exists(),
        "Database file '{}' not found. Please build the database first using 'cargo run --release --bin build_database'.",
        database_path
    );
    let query_engine = QueryEngine::new(database_path).unwrap();
    let points = [
        LatLng::from_degrees(40.5000, 117.0000),
        LatLng::from_degrees(40.2000, 116.1000),
        LatLng::from_degrees(39.6000, 115.9000),
    ];
    for point in &points {
        let cell_id = CellID::from(*point);
        let result = query_engine.query(cell_id.0).unwrap();
        assert_ne!(result, 0);
        assert!(get_level(result) <= 12);
    }
}

// Mount Everest geographic queries.

#[test]
fn test_everest_exact_l10_query() {
    let database_path = "../population_density_database.bin";
    assert!(
        Path::new(database_path).exists(),
        "Database file '{}' not found. Please build the database first using 'cargo run --release --bin build_database'.",
        database_path
    );
    let query_engine = QueryEngine::new(database_path).unwrap();
    let everest_point = LatLng::from_degrees(27.9881, 86.9250);
    let everest_cell_id = CellID::from(everest_point);
    let result = query_engine.query(everest_cell_id.parent(10).0).unwrap();
    assert_ne!(result, 0);
    assert_eq!(get_level(result), 10);
}

#[test]
fn test_everest_finer_levels_l11_to_l30() {
    let database_path = "../population_density_database.bin";
    assert!(
        Path::new(database_path).exists(),
        "Database file '{}' not found. Please build the database first using 'cargo run --release --bin build_database'.",
        database_path
    );
    let query_engine = QueryEngine::new(database_path).unwrap();
    let everest_point = LatLng::from_degrees(27.9881, 86.9250);
    let everest_cell_id = CellID::from(everest_point);
    for level in 11..=30 {
        let parent = everest_cell_id.parent(level as u64);
        let result = query_engine.query(parent.0).unwrap();
        assert_eq!(get_level(result), 10);
    }
}

#[test]
fn test_everest_coarser_levels_l0_to_l9() {
    let database_path = "../population_density_database.bin";
    assert!(
        Path::new(database_path).exists(),
        "Database file '{}' not found. Please build the database first using 'cargo run --release --bin build_database'.",
        database_path
    );
    let query_engine = QueryEngine::new(database_path).unwrap();
    let everest_point = LatLng::from_degrees(27.9881, 86.9250);
    let everest_cell_id = CellID::from(everest_point);
    for level in 0..=9 {
        let parent = everest_cell_id.parent(level as u64);
        let result = query_engine.query(parent.0).unwrap();
        assert_ne!(result, 0);
        assert!(get_level(result) <= level);
    }
}

#[test]
fn test_everest_valleys_vs_peaks() {
    let database_path = "../population_density_database.bin";
    assert!(
        Path::new(database_path).exists(),
        "Database file '{}' not found. Please build the database first using 'cargo run --release --bin build_database'.",
        database_path
    );
    let query_engine = QueryEngine::new(database_path).unwrap();
    let kathmandu = LatLng::from_degrees(27.7172, 85.3240);
    let lhasa = LatLng::from_degrees(29.6524, 91.1172);
    let kathmandu_result = query_engine.query(CellID::from(kathmandu).0).unwrap();
    let lhasa_result = query_engine.query(CellID::from(lhasa).0).unwrap();
    assert_eq!(get_level(kathmandu_result), 12);
    assert_eq!(get_level(lhasa_result), 12);
}

#[test]
fn test_everest_unpopulated_bounding_box() {
    let database_path = "../population_density_database.bin";
    assert!(
        Path::new(database_path).exists(),
        "Database file '{}' not found. Please build the database first using 'cargo run --release --bin build_database'.",
        database_path
    );
    let query_engine = QueryEngine::new(database_path).unwrap();
    let point = LatLng::from_degrees(27.9881 + 0.5, 86.9250 + 0.5);
    let cell_id = CellID::from(point);
    let result = query_engine.query(cell_id.0).unwrap();
    assert_ne!(result, 0);
    assert!(get_level(result) < 10);
}

// Ocean geographic queries.

#[test]
fn test_ocean_deep_sea_pacific() {
    let database_path = "../population_density_database.bin";
    assert!(
        Path::new(database_path).exists(),
        "Database file '{}' not found. Please build the database first using 'cargo run --release --bin build_database'.",
        database_path
    );
    let query_engine = QueryEngine::new(database_path).unwrap();
    let point = LatLng::from_degrees(0.0, -140.0);
    let cell_id = CellID::from(point);
    let result = query_engine.query(cell_id.0).unwrap();
    assert_ne!(result, 0);
    assert!(get_level(result) <= 3);
}

#[test]
fn test_ocean_coastline_gradient() {
    let database_path = "../population_density_database.bin";
    assert!(
        Path::new(database_path).exists(),
        "Database file '{}' not found. Please build the database first using 'cargo run --release --bin build_database'.",
        database_path
    );
    let query_engine = QueryEngine::new(database_path).unwrap();
    let points = [
        LatLng::from_degrees(31.2304, 126.0),
        LatLng::from_degrees(31.2304, 124.0),
        LatLng::from_degrees(31.2304, 122.5),
        LatLng::from_degrees(31.2304, 121.8),
        LatLng::from_degrees(31.2304, 121.4737),
    ];
    let mut previous_level = 0;
    for point in &points {
        let result = query_engine.query(CellID::from(*point).0).unwrap();
        let level = get_level(result);
        assert!(level >= previous_level);
        previous_level = level;
    }
}

#[test]
fn test_ocean_unpopulated_island() {
    let database_path = "../population_density_database.bin";
    assert!(
        Path::new(database_path).exists(),
        "Database file '{}' not found. Please build the database first using 'cargo run --release --bin build_database'.",
        database_path
    );
    let query_engine = QueryEngine::new(database_path).unwrap();
    let point = LatLng::from_degrees(-24.37, -128.3);
    let result = query_engine.query(CellID::from(point).0).unwrap();
    assert_ne!(result, 0);
    assert!(get_level(result) <= 6);
}

#[test]
fn test_ocean_polar_regions() {
    let database_path = "../population_density_database.bin";
    assert!(
        Path::new(database_path).exists(),
        "Database file '{}' not found. Please build the database first using 'cargo run --release --bin build_database'.",
        database_path
    );
    let query_engine = QueryEngine::new(database_path).unwrap();
    let point = LatLng::from_degrees(-80.0, 0.0);
    let result = query_engine.query(CellID::from(point).0).unwrap();
    assert_ne!(result, 0);
    assert!(get_level(result) <= 3);
}

#[test]
fn test_ocean_trench_bounds() {
    let database_path = "../population_density_database.bin";
    assert!(
        Path::new(database_path).exists(),
        "Database file '{}' not found. Please build the database first using 'cargo run --release --bin build_database'.",
        database_path
    );
    let query_engine = QueryEngine::new(database_path).unwrap();
    let point = LatLng::from_degrees(11.3493, 142.1996);
    let result = query_engine.query(CellID::from(point).0).unwrap();
    assert_ne!(result, 0);
    assert!(get_level(result) <= 12);
}

// S2 utility calculations.

#[test]
fn test_utility_get_level_exhaustive() {
    let point = LatLng::from_degrees(39.9042, 116.4074);
    let cell_id = CellID::from(point);
    for level in 0..=30 {
        let parent = cell_id.parent(level as u64);
        assert_eq!(get_level(parent.0), level);
    }
}

#[test]
fn test_utility_get_ancestor_all_parents() {
    let point = LatLng::from_degrees(39.9042, 116.4074);
    let cell_id = CellID::from(point);
    for level in 0..=30 {
        let expected = cell_id.parent(level as u64).0;
        let actual = get_ancestor(cell_id.0, level);
        assert_eq!(actual, expected);
    }
}

#[test]
fn test_utility_get_parent_id() {
    let point = LatLng::from_degrees(39.9042, 116.4074);
    let cell_id = CellID::from(point);
    for level in 0..=30 {
        let ancestor = get_ancestor(cell_id.0, level);
        let parent = population_density::get_parent_id(cell_id.0, level);
        assert_eq!(ancestor, parent);
    }
}

#[test]
fn test_utility_get_children_ids() {
    let point = LatLng::from_degrees(39.9042, 116.4074);
    let cell_id = CellID::from(point);
    for level in 0..29 {
        let parent = cell_id.parent(level as u64);
        let children = population_density::get_children_ids(parent.0, level);
        for &child in &children {
            assert_eq!(get_level(child), level + 1);
            assert_eq!(get_ancestor(child, level), parent.0);
        }
    }
}

#[test]
fn test_utility_precondition_panics() {
    let result_get_level = std::panic::catch_unwind(|| {
        get_level(0);
    });
    assert!(result_get_level.is_err());

    let result_get_ancestor = std::panic::catch_unwind(|| {
        get_ancestor(1, 31);
    });
    assert!(result_get_ancestor.is_err());

    let result_children_0 = std::panic::catch_unwind(|| {
        population_density::get_children_ids(0, 5);
    });
    assert!(result_children_0.is_err());

    let result_children_30 = std::panic::catch_unwind(|| {
        population_density::get_children_ids(1, 30);
    });
    assert!(result_children_30.is_err());
}

// Additional production workloads.

#[test]
fn test_workload_exhaustive_global_l12_parity() {
    const S2_FACE_SHIFT: u32 = 61;
    const STEP_SHIFT: u32 = 37;

    let database_path = "../population_density_database.bin";
    assert!(
        Path::new(database_path).exists(),
        "Database file '{}' not found. Please build the database first using 'cargo run --release --bin build_database'.",
        database_path
    );

    let query_engine = QueryEngine::new(database_path).unwrap();
    let naive_oracle = NaiveOracle::new(&query_engine);

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
                }
                s2_cell_id += step;
            }
            face_mismatches
        })
        .sum();

    assert_eq!(mismatches, 0);
}

#[test]
fn test_workload_sampling_parity_l13_to_l30() {
    let database_path = "../population_density_database.bin";
    assert!(
        Path::new(database_path).exists(),
        "Database file '{}' not found. Please build the database first using 'cargo run --release --bin build_database'.",
        database_path
    );
    let query_engine = QueryEngine::new(database_path).unwrap();
    let naive_oracle = NaiveOracle::new(&query_engine);

    let mut rng = SimpleRng::new(42);
    for _ in 0..50_000 {
        let face = rng.next_u64() % 6;
        let level = (rng.next_u64() % 18) + 13;
        let sentinel_position = 60 - 2 * level;
        let position_mask = (1u64 << (60 - sentinel_position)) - 1;
        let position = rng.next_u64() & position_mask;
        let cell_id = (face << 61) | (position << sentinel_position) | (1u64 << sentinel_position);

        let optimized_result = query_engine.query(cell_id).unwrap();
        let naive_result = naive_oracle.query(cell_id);
        assert_eq!(optimized_result, naive_result);
    }
}

#[test]
fn test_workload_high_throughput_concurrent_queries() {
    let database_path = "../population_density_database.bin";
    assert!(
        Path::new(database_path).exists(),
        "Database file '{}' not found. Please build the database first using 'cargo run --release --bin build_database'.",
        database_path
    );
    let query_engine = std::sync::Arc::new(QueryEngine::new(database_path).unwrap());
    let mut handles = Vec::new();

    for thread_index in 0..64 {
        let engine_clone = std::sync::Arc::clone(&query_engine);
        let handle = std::thread::spawn(move || {
            let mut rng = SimpleRng::new(thread_index as u64 + 100);
            for _ in 0..5_000 {
                let face = rng.next_u64() % 6;
                let level = rng.next_u64() % 13;
                let sentinel_position = 60 - 2 * level;
                let position_mask = (1u64 << (60 - sentinel_position)) - 1;
                let position = rng.next_u64() & position_mask;
                let cell_id =
                    (face << 61) | (position << sentinel_position) | (1u64 << sentinel_position);

                if cell_id != 0 {
                    let _ = engine_clone.query(cell_id);
                }
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn test_workload_random_latlng_sampling() {
    let database_path = "../population_density_database.bin";
    assert!(
        Path::new(database_path).exists(),
        "Database file '{}' not found. Please build the database first using 'cargo run --release --bin build_database'.",
        database_path
    );
    let query_engine = QueryEngine::new(database_path).unwrap();
    let naive_oracle = NaiveOracle::new(&query_engine);

    let mut rng = SimpleRng::new(1337);
    for _ in 0..100_000 {
        let latitude = rng.next_range(-90.0, 90.0);
        let longitude = rng.next_range(-180.0, 180.0);
        let point = LatLng::from_degrees(latitude, longitude);
        let cell_id = CellID::from(point);
        let level = rng.next_u64() % 31;
        let parent = cell_id.parent(level);

        let optimized_result = query_engine.query(parent.0).unwrap();
        let naive_result = naive_oracle.query(parent.0);
        assert_eq!(optimized_result, naive_result);
    }
}

#[test]
fn test_workload_cache_thrashing_stress() {
    let database_path = "../population_density_database.bin";
    assert!(
        Path::new(database_path).exists(),
        "Database file '{}' not found. Please build the database first using 'cargo run --release --bin build_database'.",
        database_path
    );
    let query_engine = QueryEngine::new(database_path).unwrap();

    let mut rng = SimpleRng::new(999);
    for _ in 0..10_000 {
        let face = if rng.next_u64() % 2 == 0 { 0u64 } else { 5u64 };
        let level = 12u64;
        let sentinel_position = 60 - 2 * level;
        let position_mask = (1u64 << (60 - sentinel_position)) - 1;
        let position = rng.next_u64() & position_mask;
        let cell_id = (face << 61) | (position << sentinel_position) | (1u64 << sentinel_position);

        let result = query_engine.query(cell_id);
        assert!(result.is_ok());
    }
}
