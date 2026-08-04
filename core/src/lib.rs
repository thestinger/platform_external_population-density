//! Provides memory-mapped S2PD queries and S2 utilities for population density databases.

use anyhow::{Result, anyhow};
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;

pub use memmap2;

mod topology;
#[doc(hidden)]
pub mod topology_format;

use topology::TopologyQueryEngine;

// S2 geometry and database constants.
/// Defines the maximum S2 cell level.
pub const MAX_S2_LEVEL: u32 = 30;
/// Defines the number of S2 position bits per level.
pub const BITS_PER_LEVEL: u32 = 2;
/// Defines the number of S2 position bits below the face field.
pub const MAX_S2_BITS: u32 = 60;
/// Defines the maximum level represented by the database.
pub const MAX_DB_LEVEL: u32 = 12;
/// Defines the minimum population represented by a returned S2 cell.
pub const POPULATION_THRESHOLD: i64 = 1000;
/// Defines the fixed-point units per person used for population accumulation.
pub const POPULATION_FIXED_POINT_SCALE: i64 = 1 << 20;
/// Defines the population threshold in fixed-point units.
pub const POPULATION_THRESHOLD_FIXED: i64 = POPULATION_THRESHOLD * POPULATION_FIXED_POINT_SCALE;
/// Defines the number of S2 root faces.
pub const NUM_ROOT_FACES: usize = 6;
/// Defines the right shift used for compact database cell IDs.
pub const SHIFT_COMPACT: u32 = 36;
/// Defines the largest valid compact S2 cell ID.
pub const MAX_COMPACT_CELL_ID: u32 = 0x0FFFFFFF;

/// Defines the number of bits in the S2 face field.
pub const S2_FACE_BITS: u32 = 3;
/// Defines the bit position of the S2 face field within a cell ID.
pub const S2_FACE_SHIFT: u32 = 61;

/// Returns the level of a valid S2 cell ID.
#[allow(clippy::manual_is_multiple_of)]
pub fn get_level(s2_cell_id: u64) -> u32 {
    assert!(s2_cell_id != 0, "S2 cell ID cannot be 0");
    let face = s2_cell_id >> S2_FACE_SHIFT;
    assert!(face < NUM_ROOT_FACES as u64, "invalid S2 face ID: {}", face);
    let lowest_bit = s2_cell_id & s2_cell_id.wrapping_neg();
    let trailing_zeros = lowest_bit.trailing_zeros();
    assert!(
        trailing_zeros <= MAX_S2_BITS,
        "invalid S2 cell ID trailing zeros: {}",
        trailing_zeros
    );
    assert!(
        trailing_zeros % 2 == 0,
        "S2 cell ID lowest bit must be at even index: {}",
        trailing_zeros
    );
    (MAX_S2_BITS - trailing_zeros) / BITS_PER_LEVEL
}

/// Returns the S2 level or an error for an invalid cell ID.
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

/// Returns the ancestor of an S2 cell ID at the specified level.
pub fn get_ancestor(s2_cell_id: u64, level: u32) -> u64 {
    let cell_level = get_level(s2_cell_id);
    assert!(
        level <= cell_level,
        "level {} must be <= cell level {}",
        level,
        cell_level
    );
    assert!(level <= MAX_S2_LEVEL, "S2 level must be <= 30");
    let sentinel_position = MAX_S2_BITS - BITS_PER_LEVEL * level;
    let mask = !((1u64 << sentinel_position) - 1);
    (s2_cell_id & mask) | (1u64 << sentinel_position)
}

/// Returns the ancestor or an error for an invalid cell ID or level.
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

/// Returns the bitwise S2 parent ID used by the database builder.
pub fn get_parent_id(cell_id: u64, level: u32) -> u64 {
    get_ancestor(cell_id, level)
}

/// Returns the bitwise S2 child IDs used by the database builder.
pub fn get_children_ids(cell_id: u64, level: u32) -> [u64; 4] {
    assert!(
        level < MAX_S2_LEVEL,
        "S2 level must be < 30 to have children"
    );
    let cell_level = get_level(cell_id);
    assert_eq!(
        cell_level, level,
        "cell level {} does not match requested level {}",
        cell_level, level
    );
    let sentinel_position = MAX_S2_BITS - BITS_PER_LEVEL * level;
    let prefix = cell_id & !(1u64 << sentinel_position);
    [
        prefix | (0u64 << (sentinel_position - 1)) | (1u64 << (sentinel_position - 2)),
        prefix | (1u64 << (sentinel_position - 1)) | (1u64 << (sentinel_position - 2)),
        prefix | (2u64 << (sentinel_position - 1)) | (1u64 << (sentinel_position - 2)),
        prefix | (3u64 << (sentinel_position - 1)) | (1u64 << (sentinel_position - 2)),
    ]
}

/// Implements allocation-free successful queries over a memory-mapped S2PD database.
#[derive(Debug)]
pub struct QueryEngine {
    topology: TopologyQueryEngine,
}

impl QueryEngine {
    /// Opens and fully validates a memory-mapped database file.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the mapped file contents are not modified or truncated through
    /// any process or writable alias until the returned engine is dropped.
    pub unsafe fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        // SAFETY: The caller guarantees that the mapped file contents remain unchanged until the
        // engine drops its mapping.
        let mmap = unsafe { Mmap::map(&file)? };
        Self::from_mmap(mmap)
    }

    /// Initializes and fully validates an engine from an existing memory mapping.
    ///
    /// The engine keeps the mapping alive until it is dropped. A file-backed mapping retains the
    /// immutability obligations established when that mapping was created.
    pub fn from_mmap(mmap: Mmap) -> Result<Self> {
        Ok(Self {
            topology: TopologyQueryEngine::from_mmap(mmap)?,
        })
    }

    /// Initializes an engine from a preverified immutable memory mapping.
    ///
    /// This validates the layout, models, and indexes without decoding every Huffman mask. Use it
    /// only when the exact bytes already passed [`Self::from_mmap`] and database-level query
    /// verification. The engine keeps the mapping alive but does not itself establish that a
    /// file-backed mapping is immutable.
    pub fn from_verified_mmap(mmap: Mmap) -> Result<Self> {
        Ok(Self {
            topology: TopologyQueryEngine::from_verified_mmap(mmap)?,
        })
    }

    /// Returns the total number of compact leaf IDs stored in the database.
    #[inline]
    pub fn count(&self) -> usize {
        self.topology.leaf_count()
    }

    /// Finds the deepest qualifying database ancestor for a valid S2 cell ID.
    #[inline]
    pub fn query(&self, cell_id: u64) -> Result<u64> {
        self.topology.query(cell_id)
    }

    /// Reconstructs every sorted compact leaf ID stored in the database.
    pub fn reconstruct_all_leaves(&self) -> Result<Vec<u32>> {
        self.topology.reconstruct_all_leaves()
    }
}
