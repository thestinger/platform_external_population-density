//! Verifies canonical encoding and database-level query behavior for an S2PD database.

use anyhow::{Context, Result, anyhow};
use clap::Parser;
use population_density::{
    BITS_PER_LEVEL, MAX_DB_LEVEL, MAX_S2_BITS, NUM_ROOT_FACES, QueryEngine, S2_FACE_SHIFT,
    SHIFT_COMPACT, try_get_level,
};
use population_density_cli::encode_topology_database;
use rayon::prelude::*;
use std::fs::read;
use std::path::PathBuf;
use std::time::Instant;

const HASH_MULTIPLIER: u64 = 0x9e37_79b1_85eb_ca87;
const QUICK_PATH_COUNT_PER_FACE_LEVEL: u64 = 4_096;

/// Configures canonical database verification.
#[derive(Debug, Parser)]
#[command(
    author,
    version,
    about = "Verify canonical S2PD encoding and database-level query behavior."
)]
struct Arguments {
    /// Specifies the S2PD database path.
    #[arg(long, default_value = "population_density_database.bin")]
    database_path: PathBuf,

    /// Checks a deterministic sample instead of every valid level 0-12 cell.
    #[arg(long)]
    quick: bool,
}

/// Stores one reconstructed leaf for reference queries.
#[derive(Clone, Copy)]
struct ReferenceLeaf {
    cell_id: u64,
    level: u32,
}

/// Answers reference queries using only the canonical sorted leaf vector.
struct LeafVectorOracle {
    leaves: Vec<ReferenceLeaf>,
    face_offsets: [usize; NUM_ROOT_FACES + 1],
}

impl LeafVectorOracle {
    /// Builds a query oracle from sorted reconstructed leaves.
    fn new(compact_leaves: &[u32]) -> Result<Self> {
        if compact_leaves.is_empty() {
            return Err(anyhow!("database reconstructed no leaves"));
        }

        let mut leaves = Vec::with_capacity(compact_leaves.len());
        let mut previous_cell_id = None;
        for (leaf_index, &compact_leaf) in compact_leaves.iter().enumerate() {
            let cell_id = u64::from(compact_leaf) << SHIFT_COMPACT;
            if previous_cell_id.is_some_and(|previous| cell_id <= previous) {
                return Err(anyhow!(
                    "reconstructed leaves are out of order at index {}",
                    leaf_index
                ));
            }
            let level = try_get_level(cell_id)
                .with_context(|| format!("invalid reconstructed leaf at index {}", leaf_index))?;
            if level > MAX_DB_LEVEL {
                return Err(anyhow!(
                    "reconstructed leaf at index {} exceeds database level {}",
                    leaf_index,
                    MAX_DB_LEVEL
                ));
            }
            leaves.push(ReferenceLeaf { cell_id, level });
            previous_cell_id = Some(cell_id);
        }

        let mut face_offsets = [0usize; NUM_ROOT_FACES + 1];
        let mut leaf_index = 0usize;
        for (face, face_offset) in face_offsets.iter_mut().take(NUM_ROOT_FACES).enumerate() {
            *face_offset = leaf_index;
            while leaf_index < leaves.len()
                && (leaves[leaf_index].cell_id >> S2_FACE_SHIFT) as usize == face
            {
                leaf_index += 1;
            }
            if *face_offset == leaf_index {
                return Err(anyhow!("reconstructed leaves omit S2 face {}", face));
            }
        }
        face_offsets[NUM_ROOT_FACES] = leaf_index;
        if leaf_index != leaves.len() {
            return Err(anyhow!("reconstructed leaf has an invalid S2 face"));
        }

        Ok(Self {
            leaves,
            face_offsets,
        })
    }

    /// Returns the sorted reference leaves for one S2 face.
    fn face_leaves(&self, face: usize) -> &[ReferenceLeaf] {
        &self.leaves[self.face_offsets[face]..self.face_offsets[face + 1]]
    }
}

/// Verifies leaf reconstruction, canonical bytes, and query behavior.
fn main() -> Result<()> {
    let arguments = Arguments::parse();
    let database = read(&arguments.database_path).with_context(|| {
        format!(
            "failed to read database '{}'",
            arguments.database_path.display()
        )
    })?;
    let strictly_validated_engine = engine_from_snapshot(&database, true)?;
    let verified_resource_engine = engine_from_snapshot(&database, false)?;

    let start_time = Instant::now();
    let strict_leaves = strictly_validated_engine.reconstruct_all_leaves()?;
    let verified_resource_leaves = verified_resource_engine.reconstruct_all_leaves()?;
    if verified_resource_leaves != strict_leaves {
        return Err(anyhow!(
            "strict and verified-resource leaf reconstructions differ"
        ));
    }
    verify_canonical_encoding(&database, &strict_leaves)?;
    let oracle = LeafVectorOracle::new(&strict_leaves)?;
    println!(
        "Canonical encoding and strict/verified-resource reconstruction: {} leaves passed in {:.2?}.",
        strict_leaves.len(),
        start_time.elapsed()
    );

    verify_queries(&verified_resource_engine, &oracle, arguments.quick)
}

/// Creates an engine over an immutable anonymous mapping.
fn engine_from_snapshot(database: &[u8], validate_payload: bool) -> Result<QueryEngine> {
    if database.is_empty() {
        return Err(anyhow!("database file is empty"));
    }
    let mut mapping = population_density::memmap2::MmapMut::map_anon(database.len())?;
    mapping.copy_from_slice(database);
    let mapping = mapping.make_read_only()?;
    if validate_payload {
        QueryEngine::from_mmap(mapping)
    } else {
        QueryEngine::from_verified_mmap(mapping)
    }
}

/// Re-encodes reconstructed leaves and requires byte-for-byte canonical encoding.
fn verify_canonical_encoding(actual_database: &[u8], compact_leaves: &[u32]) -> Result<()> {
    let expected_database = encode_topology_database(compact_leaves)?;
    if expected_database.len() != actual_database.len() {
        return Err(anyhow!(
            "canonical encoding size mismatch: expected {} bytes, actual {} bytes",
            expected_database.len(),
            actual_database.len()
        ));
    }
    if let Some(byte_offset) = expected_database
        .iter()
        .zip(actual_database)
        .position(|(expected, actual)| expected != actual)
    {
        return Err(anyhow!(
            "canonical encoding mismatch at byte {}: expected {:02x}, actual {:02x}",
            byte_offset,
            expected_database[byte_offset],
            actual_database[byte_offset]
        ));
    }
    Ok(())
}

/// Verifies every valid database-level cell or a deterministic sample.
fn verify_queries(engine: &QueryEngine, oracle: &LeafVectorOracle, quick: bool) -> Result<()> {
    let start_time = Instant::now();
    let mut total_query_count = 0u64;
    let mut total_hash = 0u64;
    for level in 0..=MAX_DB_LEVEL {
        let sentinel_position = MAX_S2_BITS - BITS_PER_LEVEL * level;
        let path_count = 1u64 << (BITS_PER_LEVEL * level);
        let checked_path_count = if quick {
            path_count.min(QUICK_PATH_COUNT_PER_FACE_LEVEL)
        } else {
            path_count
        };
        let face_results = (0..NUM_ROOT_FACES)
            .into_par_iter()
            .map(|face| -> Result<(u64, u64)> {
                let leaves = oracle.face_leaves(face);
                let mut insertion_index = 0usize;
                let mut face_hash = 0u64;
                for query_index in 0..checked_path_count {
                    let path = sampled_path(query_index, checked_path_count, path_count);
                    let cell_id = ((face as u64) << S2_FACE_SHIFT)
                        | (path << (sentinel_position + 1))
                        | (1u64 << sentinel_position);
                    while insertion_index < leaves.len()
                        && leaves[insertion_index].cell_id < cell_id
                    {
                        insertion_index += 1;
                    }
                    let expected =
                        reference_query(cell_id, level, leaves, insertion_index)?;
                    let actual = engine.query(cell_id)?;
                    if actual != expected {
                        return Err(anyhow!(
                            "query mismatch at level {} cell {:016x}: expected {:016x}, actual {:016x}",
                            level,
                            cell_id,
                            expected,
                            actual
                        ));
                    }
                    face_hash = update_hash(face_hash, actual);
                }
                Ok((face_hash, checked_path_count))
            })
            .collect::<Result<Vec<_>>>()?;
        for (face_hash, face_query_count) in face_results {
            total_hash = update_hash(total_hash, face_hash);
            total_query_count += face_query_count;
        }
        println!(
            "Verification through level {}: {} queries, hash {:016x}.",
            level, total_query_count, total_hash
        );
    }

    let verification_kind = if quick { "Quick" } else { "Exhaustive" };
    println!(
        "{} verification: {} queries passed in {:.2?}, hash {:016x}.",
        verification_kind,
        total_query_count,
        start_time.elapsed(),
        total_hash
    );
    Ok(())
}

/// Maps a checked query index to an evenly distributed path.
fn sampled_path(query_index: u64, checked_path_count: u64, path_count: u64) -> u64 {
    if checked_path_count == path_count {
        query_index
    } else {
        query_index * path_count / checked_path_count
    }
}

/// Answers one sorted query from its neighboring leaf-vector entries.
fn reference_query(
    cell_id: u64,
    cell_level: u32,
    leaves: &[ReferenceLeaf],
    insertion_index: usize,
) -> Result<u64> {
    let preceding_result = insertion_index
        .checked_sub(1)
        .map(|leaf_index| shared_ancestor(cell_id, cell_level, leaves[leaf_index]));
    let following_result = leaves
        .get(insertion_index)
        .copied()
        .map(|leaf| shared_ancestor(cell_id, cell_level, leaf));

    match (preceding_result, following_result) {
        (Some(preceding), Some(following)) => {
            if following.1 > preceding.1 {
                Ok(following.0)
            } else {
                Ok(preceding.0)
            }
        }
        (Some((ancestor, _)), None) | (None, Some((ancestor, _))) => Ok(ancestor),
        (None, None) => Err(anyhow!("reference query has no leaves on its S2 face")),
    }
}

/// Returns the deepest common ancestor of one cell and one reference leaf.
fn shared_ancestor(cell_id: u64, cell_level: u32, leaf: ReferenceLeaf) -> (u64, u32) {
    debug_assert_eq!(cell_id >> S2_FACE_SHIFT, leaf.cell_id >> S2_FACE_SHIFT);
    let compared_level = cell_level.min(leaf.level);
    let compared_path_bit_count = BITS_PER_LEVEL * compared_level;
    let sentinel_position = MAX_S2_BITS - compared_path_bit_count;
    let different_path_bits = (cell_id ^ leaf.cell_id) >> (sentinel_position + 1);
    let shared_level = if different_path_bits == 0 {
        compared_level
    } else {
        let significant_bit_count = u64::BITS - different_path_bits.leading_zeros();
        (compared_path_bit_count - significant_bit_count) / BITS_PER_LEVEL
    };
    (ancestor_at_level(cell_id, shared_level), shared_level)
}

/// Returns a known-valid cell's ancestor without repeating input validation.
fn ancestor_at_level(cell_id: u64, level: u32) -> u64 {
    let sentinel_position = MAX_S2_BITS - BITS_PER_LEVEL * level;
    let lower_bit_mask = (1u64 << sentinel_position) - 1;
    (cell_id & !lower_bit_mask) | (1u64 << sentinel_position)
}

/// Advances an order-sensitive verification hash.
#[inline]
fn update_hash(hash: u64, value: u64) -> u64 {
    hash.rotate_left(9)
        .wrapping_add(value)
        .wrapping_mul(HASH_MULTIPLIER)
}

#[cfg(test)]
mod tests {
    use super::*;
    use population_density::{get_ancestor, try_get_ancestor};

    /// Creates one valid S2 cell from a face, level, and path.
    fn cell_id(face: u64, level: u32, path: u64) -> u64 {
        let sentinel_position = MAX_S2_BITS - BITS_PER_LEVEL * level;
        (face << S2_FACE_SHIFT) | (path << (sentinel_position + 1)) | (1u64 << sentinel_position)
    }

    /// Verifies the reference ancestor arithmetic against validated S2 operations.
    #[test]
    fn shared_ancestor_matches_validated_s2_operations() {
        let left = cell_id(2, 12, 0x12_3456);
        let right = cell_id(2, 9, 0x1_2345);
        let expected_level = (0..=9)
            .rev()
            .find(|&level| get_ancestor(left, level) == get_ancestor(right, level))
            .unwrap();
        let actual = shared_ancestor(
            left,
            12,
            ReferenceLeaf {
                cell_id: right,
                level: 9,
            },
        );
        assert_eq!(actual.1, expected_level);
        assert_eq!(actual.0, try_get_ancestor(left, expected_level).unwrap());
    }

    /// Verifies that quick-mode sampling is deterministic and evenly distributed.
    #[test]
    fn sampled_paths_are_deterministic_and_evenly_distributed() {
        let paths: Vec<u64> = (0..4).map(|index| sampled_path(index, 4, 16)).collect();
        assert_eq!(paths, [0, 4, 8, 12]);
    }
}
