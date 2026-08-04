//! Builds the versioned mmap-only S2 quadtree topology database.

use anyhow::{Context, Result, anyhow};
use population_density::topology_format::*;
use population_density::{
    BITS_PER_LEVEL, MAX_COMPACT_CELL_ID, MAX_DB_LEVEL, MAX_S2_BITS, NUM_ROOT_FACES, S2_FACE_SHIFT,
    SHIFT_COMPACT, get_ancestor, try_get_level,
};
use std::io::{BufWriter, Write};
use std::path::Path;
use tempfile::Builder;

const INVALID_SYMBOL: u8 = u8::MAX;

/// Stores the nodes and child masks at every database level.
struct Topology {
    nodes: Vec<Vec<u32>>,
    masks: Vec<Vec<u8>>,
    leaf_count: usize,
}

impl Topology {
    /// Builds a breadth-first topology from sorted compact leaf IDs.
    fn new(compact_leaves: &[u32]) -> Result<Self> {
        let leaf_levels = validate_compact_leaves(compact_leaves)?;
        let mut nodes = Vec::with_capacity(LEVEL_COUNT + 1);
        for level_index in 0..=LEVEL_COUNT {
            let mut level_nodes = Vec::new();
            for (&compact_leaf, &leaf_level) in compact_leaves.iter().zip(&leaf_levels) {
                if leaf_level < level_index as u32 {
                    continue;
                }
                let ancestor = compact_ancestor(compact_leaf, level_index as u32);
                if level_nodes.last().copied() != Some(ancestor) {
                    level_nodes.push(ancestor);
                }
            }
            if level_nodes.is_empty() {
                return Err(anyhow!("topology level {} has no nodes", level_index));
            }
            nodes.push(level_nodes);
        }

        let expected_roots: Vec<u32> = (0..NUM_ROOT_FACES)
            .map(|face| {
                ((((face as u64) << S2_FACE_SHIFT) | (1u64 << MAX_S2_BITS)) >> SHIFT_COMPACT) as u32
            })
            .collect();
        if nodes[0] != expected_roots {
            return Err(anyhow!("compact leaves must represent every S2 root face"));
        }

        let mut masks = Vec::with_capacity(LEVEL_COUNT);
        for level_index in 0..LEVEL_COUNT {
            let mut level_masks = vec![0u8; nodes[level_index].len()];
            let mut parent_index = 0usize;
            for &child in &nodes[level_index + 1] {
                let parent = compact_ancestor(child, level_index as u32);
                while nodes[level_index][parent_index] < parent {
                    parent_index += 1;
                }
                if nodes[level_index][parent_index] != parent {
                    return Err(anyhow!(
                        "topology level {} contains a child without its parent",
                        level_index + 1
                    ));
                }
                let child_position = child_position(child, level_index as u32);
                level_masks[parent_index] |= 1u8 << child_position;
            }
            masks.push(level_masks);
        }

        Ok(Self {
            nodes,
            masks,
            leaf_count: compact_leaves.len(),
        })
    }
}

/// Stores one canonical Huffman codeword.
#[derive(Clone, Copy, Default)]
struct Codeword {
    bits: u16,
    length: u8,
}

/// Stores one previous-mask-conditioned canonical Huffman model.
struct ContextModel {
    codewords: [Codeword; MASK_SYMBOL_COUNT],
    length_counts: [u8; MASK_SYMBOL_COUNT],
    ordered_symbols: [u8; MASK_SYMBOL_COUNT],
    sole_symbol: u8,
}

impl ContextModel {
    /// Builds a deterministic optimal canonical Huffman model.
    fn new(symbol_counts: &[u32; MASK_SYMBOL_COUNT]) -> Result<Self> {
        let mut code_lengths = [0u8; MASK_SYMBOL_COUNT];
        let mut active_nodes: Vec<(u64, Vec<usize>)> = symbol_counts
            .iter()
            .enumerate()
            .filter_map(|(symbol, &count)| (count != 0).then_some((u64::from(count), vec![symbol])))
            .collect();

        while active_nodes.len() > 1 {
            active_nodes.sort_unstable_by(|left, right| {
                left.0
                    .cmp(&right.0)
                    .then_with(|| left.1[0].cmp(&right.1[0]))
            });
            let (left_count, mut left_symbols) = active_nodes.remove(0);
            let (right_count, right_symbols) = active_nodes.remove(0);
            for &symbol in left_symbols.iter().chain(&right_symbols) {
                code_lengths[symbol] = code_lengths[symbol]
                    .checked_add(1)
                    .ok_or_else(|| anyhow!("Huffman code length overflow"))?;
            }
            left_symbols.extend(right_symbols);
            left_symbols.sort_unstable();
            active_nodes.push((
                left_count
                    .checked_add(right_count)
                    .ok_or_else(|| anyhow!("Huffman symbol count overflow"))?,
                left_symbols,
            ));
        }

        let present_symbols: Vec<usize> = symbol_counts
            .iter()
            .enumerate()
            .filter_map(|(symbol, &count)| (count != 0).then_some(symbol))
            .collect();
        let sole_symbol = if present_symbols.len() == 1 {
            present_symbols[0] as u8
        } else {
            INVALID_SYMBOL
        };
        let mut canonical_symbols: Vec<(u8, usize)> = present_symbols
            .iter()
            .copied()
            .filter_map(|symbol| {
                let length = code_lengths[symbol];
                (length != 0).then_some((length, symbol))
            })
            .collect();
        canonical_symbols.sort_unstable();
        if canonical_symbols
            .last()
            .is_some_and(|&(length, _)| length as usize > MAX_CODE_LENGTH)
        {
            return Err(anyhow!("Huffman code exceeds the format length limit"));
        }

        let mut codewords = [Codeword::default(); MASK_SYMBOL_COUNT];
        let mut length_counts = [0u8; MASK_SYMBOL_COUNT];
        let mut ordered_symbols = [0u8; MASK_SYMBOL_COUNT];
        let mut code = 0u16;
        let mut previous_length = 0u8;
        for (ordered_index, &(length, symbol)) in canonical_symbols.iter().enumerate() {
            code <<= length - previous_length;
            codewords[symbol] = Codeword { bits: code, length };
            length_counts[length as usize] += 1;
            ordered_symbols[ordered_index] = symbol as u8;
            code = code
                .checked_add(1)
                .ok_or_else(|| anyhow!("canonical Huffman code overflow"))?;
            previous_length = length;
        }

        Ok(Self {
            codewords,
            length_counts,
            ordered_symbols,
            sole_symbol,
        })
    }

    /// Serializes the compact canonical decoder table.
    fn append_serialized(&self, bytes: &mut Vec<u8>) {
        let model_offset = bytes.len();
        bytes.resize(model_offset + CONTEXT_SIZE, 0);
        if self.sole_symbol != INVALID_SYMBOL {
            bytes[model_offset] = 1;
            bytes[model_offset + 16] = self.sole_symbol;
            return;
        }
        bytes[model_offset + 1..model_offset + 16].copy_from_slice(&self.length_counts[1..16]);
        let symbol_count = self
            .length_counts
            .iter()
            .map(|&count| count as usize)
            .sum::<usize>();
        for symbol_index in 0..symbol_count {
            bytes[model_offset + 16 + symbol_index / 2] |=
                self.ordered_symbols[symbol_index] << (4 * (symbol_index % 2));
        }
    }
}

/// Appends bits in most-significant-bit-first order.
#[derive(Default)]
struct BitWriter {
    bytes: Vec<u8>,
    bit_length: usize,
}

impl BitWriter {
    /// Appends the requested low bits of a value.
    fn write(&mut self, value: u16, bit_count: u8) -> Result<()> {
        if bit_count > u16::BITS as u8 {
            return Err(anyhow!("bit count exceeds the source value width"));
        }
        self.bit_length = self
            .bit_length
            .checked_add(bit_count as usize)
            .ok_or_else(|| anyhow!("Huffman payload bit length overflow"))?;
        let new_byte_count = self.bit_length.div_ceil(8);
        if new_byte_count > self.bytes.len() {
            self.bytes.resize(new_byte_count, 0);
        }
        let first_bit = self.bit_length - bit_count as usize;
        for source_bit in (0..bit_count).rev() {
            let target_bit = first_bit + (bit_count - 1 - source_bit) as usize;
            let bit = ((value >> source_bit) & 1) as u8;
            self.bytes[target_bit / 8] |= bit << (7 - target_bit % 8);
        }
        Ok(())
    }
}

/// Stores one encoded Huffman level and its indexes.
struct EncodedHuffmanLevel {
    data_offset: usize,
    data_length: usize,
    models: Vec<u8>,
    group_bit_offsets: Vec<u8>,
    group_child_ranks: Vec<u8>,
    block_bit_lengths: Vec<u8>,
    block_child_counts: Vec<u8>,
    bit_length_width: u8,
    child_count_width: u8,
}

impl EncodedHuffmanLevel {
    /// Encodes one topology level into a shared payload.
    fn new(masks: &[u8], final_level: bool, payload: &mut BitWriter) -> Result<Self> {
        if masks.is_empty() {
            return Err(anyhow!("cannot Huffman-encode an empty topology level"));
        }
        let mut transition_counts = [[0u32; MASK_SYMBOL_COUNT]; MASK_SYMBOL_COUNT];
        for block in masks.chunks(RESTART_INTERVAL) {
            for transition in block.windows(2) {
                let count = &mut transition_counts[transition[0] as usize][transition[1] as usize];
                *count = count
                    .checked_add(1)
                    .ok_or_else(|| anyhow!("Huffman transition count overflow"))?;
            }
        }
        let models = transition_counts
            .iter()
            .map(ContextModel::new)
            .collect::<Result<Vec<_>>>()?;
        let mut serialized_models = Vec::with_capacity(LEVEL_MODEL_SIZE);
        for model in &models {
            model.append_serialized(&mut serialized_models);
        }

        let data_offset = payload.bit_length;
        let block_count = masks.len().div_ceil(RESTART_INTERVAL);
        let mut checkpoints = Vec::with_capacity(block_count);
        let mut block_bit_lengths = Vec::with_capacity(block_count);
        let mut block_child_counts = Vec::with_capacity(block_count);
        let mut child_rank = 0u32;
        for block in masks.chunks(RESTART_INTERVAL) {
            let block_start = payload.bit_length;
            checkpoints.push((u32_from_usize(block_start, "group bit offset")?, child_rank));
            payload.write(u16::from(block[0]), 4)?;
            for transition in block.windows(2) {
                let model = &models[transition[0] as usize];
                let codeword = model.codewords[transition[1] as usize];
                if codeword.length == 0 {
                    if model.sole_symbol == transition[1] {
                        continue;
                    }
                    return Err(anyhow!("missing Huffman transition codeword"));
                }
                payload.write(codeword.bits, codeword.length)?;
            }
            let block_bit_length = payload.bit_length - block_start;
            block_bit_lengths.push(u32_from_usize(block_bit_length, "block bit length")?);
            let block_child_count = block.iter().map(|mask| mask.count_ones()).sum::<u32>();
            block_child_counts.push(block_child_count);
            child_rank = child_rank
                .checked_add(block_child_count)
                .ok_or_else(|| anyhow!("topology child rank overflow"))?;
        }

        let group_bit_values: Vec<u32> = checkpoints
            .iter()
            .step_by(BLOCKS_PER_GROUP)
            .map(|&(bit_offset, _)| bit_offset)
            .collect();
        let group_child_values: Vec<u32> = if final_level {
            Vec::new()
        } else {
            checkpoints
                .iter()
                .step_by(BLOCKS_PER_GROUP)
                .map(|&(_, child_rank)| child_rank)
                .collect()
        };
        let bit_length_width =
            required_width(block_bit_lengths.iter().copied().max().unwrap_or_default());
        let child_count_width = if final_level {
            0
        } else {
            required_width(block_child_counts.iter().copied().max().unwrap_or_default())
        };
        if bit_length_width > MAX_BLOCK_BIT_LENGTH_WIDTH
            || child_count_width > MAX_BLOCK_CHILD_COUNT_WIDTH
        {
            return Err(anyhow!(
                "topology restart block exceeds the format width limits"
            ));
        }
        let packed_bit_lengths = pack_values(&block_bit_lengths, bit_length_width)?;
        let packed_child_counts = if final_level {
            Vec::new()
        } else {
            pack_values(&block_child_counts, child_count_width)?
        };

        Ok(Self {
            data_offset,
            data_length: payload.bit_length - data_offset,
            models: serialized_models,
            group_bit_offsets: pack_u24(&group_bit_values, "group bit offset")?,
            group_child_ranks: pack_u24(&group_child_values, "group child rank")?,
            block_bit_lengths: packed_bit_lengths,
            block_child_counts: packed_child_counts,
            bit_length_width,
            child_count_width,
        })
    }
}

/// Stores one serialized level-directory entry.
#[derive(Clone, Copy, Default)]
struct LevelEntry {
    node_count: usize,
    data_offset: usize,
    data_length: usize,
    encoding: u8,
    bit_length_width: u8,
    child_count_width: u8,
    flags: u8,
}

/// Encodes sorted compact leaves into the mmap-only topology format.
pub fn encode_topology_database(compact_leaves: &[u32]) -> Result<Vec<u8>> {
    let topology = Topology::new(compact_leaves)?;
    let mut level_entries = [LevelEntry::default(); LEVEL_COUNT];

    let mut raw_masks = Vec::new();
    let mut raw_ranks = Vec::new();
    for (level_index, level_entry) in level_entries.iter_mut().enumerate().take(RAW_LEVEL_COUNT) {
        let masks = &topology.masks[level_index];
        let data_offset = raw_masks.len();
        append_packed_masks(masks, &mut raw_masks);
        let data_length = raw_masks.len() - data_offset;
        append_raw_ranks(masks, &mut raw_ranks)?;
        *level_entry = LevelEntry {
            node_count: masks.len(),
            data_offset,
            data_length,
            encoding: ENCODING_RAW,
            bit_length_width: 0,
            child_count_width: 0,
            flags: CHILD_RANK_FLAG,
        };
    }

    let mut payload = BitWriter::default();
    let mut encoded_huffman_levels = Vec::with_capacity(LEVEL_COUNT - RAW_LEVEL_COUNT);
    for (level_index, level_entry) in level_entries.iter_mut().enumerate().skip(RAW_LEVEL_COUNT) {
        let encoded = EncodedHuffmanLevel::new(
            &topology.masks[level_index],
            level_index + 1 == LEVEL_COUNT,
            &mut payload,
        )?;
        *level_entry = LevelEntry {
            node_count: topology.masks[level_index].len(),
            data_offset: encoded.data_offset,
            data_length: encoded.data_length,
            encoding: ENCODING_HUFFMAN,
            bit_length_width: encoded.bit_length_width,
            child_count_width: encoded.child_count_width,
            flags: if level_index + 1 == LEVEL_COUNT {
                0
            } else {
                CHILD_RANK_FLAG
            },
        };
        encoded_huffman_levels.push(encoded);
    }

    let mut models = Vec::new();
    let mut group_bit_offsets = Vec::new();
    let mut group_child_ranks = Vec::new();
    let mut block_bit_lengths = Vec::new();
    let mut block_child_counts = Vec::new();
    for level in &encoded_huffman_levels {
        models.extend_from_slice(&level.models);
        group_bit_offsets.extend_from_slice(&level.group_bit_offsets);
        group_child_ranks.extend_from_slice(&level.group_child_ranks);
        block_bit_lengths.extend_from_slice(&level.block_bit_lengths);
        block_child_counts.extend_from_slice(&level.block_child_counts);
    }

    let raw_masks_offset = HEADER_SIZE;
    let raw_ranks_offset = section_end(raw_masks_offset, raw_masks.len(), "raw masks")?;
    let models_offset = section_end(raw_ranks_offset, raw_ranks.len(), "raw ranks")?;
    let group_bit_offsets_offset = section_end(models_offset, models.len(), "models")?;
    let group_child_ranks_offset = section_end(
        group_bit_offsets_offset,
        group_bit_offsets.len(),
        "group bit offsets",
    )?;
    let block_bit_lengths_offset = section_end(
        group_child_ranks_offset,
        group_child_ranks.len(),
        "group child ranks",
    )?;
    let block_child_counts_offset = section_end(
        block_bit_lengths_offset,
        block_bit_lengths.len(),
        "block bit lengths",
    )?;
    let payload_offset = section_end(
        block_child_counts_offset,
        block_child_counts.len(),
        "block child counts",
    )?;
    let file_size = section_end(payload_offset, payload.bytes.len(), "Huffman payload")?;

    let mut database = vec![0u8; HEADER_SIZE];
    database[MAGIC_OFFSET..MAGIC_OFFSET + MAGIC.len()].copy_from_slice(&MAGIC);
    write_u16(&mut database, VERSION_OFFSET, VERSION);
    write_u16(
        &mut database,
        HEADER_SIZE_OFFSET,
        u16::try_from(HEADER_SIZE).expect("header size fits u16"),
    );
    write_u32_checked(&mut database, FILE_SIZE_OFFSET, file_size, "file size")?;
    write_u32_checked(
        &mut database,
        PAYLOAD_BIT_LENGTH_OFFSET,
        payload.bit_length,
        "payload bit length",
    )?;
    write_u32_checked(
        &mut database,
        RAW_MASKS_OFFSET,
        raw_masks_offset,
        "raw masks offset",
    )?;
    write_u32_checked(
        &mut database,
        RAW_RANKS_OFFSET,
        raw_ranks_offset,
        "raw ranks offset",
    )?;
    write_u32_checked(&mut database, MODELS_OFFSET, models_offset, "models offset")?;
    write_u32_checked(
        &mut database,
        GROUP_BIT_OFFSETS_OFFSET,
        group_bit_offsets_offset,
        "group bit offsets offset",
    )?;
    write_u32_checked(
        &mut database,
        GROUP_CHILD_RANKS_OFFSET,
        group_child_ranks_offset,
        "group child ranks offset",
    )?;
    write_u32_checked(
        &mut database,
        BLOCK_BIT_LENGTHS_OFFSET,
        block_bit_lengths_offset,
        "block bit lengths offset",
    )?;
    write_u32_checked(
        &mut database,
        BLOCK_CHILD_COUNTS_OFFSET,
        block_child_counts_offset,
        "block child counts offset",
    )?;
    write_u32_checked(
        &mut database,
        PAYLOAD_OFFSET,
        payload_offset,
        "payload offset",
    )?;
    write_u16(
        &mut database,
        RESTART_INTERVAL_OFFSET,
        u16::try_from(RESTART_INTERVAL).expect("restart interval fits u16"),
    );
    write_u16(
        &mut database,
        RAW_RANK_INTERVAL_OFFSET,
        u16::try_from(RAW_RANK_INTERVAL).expect("raw rank interval fits u16"),
    );
    database[LEVEL_COUNT_OFFSET] = LEVEL_COUNT as u8;
    database[RAW_LEVEL_COUNT_OFFSET] = RAW_LEVEL_COUNT as u8;
    database[MAX_DATABASE_LEVEL_OFFSET] = MAX_DB_LEVEL as u8;
    database[BLOCKS_PER_GROUP_OFFSET] = BLOCKS_PER_GROUP as u8;
    write_u32_checked(
        &mut database,
        LEAF_COUNT_OFFSET,
        topology.leaf_count,
        "leaf count",
    )?;
    write_u32_checked(
        &mut database,
        TERMINAL_NODE_COUNT_OFFSET,
        topology.nodes[LEVEL_COUNT].len(),
        "terminal node count",
    )?;

    for (level_index, entry) in level_entries.iter().enumerate() {
        let entry_offset = FIXED_HEADER_SIZE + level_index * LEVEL_ENTRY_SIZE;
        write_u32_checked(
            &mut database,
            entry_offset + LEVEL_NODE_COUNT_OFFSET,
            entry.node_count,
            "level node count",
        )?;
        write_u32_checked(
            &mut database,
            entry_offset + LEVEL_DATA_OFFSET_OFFSET,
            entry.data_offset,
            "level data offset",
        )?;
        write_u32_checked(
            &mut database,
            entry_offset + LEVEL_DATA_LENGTH_OFFSET,
            entry.data_length,
            "level data length",
        )?;
        database[entry_offset + LEVEL_ENCODING_OFFSET] = entry.encoding;
        database[entry_offset + LEVEL_BIT_LENGTH_WIDTH_OFFSET] = entry.bit_length_width;
        database[entry_offset + LEVEL_CHILD_COUNT_WIDTH_OFFSET] = entry.child_count_width;
        database[entry_offset + LEVEL_FLAGS_OFFSET] = entry.flags;
    }

    database.extend_from_slice(&raw_masks);
    database.extend_from_slice(&raw_ranks);
    database.extend_from_slice(&models);
    database.extend_from_slice(&group_bit_offsets);
    database.extend_from_slice(&group_child_ranks);
    database.extend_from_slice(&block_bit_lengths);
    database.extend_from_slice(&block_child_counts);
    database.extend_from_slice(&payload.bytes);
    if database.len() != file_size {
        return Err(anyhow!(
            "serialized topology size {} does not match header size {}",
            database.len(),
            file_size
        ));
    }
    Ok(database)
}

/// Writes a topology database through a same-directory temporary file.
pub fn write_topology_database<P: AsRef<Path>>(
    database_path: P,
    compact_leaves: &[u32],
) -> Result<()> {
    let database = encode_topology_database(compact_leaves)?;
    let database_path = database_path.as_ref();
    let output_directory = database_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let mut temporary_file = Builder::new()
        .prefix(".population-density-database-")
        .tempfile_in(output_directory)
        .with_context(|| {
            format!(
                "failed to create a temporary topology database in '{}'",
                output_directory.display()
            )
        })?;
    {
        let mut writer = BufWriter::new(temporary_file.as_file_mut());
        writer.write_all(&database)?;
        writer.flush()?;
    }
    temporary_file
        .persist(database_path)
        .map_err(|error| error.error)
        .with_context(|| {
            format!(
                "failed to replace topology database '{}'",
                database_path.display()
            )
        })?;
    Ok(())
}

/// Validates sorted compact leaves and returns their S2 levels.
fn validate_compact_leaves(compact_leaves: &[u32]) -> Result<Vec<u32>> {
    if compact_leaves.is_empty() {
        return Err(anyhow!(
            "no valid population data found to build a topology database"
        ));
    }
    let mut leaf_levels = Vec::with_capacity(compact_leaves.len());
    let mut previous_compact_leaf = None;
    let mut previous_range_end = 0u64;
    let mut represented_faces = 0u8;
    for (leaf_index, &compact_leaf) in compact_leaves.iter().enumerate() {
        if compact_leaf > MAX_COMPACT_CELL_ID {
            return Err(anyhow!(
                "compact leaf ID {} exceeds the format limit",
                compact_leaf
            ));
        }
        if previous_compact_leaf.is_some_and(|previous| compact_leaf <= previous) {
            return Err(anyhow!(
                "compact leaves are out of order or duplicate at index {}",
                leaf_index
            ));
        }
        let cell_id = u64::from(compact_leaf) << SHIFT_COMPACT;
        let leaf_level = try_get_level(cell_id)
            .with_context(|| format!("invalid compact leaf ID at index {}", leaf_index))?;
        if leaf_level > MAX_DB_LEVEL {
            return Err(anyhow!(
                "compact leaf at index {} exceeds database level {}",
                leaf_index,
                MAX_DB_LEVEL
            ));
        }
        let lowest_bit = cell_id & cell_id.wrapping_neg();
        let range_radius = lowest_bit - 1;
        let range_start = cell_id - range_radius;
        let range_end = cell_id + range_radius;
        if leaf_index != 0 && range_start <= previous_range_end {
            return Err(anyhow!("compact leaves overlap at index {}", leaf_index));
        }
        previous_range_end = range_end;
        previous_compact_leaf = Some(compact_leaf);
        represented_faces |= 1u8 << (cell_id >> S2_FACE_SHIFT);
        leaf_levels.push(leaf_level);
    }
    let all_faces = (1u8 << NUM_ROOT_FACES) - 1;
    if represented_faces != all_faces {
        return Err(anyhow!("compact leaves must represent every S2 root face"));
    }
    Ok(leaf_levels)
}

/// Returns a compact cell's ancestor.
fn compact_ancestor(compact_cell_id: u32, level: u32) -> u32 {
    let cell_id = u64::from(compact_cell_id) << SHIFT_COMPACT;
    (get_ancestor(cell_id, level) >> SHIFT_COMPACT) as u32
}

/// Returns a compact child's position below a parent level.
fn child_position(compact_child_id: u32, parent_level: u32) -> usize {
    let cell_id = u64::from(compact_child_id) << SHIFT_COMPACT;
    let parent_sentinel_position = MAX_S2_BITS - BITS_PER_LEVEL * parent_level;
    ((cell_id >> (parent_sentinel_position - 1)) & 3) as usize
}

/// Appends two masks per byte with the even node in the low nibble.
fn append_packed_masks(masks: &[u8], bytes: &mut Vec<u8>) {
    for pair in masks.chunks(2) {
        let high_mask = pair.get(1).copied().unwrap_or(0);
        bytes.push(pair[0] | (high_mask << 4));
    }
}

/// Appends little-endian 24-bit raw rank checkpoints.
fn append_raw_ranks(masks: &[u8], bytes: &mut Vec<u8>) -> Result<()> {
    let mut child_rank = 0u32;
    for (node_index, &mask) in masks.iter().enumerate() {
        if node_index % RAW_RANK_INTERVAL == 0 {
            append_u24(bytes, child_rank, "raw child rank")?;
        }
        child_rank = child_rank
            .checked_add(mask.count_ones())
            .ok_or_else(|| anyhow!("raw child rank overflow"))?;
    }
    Ok(())
}

/// Packs unsigned values in least-significant-bit-first order.
fn pack_values(values: &[u32], width: u8) -> Result<Vec<u8>> {
    if width == 0 || width > 24 {
        return Err(anyhow!("packed value width {} is unsupported", width));
    }
    let bit_count = values
        .len()
        .checked_mul(width as usize)
        .ok_or_else(|| anyhow!("packed value bit count overflow"))?;
    let mut bytes = vec![0u8; bit_count.div_ceil(8)];
    let value_limit = 1u32 << width;
    for (value_index, &value) in values.iter().enumerate() {
        if value >= value_limit {
            return Err(anyhow!("packed value {} exceeds {} bits", value, width));
        }
        let bit_offset = value_index * width as usize;
        for value_bit in 0..width as usize {
            if value & (1u32 << value_bit) != 0 {
                let target_bit = bit_offset + value_bit;
                bytes[target_bit / 8] |= 1u8 << (target_bit % 8);
            }
        }
    }
    Ok(bytes)
}

/// Packs unsigned values into little-endian 24-bit fields.
fn pack_u24(values: &[u32], name: &str) -> Result<Vec<u8>> {
    let mut bytes = Vec::with_capacity(values.len() * 3);
    for &value in values {
        append_u24(&mut bytes, value, name)?;
    }
    Ok(bytes)
}

/// Appends one little-endian 24-bit field.
fn append_u24(bytes: &mut Vec<u8>, value: u32, name: &str) -> Result<()> {
    if value > MAX_U24 {
        return Err(anyhow!("{} {} exceeds the 24-bit limit", name, value));
    }
    bytes.extend_from_slice(&value.to_le_bytes()[..3]);
    Ok(())
}

/// Returns the minimum nonzero width needed to store a value.
fn required_width(maximum_value: u32) -> u8 {
    (u32::BITS - maximum_value.leading_zeros()).max(1) as u8
}

/// Returns the checked end of a serialized section.
fn section_end(offset: usize, length: usize, name: &str) -> Result<usize> {
    offset
        .checked_add(length)
        .ok_or_else(|| anyhow!("{} section end overflow", name))
}

/// Converts a size to a format field.
fn u32_from_usize(value: usize, name: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| anyhow!("{} {} exceeds the 32-bit limit", name, value))
}

/// Writes a little-endian 16-bit header field.
fn write_u16(bytes: &mut [u8], offset: usize, value: u16) {
    bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

/// Writes a checked little-endian 32-bit header field.
fn write_u32_checked(bytes: &mut [u8], offset: usize, value: usize, name: &str) -> Result<()> {
    bytes[offset..offset + 4].copy_from_slice(&u32_from_usize(value, name)?.to_le_bytes());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use population_density::memmap2::MmapMut;
    use population_density::{QueryEngine, get_children_ids};

    const MINIMAL_DATABASE_SIZE: usize = 2_245;
    const MINIMAL_LEAVES: [u32; 6] = [
        0x0000_0001,
        0x0300_0000,
        0x0500_0000,
        0x0700_0000,
        0x0900_0000,
        0x0b00_0000,
    ];

    /// Creates an engine from database bytes without filesystem state.
    fn load_database(database: &[u8]) -> Result<QueryEngine> {
        let mut mapping = MmapMut::map_anon(database.len())?;
        mapping.copy_from_slice(database);
        QueryEngine::from_mmap(mapping.make_read_only()?)
    }

    /// Creates a verified-resource engine from database bytes without filesystem state.
    fn load_verified_database(database: &[u8]) -> Result<QueryEngine> {
        let mut mapping = MmapMut::map_anon(database.len())?;
        mapping.copy_from_slice(database);
        QueryEngine::from_verified_mmap(mapping.make_read_only()?)
    }

    /// Returns one little-endian header field.
    fn read_header_u32(database: &[u8], offset: usize) -> u32 {
        u32::from_le_bytes(
            database[offset..offset + 4]
                .try_into()
                .expect("four-byte header field"),
        )
    }

    /// Writes one little-endian header field.
    fn write_header_u32(database: &mut [u8], offset: usize, value: u32) {
        database[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    /// Verifies deterministic minimal serialization and mmap query roundtripping.
    #[test]
    fn minimal_database_roundtrips() {
        let database = encode_topology_database(&MINIMAL_LEAVES).unwrap();
        assert_eq!(database, encode_topology_database(&MINIMAL_LEAVES).unwrap());
        assert_eq!(database.len(), MINIMAL_DATABASE_SIZE);
        assert_eq!(&database[MAGIC_OFFSET..MAGIC_OFFSET + MAGIC.len()], &MAGIC);
        assert_eq!(
            read_header_u32(&database, LEAF_COUNT_OFFSET),
            MINIMAL_LEAVES.len() as u32
        );
        assert_eq!(read_header_u32(&database, TERMINAL_NODE_COUNT_OFFSET), 1);

        let engine = load_database(&database).unwrap();
        let verified_engine = load_verified_database(&database).unwrap();
        assert_eq!(engine.count(), MINIMAL_LEAVES.len());
        assert_eq!(verified_engine.count(), engine.count());
        assert_eq!(engine.reconstruct_all_leaves().unwrap(), MINIMAL_LEAVES);
        assert_eq!(
            verified_engine.reconstruct_all_leaves().unwrap(),
            MINIMAL_LEAVES
        );

        for &compact_leaf in &MINIMAL_LEAVES {
            let cell_id = u64::from(compact_leaf) << SHIFT_COMPACT;
            assert_eq!(engine.query(cell_id).unwrap(), cell_id);
        }
        let root_cell_id = 1u64 << 60;
        let missing_child = get_children_ids(root_cell_id, 0)[1];
        assert_eq!(engine.query(missing_child).unwrap(), root_cell_id);
        let level_12_leaf = u64::from(MINIMAL_LEAVES[0]) << SHIFT_COMPACT;
        let level_13_child = get_children_ids(level_12_leaf, MAX_DB_LEVEL)[0];
        assert_eq!(engine.query(level_13_child).unwrap(), level_12_leaf);
    }

    /// Verifies representative corruption in each format region is rejected.
    #[test]
    fn representative_corruption_is_rejected() {
        let database = encode_topology_database(&MINIMAL_LEAVES).unwrap();

        let mut wrong_leaf_count = database.clone();
        write_header_u32(
            &mut wrong_leaf_count,
            LEAF_COUNT_OFFSET,
            MINIMAL_LEAVES.len() as u32 + 1,
        );
        assert!(load_database(&wrong_leaf_count).is_err());

        let mut wrong_raw_rank = database.clone();
        let raw_ranks_offset = read_header_u32(&database, RAW_RANKS_OFFSET) as usize;
        wrong_raw_rank[raw_ranks_offset] = 1;
        assert!(load_database(&wrong_raw_rank).is_err());

        let mut invalid_model = database.clone();
        let models_offset = read_header_u32(&database, MODELS_OFFSET) as usize;
        invalid_model[models_offset] = 2;
        assert!(load_database(&invalid_model).is_err());

        let mut invalid_payload_padding = database;
        let payload_offset = read_header_u32(&invalid_payload_padding, PAYLOAD_OFFSET) as usize;
        let payload_bit_length =
            read_header_u32(&invalid_payload_padding, PAYLOAD_BIT_LENGTH_OFFSET) as usize;
        invalid_payload_padding[payload_offset + payload_bit_length / 8] |= 1;
        assert!(load_database(&invalid_payload_padding).is_err());
    }

    /// Verifies verified-resource construction rejects structural corruption.
    #[test]
    fn verified_resource_structural_corruption_is_rejected() {
        let database = encode_topology_database(&MINIMAL_LEAVES).unwrap();

        let mut wrong_raw_rank = database.clone();
        let raw_ranks_offset = read_header_u32(&database, RAW_RANKS_OFFSET) as usize;
        wrong_raw_rank[raw_ranks_offset] = 1;
        assert!(load_verified_database(&wrong_raw_rank).is_err());

        let mut invalid_model = database.clone();
        let models_offset = read_header_u32(&database, MODELS_OFFSET) as usize;
        invalid_model[models_offset] = 2;
        assert!(load_verified_database(&invalid_model).is_err());

        let mut wrong_group_bit_offset = database.clone();
        let group_bit_offsets_offset =
            read_header_u32(&database, GROUP_BIT_OFFSETS_OFFSET) as usize;
        wrong_group_bit_offset[group_bit_offsets_offset] ^= 1;
        assert!(load_verified_database(&wrong_group_bit_offset).is_err());

        let mut wrong_group_child_rank = database.clone();
        let group_child_ranks_offset =
            read_header_u32(&database, GROUP_CHILD_RANKS_OFFSET) as usize;
        wrong_group_child_rank[group_child_ranks_offset] = 1;
        assert!(load_verified_database(&wrong_group_child_rank).is_err());

        let mut wrong_block_bit_length = database.clone();
        let block_bit_lengths_offset =
            read_header_u32(&database, BLOCK_BIT_LENGTHS_OFFSET) as usize;
        wrong_block_bit_length[block_bit_lengths_offset] ^= 1;
        assert!(load_verified_database(&wrong_block_bit_length).is_err());

        let mut wrong_block_child_count = database.clone();
        let block_child_counts_offset =
            read_header_u32(&database, BLOCK_CHILD_COUNTS_OFFSET) as usize;
        wrong_block_child_count[block_child_counts_offset] ^= 1;
        assert!(load_verified_database(&wrong_block_child_count).is_err());

        let mut invalid_payload_padding = database;
        let payload_offset = read_header_u32(&invalid_payload_padding, PAYLOAD_OFFSET) as usize;
        let payload_bit_length =
            read_header_u32(&invalid_payload_padding, PAYLOAD_BIT_LENGTH_OFFSET) as usize;
        invalid_payload_padding[payload_offset + payload_bit_length / 8] |= 1;
        assert!(load_verified_database(&invalid_payload_padding).is_err());
    }

    /// Verifies verified-resource construction relies on offline semantic validation.
    #[test]
    fn verified_resource_skips_semantic_validation() {
        let mut wrong_leaf_count = encode_topology_database(&MINIMAL_LEAVES).unwrap();
        write_header_u32(
            &mut wrong_leaf_count,
            LEAF_COUNT_OFFSET,
            MINIMAL_LEAVES.len() as u32 + 1,
        );
        assert!(load_database(&wrong_leaf_count).is_err());
        assert!(load_verified_database(&wrong_leaf_count).is_ok());
    }

    /// Verifies invalid or incomplete compact leaf sets fail fast.
    #[test]
    fn invalid_leaf_sets_are_rejected() {
        assert!(encode_topology_database(&[]).is_err());

        let mut duplicate = MINIMAL_LEAVES.to_vec();
        duplicate.insert(1, duplicate[0]);
        assert!(encode_topology_database(&duplicate).is_err());

        let mut missing_face = MINIMAL_LEAVES.to_vec();
        missing_face.pop();
        assert!(encode_topology_database(&missing_face).is_err());

        let mut overlapping = MINIMAL_LEAVES.to_vec();
        overlapping.insert(1, 0x0100_0000);
        assert!(encode_topology_database(&overlapping).is_err());
    }
}
