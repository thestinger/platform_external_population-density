//! Provides mmap-only queries for the versioned S2 quadtree topology format.

use crate::topology_format::*;
use crate::{
    BITS_PER_LEVEL, MAX_DB_LEVEL, MAX_S2_BITS, NUM_ROOT_FACES, S2_FACE_SHIFT, SHIFT_COMPACT,
    get_ancestor, get_children_ids, try_get_level,
};
use anyhow::{Result, anyhow};
use memmap2::Mmap;

const NO_OFFSET: usize = usize::MAX;

/// Stores the mmap locations and widths needed to decode one topology level.
#[derive(Clone, Copy, Debug)]
struct LevelLayout {
    node_count: usize,
    data_offset: usize,
    data_length: usize,
    raw_rank_offset: usize,
    model_offset: usize,
    group_bit_offset: usize,
    group_child_rank_offset: usize,
    block_bit_length_offset: usize,
    block_bit_length_bytes: usize,
    block_child_count_offset: usize,
    block_child_count_bytes: usize,
    block_count: usize,
    group_count: usize,
    bit_length_width: u8,
    child_count_width: u8,
    has_child_ranks: bool,
}

const EMPTY_LEVEL_LAYOUT: LevelLayout = LevelLayout {
    node_count: 0,
    data_offset: 0,
    data_length: 0,
    raw_rank_offset: NO_OFFSET,
    model_offset: NO_OFFSET,
    group_bit_offset: NO_OFFSET,
    group_child_rank_offset: NO_OFFSET,
    block_bit_length_offset: NO_OFFSET,
    block_bit_length_bytes: 0,
    block_child_count_offset: NO_OFFSET,
    block_child_count_bytes: 0,
    block_count: 0,
    group_count: 0,
    bit_length_width: 0,
    child_count_width: 0,
    has_child_ranks: false,
};

/// Implements allocation-free successful queries over a memory-mapped topology database.
#[derive(Debug)]
pub(crate) struct TopologyQueryEngine {
    mmap: Mmap,
    levels: [LevelLayout; LEVEL_COUNT],
    leaf_count: usize,
    terminal_node_count: usize,
    payload_offset: usize,
    payload_bit_length: usize,
}

impl TopologyQueryEngine {
    /// Initializes and validates an engine from an existing memory mapping.
    pub(crate) fn from_mmap(mmap: Mmap) -> Result<Self> {
        Self::from_mmap_with_validation(mmap, true)
    }

    /// Initializes an engine from a preverified immutable memory mapping.
    pub(crate) fn from_verified_mmap(mmap: Mmap) -> Result<Self> {
        Self::from_mmap_with_validation(mmap, false)
    }

    /// Initializes an engine with the requested Huffman payload validation depth.
    fn from_mmap_with_validation(mmap: Mmap, validate_payload: bool) -> Result<Self> {
        if mmap.len() < HEADER_SIZE {
            return Err(anyhow!(
                "topology database file is too small to contain a valid header"
            ));
        }
        let magic_range = MAGIC_OFFSET..MAGIC_OFFSET + MAGIC.len();
        if mmap[magic_range.clone()] != MAGIC {
            return Err(anyhow!(
                "invalid topology database magic bytes: {:?}",
                &mmap[magic_range]
            ));
        }
        if read_u16(&mmap, VERSION_OFFSET) != VERSION {
            return Err(anyhow!(
                "unsupported topology database version: {}",
                read_u16(&mmap, VERSION_OFFSET)
            ));
        }
        if read_u16(&mmap, HEADER_SIZE_OFFSET) as usize != HEADER_SIZE {
            return Err(anyhow!(
                "invalid topology database header size: {}",
                read_u16(&mmap, HEADER_SIZE_OFFSET)
            ));
        }
        if read_u16(&mmap, RESTART_INTERVAL_OFFSET) as usize != RESTART_INTERVAL
            || read_u16(&mmap, RAW_RANK_INTERVAL_OFFSET) as usize != RAW_RANK_INTERVAL
            || mmap[LEVEL_COUNT_OFFSET] as usize != LEVEL_COUNT
            || mmap[RAW_LEVEL_COUNT_OFFSET] as usize != RAW_LEVEL_COUNT
            || mmap[MAX_DATABASE_LEVEL_OFFSET] as u32 != MAX_DB_LEVEL
            || mmap[BLOCKS_PER_GROUP_OFFSET] as usize != BLOCKS_PER_GROUP
        {
            return Err(anyhow!("unsupported topology database encoding parameters"));
        }

        let declared_file_size = read_u32(&mmap, FILE_SIZE_OFFSET) as usize;
        if declared_file_size != mmap.len() {
            return Err(anyhow!(
                "topology database file size {} does not match declared size {}",
                mmap.len(),
                declared_file_size
            ));
        }
        let payload_bit_length = read_u32(&mmap, PAYLOAD_BIT_LENGTH_OFFSET) as usize;
        let leaf_count = read_u32(&mmap, LEAF_COUNT_OFFSET) as usize;
        let terminal_node_count = read_u32(&mmap, TERMINAL_NODE_COUNT_OFFSET) as usize;
        if payload_bit_length == 0 || leaf_count == 0 || terminal_node_count == 0 {
            return Err(anyhow!(
                "topology database counts and payload length must be nonzero"
            ));
        }

        let raw_masks_offset = read_u32(&mmap, RAW_MASKS_OFFSET) as usize;
        let raw_ranks_offset = read_u32(&mmap, RAW_RANKS_OFFSET) as usize;
        let models_offset = read_u32(&mmap, MODELS_OFFSET) as usize;
        let group_bit_offsets_offset = read_u32(&mmap, GROUP_BIT_OFFSETS_OFFSET) as usize;
        let group_child_ranks_offset = read_u32(&mmap, GROUP_CHILD_RANKS_OFFSET) as usize;
        let block_bit_lengths_offset = read_u32(&mmap, BLOCK_BIT_LENGTHS_OFFSET) as usize;
        let block_child_counts_offset = read_u32(&mmap, BLOCK_CHILD_COUNTS_OFFSET) as usize;
        let payload_offset = read_u32(&mmap, PAYLOAD_OFFSET) as usize;

        let mut levels = [EMPTY_LEVEL_LAYOUT; LEVEL_COUNT];
        let mut expected_raw_data_offset = 0usize;
        let mut expected_payload_bit_offset = 0usize;
        for (level_index, level) in levels.iter_mut().enumerate() {
            let entry_offset = FIXED_HEADER_SIZE + level_index * LEVEL_ENTRY_SIZE;
            let node_count = read_u32(&mmap, entry_offset + LEVEL_NODE_COUNT_OFFSET) as usize;
            if node_count == 0 {
                return Err(anyhow!("topology level {} has no nodes", level_index));
            }
            let maximum_node_count = maximum_node_count(level_index)?;
            if node_count > maximum_node_count {
                return Err(anyhow!(
                    "topology level {} node count {} exceeds geometric limit {}",
                    level_index,
                    node_count,
                    maximum_node_count
                ));
            }
            let data_offset = read_u32(&mmap, entry_offset + LEVEL_DATA_OFFSET_OFFSET) as usize;
            let data_length = read_u32(&mmap, entry_offset + LEVEL_DATA_LENGTH_OFFSET) as usize;
            let encoding = mmap[entry_offset + LEVEL_ENCODING_OFFSET];
            let bit_length_width = mmap[entry_offset + LEVEL_BIT_LENGTH_WIDTH_OFFSET];
            let child_count_width = mmap[entry_offset + LEVEL_CHILD_COUNT_WIDTH_OFFSET];
            let flags = mmap[entry_offset + LEVEL_FLAGS_OFFSET];
            if flags & !CHILD_RANK_FLAG != 0 {
                return Err(anyhow!(
                    "topology level {} has unsupported flags",
                    level_index
                ));
            }
            let has_child_ranks = flags & CHILD_RANK_FLAG != 0;

            if level_index < RAW_LEVEL_COUNT {
                let expected_data_length = node_count.div_ceil(2);
                if encoding != ENCODING_RAW
                    || data_offset != expected_raw_data_offset
                    || data_length != expected_data_length
                    || bit_length_width != 0
                    || child_count_width != 0
                    || !has_child_ranks
                {
                    return Err(anyhow!(
                        "invalid raw topology level {} directory entry",
                        level_index
                    ));
                }
                expected_raw_data_offset = checked_add(
                    expected_raw_data_offset,
                    expected_data_length,
                    "raw mask section size",
                )?;
            } else {
                let final_level = level_index + 1 == LEVEL_COUNT;
                if encoding != ENCODING_HUFFMAN
                    || data_offset != expected_payload_bit_offset
                    || data_length == 0
                    || !(1..=MAX_BLOCK_BIT_LENGTH_WIDTH).contains(&bit_length_width)
                    || has_child_ranks == final_level
                    || (final_level && child_count_width != 0)
                    || (!final_level
                        && !(1..=MAX_BLOCK_CHILD_COUNT_WIDTH).contains(&child_count_width))
                {
                    return Err(anyhow!(
                        "invalid Huffman topology level {} directory entry",
                        level_index
                    ));
                }
                expected_payload_bit_offset = checked_add(
                    expected_payload_bit_offset,
                    data_length,
                    "Huffman payload bit length",
                )?;
            }
            level.node_count = node_count;
            level.data_offset = data_offset;
            level.data_length = data_length;
            level.bit_length_width = bit_length_width;
            level.child_count_width = child_count_width;
            level.has_child_ranks = has_child_ranks;
        }

        if levels[0].node_count != NUM_ROOT_FACES || levels[0].node_count != ROOT_FACE_COUNT {
            return Err(anyhow!(
                "topology database must contain all {} S2 root faces",
                ROOT_FACE_COUNT
            ));
        }
        let maximum_terminal_node_count = maximum_node_count(LEVEL_COUNT)?;
        if terminal_node_count > maximum_terminal_node_count
            || leaf_count > maximum_terminal_node_count
        {
            return Err(anyhow!(
                "topology terminal or leaf count exceeds the level-{} geometric limit",
                LEVEL_COUNT
            ));
        }
        if expected_payload_bit_offset != payload_bit_length {
            return Err(anyhow!(
                "topology level payload lengths do not match the declared payload length"
            ));
        }

        require_offset("raw masks", raw_masks_offset, HEADER_SIZE)?;
        let expected_raw_ranks_offset = checked_add(
            raw_masks_offset,
            expected_raw_data_offset,
            "raw mask section end",
        )?;
        require_offset("raw ranks", raw_ranks_offset, expected_raw_ranks_offset)?;

        let raw_rank_size = levels[..RAW_LEVEL_COUNT]
            .iter()
            .try_fold(0usize, |size, level| {
                checked_add(
                    size,
                    checked_mul(
                        level.node_count.div_ceil(RAW_RANK_INTERVAL),
                        3,
                        "raw rank size",
                    )?,
                    "raw rank section size",
                )
            })?;
        let expected_models_offset =
            checked_add(raw_ranks_offset, raw_rank_size, "raw rank section end")?;
        require_offset("models", models_offset, expected_models_offset)?;
        let model_size = checked_mul(
            LEVEL_COUNT - RAW_LEVEL_COUNT,
            LEVEL_MODEL_SIZE,
            "model section size",
        )?;
        let expected_group_bit_offsets_offset =
            checked_add(models_offset, model_size, "model section end")?;
        require_offset(
            "group bit offsets",
            group_bit_offsets_offset,
            expected_group_bit_offsets_offset,
        )?;

        let mut raw_rank_cursor = raw_ranks_offset;
        let mut group_bit_cursor = group_bit_offsets_offset;
        let mut group_child_cursor = group_child_ranks_offset;
        let mut block_length_cursor = block_bit_lengths_offset;
        let mut block_child_cursor = block_child_counts_offset;
        for (level_index, level) in levels.iter_mut().enumerate() {
            if level_index < RAW_LEVEL_COUNT {
                level.data_offset =
                    checked_add(raw_masks_offset, level.data_offset, "raw mask level offset")?;
                level.raw_rank_offset = raw_rank_cursor;
                raw_rank_cursor = checked_add(
                    raw_rank_cursor,
                    checked_mul(
                        level.node_count.div_ceil(RAW_RANK_INTERVAL),
                        3,
                        "raw rank level size",
                    )?,
                    "raw rank cursor",
                )?;
                continue;
            }

            level.block_count = level.node_count.div_ceil(RESTART_INTERVAL);
            level.group_count = level.block_count.div_ceil(BLOCKS_PER_GROUP);
            level.model_offset = models_offset + (level_index - RAW_LEVEL_COUNT) * LEVEL_MODEL_SIZE;
            level.group_bit_offset = group_bit_cursor;
            group_bit_cursor = checked_add(
                group_bit_cursor,
                checked_mul(level.group_count, 3, "group bit offset size")?,
                "group bit offset cursor",
            )?;
            if level.has_child_ranks {
                level.group_child_rank_offset = group_child_cursor;
                group_child_cursor = checked_add(
                    group_child_cursor,
                    checked_mul(level.group_count, 3, "group child rank size")?,
                    "group child rank cursor",
                )?;
            }
            level.block_bit_length_offset = block_length_cursor;
            level.block_bit_length_bytes =
                packed_byte_count(level.block_count, level.bit_length_width)?;
            block_length_cursor = checked_add(
                block_length_cursor,
                level.block_bit_length_bytes,
                "block bit length cursor",
            )?;
            if level.has_child_ranks {
                level.block_child_count_offset = block_child_cursor;
                level.block_child_count_bytes =
                    packed_byte_count(level.block_count, level.child_count_width)?;
                block_child_cursor = checked_add(
                    block_child_cursor,
                    level.block_child_count_bytes,
                    "block child count cursor",
                )?;
            }
        }

        require_offset(
            "group child ranks",
            group_child_ranks_offset,
            group_bit_cursor,
        )?;
        require_offset(
            "block bit lengths",
            block_bit_lengths_offset,
            group_child_cursor,
        )?;
        require_offset(
            "block child counts",
            block_child_counts_offset,
            block_length_cursor,
        )?;
        require_offset("payload", payload_offset, block_child_cursor)?;
        let expected_file_size = checked_add(
            payload_offset,
            payload_bit_length.div_ceil(8),
            "payload section end",
        )?;
        require_offset("file size", declared_file_size, expected_file_size)?;

        let engine = Self {
            mmap,
            levels,
            leaf_count,
            terminal_node_count,
            payload_offset,
            payload_bit_length,
        };
        if validate_payload {
            engine.validate()?;
        } else {
            engine.validate_verified_layout()?;
        }
        Ok(engine)
    }

    /// Returns the number of leaves represented by the topology.
    pub(crate) fn leaf_count(&self) -> usize {
        self.leaf_count
    }

    /// Finds the deepest database ancestor for a valid S2 cell ID.
    #[inline]
    pub(crate) fn query(&self, cell_id: u64) -> Result<u64> {
        let query_level = try_get_level(cell_id)
            .map_err(|error| anyhow!("invalid query S2 cell ID: {}", error))?;
        let target_level = query_level.min(MAX_DB_LEVEL) as usize;
        let mut node_index = (cell_id >> S2_FACE_SHIFT) as usize;

        for level_index in 0..target_level {
            let final_visited_level = level_index + 1 == target_level;
            let (mask, child_rank) =
                self.mask_and_rank(level_index, node_index, !final_visited_level);
            let child_index = child_index(cell_id, level_index as u32);
            let child_bit = 1u8 << child_index;
            if mask & child_bit == 0 {
                return Ok(get_ancestor(cell_id, level_index as u32));
            }
            if final_visited_level {
                return Ok(get_ancestor(cell_id, target_level as u32));
            }
            node_index = child_rank + (mask & child_bit.wrapping_sub(1)).count_ones() as usize;
        }

        Ok(get_ancestor(cell_id, target_level as u32))
    }

    /// Reconstructs the sorted compact leaf IDs represented by the topology.
    pub(crate) fn reconstruct_all_leaves(&self) -> Result<Vec<u32>> {
        let mut leaves = Vec::with_capacity(self.leaf_count);
        for face in 0..ROOT_FACE_COUNT {
            let root_cell_id = ((face as u64) << S2_FACE_SHIFT) | (1u64 << MAX_S2_BITS);
            self.reconstruct_node(face, root_cell_id, 0, &mut leaves);
        }
        if leaves.len() != self.leaf_count {
            return Err(anyhow!(
                "validated topology reconstructed {} leaves instead of {}",
                leaves.len(),
                self.leaf_count
            ));
        }
        Ok(leaves)
    }

    /// Reconstructs leaves below one topology node in S2 traversal order.
    fn reconstruct_node(
        &self,
        node_index: usize,
        cell_id: u64,
        level_index: usize,
        leaves: &mut Vec<u32>,
    ) {
        let has_indexed_children = level_index + 1 < LEVEL_COUNT;
        let (mask, child_rank) = self.mask_and_rank(level_index, node_index, has_indexed_children);
        if mask == 0 {
            leaves.push((cell_id >> SHIFT_COMPACT) as u32);
            return;
        }

        let children = get_children_ids(cell_id, level_index as u32);
        for (child_index, &child_cell_id) in children.iter().enumerate() {
            let child_bit = 1u8 << child_index;
            if mask & child_bit == 0 {
                continue;
            }
            if has_indexed_children {
                let child_node_index =
                    child_rank + (mask & child_bit.wrapping_sub(1)).count_ones() as usize;
                self.reconstruct_node(child_node_index, child_cell_id, level_index + 1, leaves);
            } else {
                leaves.push((child_cell_id >> SHIFT_COMPACT) as u32);
            }
        }
    }

    /// Validates models, indexes, masks, ranks, child totals, and padding.
    fn validate(&self) -> Result<()> {
        self.validate_models()?;
        let mut zero_mask_count = 0usize;
        for level_index in 0..LEVEL_COUNT {
            let child_count = if level_index + 1 < LEVEL_COUNT {
                self.levels[level_index + 1].node_count
            } else {
                self.terminal_node_count
            };
            let (decoded_child_count, level_zero_mask_count) = if level_index < RAW_LEVEL_COUNT {
                self.validate_raw_level(level_index)?
            } else {
                self.validate_huffman_level(level_index)?
            };
            if decoded_child_count != child_count {
                return Err(anyhow!(
                    "topology level {} has {} children instead of {}",
                    level_index,
                    decoded_child_count,
                    child_count
                ));
            }
            zero_mask_count = checked_add(
                zero_mask_count,
                level_zero_mask_count,
                "topology leaf count",
            )?;
        }
        let decoded_leaf_count = checked_add(
            zero_mask_count,
            self.terminal_node_count,
            "topology leaf count",
        )?;
        if decoded_leaf_count != self.leaf_count {
            return Err(anyhow!(
                "topology contains {} leaves instead of the declared {}",
                decoded_leaf_count,
                self.leaf_count
            ));
        }
        self.validate_payload_padding()
    }

    /// Validates the layout used by a preverified immutable payload.
    fn validate_verified_layout(&self) -> Result<()> {
        self.validate_models()?;
        for level_index in 0..RAW_LEVEL_COUNT {
            let expected_child_count = self.levels[level_index + 1].node_count;
            let (child_count, _) = self.validate_raw_level(level_index)?;
            if child_count != expected_child_count {
                return Err(anyhow!(
                    "topology level {} has {} children instead of {}",
                    level_index,
                    child_count,
                    expected_child_count
                ));
            }
        }
        for level_index in RAW_LEVEL_COUNT..LEVEL_COUNT {
            self.validate_huffman_indexes(level_index)?;
        }
        self.validate_payload_padding()
    }

    /// Validates every compact canonical Huffman context table.
    fn validate_models(&self) -> Result<()> {
        for level_index in RAW_LEVEL_COUNT..LEVEL_COUNT {
            let level = &self.levels[level_index];
            for context in 0..MASK_SYMBOL_COUNT {
                let model_offset = level.model_offset + context * CONTEXT_SIZE;
                let model = &self.mmap[model_offset..model_offset + CONTEXT_SIZE];
                match model[0] {
                    0 => self.validate_canonical_model(level_index, context, model)?,
                    1 => {
                        if model[16] >= MASK_SYMBOL_COUNT as u8
                            || model[1..16].iter().any(|&value| value != 0)
                            || model[17..].iter().any(|&value| value != 0)
                        {
                            return Err(anyhow!(
                                "invalid sole-symbol model at level {} context {}",
                                level_index,
                                context
                            ));
                        }
                    }
                    flag => {
                        return Err(anyhow!(
                            "invalid model flag {} at level {} context {}",
                            flag,
                            level_index,
                            context
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    /// Validates one empty or canonical context model.
    fn validate_canonical_model(
        &self,
        level_index: usize,
        context: usize,
        model: &[u8],
    ) -> Result<()> {
        let symbol_count = model[1..16]
            .iter()
            .map(|&count| count as usize)
            .sum::<usize>();
        if symbol_count == 0 {
            if model[16..].iter().any(|&value| value != 0) {
                return Err(anyhow!(
                    "empty model has symbols at level {} context {}",
                    level_index,
                    context
                ));
            }
            return Ok(());
        }
        if !(2..=MASK_SYMBOL_COUNT).contains(&symbol_count) {
            return Err(anyhow!(
                "invalid canonical symbol count at level {} context {}",
                level_index,
                context
            ));
        }

        let mut available_codes = 1u32;
        for &count in &model[1..16] {
            available_codes = available_codes
                .checked_mul(2)
                .ok_or_else(|| anyhow!("canonical code-space overflow"))?;
            if u32::from(count) > available_codes {
                return Err(anyhow!(
                    "oversubscribed canonical model at level {} context {}",
                    level_index,
                    context
                ));
            }
            available_codes -= u32::from(count);
        }
        if available_codes != 0 {
            return Err(anyhow!(
                "incomplete canonical model at level {} context {}",
                level_index,
                context
            ));
        }

        let mut seen_symbols = 0u16;
        for symbol_index in 0..MASK_SYMBOL_COUNT {
            let packed = model[16 + symbol_index / 2];
            let symbol = (packed >> (4 * (symbol_index % 2))) & 0x0f;
            if symbol_index < symbol_count {
                let symbol_bit = 1u16 << symbol;
                if seen_symbols & symbol_bit != 0 {
                    return Err(anyhow!(
                        "duplicate canonical symbol at level {} context {}",
                        level_index,
                        context
                    ));
                }
                seen_symbols |= symbol_bit;
            } else if symbol != 0 {
                return Err(anyhow!(
                    "nonzero canonical model padding at level {} context {}",
                    level_index,
                    context
                ));
            }
        }
        Ok(())
    }

    /// Validates one raw-mask level and returns its child and zero-mask counts.
    fn validate_raw_level(&self, level_index: usize) -> Result<(usize, usize)> {
        let level = &self.levels[level_index];
        if !level.node_count.is_multiple_of(2)
            && self.mmap[level.data_offset + level.data_length - 1] & 0xf0 != 0
        {
            return Err(anyhow!("nonzero raw mask padding at level {}", level_index));
        }

        let mut child_count = 0usize;
        let mut zero_mask_count = 0usize;
        for node_index in 0..level.node_count {
            if node_index % RAW_RANK_INTERVAL == 0 {
                let checkpoint_index = node_index / RAW_RANK_INTERVAL;
                let checkpoint =
                    read_u24(&self.mmap, level.raw_rank_offset + checkpoint_index * 3) as usize;
                if checkpoint != child_count {
                    return Err(anyhow!(
                        "invalid raw rank checkpoint at level {} node {}",
                        level_index,
                        node_index
                    ));
                }
            }
            let mask = self.raw_mask(level, node_index);
            child_count = checked_add(child_count, mask.count_ones() as usize, "raw child count")?;
            zero_mask_count += usize::from(mask == 0);
        }
        Ok((child_count, zero_mask_count))
    }

    /// Validates one Huffman level and returns its child and zero-mask counts.
    fn validate_huffman_level(&self, level_index: usize) -> Result<(usize, usize)> {
        let level = &self.levels[level_index];
        validate_packed_padding(
            &self.mmap,
            level.block_bit_length_offset,
            level.block_bit_length_bytes,
            level.block_count,
            level.bit_length_width,
            "block bit lengths",
        )?;
        if level.has_child_ranks {
            validate_packed_padding(
                &self.mmap,
                level.block_child_count_offset,
                level.block_child_count_bytes,
                level.block_count,
                level.child_count_width,
                "block child counts",
            )?;
        }

        let level_end = level.data_offset + level.data_length;
        let mut bit_offset = level.data_offset;
        let mut child_count = 0usize;
        let mut zero_mask_count = 0usize;
        let mut maximum_block_bit_length = 0u32;
        let mut maximum_block_child_count = 0u32;
        for block_index in 0..level.block_count {
            if block_index % BLOCKS_PER_GROUP == 0 {
                let group_index = block_index / BLOCKS_PER_GROUP;
                let stored_bit_offset =
                    read_u24(&self.mmap, level.group_bit_offset + group_index * 3) as usize;
                if stored_bit_offset != bit_offset {
                    return Err(anyhow!(
                        "invalid group bit offset at level {} group {}",
                        level_index,
                        group_index
                    ));
                }
                if level.has_child_ranks {
                    let stored_child_rank =
                        read_u24(&self.mmap, level.group_child_rank_offset + group_index * 3)
                            as usize;
                    if stored_child_rank != child_count {
                        return Err(anyhow!(
                            "invalid group child rank at level {} group {}",
                            level_index,
                            group_index
                        ));
                    }
                }
            }

            let block_start = bit_offset;
            let block_node_start = block_index * RESTART_INTERVAL;
            let block_node_count = (level.node_count - block_node_start).min(RESTART_INTERVAL);
            let mut mask = self.read_payload_bits_checked(&mut bit_offset, 4, level_end)? as u8;
            let mut block_child_count = 0usize;
            for node_in_block in 0..block_node_count {
                if node_in_block != 0 {
                    mask = self.decode_symbol_checked(level, mask, &mut bit_offset, level_end)?;
                }
                block_child_count += mask.count_ones() as usize;
                zero_mask_count += usize::from(mask == 0);
            }
            let stored_bit_length = read_packed(
                &self.mmap,
                level.block_bit_length_offset,
                level.block_bit_length_bytes,
                block_index,
                level.bit_length_width,
            ) as usize;
            maximum_block_bit_length = maximum_block_bit_length.max(stored_bit_length as u32);
            if bit_offset - block_start != stored_bit_length {
                return Err(anyhow!(
                    "invalid block bit length at level {} block {}",
                    level_index,
                    block_index
                ));
            }
            if level.has_child_ranks {
                let stored_child_count = read_packed(
                    &self.mmap,
                    level.block_child_count_offset,
                    level.block_child_count_bytes,
                    block_index,
                    level.child_count_width,
                ) as usize;
                maximum_block_child_count =
                    maximum_block_child_count.max(stored_child_count as u32);
                if stored_child_count != block_child_count {
                    return Err(anyhow!(
                        "invalid block child count at level {} block {}",
                        level_index,
                        block_index
                    ));
                }
            }
            child_count = checked_add(child_count, block_child_count, "Huffman child count")?;
        }
        if bit_offset != level_end {
            return Err(anyhow!(
                "Huffman level {} does not consume its declared payload",
                level_index
            ));
        }
        if required_width(maximum_block_bit_length) != level.bit_length_width
            || (level.has_child_ranks
                && required_width(maximum_block_child_count) != level.child_count_width)
        {
            return Err(anyhow!(
                "topology level {} uses noncanonical packed widths",
                level_index
            ));
        }
        Ok((child_count, zero_mask_count))
    }

    /// Validates one Huffman level's restart indexes without decoding its payload.
    fn validate_huffman_indexes(&self, level_index: usize) -> Result<()> {
        let level = &self.levels[level_index];
        validate_packed_padding(
            &self.mmap,
            level.block_bit_length_offset,
            level.block_bit_length_bytes,
            level.block_count,
            level.bit_length_width,
            "block bit lengths",
        )?;
        if level.has_child_ranks {
            validate_packed_padding(
                &self.mmap,
                level.block_child_count_offset,
                level.block_child_count_bytes,
                level.block_count,
                level.child_count_width,
                "block child counts",
            )?;
        }

        let level_end = checked_add(level.data_offset, level.data_length, "Huffman level end")?;
        let mut bit_offset = level.data_offset;
        let mut child_count = 0usize;
        let mut maximum_block_bit_length = 0u32;
        let mut maximum_block_child_count = 0u32;
        for block_index in 0..level.block_count {
            if block_index % BLOCKS_PER_GROUP == 0 {
                let group_index = block_index / BLOCKS_PER_GROUP;
                let stored_bit_offset =
                    read_u24(&self.mmap, level.group_bit_offset + group_index * 3) as usize;
                if stored_bit_offset != bit_offset {
                    return Err(anyhow!(
                        "invalid group bit offset at level {} group {}",
                        level_index,
                        group_index
                    ));
                }
                if level.has_child_ranks {
                    let stored_child_rank =
                        read_u24(&self.mmap, level.group_child_rank_offset + group_index * 3)
                            as usize;
                    if stored_child_rank != child_count {
                        return Err(anyhow!(
                            "invalid group child rank at level {} group {}",
                            level_index,
                            group_index
                        ));
                    }
                }
            }

            let block_node_start = block_index * RESTART_INTERVAL;
            let block_node_count = (level.node_count - block_node_start).min(RESTART_INTERVAL);
            let stored_bit_length = read_packed(
                &self.mmap,
                level.block_bit_length_offset,
                level.block_bit_length_bytes,
                block_index,
                level.bit_length_width,
            ) as usize;
            let maximum_bit_length = 4 + (block_node_count - 1) * MAX_CODE_LENGTH;
            if !(4..=maximum_bit_length).contains(&stored_bit_length) {
                return Err(anyhow!(
                    "invalid block bit length at level {} block {}",
                    level_index,
                    block_index
                ));
            }
            maximum_block_bit_length = maximum_block_bit_length.max(stored_bit_length as u32);
            bit_offset = checked_add(bit_offset, stored_bit_length, "Huffman bit offset")?;
            if bit_offset > level_end {
                return Err(anyhow!(
                    "block bit lengths exceed the payload at level {} block {}",
                    level_index,
                    block_index
                ));
            }

            if level.has_child_ranks {
                let stored_child_count = read_packed(
                    &self.mmap,
                    level.block_child_count_offset,
                    level.block_child_count_bytes,
                    block_index,
                    level.child_count_width,
                ) as usize;
                if stored_child_count > block_node_count * 4 {
                    return Err(anyhow!(
                        "invalid block child count at level {} block {}",
                        level_index,
                        block_index
                    ));
                }
                maximum_block_child_count =
                    maximum_block_child_count.max(stored_child_count as u32);
                child_count = checked_add(
                    child_count,
                    stored_child_count,
                    "Huffman indexed child count",
                )?;
            }
        }

        if bit_offset != level_end {
            return Err(anyhow!(
                "Huffman level {} indexes do not consume its declared payload",
                level_index
            ));
        }
        if required_width(maximum_block_bit_length) != level.bit_length_width
            || (level.has_child_ranks
                && required_width(maximum_block_child_count) != level.child_count_width)
        {
            return Err(anyhow!(
                "topology level {} uses noncanonical packed widths",
                level_index
            ));
        }

        if level.has_child_ranks {
            let expected_child_count = self.levels[level_index + 1].node_count;
            if child_count != expected_child_count {
                return Err(anyhow!(
                    "topology level {} indexes {} children instead of {}",
                    level_index,
                    child_count,
                    expected_child_count
                ));
            }
        } else {
            let maximum_terminal_node_count =
                checked_mul(level.node_count, 4, "maximum terminal node count")?;
            if self.terminal_node_count > maximum_terminal_node_count {
                return Err(anyhow!(
                    "topology terminal node count exceeds the final-level geometric limit"
                ));
            }
        }
        Ok(())
    }

    /// Validates the unused low bits at the end of the MSB-first payload.
    fn validate_payload_padding(&self) -> Result<()> {
        let used_bits = self.payload_bit_length % 8;
        if used_bits == 0 {
            return Ok(());
        }
        let padding_mask = (1u8 << (8 - used_bits)) - 1;
        let final_byte = self.mmap[self.payload_offset + self.payload_bit_length / 8];
        if final_byte & padding_mask != 0 {
            return Err(anyhow!("nonzero Huffman payload padding"));
        }
        Ok(())
    }

    /// Returns one level's mask and the rank of children before its node.
    #[inline]
    fn mask_and_rank(
        &self,
        level_index: usize,
        node_index: usize,
        calculate_rank: bool,
    ) -> (u8, usize) {
        let level = &self.levels[level_index];
        debug_assert!(node_index < level.node_count);
        if level_index < RAW_LEVEL_COUNT {
            self.raw_mask_and_rank(level, node_index, calculate_rank)
        } else {
            self.huffman_mask_and_rank(level, node_index, calculate_rank)
        }
    }

    /// Returns one raw mask and its optional child rank.
    #[inline]
    fn raw_mask_and_rank(
        &self,
        level: &LevelLayout,
        node_index: usize,
        calculate_rank: bool,
    ) -> (u8, usize) {
        let mask = self.raw_mask(level, node_index);
        if !calculate_rank {
            return (mask, 0);
        }
        let checkpoint_index = node_index / RAW_RANK_INTERVAL;
        let checkpoint_node = checkpoint_index * RAW_RANK_INTERVAL;
        let mut child_rank =
            read_u24(&self.mmap, level.raw_rank_offset + checkpoint_index * 3) as usize;
        let start_byte = level.data_offset + checkpoint_node / 2;
        let end_byte = level.data_offset + node_index / 2;
        let bytes = &self.mmap[start_byte..end_byte];
        let mut chunks = bytes.chunks_exact(8);
        for chunk in &mut chunks {
            child_rank += u64::from_le_bytes(chunk.try_into().expect("eight-byte chunk"))
                .count_ones() as usize;
        }
        for &byte in chunks.remainder() {
            child_rank += byte.count_ones() as usize;
        }
        if !node_index.is_multiple_of(2) {
            child_rank += (self.mmap[end_byte] & 0x0f).count_ones() as usize;
        }
        (mask, child_rank)
    }

    /// Returns one Huffman mask and its optional child rank.
    #[inline]
    fn huffman_mask_and_rank(
        &self,
        level: &LevelLayout,
        node_index: usize,
        calculate_rank: bool,
    ) -> (u8, usize) {
        let block_index = node_index / RESTART_INTERVAL;
        let index_in_block = node_index % RESTART_INTERVAL;
        let group_index = block_index / BLOCKS_PER_GROUP;
        let group_start = group_index * BLOCKS_PER_GROUP;
        let mut bit_offset =
            read_u24(&self.mmap, level.group_bit_offset + group_index * 3) as usize;
        let mut child_rank = if calculate_rank {
            read_u24(&self.mmap, level.group_child_rank_offset + group_index * 3) as usize
        } else {
            0
        };
        for preceding_block in group_start..block_index {
            bit_offset += read_packed(
                &self.mmap,
                level.block_bit_length_offset,
                level.block_bit_length_bytes,
                preceding_block,
                level.bit_length_width,
            ) as usize;
            if calculate_rank {
                child_rank += read_packed(
                    &self.mmap,
                    level.block_child_count_offset,
                    level.block_child_count_bytes,
                    preceding_block,
                    level.child_count_width,
                ) as usize;
            }
        }

        let mut mask = self.read_payload_bits(&mut bit_offset, 4) as u8;
        for _ in 0..index_in_block {
            if calculate_rank {
                child_rank += mask.count_ones() as usize;
            }
            mask = self.decode_symbol(level, mask, &mut bit_offset);
        }
        (mask, child_rank)
    }

    /// Returns one raw packed mask.
    #[inline]
    fn raw_mask(&self, level: &LevelLayout, node_index: usize) -> u8 {
        (self.mmap[level.data_offset + node_index / 2] >> (4 * (node_index % 2))) & 0x0f
    }

    /// Decodes one symbol from a validated canonical model.
    #[inline]
    fn decode_symbol(&self, level: &LevelLayout, context: u8, bit_offset: &mut usize) -> u8 {
        let model_offset = level.model_offset + context as usize * CONTEXT_SIZE;
        if self.mmap[model_offset] == 1 {
            return self.mmap[model_offset + 16];
        }
        let mut code = 0u16;
        let mut first_code = 0u16;
        let mut symbol_offset = 0usize;
        let mut previous_count = 0u16;
        for length in 1..=MAX_CODE_LENGTH {
            first_code = (first_code + previous_count) << 1;
            symbol_offset += previous_count as usize;
            code = (code << 1) | self.read_payload_bit(bit_offset);
            let count = self.mmap[model_offset + length] as u16;
            if code >= first_code && code - first_code < count {
                let symbol_index = symbol_offset + (code - first_code) as usize;
                let packed = self.mmap[model_offset + 16 + symbol_index / 2];
                return (packed >> (4 * (symbol_index % 2))) & 0x0f;
            }
            previous_count = count;
        }
        unreachable!("validated canonical model contains no matching code")
    }

    /// Decodes one symbol while enforcing its level payload boundary.
    fn decode_symbol_checked(
        &self,
        level: &LevelLayout,
        context: u8,
        bit_offset: &mut usize,
        level_end: usize,
    ) -> Result<u8> {
        let model_offset = level.model_offset + context as usize * CONTEXT_SIZE;
        if self.mmap[model_offset] == 1 {
            return Ok(self.mmap[model_offset + 16]);
        }
        let mut code = 0u16;
        let mut first_code = 0u16;
        let mut symbol_offset = 0usize;
        let mut previous_count = 0u16;
        for length in 1..=MAX_CODE_LENGTH {
            first_code = (first_code + previous_count) << 1;
            symbol_offset += previous_count as usize;
            code = (code << 1) | self.read_payload_bits_checked(bit_offset, 1, level_end)?;
            let count = self.mmap[model_offset + length] as u16;
            if code >= first_code && code - first_code < count {
                let symbol_index = symbol_offset + (code - first_code) as usize;
                let packed = self.mmap[model_offset + 16 + symbol_index / 2];
                return Ok((packed >> (4 * (symbol_index % 2))) & 0x0f);
            }
            previous_count = count;
        }
        Err(anyhow!("invalid Huffman code in topology payload"))
    }

    /// Reads an MSB-first payload integer.
    #[inline]
    fn read_payload_bits(&self, bit_offset: &mut usize, bit_count: usize) -> u16 {
        let mut value = 0u16;
        for _ in 0..bit_count {
            value = (value << 1) | self.read_payload_bit(bit_offset);
        }
        value
    }

    /// Reads one MSB-first payload bit.
    #[inline]
    fn read_payload_bit(&self, bit_offset: &mut usize) -> u16 {
        let offset = *bit_offset;
        let bit = (self.mmap[self.payload_offset + offset / 8] >> (7 - offset % 8)) & 1;
        *bit_offset += 1;
        u16::from(bit)
    }

    /// Reads an MSB-first payload integer while enforcing a bit boundary.
    fn read_payload_bits_checked(
        &self,
        bit_offset: &mut usize,
        bit_count: usize,
        bit_end: usize,
    ) -> Result<u16> {
        if bit_offset
            .checked_add(bit_count)
            .is_none_or(|end| end > bit_end)
        {
            return Err(anyhow!("truncated Huffman topology payload"));
        }
        Ok(self.read_payload_bits(bit_offset, bit_count))
    }
}

/// Returns the child position below a parent level.
#[inline]
fn child_index(cell_id: u64, parent_level: u32) -> usize {
    let parent_sentinel_position = MAX_S2_BITS - BITS_PER_LEVEL * parent_level;
    ((cell_id >> (parent_sentinel_position - 1)) & 3) as usize
}

/// Returns a little-endian 16-bit field.
fn read_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(bytes[offset..offset + 2].try_into().expect("u16 field"))
}

/// Returns a little-endian 32-bit field.
fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("u32 field"))
}

/// Returns a little-endian 24-bit field.
#[inline]
fn read_u24(bytes: &[u8], offset: usize) -> u32 {
    u32::from(bytes[offset])
        | (u32::from(bytes[offset + 1]) << 8)
        | (u32::from(bytes[offset + 2]) << 16)
}

/// Returns one LSB-first fixed-width packed value.
#[inline]
fn read_packed(
    bytes: &[u8],
    offset: usize,
    byte_count: usize,
    value_index: usize,
    width: u8,
) -> u32 {
    let bit_offset = value_index * width as usize;
    let byte_offset = bit_offset / 8;
    let shift = bit_offset % 8;
    let available_bytes = (byte_count - byte_offset).min(4);
    let mut word = 0u32;
    for source_byte in 0..available_bytes {
        word |= u32::from(bytes[offset + byte_offset + source_byte]) << (8 * source_byte);
    }
    (word >> shift) & ((1u32 << width) - 1)
}

/// Returns the byte count needed for fixed-width packed values.
fn packed_byte_count(value_count: usize, width: u8) -> Result<usize> {
    Ok(checked_mul(value_count, width as usize, "packed bit count")?.div_ceil(8))
}

/// Validates unused high bits in an LSB-first packed array.
fn validate_packed_padding(
    bytes: &[u8],
    offset: usize,
    byte_count: usize,
    value_count: usize,
    width: u8,
    name: &str,
) -> Result<()> {
    let used_bits = checked_mul(value_count, width as usize, "packed padding bit count")? % 8;
    if used_bits == 0 {
        return Ok(());
    }
    let padding_mask = !((1u8 << used_bits) - 1);
    if bytes[offset + byte_count - 1] & padding_mask != 0 {
        return Err(anyhow!("nonzero {} padding", name));
    }
    Ok(())
}

/// Requires an actual section offset to equal its canonical value.
fn require_offset(name: &str, actual: usize, expected: usize) -> Result<()> {
    if actual != expected {
        return Err(anyhow!(
            "invalid {} offset {} instead of {}",
            name,
            actual,
            expected
        ));
    }
    Ok(())
}

/// Adds two sizes while reporting format arithmetic overflow.
fn checked_add(left: usize, right: usize, name: &str) -> Result<usize> {
    left.checked_add(right)
        .ok_or_else(|| anyhow!("{} overflow", name))
}

/// Multiplies two sizes while reporting format arithmetic overflow.
fn checked_mul(left: usize, right: usize, name: &str) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| anyhow!("{} overflow", name))
}

/// Returns the maximum number of S2 nodes at a level.
fn maximum_node_count(level_index: usize) -> Result<usize> {
    let path_bits = level_index
        .checked_mul(BITS_PER_LEVEL as usize)
        .ok_or_else(|| anyhow!("geometric level bit count overflow"))?;
    let paths_per_face = 1usize
        .checked_shl(path_bits as u32)
        .ok_or_else(|| anyhow!("geometric node count overflow"))?;
    checked_mul(ROOT_FACE_COUNT, paths_per_face, "geometric node count")
}

/// Returns the minimum nonzero width needed to store a value.
fn required_width(maximum_value: u32) -> u8 {
    (u32::BITS - maximum_value.leading_zeros()).max(1) as u8
}
