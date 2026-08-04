//! Defines the versioned mmap topology database format.

/// Identifies a topology database.
pub const MAGIC: [u8; 4] = *b"S2PD";
/// Identifies the supported topology database format version.
pub const VERSION: u16 = 1;
/// Defines the fixed header and level-directory size.
pub const HEADER_SIZE: usize = 256;
/// Defines the number of bytes in the fixed header before the level directory.
pub const FIXED_HEADER_SIZE: usize = 64;
/// Defines the number of bytes in one level-directory entry.
pub const LEVEL_ENTRY_SIZE: usize = 16;
/// Locates the format magic bytes.
pub const MAGIC_OFFSET: usize = 0;
/// Locates the format version field.
pub const VERSION_OFFSET: usize = MAGIC_OFFSET + MAGIC.len();
/// Locates the format header-size field.
pub const HEADER_SIZE_OFFSET: usize = VERSION_OFFSET + std::mem::size_of::<u16>();
/// Defines the number of levels containing child masks.
pub const LEVEL_COUNT: usize = 12;
/// Defines the number of upper levels stored as raw masks.
pub const RAW_LEVEL_COUNT: usize = 7;
/// Defines the restart interval for Huffman-coded masks.
pub const RESTART_INTERVAL: usize = 208;
/// Defines the raw-mask rank checkpoint interval.
pub const RAW_RANK_INTERVAL: usize = 128;
/// Defines the number of restart blocks sharing one absolute index base.
pub const BLOCKS_PER_GROUP: usize = 16;
/// Defines the number of mask symbols and previous-mask contexts.
pub const MASK_SYMBOL_COUNT: usize = 16;
/// Defines the number of bytes in one compact canonical Huffman context.
pub const CONTEXT_SIZE: usize = 24;
/// Defines the number of bytes in one level's canonical Huffman models.
pub const LEVEL_MODEL_SIZE: usize = MASK_SYMBOL_COUNT * CONTEXT_SIZE;
/// Defines the maximum canonical Huffman code length.
pub const MAX_CODE_LENGTH: usize = 15;
/// Defines the largest supported packed restart-block bit-length width.
pub const MAX_BLOCK_BIT_LENGTH_WIDTH: u8 = 12;
/// Defines the largest supported packed restart-block child-count width.
pub const MAX_BLOCK_CHILD_COUNT_WIDTH: u8 = 10;
/// Defines the largest value representable by a three-byte index.
pub const MAX_U24: u32 = 0x00ff_ffff;
/// Defines the number of S2 root faces required by the database.
pub const ROOT_FACE_COUNT: usize = 6;

/// Locates the declared file size.
pub const FILE_SIZE_OFFSET: usize = 8;
/// Locates the Huffman payload bit length.
pub const PAYLOAD_BIT_LENGTH_OFFSET: usize = 12;
/// Locates the raw-mask section offset.
pub const RAW_MASKS_OFFSET: usize = 16;
/// Locates the raw-rank section offset.
pub const RAW_RANKS_OFFSET: usize = 20;
/// Locates the canonical-model section offset.
pub const MODELS_OFFSET: usize = 24;
/// Locates the group bit-offset section offset.
pub const GROUP_BIT_OFFSETS_OFFSET: usize = 28;
/// Locates the group child-rank section offset.
pub const GROUP_CHILD_RANKS_OFFSET: usize = 32;
/// Locates the packed block-bit-length section offset.
pub const BLOCK_BIT_LENGTHS_OFFSET: usize = 36;
/// Locates the packed block-child-count section offset.
pub const BLOCK_CHILD_COUNTS_OFFSET: usize = 40;
/// Locates the Huffman payload section offset.
pub const PAYLOAD_OFFSET: usize = 44;
/// Locates the Huffman restart interval.
pub const RESTART_INTERVAL_OFFSET: usize = 48;
/// Locates the raw-rank checkpoint interval.
pub const RAW_RANK_INTERVAL_OFFSET: usize = 50;
/// Locates the mask level count.
pub const LEVEL_COUNT_OFFSET: usize = 52;
/// Locates the raw level count.
pub const RAW_LEVEL_COUNT_OFFSET: usize = 53;
/// Locates the maximum database level.
pub const MAX_DATABASE_LEVEL_OFFSET: usize = 54;
/// Locates the number of restart blocks per index group.
pub const BLOCKS_PER_GROUP_OFFSET: usize = 55;
/// Locates the total compact leaf count.
pub const LEAF_COUNT_OFFSET: usize = 56;
/// Locates the level-12 terminal node count.
pub const TERMINAL_NODE_COUNT_OFFSET: usize = 60;

/// Locates a directory entry's node count.
pub const LEVEL_NODE_COUNT_OFFSET: usize = 0;
/// Locates a directory entry's section-relative data offset.
pub const LEVEL_DATA_OFFSET_OFFSET: usize = 4;
/// Locates a directory entry's data length.
pub const LEVEL_DATA_LENGTH_OFFSET: usize = 8;
/// Locates a directory entry's encoding identifier.
pub const LEVEL_ENCODING_OFFSET: usize = 12;
/// Locates a directory entry's packed block-bit-length width.
pub const LEVEL_BIT_LENGTH_WIDTH_OFFSET: usize = 13;
/// Locates a directory entry's packed block-child-count width.
pub const LEVEL_CHILD_COUNT_WIDTH_OFFSET: usize = 14;
/// Locates a directory entry's flags.
pub const LEVEL_FLAGS_OFFSET: usize = 15;

/// Identifies a raw-mask directory entry.
pub const ENCODING_RAW: u8 = 0;
/// Identifies a Huffman-coded directory entry.
pub const ENCODING_HUFFMAN: u8 = 1;
/// Marks a level with child-rank indexes.
pub const CHILD_RANK_FLAG: u8 = 1;
