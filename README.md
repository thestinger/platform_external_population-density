# population-density

A Rust library for generating and querying a highly-optimized database for S2 population density.

## Architecture

The engine implements the S2PP (flat block-compressed Patched Frame of Reference) architecture. It is designed to satisfy strict mobile memory and cache constraints:

1. **Memory-mapped I/O**: The database binary is mapped directly into memory (mmap), allowing the OS to load database pages into physical RAM page caches on demand with zero heap allocations or copies.
2. **High cache locality**: S2 leaf cells are sorted and stored in flat, compressed blocks. The database index contains contiguous block headers. Querying requires a single binary search over block headers (perfectly utilizing CPU pre-fetching) followed by stack-based decoding of the target block.
3. **Patched Frame of Reference (PFOR)**: Compressed blocks are decompressed in sub-microsecond time on the stack using PFOR delta decoding, avoiding heap overhead or object boxing in hot execution paths.

## Data source

The binary database is compiled from the WorldPop global population density GeoTIFF dataset.

* **Dataset**: global_pop_2026_CN_1km_R2025A_UA_v1.tif
* **SHA-256**: `bdfe7081506bd6123d43a29f22cf75ccdd6ea5fee3fdf7c3e9835debf71c6bc3`
* **Download source**: [WorldPop Hub - global_pop_2026](https://hub.worldpop.org/geodata/summary?id=80032)

Verify the downloaded file with `sha256sum` before building — the deterministic build reproduces the shipped database byte-for-byte only from this exact input.

The build pipeline aggregates pixel coordinates into level 12 S2 cell IDs in parallel using rayon, propagates population values up the S2 quadtree, and prunes cells with population counts below the 1,000 person threshold. The remaining cells are then compact-serialized into the final flat block-compressed database format.

## Database purpose

The generated database is used for the GrapheneOS system population density provider to protect user location privacy.

By performing highly-efficient, on-device lookups, the query engine determines if the user is in a location containing at least 1,000 people. If the current location has a population below this threshold, the engine returns a coarser, parent S2 cell ID representing a larger geographic region. This dynamic on-device masking ensures consistent privacy protection for everyone, regardless of whether they are in a dense city or a sparse rural area. By adapting the coarsening scale dynamically based on local density, it aims to ensure the returned S2 cell contains an estimated population of at least 1,000 people, replacing the legacy approach of applying a globally uniform static coarseness level (a flat 2,000-meter offset) which provides high anonymity in cities but can completely fail to protect privacy in sparsely populated regions.

## Testing

To execute the test suite, run:

```bash
cargo test --release
```

> [!NOTE]
> Running the integration and comprehensive parity tests without `--release` takes significantly longer (over 2 minutes) due to the large volume of coordinate lookups and cell validations. Using `--release` ensures they complete in well under a minute.

