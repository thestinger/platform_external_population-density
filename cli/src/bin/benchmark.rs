//! Benchmarks warm-cache S2PD throughput on synthetic level-30 inputs.

use anyhow::Result;
use clap::Parser;
use population_density::{
    MAX_DB_LEVEL, MAX_S2_BITS, NUM_ROOT_FACES, QueryEngine, S2_FACE_SHIFT, SHIFT_COMPACT,
    try_get_level,
};
use population_density_cli::open_database_snapshot;
use std::fmt::Write as _;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::{Duration, Instant};

const DEFAULT_QUERY_COUNT: usize = 1_000_000;
const DEFAULT_REPEAT_COUNT: usize = 6;
const HASH_MULTIPLIER: u64 = 0x9e37_79b1_85eb_ca87;
const RANDOM_SEED: u64 = 0;
const SPLITMIX_INCREMENT: u64 = 0x9e37_79b9_7f4a_7c15;
const SPLITMIX_MULTIPLIER_1: u64 = 0xbf58_476d_1ce4_e5b9;
const SPLITMIX_MULTIPLIER_2: u64 = 0x94d0_49bb_1331_11eb;
const S2_POSITION_MASK: u64 = (1u64 << MAX_S2_BITS) - 1;
const RETURN_LEVEL_COUNT: usize = MAX_DB_LEVEL as usize + 1;

/// Configures S2PD query benchmarks.
#[derive(Debug, Parser)]
#[command(version, about = "Benchmark S2PD on synthetic level-30 inputs.")]
struct Arguments {
    /// Specifies the S2PD database path.
    #[arg(long, default_value = "population_density_database.bin")]
    database_path: PathBuf,

    /// Specifies the number of queries in each measured run.
    #[arg(long, default_value_t = DEFAULT_QUERY_COUNT)]
    query_count: usize,

    /// Specifies the number of measured repeats.
    #[arg(long, default_value_t = DEFAULT_REPEAT_COUNT)]
    repeat_count: usize,
}

/// Stores one measured benchmark result.
struct BenchmarkResult {
    elapsed: Duration,
    hash: u64,
}

/// Benchmarks deterministic synthetic workloads.
fn main() -> Result<()> {
    let arguments = Arguments::parse();
    if arguments.query_count == 0 || arguments.repeat_count == 0 {
        anyhow::bail!("query count and repeat count must be nonzero");
    }

    let engine = open_database_snapshot(&arguments.database_path)?;
    let leaves = engine.reconstruct_all_leaves()?;
    if leaves.is_empty() {
        anyhow::bail!("database reconstructed no leaves");
    }
    println!("Loaded {} leaves.", leaves.len());

    {
        let queries = create_stored_cell_descendant_queries(&leaves, arguments.query_count);
        benchmark_workload(
            "stored-cell descendants (synthetic level-30 inputs)",
            &engine,
            &queries,
            arguments.repeat_count,
        )?;
    }
    {
        let queries = create_global_level_30_queries(arguments.query_count);
        benchmark_workload(
            "global cells (synthetic level-30 inputs)",
            &engine,
            &queries,
            arguments.repeat_count,
        )?;
    }
    Ok(())
}

/// Creates deterministic level-30 descendants of stored database cells.
fn create_stored_cell_descendant_queries(leaves: &[u32], query_count: usize) -> Vec<u64> {
    let mut random_state = RANDOM_SEED;
    (0..query_count)
        .map(|_| {
            let leaf_index = (next_random(&mut random_state) % leaves.len() as u64) as usize;
            let stored_cell_id = u64::from(leaves[leaf_index]) << SHIFT_COMPACT;
            create_level_30_descendant(stored_cell_id, next_random(&mut random_state))
        })
        .collect()
}

/// Creates deterministic level-30 inputs assigned round-robin across S2 faces.
fn create_global_level_30_queries(query_count: usize) -> Vec<u64> {
    let mut random_state = RANDOM_SEED;
    (0..query_count)
        .map(|query_index| {
            let face = (query_index % NUM_ROOT_FACES) as u64;
            let position = next_random(&mut random_state) & S2_POSITION_MASK;
            (face << S2_FACE_SHIFT) | (position << 1) | 1
        })
        .collect()
}

/// Returns a deterministic level-30 descendant of an S2 cell.
fn create_level_30_descendant(cell_id: u64, random_value: u64) -> u64 {
    let least_significant_bit = cell_id & cell_id.wrapping_neg();
    assert_ne!(least_significant_bit, 0, "cell id must be nonzero");
    let first_descendant = cell_id - least_significant_bit + 1;
    let descendant_index = random_value % least_significant_bit;
    first_descendant + descendant_index * 2
}

/// Advances a deterministic SplitMix64 generator.
fn next_random(random_state: &mut u64) -> u64 {
    *random_state = random_state.wrapping_add(SPLITMIX_INCREMENT);
    mix_random_value(*random_state)
}

/// Mixes one value using the SplitMix64 output permutation.
fn mix_random_value(value: u64) -> u64 {
    let value = (value ^ (value >> 30)).wrapping_mul(SPLITMIX_MULTIPLIER_1);
    let value = (value ^ (value >> 27)).wrapping_mul(SPLITMIX_MULTIPLIER_2);
    value ^ (value >> 31)
}

/// Benchmarks one warm-cache workload and prints absolute throughput.
fn benchmark_workload(
    name: &str,
    engine: &QueryEngine,
    queries: &[u64],
    repeat_count: usize,
) -> Result<()> {
    println!("Workload: {name}");
    let (return_level_counts, expected_hash) = warm_up(engine, queries)?;
    println!(
        "  Return levels: {}",
        format_return_level_counts(&return_level_counts)
    );

    let mut elapsed_times = Vec::with_capacity(repeat_count);
    for repeat_index in 0..repeat_count {
        let result = benchmark(engine, queries)?;
        if result.hash != expected_hash {
            anyhow::bail!("benchmark result hash changed between runs");
        }
        let queries_per_second = queries.len() as f64 / result.elapsed.as_secs_f64();
        println!(
            "  Repeat {}: {:.2} QPS in {:.2?}, hash {:016x}.",
            repeat_index + 1,
            queries_per_second,
            result.elapsed,
            result.hash
        );
        elapsed_times.push(result.elapsed);
    }
    println!(
        "  Aggregate: {:.2} QPS; median: {:.2} QPS.",
        aggregate_queries_per_second(queries.len(), &elapsed_times),
        median_queries_per_second(queries.len(), &elapsed_times)
    );
    Ok(())
}

/// Warms one engine and returns its output distribution and stable hash.
fn warm_up(engine: &QueryEngine, queries: &[u64]) -> Result<([usize; RETURN_LEVEL_COUNT], u64)> {
    let mut return_level_counts = [0usize; RETURN_LEVEL_COUNT];
    let mut hash = 0u64;
    for &query in queries {
        let result = black_box(engine.query(black_box(query))?);
        let return_level = try_get_level(result)? as usize;
        return_level_counts[return_level] += 1;
        hash = update_hash(hash, result);
    }
    Ok((return_level_counts, black_box(hash)))
}

/// Formats nonempty returned-level buckets.
fn format_return_level_counts(return_level_counts: &[usize; RETURN_LEVEL_COUNT]) -> String {
    let mut output = String::new();
    for (level, &count) in return_level_counts.iter().enumerate() {
        if count == 0 {
            continue;
        }
        if !output.is_empty() {
            output.push_str(", ");
        }
        write!(output, "L{level}: {count}").expect("writing to a string cannot fail");
    }
    output
}

/// Measures one engine on a fixed query workload.
fn benchmark(engine: &QueryEngine, queries: &[u64]) -> Result<BenchmarkResult> {
    let start_time = Instant::now();
    let mut hash = 0u64;
    for &query in queries {
        let result = black_box(engine.query(black_box(query))?);
        hash = update_hash(hash, result);
    }
    Ok(BenchmarkResult {
        elapsed: start_time.elapsed(),
        hash: black_box(hash),
    })
}

/// Returns aggregate throughput across measured runs.
fn aggregate_queries_per_second(query_count: usize, elapsed_times: &[Duration]) -> f64 {
    assert!(!elapsed_times.is_empty(), "elapsed times must be nonempty");
    let total_elapsed = elapsed_times.iter().copied().sum::<Duration>();
    query_count as f64 * elapsed_times.len() as f64 / total_elapsed.as_secs_f64()
}

/// Returns median throughput across measured runs.
fn median_queries_per_second(query_count: usize, elapsed_times: &[Duration]) -> f64 {
    assert!(!elapsed_times.is_empty(), "elapsed times must be nonempty");
    let mut throughputs: Vec<f64> = elapsed_times
        .iter()
        .map(|elapsed| query_count as f64 / elapsed.as_secs_f64())
        .collect();
    throughputs.sort_by(f64::total_cmp);
    let upper_index = throughputs.len() / 2;
    if throughputs.len().is_multiple_of(2) {
        (throughputs[upper_index - 1] + throughputs[upper_index]) / 2.0
    } else {
        throughputs[upper_index]
    }
}

/// Advances an order-sensitive benchmark hash.
#[inline]
fn update_hash(hash: u64, value: u64) -> u64 {
    hash.rotate_left(9)
        .wrapping_add(value)
        .wrapping_mul(HASH_MULTIPLIER)
}

#[cfg(test)]
mod tests {
    use super::*;
    use population_density::try_get_ancestor;
    use s2::cellid::CellID;
    use s2::latlng::LatLng;

    /// Verifies generated level-30 cells remain descendants of their stored parents.
    #[test]
    fn level_30_descendants_remain_within_their_parent() {
        let parent =
            CellID::from(LatLng::from_degrees(39.9042, 116.4074)).parent(u64::from(MAX_DB_LEVEL));
        for random_value in [0, 1, u64::MAX] {
            let descendant = create_level_30_descendant(parent.0, random_value);
            assert_eq!(try_get_level(descendant).unwrap(), 30);
            assert_eq!(
                try_get_ancestor(descendant, MAX_DB_LEVEL).unwrap(),
                parent.0
            );
        }
    }

    /// Verifies global queries are deterministic level-30 cells covering every S2 face.
    #[test]
    fn global_queries_are_deterministic_level_30_cells_on_every_face() {
        let first_queries = create_global_level_30_queries(NUM_ROOT_FACES * 100);
        let second_queries = create_global_level_30_queries(NUM_ROOT_FACES * 100);
        assert_eq!(first_queries, second_queries);

        let mut face_counts = [0usize; NUM_ROOT_FACES];
        for cell_id in first_queries {
            assert_eq!(try_get_level(cell_id).unwrap(), 30);
            face_counts[(cell_id >> S2_FACE_SHIFT) as usize] += 1;
        }
        assert_eq!(face_counts, [100; NUM_ROOT_FACES]);
    }

    /// Verifies aggregate and median QPS calculations.
    #[test]
    fn aggregate_and_median_statistics_are_correct() {
        let elapsed_times = [Duration::from_secs(2), Duration::from_secs(4)];
        assert_eq!(aggregate_queries_per_second(10, &elapsed_times), 20.0 / 6.0);
        assert_eq!(median_queries_per_second(10, &elapsed_times), 3.75);
    }
}
