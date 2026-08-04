//! Provides a benchmark suite for the S2 population density database engine.
//!
//! Measures Queries Per Second (QPS) performance with and without safe indexing.

use population_density::SHIFT_COMPACT;
use population_density_cli::open_database_snapshot;
use std::time::Instant;

/// Configures the total number of benchmark queries to execute.
const TOTAL_BENCHMARK_QUERIES: usize = 1_000_000;

/// Configures the number of warmup queries to run before the main benchmark.
const WARMUP_QUERIES_COUNT: usize = 50_000;

/// Runs the benchmark suite to evaluate the query throughput of the database engine.
fn main() {
    let database_path = "population_density_database.bin";
    if !std::path::Path::new(database_path).exists() {
        eprintln!(
            "Database not found at {}, please build it first.",
            database_path
        );
        return;
    }

    let engine = open_database_snapshot(database_path).unwrap();
    let leaves = engine.reconstruct_all_leaves().unwrap();

    if leaves.is_empty() {
        eprintln!("Reconstructed leaves array is empty, benchmark cannot proceed.");
        return;
    }

    println!("Loaded database with {} leaves.", leaves.len());

    // Generate queries based on the leaves.
    let mut queries = Vec::with_capacity(TOTAL_BENCHMARK_QUERIES);
    for index in 0..TOTAL_BENCHMARK_QUERIES {
        let compact_cell_id = leaves[index % leaves.len()];
        // Reconstruct the full Level 12 Cell ID by shifting left.
        let s2_cell_id = (compact_cell_id as u64) << SHIFT_COMPACT;
        queries.push(s2_cell_id);
    }

    println!("Running warmup...");
    for &query in &queries[0..WARMUP_QUERIES_COUNT] {
        let _ = engine.query(query).unwrap();
    }

    println!(
        "Running benchmark with {} queries...",
        TOTAL_BENCHMARK_QUERIES
    );
    let start = Instant::now();
    let mut hash = 0u64;
    for &query in &queries {
        hash = hash.wrapping_add(engine.query(query).unwrap());
    }
    let elapsed = start.elapsed();

    let queries_per_second = queries.len() as f64 / elapsed.as_secs_f64();
    println!("Benchmark completed.");
    println!("Elapsed time: {:?}", elapsed);
    println!("Throughput: {:.2} QPS", queries_per_second);
    println!("Verification hash: {:x}", hash);
}
