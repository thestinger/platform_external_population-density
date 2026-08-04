//! Analyzes and prints the distribution of S2 cell levels in the population density database.

use anyhow::{Result, anyhow};
use population_density::{SHIFT_COMPACT, try_get_level};
use population_density_cli::open_database_snapshot;
use std::collections::BTreeMap;
use std::path::Path;

/// Prints the S2 level distribution reconstructed from the database.
fn main() -> Result<()> {
    let database_path = Path::new("population_density_database.bin");
    if !database_path.exists() {
        return Err(anyhow!(
            "database file '{}' not found",
            database_path.display()
        ));
    }

    println!("Loading database and reconstructing all leaf cells...");
    let query_engine = open_database_snapshot(database_path)?;
    let leaves = query_engine.reconstruct_all_leaves()?;
    let total_leaves = leaves.len();
    println!("Total reconstructed leaf cells: {}", total_leaves);

    let mut level_counts = BTreeMap::new();

    for &compact_id in &leaves {
        let database_cell_id = (compact_id as u64) << SHIFT_COMPACT;
        let level = try_get_level(database_cell_id)?;
        *level_counts.entry(level).or_insert(0usize) += 1;
    }

    println!("\nS2 Level Distribution:");
    println!("--------------------------------------------------");
    println!("| Level |      Count      |      Percentage      |");
    println!("--------------------------------------------------");
    for (level, count) in &level_counts {
        let percentage = (*count as f64 / total_leaves as f64) * 100.0;
        println!("| {:>5} | {:>15} | {:>18.4}% |", level, count, percentage);
    }
    println!("--------------------------------------------------");

    Ok(())
}
