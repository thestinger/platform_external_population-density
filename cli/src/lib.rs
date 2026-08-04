//! Provides host-side population density database build helpers.

use anyhow::{Result, anyhow};
use population_density::memmap2::MmapMut;
use population_density::{POPULATION_FIXED_POINT_SCALE, QueryEngine};
use rustc_hash::FxHashMap;
use std::path::Path;

pub mod geotiff;
pub mod topology_database;

pub use topology_database::{encode_topology_database, write_topology_database};

/// Opens a fully validated immutable snapshot of a database file.
pub fn open_database_snapshot(path: impl AsRef<Path>) -> Result<QueryEngine> {
    let database = std::fs::read(path)?;
    if database.is_empty() {
        return Err(anyhow!("database file is empty"));
    }
    let mut mapping = MmapMut::map_anon(database.len())?;
    mapping.copy_from_slice(&database);
    QueryEngine::from_mmap(mapping.make_read_only()?)
}

/// Converts a population value to fixed-point units.
pub fn convert_population_to_fixed(population: f32) -> Result<i64> {
    if !population.is_finite() || population < 0.0 {
        return Err(anyhow!(
            "population value must be finite and non-negative: {}",
            population
        ));
    }

    let fixed_population = (f64::from(population) * POPULATION_FIXED_POINT_SCALE as f64).round();
    if fixed_population >= i64::MAX as f64 {
        return Err(anyhow!(
            "population value {} exceeds the fixed-point range",
            population
        ));
    }
    Ok(fixed_population as i64)
}

/// Adds a non-negative fixed-point population to an S2 cell.
pub fn add_population(
    cell_populations: &mut FxHashMap<u64, i64>,
    cell_id: u64,
    population: i64,
) -> Result<()> {
    if population < 0 {
        return Err(anyhow!(
            "population increment must be non-negative: {}",
            population
        ));
    }

    let accumulated_population = cell_populations.entry(cell_id).or_insert(0);
    *accumulated_population = accumulated_population
        .checked_add(population)
        .ok_or_else(|| {
            anyhow!(
                "fixed-point population overflow for S2 cell {:016x}: {} + {}",
                cell_id,
                accumulated_population,
                population
            )
        })?;
    Ok(())
}
