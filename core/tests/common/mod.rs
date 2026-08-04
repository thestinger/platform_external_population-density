//! Provides shared integration-test database helpers.

use anyhow::Result;
use population_density::QueryEngine;
use population_density::memmap2::MmapMut;
use std::path::Path;

/// Opens a fully validated snapshot of a database file.
pub fn open_database(path: impl AsRef<Path>) -> Result<QueryEngine> {
    let database = std::fs::read(path)?;
    let mut mapping = MmapMut::map_anon(database.len())?;
    mapping.copy_from_slice(&database);
    QueryEngine::from_mmap(mapping.make_read_only()?)
}
