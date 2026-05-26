//! Shared S2 quadtree leaf-extraction helpers for the database build and inspection tools.

use population_density::{MAX_DB_LEVEL, POPULATION_THRESHOLD_FIXED, get_children_ids};
use rustc_hash::FxHashMap;

/// Defines the bit position of the level-0 sentinel within a 64-bit S2 cell ID.
pub const LEVEL_0_SENTINEL_SHIFT: u32 = 60;

/// Recursively finds the leaves of the pruned S2 cell quadtree.
pub fn find_leaves(
    cell_id: u64,
    level: u32,
    cell_populations: &FxHashMap<u64, i64>,
    leaves: &mut Vec<u64>,
) {
    let population = *cell_populations.get(&cell_id).unwrap_or(&0);
    if population < POPULATION_THRESHOLD_FIXED {
        return;
    }

    if level == MAX_DB_LEVEL {
        leaves.push(cell_id);
        return;
    }

    let children = get_children_ids(cell_id, level);
    let mut children_in_tree = false;
    for &child in &children {
        if *cell_populations.get(&child).unwrap_or(&0) >= POPULATION_THRESHOLD_FIXED {
            children_in_tree = true;
            break;
        }
    }

    if !children_in_tree {
        leaves.push(cell_id);
        return;
    }

    for &child in &children {
        find_leaves(child, level + 1, cell_populations, leaves);
    }
}
