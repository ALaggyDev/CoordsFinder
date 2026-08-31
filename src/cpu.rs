//! Multithreaded CPU search backend.
//!
//! The scan is generic over [`TextureSampler`]. A small dispatch in
//! [`CpuScanner::scan`] selects one monomorphized implementation before worker
//! threads start, keeping algorithm branches out of the inner filter loop.

use std::cmp;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::thread;

use crate::config::ScanConfig;
use crate::filter::prepare_filters;
use crate::sample_plan::{SamplePlan, anchor_grid, select_sample_plan, visible_four_way_symbol};
use crate::scan::{ScanPlan, WorkItem, candidate_count};
use crate::texture::{Sodium1, Sodium2, TextureSampler, Vanilla1, Vanilla2, Vanilla3};
use crate::types::{CompiledRotation, Match, TextureAlgorithm};

// Deliver each match immediately so piping to another process does not delay it
// behind a worker-local result batch.
const RESULT_BATCH_SIZE: usize = 1;

/// A reusable CPU scanner with a fixed number of worker threads.
pub struct CpuScanner {
    threads: usize,
}

#[derive(Clone, Copy)]
struct DirectionSearch<'a> {
    filter: &'a [CompiledRotation],
    forced_errors: i32,
    sample_plan: Option<&'a SamplePlan>,
}

impl CpuScanner {
    /// Creates a scanner that may run up to `threads` workers concurrently.
    pub fn new(threads: usize) -> Result<Self, String> {
        if threads == 0 {
            return Err("CPU thread count must be positive".to_owned());
        }
        Ok(Self { threads })
    }

    /// Returns the configured maximum worker count.
    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Runs the plan and reports completed work at tile boundaries.
    pub fn scan(
        &self,
        config: &ScanConfig,
        plan: &ScanPlan<'_>,
        sink: impl FnMut(&[Match]) + Send,
        progress: impl FnMut(u64, usize) + Send,
        cancelled: impl Fn() -> bool + Sync,
    ) -> Result<(), String> {
        // Runtime selection happens exactly once; run_mode is specialized for A.
        match config.algorithm {
            TextureAlgorithm::Vanilla1 => {
                self.run_mode::<Vanilla1>(config, plan, sink, progress, cancelled)
            }
            TextureAlgorithm::Vanilla2 => {
                self.run_mode::<Vanilla2>(config, plan, sink, progress, cancelled)
            }
            TextureAlgorithm::Vanilla3 => {
                self.run_mode::<Vanilla3>(config, plan, sink, progress, cancelled)
            }
            TextureAlgorithm::Sodium1 => {
                self.run_mode::<Sodium1>(config, plan, sink, progress, cancelled)
            }
            TextureAlgorithm::Sodium2 => {
                self.run_mode::<Sodium2>(config, plan, sink, progress, cancelled)
            }
        }
    }

    fn run_mode<A: TextureSampler>(
        &self,
        config: &ScanConfig,
        plan: &ScanPlan<'_>,
        sink: impl FnMut(&[Match]) + Send,
        progress: impl FnMut(u64, usize) + Send,
        cancelled: impl Fn() -> bool + Sync,
    ) -> Result<(), String> {
        let filters = prepare_filters(
            &config.filter,
            config.algorithm,
            &config.directions,
            config.error_tolerance,
        )?;
        let sample_plans = filters
            .directions
            .iter()
            .map(|direction| {
                select_sample_plan(
                    &direction.constraints,
                    config.algorithm,
                    direction.forced_errors,
                    config.error_tolerance,
                    config.search_mode,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let next_item = AtomicUsize::new(0);
        let candidates = AtomicU64::new(0);
        let completed = AtomicUsize::new(0);

        // Callbacks are serialized so callers can use ordinary FnMut state and
        // batches from different workers never interleave on stdout.
        let sink = Mutex::new(sink);
        let progress = Mutex::new(progress);
        let worker_count = cmp::min(self.threads, plan.total_items());

        thread::scope(|scope| {
            for _ in 0..worker_count {
                scope.spawn(|| {
                    let mut matches = Vec::with_capacity(RESULT_BATCH_SIZE);
                    while !cancelled() {
                        let work_num = next_item.fetch_add(1, Ordering::Relaxed);
                        let Some(item) = plan.work_item(work_num) else {
                            break;
                        };

                        let direction = &filters.directions[item.direction_index];
                        scan_item::<A>(
                            config,
                            &item,
                            DirectionSearch {
                                filter: &direction.constraints,
                                forced_errors: direction.forced_errors,
                                sample_plan: sample_plans[item.direction_index].as_ref(),
                            },
                            &cancelled,
                            &mut matches,
                            &sink,
                        );
                        flush(&mut matches, &sink);

                        // A cancelled partial tile is not counted as complete.
                        if cancelled() {
                            break;
                        }
                        let item_candidates = candidate_count(&item).0;
                        let total = candidates
                            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                                Some(current.saturating_add(item_candidates))
                            })
                            .unwrap()
                            .saturating_add(item_candidates);
                        let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                        progress.lock().unwrap()(total, done);
                    }
                    flush(&mut matches, &sink);
                });
            }
        });
        Ok(())
    }
}

#[inline(always)]
fn count_mismatches<A: TextureSampler>(
    x: i32,
    y: i32,
    z: i32,
    filter: &[CompiledRotation],
    forced_errors: i32,
    tolerance: i32,
) -> i32 {
    let mut mismatches = forced_errors;
    for sample in filter {
        // Minecraft performs these additions as wrapping Java int operations.
        let variant = A::sample(
            x.wrapping_add(i32::from(sample.x)),
            y.wrapping_add(i32::from(sample.y)),
            z.wrapping_add(i32::from(sample.z)),
            16,
        );
        if sample.accepted_indices & (1 << variant) == 0 {
            mismatches += 1;
            if mismatches > tolerance {
                break;
            }
        }
    }
    mismatches
}

fn scan_item<A: TextureSampler>(
    config: &ScanConfig,
    item: &WorkItem,
    direction: DirectionSearch<'_>,
    cancelled: &impl Fn() -> bool,
    matches: &mut Vec<Match>,
    sink: &Mutex<impl FnMut(&[Match])>,
) {
    if direction.forced_errors > config.error_tolerance {
        return;
    }
    if let Some(sample_plan) = direction.sample_plan {
        if scan_item_sample::<A>(
            config,
            item,
            direction.filter,
            sample_plan,
            cancelled,
            matches,
            sink,
        ) {
            return;
        }
    }
    scan_item_naive::<A>(
        config,
        item,
        direction.filter,
        direction.forced_errors,
        cancelled,
        matches,
        sink,
    );
}

fn scan_item_naive<A: TextureSampler>(
    config: &ScanConfig,
    item: &WorkItem,
    filter: &[CompiledRotation],
    forced_errors: i32,
    cancelled: &impl Fn() -> bool,
    matches: &mut Vec<Match>,
    sink: &Mutex<impl FnMut(&[Match])>,
) {
    for x in item.start.x..item.end.x {
        for z in item.start.z..item.end.z {
            if cancelled() {
                return;
            }
            for y in item.start.y..item.end.y {
                let mismatches =
                    count_mismatches::<A>(x, y, z, filter, forced_errors, config.error_tolerance);
                if mismatches <= config.error_tolerance {
                    matches.push(Match {
                        x,
                        y,
                        z,
                        mismatches,
                        direction: item.direction,
                    });
                    if matches.len() == RESULT_BATCH_SIZE {
                        flush(matches, sink);
                    }
                }
            }
        }
    }
}

/// Returns false only when an integer-boundary halo asks the caller to use the
/// naive loop for this tile.
fn scan_item_sample<A: TextureSampler>(
    config: &ScanConfig,
    item: &WorkItem,
    filter: &[CompiledRotation],
    plan: &SamplePlan,
    cancelled: &impl Fn() -> bool,
    matches: &mut Vec<Match>,
    sink: &Mutex<impl FnMut(&[Match])>,
) -> bool {
    let Some(grid) = anchor_grid(plan, item) else {
        return false;
    };
    for x_index in 0..grid.counts[0] {
        let anchor_x =
            (i64::from(grid.start.x) + i64::from(x_index) * i64::from(grid.steps[0])) as i32;
        for z_index in 0..grid.counts[2] {
            if cancelled() {
                return true;
            }
            let anchor_z =
                (i64::from(grid.start.z) + i64::from(z_index) * i64::from(grid.steps[2])) as i32;
            for y_index in 0..grid.counts[1] {
                let anchor_y = (i64::from(grid.start.y)
                    + i64::from(y_index) * i64::from(grid.steps[1]))
                    as i32;
                let mut node_index = 0_usize;
                let mut rejected = false;
                for offset in &plan.offsets {
                    let variant = A::sample(
                        anchor_x.wrapping_add(offset.x),
                        anchor_y.wrapping_add(offset.y),
                        anchor_z.wrapping_add(offset.z),
                        16,
                    );
                    let symbol = visible_four_way_symbol(config.algorithm, variant) as usize;
                    let child = plan.nodes[node_index].children[symbol];
                    if child == crate::sample_plan::MISSING_TRIE_CHILD {
                        rejected = true;
                        break;
                    }
                    node_index = child as usize;
                }
                if rejected {
                    continue;
                }

                let leaf = plan.nodes[node_index];
                let outputs = &plan.placements
                    [leaf.output_start as usize..(leaf.output_start + leaf.output_count) as usize];
                for placement in outputs {
                    let candidate_x = i64::from(anchor_x) - i64::from(placement.origin.x);
                    let candidate_y = i64::from(anchor_y) - i64::from(placement.origin.y);
                    let candidate_z = i64::from(anchor_z) - i64::from(placement.origin.z);
                    if candidate_x < i64::from(item.start.x)
                        || candidate_x >= i64::from(item.end.x)
                        || candidate_y < i64::from(item.start.y)
                        || candidate_y >= i64::from(item.end.y)
                        || candidate_z < i64::from(item.start.z)
                        || candidate_z >= i64::from(item.end.z)
                    {
                        continue;
                    }
                    let candidate = (candidate_x as i32, candidate_y as i32, candidate_z as i32);
                    if remaining_filter_matches::<A>(
                        candidate,
                        filter,
                        &placement.witness_indices[..plan.sample_size()],
                    ) {
                        matches.push(Match {
                            x: candidate.0,
                            y: candidate.1,
                            z: candidate.2,
                            mismatches: 0,
                            direction: item.direction,
                        });
                        if matches.len() == RESULT_BATCH_SIZE {
                            flush(matches, sink);
                        }
                    }
                }
            }
        }
    }
    true
}

#[inline(always)]
fn remaining_filter_matches<A: TextureSampler>(
    candidate: (i32, i32, i32),
    filter: &[CompiledRotation],
    witness_indices: &[u8],
) -> bool {
    for (index, sample) in filter.iter().enumerate() {
        if witness_indices.contains(&(index as u8)) {
            continue;
        }
        let variant = A::sample(
            candidate.0.wrapping_add(i32::from(sample.x)),
            candidate.1.wrapping_add(i32::from(sample.y)),
            candidate.2.wrapping_add(i32::from(sample.z)),
            16,
        );
        if sample.accepted_indices & (1 << variant) == 0 {
            return false;
        }
    }
    true
}

fn flush(matches: &mut Vec<Match>, sink: &Mutex<impl FnMut(&[Match])>) {
    if matches.is_empty() {
        return;
    }
    sink.lock().unwrap()(matches);
    matches.clear();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{IntRange, ScanOrder, TileSize};
    use crate::scan::make_plan;
    use crate::texture::get_texture;
    use crate::types::{RotationInfo, SearchMode};

    #[test]
    fn matches_all_texture_algorithms() {
        let coordinate = (17, -4, -31);
        for algorithm in [
            TextureAlgorithm::Vanilla1,
            TextureAlgorithm::Vanilla2,
            TextureAlgorithm::Vanilla3,
            TextureAlgorithm::Sodium1,
            TextureAlgorithm::Sodium2,
        ] {
            let config = ScanConfig {
                algorithm,
                scan_order: ScanOrder::Linear,
                directions: vec![0],
                x_range: IntRange {
                    start: coordinate.0,
                    end: coordinate.0 + 1,
                },
                y_range: IntRange {
                    start: coordinate.1,
                    end: coordinate.1 + 1,
                },
                z_range: IntRange {
                    start: coordinate.2,
                    end: coordinate.2 + 1,
                },
                cpu_tile_size: TileSize { x: 1, z: 1 },
                filter: vec![RotationInfo::new(
                    0,
                    0,
                    0,
                    get_texture(algorithm, coordinate.0, coordinate.1, coordinate.2, 4),
                    false,
                )],
                ..ScanConfig::default()
            };
            let plan = make_plan(&config, config.cpu_tile_size).unwrap();
            let mut found = Vec::new();
            CpuScanner::new(2)
                .unwrap()
                .scan(
                    &config,
                    &plan,
                    |batch| found.extend_from_slice(batch),
                    |_, _| {},
                    || false,
                )
                .unwrap();
            assert_eq!(
                found,
                vec![Match {
                    x: coordinate.0,
                    y: coordinate.1,
                    z: coordinate.2,
                    mismatches: 0,
                    direction: 0,
                }],
                "{algorithm}"
            );
        }
    }

    #[test]
    fn counts_conflicting_faces_once_per_block() {
        let expected = get_texture(TextureAlgorithm::Vanilla3, 1, 0, 0, 4);
        let config = ScanConfig {
            algorithm: TextureAlgorithm::Vanilla3,
            scan_order: ScanOrder::Linear,
            directions: vec![0],
            x_range: IntRange { start: 0, end: 1 },
            y_range: IntRange { start: 0, end: 1 },
            z_range: IntRange { start: 0, end: 1 },
            error_tolerance: 1,
            cpu_tile_size: TileSize { x: 1, z: 1 },
            filter: vec![
                RotationInfo::netherrack(0, 0, 0, 0, crate::types::Face::Up),
                RotationInfo::netherrack(0, 0, 0, 1, crate::types::Face::Up),
                RotationInfo::new(1, 0, 0, expected, false),
            ],
            ..ScanConfig::default()
        };
        let plan = make_plan(&config, config.cpu_tile_size).unwrap();
        let mut found = Vec::new();
        CpuScanner::new(1)
            .unwrap()
            .scan(
                &config,
                &plan,
                |batch| found.extend_from_slice(batch),
                |_, _| {},
                || false,
            )
            .unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].mismatches, 1);
    }

    #[test]
    fn skips_direction_ruled_out_by_combined_faces() {
        let expected = get_texture(TextureAlgorithm::Vanilla1, 0, 0, 0, 4);
        let config = ScanConfig {
            algorithm: TextureAlgorithm::Vanilla1,
            scan_order: ScanOrder::Linear,
            directions: vec![0, 90],
            x_range: IntRange { start: 0, end: 1 },
            y_range: IntRange { start: 0, end: 1 },
            z_range: IntRange { start: 0, end: 1 },
            cpu_tile_size: TileSize { x: 1, z: 1 },
            filter: vec![
                RotationInfo::new(0, 0, 0, expected, false),
                RotationInfo::new(0, 0, 0, expected & 1, true),
            ],
            ..ScanConfig::default()
        };
        let plan = make_plan(&config, config.cpu_tile_size).unwrap();
        let mut found = Vec::new();
        CpuScanner::new(1)
            .unwrap()
            .scan(
                &config,
                &plan,
                |batch| found.extend_from_slice(batch),
                |_, _| {},
                || false,
            )
            .unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].direction, 0);
    }

    #[test]
    fn sample_trie_matches_naive_search() {
        let target = (1_i32, 0_i32, -2_i32);
        for algorithm in [
            TextureAlgorithm::Vanilla1,
            TextureAlgorithm::Vanilla2,
            TextureAlgorithm::Vanilla3,
            TextureAlgorithm::Sodium1,
            TextureAlgorithm::Sodium2,
        ] {
            let mut filter = Vec::new();
            for x in 0..4_i8 {
                for z in 0..4_i8 {
                    filter.push(RotationInfo::new(
                        x,
                        0,
                        z,
                        get_texture(
                            algorithm,
                            target.0 + i32::from(x),
                            target.1,
                            target.2 + i32::from(z),
                            4,
                        ),
                        false,
                    ));
                }
            }
            let base = ScanConfig {
                algorithm,
                scan_order: ScanOrder::Linear,
                directions: vec![0, 90, 180, 270],
                x_range: IntRange { start: -4, end: 5 },
                y_range: IntRange { start: -1, end: 2 },
                z_range: IntRange { start: -5, end: 4 },
                cpu_tile_size: TileSize { x: 5, z: 4 },
                filter,
                ..ScanConfig::default()
            };

            let run = |mode| {
                let mut config = base.clone();
                config.search_mode = mode;
                let plan = make_plan(&config, config.cpu_tile_size).unwrap();
                let mut found = Vec::new();
                CpuScanner::new(1)
                    .unwrap()
                    .scan(
                        &config,
                        &plan,
                        |batch| found.extend_from_slice(batch),
                        |_, _| {},
                        || false,
                    )
                    .unwrap();
                found.sort_by_key(|item| (item.x, item.y, item.z, item.direction));
                found
            };

            let naive = run(SearchMode::Naive);
            let sampled = run(SearchMode::SampleTrie);
            assert_eq!(sampled, naive, "{algorithm}");
            assert!(
                sampled
                    .iter()
                    .any(|found| { (found.x, found.y, found.z) == (target.0, target.1, target.2) })
            );
        }
    }
}
