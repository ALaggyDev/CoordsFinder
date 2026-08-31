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
use crate::cpu_packed;
use crate::filter::prepare_filters;
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
        let next_item = AtomicUsize::new(0);
        let candidates = AtomicU64::new(0);
        let completed = AtomicUsize::new(0);
        let packed_filters = filters
            .directions
            .iter()
            .map(|direction| {
                cpu_packed::prepare(config, &direction.constraints, direction.forced_errors)
            })
            .collect::<Vec<_>>();

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
                        if let Some(packed) = &packed_filters[item.direction_index] {
                            cpu_packed::scan_item(
                                &item,
                                packed,
                                &direction.constraints,
                                &cancelled,
                                &mut matches,
                                &sink,
                            );
                        } else {
                            scan_item::<A>(
                                config,
                                &item,
                                &direction.constraints,
                                direction.forced_errors,
                                &cancelled,
                                &mut matches,
                                &sink,
                            );
                        }
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
    filter: &[CompiledRotation],
    forced_errors: i32,
    cancelled: &impl Fn() -> bool,
    matches: &mut Vec<Match>,
    sink: &Mutex<impl FnMut(&[Match])>,
) {
    if forced_errors > config.error_tolerance {
        return;
    }
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
    use crate::config::{IntRange, ScanOrder, TileSize, load};
    use crate::scan::make_plan;
    use crate::texture::get_texture;
    use crate::types::RotationInfo;

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
    fn packed_path_matches_brute_force_for_sparse_and_dense_covers() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let original = load(root.join("examples/packed-cpu-benchmark.conf")).unwrap();

        for observation_count in [6, original.filter.len()] {
            let mut config = original.clone();
            config.scan_order = ScanOrder::Linear;
            config.x_range = IntRange {
                start: -140,
                end: -130,
            };
            config.y_range = IntRange {
                start: -32,
                end: -22,
            };
            config.z_range = IntRange {
                start: -583,
                end: -573,
            };
            config.cpu_tile_size = TileSize { x: 10, z: 10 };
            config.filter.truncate(observation_count);

            let filters = prepare_filters(
                &config.filter,
                config.algorithm,
                &config.directions,
                config.error_tolerance,
            )
            .unwrap();
            let constraints = &filters.directions[0].constraints;
            assert!(cpu_packed::prepare(&config, constraints, 0).is_some());

            let plan = make_plan(&config, config.cpu_tile_size).unwrap();
            let mut packed = Vec::new();
            CpuScanner::new(1)
                .unwrap()
                .scan(
                    &config,
                    &plan,
                    |batch| packed.extend_from_slice(batch),
                    |_, _| {},
                    || false,
                )
                .unwrap();

            let mut brute = Vec::new();
            for x in config.x_range.start..config.x_range.end {
                for z in config.z_range.start..config.z_range.end {
                    for y in config.y_range.start..config.y_range.end {
                        if count_mismatches::<Vanilla3>(x, y, z, constraints, 0, 0) == 0 {
                            brute.push(Match {
                                x,
                                y,
                                z,
                                mismatches: 0,
                                direction: 0,
                            });
                        }
                    }
                }
            }
            packed.sort_by_key(|found| (found.x, found.y, found.z));
            brute.sort_by_key(|found| (found.x, found.y, found.z));
            assert_eq!(packed, brute, "observation_count={observation_count}");
        }
    }
}
