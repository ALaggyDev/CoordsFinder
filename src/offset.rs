//! CPU block offset search using fanda857's low-bit lifting algorithm.
//!
//! Enumerate the low 16 X/Z bits, then lift one bit per axis up to bit 28.
//! Each determined seed bit prunes whole subtrees before texture sampling.
//!
//! This is slop code - read `/docs/fast_bit_lifting.rs` for a better understanding of the algorithm.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

use crate::config::{IntRange, ScanConfig};
use crate::cpu::count_mismatches;
use crate::filter::{PreparedDirection, prepare_filters};
use crate::texture::{Sodium1, Sodium2, TextureSampler, Vanilla1, Vanilla2, Vanilla3};
use crate::types::{Match, TextureAlgorithm};

const BASE: u32 = 1 << 16;

/// Allowed low nibble prefixes for each number of known bits (0 through 4).
#[derive(Clone, Debug)]
pub(crate) struct Constraint {
    dx: u32,
    dz: u32,
    possible: [[u16; 5]; 3],
}

fn nibble_mask(nibble: Option<u8>, clipped: bool) -> u16 {
    match nibble {
        None => u16::MAX,
        Some(0..=3) if clipped => 0x000f,
        Some(12..=15) if clipped => 0xf000,
        Some(n) => 1 << n,
    }
}

fn prefixes(accepted: u16) -> [u16; 5] {
    std::array::from_fn(|bits| {
        let mut possible = 0;
        for n in 0..16 {
            if accepted & (1 << n) != 0 {
                possible |= 1 << (n & ((1 << bits) - 1));
            }
        }
        possible
    })
}

/// Combines observations at the same X/Z, including different block heights.
pub(crate) fn prepare(config: &ScanConfig) -> Result<Vec<Vec<Constraint>>, String> {
    for range in [config.x_range, config.z_range] {
        if range.start < -30_000_000 || range.end > 30_000_001 || range.start >= range.end {
            return Err("block offset X/Z ranges must lie within [-30000000, 30000000]".to_owned());
        }
    }
    let mut groups = HashMap::new();
    for observation in &config.offsets {
        if observation.nibbles.iter().flatten().any(|&n| n > 15)
            || observation.nibbles.iter().all(Option::is_none)
            || (observation.dripstone && observation.nibbles[1].is_some())
        {
            return Err("invalid block offset nibbles or dripstone vertical offset".to_owned());
        }
        let masks = groups
            .entry((observation.position.x, observation.position.z))
            .or_insert([u16::MAX; 3]);
        for (axis, mask) in masks.iter_mut().enumerate() {
            *mask &= nibble_mask(
                observation.nibbles[axis],
                observation.dripstone && axis != 1,
            );
        }
        if masks.contains(&0) {
            return Err(format!(
                "contradictory block offsets at ({}, {})",
                observation.position.x, observation.position.z
            ));
        }
    }
    // Check the most restrictive early nibble first, independent of hash order.
    let mut groups: Vec<_> = groups.into_iter().collect();
    groups.sort_unstable_by_key(|&(position, masks)| {
        (
            masks[0].count_ones(),
            masks[1].count_ones(),
            masks[2].count_ones(),
            position,
        )
    });
    Ok(config
        .directions
        .iter()
        .map(|direction| {
            groups
                .iter()
                .map(|&((dx, dz), [x, y, z])| {
                    // Negation mirrors displacement, so nibble n becomes 15-n.
                    let (dx, dz, x, z) = match direction {
                        90 => (dz.wrapping_neg(), dx, z.reverse_bits(), x),
                        180 => (
                            dx.wrapping_neg(),
                            dz.wrapping_neg(),
                            x.reverse_bits(),
                            z.reverse_bits(),
                        ),
                        270 => (dz, dx.wrapping_neg(), z, x.reverse_bits()),
                        _ => (dx, dz, x, z),
                    };
                    Constraint {
                        dx: dx as u32,
                        dz: dz as u32,
                        possible: [prefixes(x), prefixes(y), prefixes(z)],
                    }
                })
                .collect()
        })
        .collect())
}

#[inline(always)]
fn coordinate_seed(x: u32, z: u32) -> u32 {
    let seed = x.wrapping_mul(3_129_871) ^ z.wrapping_mul(116_129_781);
    seed.wrapping_mul(seed)
        .wrapping_mul(42_317_861)
        .wrapping_add(seed.wrapping_mul(11))
}

impl Constraint {
    #[inline(always)]
    fn seed(&self, x: u32, z: u32) -> u32 {
        coordinate_seed(x.wrapping_add(self.dx), z.wrapping_add(self.dz))
    }

    #[inline(always)]
    fn accepts(&self, seed: u32, bits: u32) -> bool {
        for axis in 0..3 {
            let shift = 16 + axis as u32 * 4;
            let known = bits.saturating_sub(shift).min(4);
            let value = (seed >> shift) & ((1 << known) - 1);
            if self.possible[axis][known as usize] & (1 << value) == 0 {
                return false;
            }
        }
        true
    }
}

/// A prefix is usable if any integer in the half-open range has these low bits.
fn in_bounds(value: u32, bits: u32, range: IntRange) -> bool {
    let distance = value.wrapping_sub(range.start as u32) & ((1 << bits) - 1);
    i64::from(range.start) + i64::from(distance) < i64::from(range.end)
}

fn root_count(range: IntRange) -> usize {
    (i64::from(range.end) - i64::from(range.start)).clamp(0, i64::from(BASE)) as usize
}

/// Number of independently scheduled low-16-bit X prefixes across directions.
pub fn work_items(config: &ScanConfig) -> usize {
    root_count(config.x_range) * config.directions.len()
}

fn lift(
    x: u32,
    z: u32,
    bits: u32,
    config: &ScanConfig,
    constraints: &[Constraint],
    cancelled: &impl Fn() -> bool,
    found: &mut impl FnMut(i32, i32),
) {
    if cancelled() {
        return;
    }
    if bits == 28 {
        found((x as i32) << 4 >> 4, (z as i32) << 4 >> 4);
        return;
    }
    // Both multipliers and the polynomial derivative are odd. At the first
    // lift, the next seed bit toggles by X_bit XOR Z_bit for every observation.
    let mut toggles = 3;
    if bits == 16 {
        for constraint in constraints {
            let mut allowed = constraint.possible[0][1];
            if constraint.seed(x, z) & (1 << 16) != 0 {
                allowed = ((allowed & 1) << 1) | (allowed >> 1);
            }
            toggles &= allowed;
            if toggles == 0 {
                return;
            }
        }
    }
    for option in 0..4 {
        if toggles & (1 << ((option & 1) ^ (option >> 1))) == 0 {
            continue;
        }
        let next_x = x | ((option & 1) << bits);
        let next_z = z | ((option >> 1) << bits);
        if in_bounds(next_x, bits + 1, config.x_range)
            && in_bounds(next_z, bits + 1, config.z_range)
            && constraints
                .iter()
                .all(|c| c.accepts(c.seed(next_x, next_z), bits + 1))
        {
            lift(
                next_x,
                next_z,
                bits + 1,
                config,
                constraints,
                cancelled,
                found,
            );
        }
    }
}

/// Runs a CPU lifting search. Progress counts completed low-16-bit X/Z roots;
/// offset observations are exact, while errorTolerance applies to textures.
pub fn scan(
    threads: usize,
    config: &ScanConfig,
    sink: impl FnMut(&[Match]) + Send,
    progress: impl FnMut(u64, usize) + Send,
    cancelled: impl Fn() -> bool + Sync,
) -> Result<(), String> {
    if threads == 0 || config.offsets.is_empty() {
        return Err("offset search requires observations and a positive thread count".to_owned());
    }
    let offsets = prepare(config)?;
    let filters = prepare_filters(
        &config.filter,
        config.algorithm,
        &config.directions,
        config.error_tolerance,
    )?;
    let sink = Mutex::new(sink);
    let progress = Mutex::new(progress);
    let next = AtomicUsize::new(0);
    let completed = AtomicUsize::new(0);
    let x_count = root_count(config.x_range);
    let z_count = root_count(config.z_range);
    let total = work_items(config);
    thread::scope(|scope| {
        for _ in 0..threads.min(total) {
            scope.spawn(|| {
                while !cancelled() {
                    let item = next.fetch_add(1, Ordering::Relaxed);
                    if item >= total {
                        break;
                    }
                    let direction = item / x_count;
                    let texture = &filters.directions[direction];
                    let x = (config.x_range.start as u32).wrapping_add((item % x_count) as u32)
                        & (BASE - 1);
                    for ordinal in 0..z_count {
                        if cancelled() {
                            break;
                        }
                        let z =
                            (config.z_range.start as u32).wrapping_add(ordinal as u32) & (BASE - 1);
                        lift(
                            x,
                            z,
                            16,
                            config,
                            &offsets[direction],
                            &cancelled,
                            &mut |x, z| {
                                // Without textures, Y is unknown: emit each X/Z only once.
                                let y_range = if config.filter.is_empty() {
                                    0..1
                                } else {
                                    config.y_range.start..config.y_range.end
                                };
                                for y in y_range {
                                    if cancelled() {
                                        break;
                                    }
                                    let mismatches = texture_mismatches(config, texture, x, y, z);
                                    if mismatches <= config.error_tolerance {
                                        sink.lock().unwrap()(&[Match {
                                            x,
                                            y,
                                            z,
                                            mismatches,
                                            direction: config.directions[direction],
                                        }]);
                                    }
                                }
                            },
                        );
                    }
                    if cancelled() {
                        break;
                    }
                    let mut progress = progress.lock().unwrap();
                    let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                    progress(done as u64 * z_count as u64, done);
                }
            });
        }
    });
    Ok(())
}

fn texture_mismatches(
    config: &ScanConfig,
    filter: &PreparedDirection,
    x: i32,
    y: i32,
    z: i32,
) -> i32 {
    fn sample<A: TextureSampler>(
        config: &ScanConfig,
        filter: &PreparedDirection,
        x: i32,
        y: i32,
        z: i32,
    ) -> i32 {
        count_mismatches::<A>(
            x,
            y,
            z,
            &filter.constraints,
            filter.forced_errors,
            config.error_tolerance,
        )
    }
    match config.algorithm {
        TextureAlgorithm::Vanilla1 => sample::<Vanilla1>(config, filter, x, y, z),
        TextureAlgorithm::Vanilla2 => sample::<Vanilla2>(config, filter, x, y, z),
        TextureAlgorithm::Vanilla3 => sample::<Vanilla3>(config, filter, x, y, z),
        TextureAlgorithm::Sodium1 => sample::<Sodium1>(config, filter, x, y, z),
        TextureAlgorithm::Sodium2 => sample::<Sodium2>(config, filter, x, y, z),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::CpuScanner;
    use crate::scan::make_plan;
    use crate::texture::get_texture;
    use crate::types::{Int3, OffsetInfo, RotationInfo};

    // Independent Java-style 64-bit coordinate hash; offset nibbles are the
    // low 12 bits after the signed shift. The actual block height is ignored.
    fn java_nibbles(x: i32, z: i32) -> [u8; 3] {
        let seed = i64::from(x.wrapping_mul(3_129_871)) ^ (i64::from(z) * 116_129_781);
        let seed = seed
            .wrapping_mul(seed)
            .wrapping_mul(42_317_861)
            .wrapping_add(seed.wrapping_mul(11))
            >> 16;
        std::array::from_fn(|axis| ((seed >> (axis * 4)) & 15) as u8)
    }

    fn collect(config: &ScanConfig, threads: usize) -> Vec<Match> {
        let mut found = Vec::new();
        // Exercise the public CPU entry point as well as the lifting search.
        let plan = make_plan(config, config.cpu_tile_size).unwrap();
        CpuScanner::new(threads)
            .unwrap()
            .scan(
                config,
                &plan,
                |batch| found.extend_from_slice(batch),
                |_, _| {},
                || false,
            )
            .unwrap();
        found.sort_unstable_by_key(|m| (m.x, m.y, m.z, m.direction));
        found
    }

    #[test]
    fn supplied_full_xyz_examples_lift_to_signed_coordinates() {
        for (origin, rows) in [
            (
                (12_541_022, 10_457_150),
                [
                    (2, -1, [11, 11, 5]),
                    (1, 0, [3, 2, 15]),
                    (3, -2, [10, 11, 12]),
                    (3, -1, [7, 8, 0]),
                    (-4, 1, [6, 15, 11]),
                ],
            ),
            (
                (-1_392_121, -10_529_167),
                [
                    (2, -1, [13, 6, 2]),
                    (1, 0, [9, 8, 13]),
                    (3, -2, [6, 2, 6]),
                    (3, -1, [6, 5, 12]),
                    (-4, 1, [15, 12, 13]),
                ],
            ),
        ] {
            let config = ScanConfig {
                x_range: IntRange {
                    start: origin.0 - 3,
                    end: origin.0 + 4,
                },
                y_range: IntRange { start: 0, end: 1 },
                z_range: IntRange {
                    start: origin.1 - 3,
                    end: origin.1 + 4,
                },
                offsets: rows
                    .into_iter()
                    .map(|(x, z, n)| OffsetInfo {
                        position: Int3 { x, y: 7, z },
                        nibbles: n.map(Some),
                        dripstone: false,
                    })
                    .collect(),
                ..ScanConfig::default()
            };
            assert_eq!(
                collect(&config, 3),
                vec![Match {
                    x: origin.0,
                    y: 0,
                    z: origin.1,
                    mismatches: 0,
                    direction: 0,
                }]
            );
        }
    }

    #[test]
    fn dripstone_masks_match_visible_java_float_clipping() {
        let displacement =
            |n: u8| ((f64::from(f32::from(n) / 15.0) - 0.5) * 0.5).clamp(-0.125, 0.125);
        for selected in 0..16 {
            let expected = (0..16)
                .filter(|&n| displacement(n) == displacement(selected))
                .fold(0, |mask, n| mask | (1 << n));
            assert_eq!(nibble_mask(Some(selected), true), expected);
        }
        // Mirroring must preserve the full clipped equivalence class.
        assert_eq!(
            nibble_mask(Some(3), true).reverse_bits(),
            nibble_mask(Some(12), true)
        );
    }

    #[test]
    fn partial_clipped_lifting_matches_brute_force_with_textures_and_rotations() {
        let origin = (-65_540, 65_534);
        for algorithm in [
            TextureAlgorithm::Vanilla1,
            TextureAlgorithm::Vanilla2,
            TextureAlgorithm::Vanilla3,
            TextureAlgorithm::Sodium1,
            TextureAlgorithm::Sodium2,
        ] {
            let config = ScanConfig {
                algorithm,
                directions: vec![0, 90, 180, 270],
                x_range: IntRange {
                    start: origin.0 - 14,
                    end: origin.0 + 15,
                },
                y_range: IntRange { start: -2, end: 2 },
                z_range: IntRange {
                    start: origin.1 - 14,
                    end: origin.1 + 15,
                },
                error_tolerance: 1,
                filter: vec![
                    RotationInfo::new(
                        0,
                        0,
                        0,
                        get_texture(algorithm, origin.0, 0, origin.1, 4),
                        false,
                    ),
                    RotationInfo::new(1, 0, 0, 0, true),
                ],
                offsets: vec![OffsetInfo {
                    // Offset positions deliberately exceed the texture int8 limits.
                    position: Int3 {
                        x: i32::MAX,
                        y: 30,
                        z: -300,
                    },
                    nibbles: [Some(3), None, Some(12)],
                    dripstone: true,
                }],
                ..ScanConfig::default()
            };
            let offsets = prepare(&config).unwrap();
            let textures =
                prepare_filters(&config.filter, algorithm, &config.directions, 1).unwrap();
            let mut expected = Vec::new();
            for x in config.x_range.start..config.x_range.end {
                for z in config.z_range.start..config.z_range.end {
                    for (index, constraints) in offsets.iter().enumerate() {
                        let c = &constraints[0];
                        let n =
                            java_nibbles(x.wrapping_add(c.dx as i32), z.wrapping_add(c.dz as i32));
                        if !(0..3).all(|axis| c.possible[axis][4] & (1 << n[axis]) != 0) {
                            continue;
                        }
                        for y in config.y_range.start..config.y_range.end {
                            let mismatches =
                                texture_mismatches(&config, &textures.directions[index], x, y, z);
                            if mismatches <= config.error_tolerance {
                                expected.push(Match {
                                    x,
                                    y,
                                    z,
                                    mismatches,
                                    direction: config.directions[index],
                                });
                            }
                        }
                    }
                }
            }
            expected.sort_unstable_by_key(|m| (m.x, m.y, m.z, m.direction));
            assert!(!expected.is_empty());
            assert_eq!(collect(&config, 4), expected, "{algorithm}");
        }
    }

    #[test]
    fn bounds_include_world_edges_and_exclude_range_ends() {
        for (x, z) in [(-30_000_000, 30_000_000), (-1, 0), (65_535, -65_536)] {
            let mut config = ScanConfig {
                x_range: IntRange {
                    start: x,
                    end: x + 1,
                },
                y_range: IntRange { start: 2, end: 60 },
                z_range: IntRange {
                    start: z,
                    end: z + 1,
                },
                offsets: vec![OffsetInfo {
                    position: Int3::default(),
                    nibbles: java_nibbles(x, z).map(Some),
                    dripstone: false,
                }],
                ..ScanConfig::default()
            };
            let found = collect(&config, 2);
            assert_eq!(found.len(), 1);
            assert_eq!(found[0].y, 0);
            assert!(found.iter().all(|m| m.x == x && m.z == z));
            config.x_range = IntRange {
                start: x - 1,
                end: x,
            };
            if x > -30_000_000 {
                assert!(collect(&config, 2).iter().all(|m| m.x < x));
            }
        }
    }

    #[test]
    fn rotates_displacements_and_merges_heights_without_losing_clipping() {
        let mut config = ScanConfig {
            directions: vec![0, 90, 180, 270],
            x_range: IntRange { start: 0, end: 1 },
            z_range: IntRange { start: 0, end: 1 },
            offsets: vec![OffsetInfo {
                position: Int3 { x: 2, y: 1, z: -3 },
                nibbles: [Some(3), None, Some(7)],
                dripstone: true,
            }],
            ..ScanConfig::default()
        };
        let constraints = prepare(&config).unwrap();
        for (index, (dx, dz, x, z)) in [
            (2, -3, 0xf, 1 << 7),
            (3, 2, 1 << 8, 0xf),
            (-2, 3, 0xf000, 1 << 8),
            (-3, -2, 1 << 7, 0xf000),
        ]
        .into_iter()
        .enumerate()
        {
            let c = &constraints[index][0];
            assert_eq!(
                (c.dx as i32, c.dz as i32, c.possible[0][4], c.possible[2][4]),
                (dx, dz, x, z)
            );
        }
        let mut second = config.offsets[0];
        second.position.y = 99;
        second.nibbles[0] = Some(0);
        config.offsets.push(second);
        assert_eq!(prepare(&config).unwrap()[0].len(), 1);
        config.offsets[1].nibbles[0] = Some(4);
        assert!(prepare(&config).unwrap_err().contains("contradictory"));
    }

    #[test]
    fn cancellation_stops_before_reporting_matches_or_progress() {
        let config =
            crate::config::load(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/offsets.conf"))
                .unwrap();
        scan(
            2,
            &config,
            |_| panic!("match after cancellation"),
            |_, _| panic!("progress after cancellation"),
            || true,
        )
        .unwrap();
    }
}
