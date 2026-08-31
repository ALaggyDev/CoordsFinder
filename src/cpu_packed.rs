//! Experimental Vanilla-3 CPU path for the common `Y=-60..0` search.
//!
//! This intentionally targets the narrow benchmark used by the linked Fas
//! implementation. One bit represents one possible origin Y, and only a sparse
//! set of source Z rows is materialized. Any survivors are checked again with
//! the ordinary exact texture sampler.

use std::collections::BTreeMap;
use std::sync::Mutex;

use crate::config::ScanConfig;
use crate::scan::WorkItem;
use crate::texture::{TextureSampler, Vanilla3};
use crate::types::{CompiledRotation, Match, TextureAlgorithm};

const JAVA_MULTIPLIER: u64 = 0x5deece66d;
const JAVA_MASK: u64 = (1 << 48) - 1;
const MIX_MULTIPLIER: u64 = 42_317_861;
const SECOND_DIFFERENCE: u64 = MIX_MULTIPLIER * 2;
const SPARSE_RESIDUES: [i32; 6] = [2, 30, 8, 24, 13, 19];
const Z_BAND: i32 = 64;

#[derive(Clone, Copy, Debug)]
pub(crate) struct MaskObservation {
    dx: i32,
    dy: u32,
    dz: i32,
    accepted_rotations: u8,
}

/// Returns a packed representation when the config fits the experimental path.
pub(crate) fn prepare(
    config: &ScanConfig,
    constraints: &[CompiledRotation],
    forced_errors: i32,
) -> Option<Vec<MaskObservation>> {
    if config.algorithm != TextureAlgorithm::Vanilla3
        || std::env::var_os("COORDSFINDER_DISABLE_PACKED_CPU").is_some()
        || config.error_tolerance != 0
        || forced_errors != 0
        || config.y_range.start < -60
        || config.y_range.end > 1
    {
        return None;
    }

    constraints
        .iter()
        .map(|constraint| {
            let dy = u32::try_from(constraint.y).ok()?;
            if dy > 1 {
                return None;
            }

            // Vanilla-3's visible four-way rotation is the upper two bits of
            // its 16-way model index. Side observations are unions of groups.
            // Netherrack produces partial groups and therefore stays on the
            // general CPU path.
            let mut accepted_rotations = 0_u8;
            let mut represented_indices = 0_u16;
            for rotation in 0..4 {
                let group = 0xf_u16 << (rotation * 4);
                if constraint.accepted_indices & group == group {
                    accepted_rotations |= 1 << rotation;
                    represented_indices |= group;
                }
            }
            if accepted_rotations == 0 || represented_indices != constraint.accepted_indices {
                return None;
            }
            Some(MaskObservation {
                dx: i32::from(constraint.x),
                dy,
                dz: i32::from(constraint.z),
                accepted_rotations,
            })
        })
        .collect()
}

pub(crate) fn scan_item(
    item: &WorkItem,
    observations: &[MaskObservation],
    exact: &[CompiledRotation],
    cancelled: &impl Fn() -> bool,
    matches: &mut Vec<Match>,
    sink: &Mutex<impl FnMut(&[Match])>,
) {
    let initial_live = origin_y_mask(item.start.y, item.end.y);
    let min_dx = observations.iter().map(|item| item.dx).min().unwrap_or(0);
    let max_dx = observations.iter().map(|item| item.dx).max().unwrap_or(0);
    let source_x_start = item.start.x.wrapping_add(min_dx);
    let source_x_end = item.end.x.wrapping_add(max_dx);
    let source_width = (i64::from(source_x_end) - i64::from(source_x_start)) as usize;
    let candidate_width = (i64::from(item.end.x) - i64::from(item.start.x)) as usize;
    let dense_stride = select_dense_stride(observations);

    for band_start in (item.start.z..item.end.z).step_by(Z_BAND as usize) {
        if cancelled() {
            return;
        }
        let band_end = item.end.z.min(band_start.saturating_add(Z_BAND));
        let band_rows = (band_end - band_start) as usize;
        let mut live = vec![initial_live; band_rows * candidate_width];
        let mut tasks: BTreeMap<i32, Vec<(usize, usize)>> = BTreeMap::new();

        // Group applications by source Z so every expensive 64-Y signature
        // row is generated once and reused by all observations that touch it.
        for (observation_index, observation) in observations.iter().enumerate() {
            for candidate_z in band_start..band_end {
                let source_z = candidate_z.wrapping_add(observation.dz);
                if covered_source_z(source_z, dense_stride) {
                    tasks
                        .entry(source_z)
                        .or_default()
                        .push(((candidate_z - band_start) as usize, observation_index));
                }
            }
        }

        let mut source_masks = vec![[0_u64; 4]; source_width];
        for (source_z, row_tasks) in tasks {
            generate_source_row(source_x_start, source_z, &mut source_masks);
            for (candidate_row, observation_index) in row_tasks {
                let observation = observations[observation_index];
                let source_offset = i64::from(item.start.x.wrapping_add(observation.dx))
                    - i64::from(source_x_start);
                let live_row = &mut live
                    [candidate_row * candidate_width..(candidate_row + 1) * candidate_width];
                apply_masks(
                    live_row,
                    &source_masks[source_offset as usize..],
                    observation,
                );
            }
        }

        for row in 0..band_rows {
            let z = band_start + row as i32;
            for local_x in 0..candidate_width {
                let x = item.start.x.wrapping_add(local_x as i32);
                let mut bits = live[row * candidate_width + local_x];
                while bits != 0 {
                    let y = bits.trailing_zeros() as i32 - 60;
                    bits &= bits - 1;
                    if exact_match(x, y, z, exact) {
                        matches.push(Match {
                            x,
                            y,
                            z,
                            mismatches: 0,
                            direction: item.direction,
                        });
                        // Matches are rare; preserve the existing immediate
                        // reporting behavior without burdening the hot path.
                        sink.lock().unwrap()(matches);
                        matches.clear();
                    }
                }
            }
        }
    }
}

#[inline]
fn origin_y_mask(start: i32, end: i32) -> u64 {
    let count = (end - start) as u32;
    let low = (start + 60) as u32;
    let ones = if count == 64 {
        u64::MAX
    } else {
        (1_u64 << count) - 1
    };
    ones << low
}

fn select_dense_stride(observations: &[MaskObservation]) -> i32 {
    let mut stride = observations.len() / 5;
    while stride >= 2 {
        let mut counts = vec![0_usize; stride];
        for observation in observations {
            counts[observation.dz.rem_euclid(stride as i32) as usize] += 1;
        }
        if counts.into_iter().all(|count| count >= 5) {
            return stride as i32;
        }
        stride -= 1;
    }
    0
}

#[inline(always)]
fn covered_source_z(z: i32, dense_stride: i32) -> bool {
    if dense_stride != 0 {
        let residue = if dense_stride % 2 == 0 {
            dense_stride / 2
        } else {
            0
        };
        z.rem_euclid(dense_stride) == residue
    } else {
        SPARSE_RESIDUES.contains(&z.rem_euclid(32))
    }
}

#[inline(always)]
fn p_bits(x: i32) -> u64 {
    i64::from(x.wrapping_mul(3_129_871)) as u64
}

#[inline(always)]
fn q_bits(z: i32) -> u64 {
    (z as i64).wrapping_mul(116_129_781) as u64
}

#[inline(always)]
fn mixed_value(seed: u64) -> u64 {
    seed.wrapping_mul(seed.wrapping_mul(MIX_MULTIPLIER).wrapping_add(11))
}

#[inline(always)]
fn rotation_from_mixed(mixed: u64) -> u8 {
    let state = ((mixed >> 16) ^ JAVA_MULTIPLIER) & JAVA_MASK;
    let state = state.wrapping_mul(JAVA_MULTIPLIER).wrapping_add(11) & JAVA_MASK;
    (state >> 46) as u8
}

/// Generates four 62-bit masks for each X. Work is interleaved in groups of
/// eight columns to expose enough independent integer chains to the CPU.
fn generate_source_row(x_start: i32, z: i32, output: &mut [[u64; 4]]) {
    let q = q_bits(z);
    for (chunk_index, chunk) in output.chunks_mut(8).enumerate() {
        let lanes = chunk.len();
        let mut bases = [0_u64; 8];
        let mut values = [0_u64; 8];
        let mut differences = [0_u64; 8];
        let mut base_lows = [0_u32; 8];
        let mut plane_zero = [0_u64; 8];
        let mut plane_one = [0_u64; 8];

        for lane in 0..lanes {
            let x = x_start.wrapping_add((chunk_index * 8 + lane) as i32);
            let base = p_bits(x) ^ q;
            let first_seed = (base ^ u64::MAX) & !63;
            bases[lane] = base;
            values[lane] = mixed_value(first_seed);
            differences[lane] = MIX_MULTIPLIER
                .wrapping_mul(first_seed.wrapping_mul(2).wrapping_add(1))
                .wrapping_add(11);
            base_lows[lane] = (base & 63) as u32;
        }

        for low in 0..64_u32 {
            for lane in 0..lanes {
                let rotation = rotation_from_mixed(values[lane]);
                let bit = 1_u64 << (low ^ base_lows[lane]);
                plane_zero[lane] |= u64::from(rotation & 1).wrapping_neg() & bit;
                plane_one[lane] |= u64::from((rotation >> 1) & 1).wrapping_neg() & bit;
                values[lane] = values[lane].wrapping_add(differences[lane]);
                differences[lane] = differences[lane].wrapping_add(SECOND_DIFFERENCE);
            }
        }

        for lane in 0..lanes {
            let y0 = rotation_from_mixed(mixed_value(bases[lane]));
            let y1 = rotation_from_mixed(mixed_value(bases[lane] ^ 1));
            let zero = plane_zero[lane];
            let one = plane_one[lane];
            let mut masks = [
                (!(zero | one)) >> 4,
                (zero & !one) >> 4,
                (!zero & one) >> 4,
                (zero & one) >> 4,
            ];
            masks[y0 as usize] |= 1 << 60;
            masks[y1 as usize] |= 1 << 61;
            chunk[lane] = masks;
        }
    }
}

#[inline]
fn apply_masks(live: &mut [u64], source: &[[u64; 4]], observation: MaskObservation) {
    match observation.accepted_rotations {
        1 => apply_one(live, source, 0, observation.dy),
        2 => apply_one(live, source, 1, observation.dy),
        4 => apply_one(live, source, 2, observation.dy),
        8 => apply_one(live, source, 3, observation.dy),
        accepted => {
            for (live, masks) in live.iter_mut().zip(source) {
                let mut mask = 0_u64;
                for (rotation, &rotation_mask) in masks.iter().enumerate() {
                    if accepted & (1 << rotation) != 0 {
                        mask |= rotation_mask;
                    }
                }
                *live &= mask >> observation.dy;
            }
        }
    }
}

#[inline(always)]
fn apply_one(live: &mut [u64], source: &[[u64; 4]], rotation: usize, shift: u32) {
    for (live, masks) in live.iter_mut().zip(source) {
        *live &= masks[rotation] >> shift;
    }
}

#[inline(always)]
fn exact_match(x: i32, y: i32, z: i32, observations: &[CompiledRotation]) -> bool {
    observations.iter().all(|observation| {
        let index = Vanilla3::sample(
            x.wrapping_add(i32::from(observation.x)),
            y.wrapping_add(i32::from(observation.y)),
            z.wrapping_add(i32::from(observation.z)),
            16,
        );
        observation.accepted_indices & (1 << index) != 0
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_masks_match_exact_rotations() {
        for &(x, z) in &[(0, 0), (17, -31), (-538, -575), (1_000_000, -1_000_000)] {
            let mut masks = [[0_u64; 4]; 1];
            generate_source_row(x, z, &mut masks);
            for y in -60..=1 {
                let expected = Vanilla3::sample(x, y, z, 16) >> 2;
                assert_ne!(masks[0][expected as usize] & (1 << (y + 60)), 0);
                for (rotation, &mask) in masks[0].iter().enumerate() {
                    assert_eq!(
                        mask & (1 << (y + 60)) != 0,
                        rotation == expected as usize,
                        "x={x}, y={y}, z={z}, rotation={rotation}"
                    );
                }
            }
        }
    }
}
