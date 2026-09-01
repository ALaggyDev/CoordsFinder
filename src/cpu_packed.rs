//! Experimental Vanilla-3 CPU path for the common `Y=-60..0` search.
//!
//! The ordinary scanner considers every `(x, y, z)` separately. This scanner
//! instead stores every possible origin Y for one `(x, z)` column in one `u64`:
//!
//! ```text
//! bit 0  = origin Y -60
//! bit 1  = origin Y -59
//! ...
//! bit 60 = origin Y   0
//! ```
//!
//! The algorithm has four stages:
//!
//! 1. Select a sparse set of source Z rows that still gives each candidate row
//!    several useful observations.
//! 2. Generate four Y masks (one per visible rotation) for each source `(x,z)`.
//! 3. Apply an observation to all 61 Y candidates with one bitwise AND.
//! 4. Run the ordinary exact sampler on the few set bits that survive.
//!
//! The mask pass only omits constraints; it never approximates an applied
//! constraint. It can therefore leave false positives for stage 4, but cannot
//! discard a genuine match.

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
const PACKED_Y_MIN: i32 = -60;
const SOURCE_Y_MAX: i32 = 1;
const MAX_ORIGIN_Y_EXCLUSIVE: i32 = SOURCE_Y_MAX;
const MAX_OBSERVATION_DY: u32 = 1;
const Y_ZERO_BIT: u32 = 60;
const Y_ONE_BIT: u32 = 61;
const ROTATION_COUNT: usize = 4;
const SIGNATURE_BATCH_SIZE: usize = 8;
const MIN_MASKS_PER_CANDIDATE_ROW: usize = 5;
const SPARSE_RESIDUES: [i32; 6] = [2, 30, 8, 24, 13, 19];
const Z_BAND: i32 = 64;

/// `masks[rotation]` has one bit set for every source Y producing that rotation.
type RotationMasks = [u64; ROTATION_COUNT];

#[derive(Clone, Copy, Debug)]
pub(crate) struct MaskObservation {
    /// Offset from a candidate origin to the observed source block.
    dx: i32,
    dy: u32,
    dz: i32,
    /// Bit `r` is set when visible rotation `r` satisfies the observation.
    accepted_rotations: u8,
}

/// One use of a generated source-Z row.
#[derive(Clone, Copy, Debug)]
struct MaskTask {
    /// Row within the current 64-Z candidate band whose live masks are updated.
    candidate_row: usize,
    observation_index: usize,
}

/// Values that remain constant while a worker scans one X/Z work item.
struct TileLayout {
    initial_live: u64,
    candidate_width: usize,
    source_x_start: i32,
    source_width: usize,
    dense_z_stride: i32,
}

impl TileLayout {
    #[inline]
    fn new(item: &WorkItem, observations: &[MaskObservation]) -> Self {
        let minimum_dx = observations.iter().map(|item| item.dx).min().unwrap_or(0);
        let maximum_dx = observations.iter().map(|item| item.dx).max().unwrap_or(0);
        let source_x_start = item.start.x.wrapping_add(minimum_dx);
        let source_x_end = item.end.x.wrapping_add(maximum_dx);

        Self {
            initial_live: origin_y_mask(item.start.y, item.end.y),
            candidate_width: (i64::from(item.end.x) - i64::from(item.start.x)) as usize,
            source_x_start,
            // The source row includes the X halo required by dx offsets.
            source_width: (i64::from(source_x_end) - i64::from(source_x_start)) as usize,
            dense_z_stride: select_dense_stride(observations),
        }
    }
}

/// Returns a packed representation when the config fits the experimental path.
///
/// Anything outside this deliberately narrow contract returns `None`, causing
/// `cpu.rs` to use the original general implementation instead.
pub(crate) fn prepare(
    config: &ScanConfig,
    constraints: &[CompiledRotation],
    forced_errors: i32,
) -> Option<Vec<MaskObservation>> {
    if config.algorithm != TextureAlgorithm::Vanilla3
        || std::env::var_os("COORDSFINDER_DISABLE_PACKED_CPU").is_some()
        || config.error_tolerance != 0
        || forced_errors != 0
        || config.y_range.start < PACKED_Y_MIN
        || config.y_range.end > MAX_ORIGIN_Y_EXCLUSIVE
    {
        return None;
    }

    constraints
        .iter()
        .map(|constraint| {
            let dy = u32::try_from(constraint.y).ok()?;
            if dy > MAX_OBSERVATION_DY {
                return None;
            }

            Some(MaskObservation {
                dx: i32::from(constraint.x),
                dy,
                dz: i32::from(constraint.z),
                accepted_rotations: visible_rotation_mask(constraint)?,
            })
        })
        .collect()
}

/// Converts the 16-way compiled model mask to the four visible Vanilla-3
/// rotations used by the packed signatures.
///
/// Vanilla-3 exposes the top two bits of its 16-way model index. Consequently,
/// visible rotation `r` represents indices `4*r .. 4*r+3`. Ordinary four-way
/// and side observations contain whole groups. Netherrack can select only part
/// of a group, so it is deliberately rejected and handled by the general path.
fn visible_rotation_mask(constraint: &CompiledRotation) -> Option<u8> {
    let mut accepted_rotations = 0_u8;
    let mut represented_indices = 0_u16;
    for rotation in 0..ROTATION_COUNT {
        let model_group = 0xf_u16 << (rotation * 4);
        if constraint.accepted_indices & model_group == model_group {
            accepted_rotations |= 1 << rotation;
            represented_indices |= model_group;
        }
    }
    (accepted_rotations != 0 && represented_indices == constraint.accepted_indices)
        .then_some(accepted_rotations)
}

pub(crate) fn scan_item(
    item: &WorkItem,
    observations: &[MaskObservation],
    exact: &[CompiledRotation],
    cancelled: &impl Fn() -> bool,
    matches: &mut Vec<Match>,
    sink: &Mutex<impl FnMut(&[Match])>,
) {
    let layout = TileLayout::new(item, observations);

    for band_start in (item.start.z..item.end.z).step_by(Z_BAND as usize) {
        if cancelled() {
            return;
        }
        let band_end = item.end.z.min(band_start.saturating_add(Z_BAND));
        scan_band(
            item,
            band_start,
            band_end,
            &layout,
            observations,
            exact,
            matches,
            sink,
        );
    }
}

/// Runs the three per-band stages: plan sparse source rows, apply their masks,
/// and exactly verify the surviving bits.
#[allow(clippy::too_many_arguments)]
#[inline]
fn scan_band(
    item: &WorkItem,
    band_start: i32,
    band_end: i32,
    layout: &TileLayout,
    observations: &[MaskObservation],
    exact: &[CompiledRotation],
    matches: &mut Vec<Match>,
    sink: &Mutex<impl FnMut(&[Match])>,
) {
    let band_rows = (band_end - band_start) as usize;

    // One live u64 is stored for each candidate (x,z) column. All configured
    // origin Y values begin alive; observations progressively clear bits.
    let mut live = vec![layout.initial_live; band_rows * layout.candidate_width];
    let source_row_tasks =
        plan_source_rows(band_start, band_end, observations, layout.dense_z_stride);

    // Reuse this allocation. Its contents are replaced for each selected Z.
    let mut source_masks = vec![[0_u64; ROTATION_COUNT]; layout.source_width];
    for (source_z, tasks) in source_row_tasks {
        generate_source_row(layout.source_x_start, source_z, &mut source_masks);
        apply_source_row(item, layout, observations, &source_masks, &mut live, &tasks);
    }

    verify_survivors(
        item, band_start, band_rows, layout, exact, &live, matches, sink,
    );
}

/// Chooses which observation applications can use each selected source Z.
/// Grouping by source Z means an expensive signature row is generated once,
/// even when several observations point into it.
fn plan_source_rows(
    band_start: i32,
    band_end: i32,
    observations: &[MaskObservation],
    dense_z_stride: i32,
) -> BTreeMap<i32, Vec<MaskTask>> {
    let mut rows = BTreeMap::<i32, Vec<MaskTask>>::new();
    for (observation_index, observation) in observations.iter().enumerate() {
        for candidate_z in band_start..band_end {
            let source_z = candidate_z.wrapping_add(observation.dz);
            if covered_source_z(source_z, dense_z_stride) {
                rows.entry(source_z).or_default().push(MaskTask {
                    candidate_row: (candidate_z - band_start) as usize,
                    observation_index,
                });
            }
        }
    }
    rows
}

/// Applies all observations that refer to one already-generated source-Z row.
#[inline(always)]
fn apply_source_row(
    item: &WorkItem,
    layout: &TileLayout,
    observations: &[MaskObservation],
    source_masks: &[RotationMasks],
    live: &mut [u64],
    tasks: &[MaskTask],
) {
    for task in tasks {
        let observation = observations[task.observation_index];

        // source_masks[0] describes layout.source_x_start. An observation with
        // dx shifts the first candidate X to the corresponding source X.
        let source_offset =
            i64::from(item.start.x.wrapping_add(observation.dx)) - i64::from(layout.source_x_start);
        let row_start = task.candidate_row * layout.candidate_width;
        let live_row = &mut live[row_start..row_start + layout.candidate_width];
        apply_observation_masks(
            live_row,
            &source_masks[source_offset as usize..],
            observation,
        );
    }
}

/// Enumerates set Y bits and performs the full exact test before reporting.
#[allow(clippy::too_many_arguments)]
#[inline]
fn verify_survivors(
    item: &WorkItem,
    band_start: i32,
    band_rows: usize,
    layout: &TileLayout,
    exact: &[CompiledRotation],
    live: &[u64],
    matches: &mut Vec<Match>,
    sink: &Mutex<impl FnMut(&[Match])>,
) {
    for candidate_row in 0..band_rows {
        let z = band_start + candidate_row as i32;
        for local_x in 0..layout.candidate_width {
            let x = item.start.x.wrapping_add(local_x as i32);
            let mut possible_y = live[candidate_row * layout.candidate_width + local_x];
            while possible_y != 0 {
                let y = possible_y.trailing_zeros() as i32 + PACKED_Y_MIN;
                possible_y &= possible_y - 1; // Clear the lowest set bit.

                if exact_match(x, y, z, exact) {
                    matches.push(Match {
                        x,
                        y,
                        z,
                        mismatches: 0,
                        direction: item.direction,
                    });
                    // Matches are rare; preserve immediate reporting without
                    // putting a lock in the rejection-heavy mask loops.
                    sink.lock().unwrap()(matches);
                    matches.clear();
                }
            }
        }
    }
}

/// Builds the initial set of possible origin Ys for one candidate column.
#[inline]
fn origin_y_mask(start: i32, end: i32) -> u64 {
    let count = (end - start) as u32;
    let lowest_bit = (start - PACKED_Y_MIN) as u32;
    let ones = if count == 64 {
        u64::MAX
    } else {
        (1_u64 << count) - 1
    };
    ones << lowest_bit
}

/// Finds the largest modulus for which every `dz` residue class contributes at
/// least five observations. For the supplied benchmark this returns 4, so only
/// one quarter of source Z rows need signatures while every candidate Z still
/// receives at least five mask intersections.
fn select_dense_stride(observations: &[MaskObservation]) -> i32 {
    let mut stride = observations.len() / MIN_MASKS_PER_CANDIDATE_ROW;
    while stride >= 2 {
        let mut counts = vec![0_usize; stride];
        for observation in observations {
            counts[observation.dz.rem_euclid(stride as i32) as usize] += 1;
        }
        if counts
            .into_iter()
            .all(|count| count >= MIN_MASKS_PER_CANDIDATE_ROW)
        {
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
        // Six rows per 32 gives the general 5.33x source-row reduction. Rows
        // come in +/- pairs so reflected coordinate patterns behave similarly.
        SPARSE_RESIDUES.contains(&z.rem_euclid(32))
    }
}

#[inline(always)]
fn p_bits(x: i32) -> u64 {
    // Minecraft multiplies X as a wrapping Java int, then sign-extends it.
    i64::from(x.wrapping_mul(3_129_871)) as u64
}

#[inline(always)]
fn q_bits(z: i32) -> u64 {
    // Z is promoted to a Java long before multiplication.
    (z as i64).wrapping_mul(116_129_781) as u64
}

#[inline(always)]
fn mixed_value(seed: u64) -> u64 {
    // Minecraft's quadratic coordinate mixer: seed * (seed * M + 11).
    seed.wrapping_mul(seed.wrapping_mul(MIX_MULTIPLIER).wrapping_add(11))
}

#[inline(always)]
fn rotation_from_mixed(mixed: u64) -> u8 {
    // Vanilla-3 seeds java.util.Random with mixed >> 16 and asks for one of
    // four visible rotations. For a power-of-two bound this is the top 2 bits
    // of the first 48-bit LCG state.
    let state = ((mixed >> 16) ^ JAVA_MULTIPLIER) & JAVA_MASK;
    let state = state.wrapping_mul(JAVA_MULTIPLIER).wrapping_add(11) & JAVA_MASK;
    (state >> 46) as u8
}

/// Generates four 62-bit masks for every X at one selected source Z.
///
/// Bits 0..59 cover source Y -60..-1; bits 60 and 61 cover Y 0 and 1. The
/// latter is required because an origin at Y=0 can have an observation at dy=1.
/// Work is interleaved in groups of eight columns to expose independent integer
/// chains to the native optimizer.
fn generate_source_row(x_start: i32, z: i32, output: &mut [RotationMasks]) {
    let q = q_bits(z);
    for (chunk_index, chunk) in output.chunks_mut(SIGNATURE_BATCH_SIZE).enumerate() {
        let lanes = chunk.len();
        let mut bases = [0_u64; SIGNATURE_BATCH_SIZE];
        let mut values = [0_u64; SIGNATURE_BATCH_SIZE];
        let mut differences = [0_u64; SIGNATURE_BATCH_SIZE];
        let mut base_lows = [0_u32; SIGNATURE_BATCH_SIZE];
        let mut rotation_low_bits = [0_u64; SIGNATURE_BATCH_SIZE];
        let mut rotation_high_bits = [0_u64; SIGNATURE_BATCH_SIZE];

        for lane in 0..lanes {
            let x = x_start.wrapping_add((chunk_index * SIGNATURE_BATCH_SIZE + lane) as i32);
            let base = p_bits(x) ^ q;

            // For Y=-64..-1, `base ^ y` has a common high 58-bit prefix. The
            // 64 low-bit combinations only appear in a different order, given
            // by `low ^ (base & 63)` below.
            let first_seed = (base ^ u64::MAX) & !63;
            bases[lane] = base;
            values[lane] = mixed_value(first_seed);

            // If f(s)=s*(s*M+11), then:
            //   f(s+1)-f(s) = M*(2*s+1)+11
            // and every following difference increases by the constant 2*M.
            // This generates 64 quadratic mixer results mostly with additions.
            differences[lane] = MIX_MULTIPLIER
                .wrapping_mul(first_seed.wrapping_mul(2).wrapping_add(1))
                .wrapping_add(11);
            base_lows[lane] = (base & 63) as u32;
        }

        for low in 0..64_u32 {
            for lane in 0..lanes {
                let rotation = rotation_from_mixed(values[lane]);
                let bit = 1_u64 << (low ^ base_lows[lane]);

                // Store the two bits of every rotation as two bitplanes.
                // wrapping_neg maps 0 -> all zeroes and 1 -> all ones, avoiding
                // unpredictable branches while conditionally inserting `bit`.
                rotation_low_bits[lane] |= u64::from(rotation & 1).wrapping_neg() & bit;
                rotation_high_bits[lane] |= u64::from((rotation >> 1) & 1).wrapping_neg() & bit;
                values[lane] = values[lane].wrapping_add(differences[lane]);
                differences[lane] = differences[lane].wrapping_add(SECOND_DIFFERENCE);
            }
        }

        for lane in 0..lanes {
            let y0 = rotation_from_mixed(mixed_value(bases[lane]));
            let y1 = rotation_from_mixed(mixed_value(bases[lane] ^ 1));
            let low_bits = rotation_low_bits[lane];
            let high_bits = rotation_high_bits[lane];

            // Decode the two bitplanes into one mask per rotation. Shifting by
            // four drops Y=-64..-61, making mask bit 0 correspond to Y=-60.
            let mut masks = [
                (!(low_bits | high_bits)) >> 4,
                (low_bits & !high_bits) >> 4,
                (!low_bits & high_bits) >> 4,
                (low_bits & high_bits) >> 4,
            ];
            // Y=0 and Y=1 do not share the negative-Y prefix and are cheaper
            // to calculate directly than to complicate the recurrence.
            masks[y0 as usize] |= 1 << Y_ZERO_BIT;
            masks[y1 as usize] |= 1 << Y_ONE_BIT;
            chunk[lane] = masks;
        }
    }
}

#[inline]
fn apply_observation_masks(
    live: &mut [u64],
    source: &[RotationMasks],
    observation: MaskObservation,
) {
    // Four-way observations normally select one arm. Side observations select
    // two arms and use the generic union below. In either case this operation
    // checks every possible origin Y at once:
    //
    //     live_y_candidates &= Ys_where_the_observation_matches
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
fn apply_one(live: &mut [u64], source: &[RotationMasks], rotation: usize, shift: u32) {
    for (live, masks) in live.iter_mut().zip(source) {
        // A dy=1 source mask is shifted down so source Y=(origin Y+1) lines up
        // with the bit representing origin Y.
        *live &= masks[rotation] >> shift;
    }
}

/// Slow but authoritative final check over every compiled observation.
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
            let mut masks = [[0_u64; ROTATION_COUNT]; 1];
            generate_source_row(x, z, &mut masks);
            for y in PACKED_Y_MIN..=SOURCE_Y_MAX {
                let expected = Vanilla3::sample(x, y, z, 16) >> 2;
                let y_bit = 1 << (y - PACKED_Y_MIN);
                assert_ne!(masks[0][expected as usize] & y_bit, 0);
                for (rotation, &mask) in masks[0].iter().enumerate() {
                    assert_eq!(
                        mask & y_bit != 0,
                        rotation == expected as usize,
                        "x={x}, y={y}, z={z}, rotation={rotation}"
                    );
                }
            }
        }
    }
}
