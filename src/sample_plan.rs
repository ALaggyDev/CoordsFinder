//! Compact-template planning for exact texture searches.
//!
//! A plan groups a rectangular box of candidate alignments behind one world
//! sample. The sample word is looked up in a flat four-way trie; leaf outputs
//! identify the candidate coordinates that still need a full filter check.

use std::collections::{HashMap, HashSet};

use crate::scan::WorkItem;
use crate::types::{CompiledRotation, Int3, SearchMode, TextureAlgorithm};

pub const MAX_SAMPLE_SIZE: usize = 4;
pub const MISSING_TRIE_CHILD: u32 = u32::MAX;
// This model counts texture hashes, not GPU invocations or trie/control cost.
// Independent-anchor GPU dispatch loses on low-volume templates, so automatic
// selection needs a deliberately wide margin. Forced mode remains available
// for benchmarking marginal plans.
const AUTO_MIN_SPEEDUP: f64 = 5.0;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlacementBox {
    pub min: Int3,
    pub lengths: [u32; 3],
}

impl PlacementBox {
    pub fn volume(self) -> u32 {
        self.lengths.into_iter().product()
    }

    fn max(self) -> Int3 {
        Int3 {
            x: self.min.x + self.lengths[0] as i32 - 1,
            y: self.min.y + self.lengths[1] as i32 - 1,
            z: self.min.z + self.lengths[2] as i32 - 1,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FlatTrieNode {
    pub children: [u32; 4],
    pub output_start: u32,
    pub output_count: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TriePlacement {
    pub origin: Int3,
    /// Indices of the constraints already proved by the sample word.
    pub witness_indices: [u8; MAX_SAMPLE_SIZE],
}

#[derive(Clone, Debug)]
pub struct SamplePlan {
    pub offsets: Vec<Int3>,
    pub placement_box: PlacementBox,
    pub nodes: Vec<FlatTrieNode>,
    pub placements: Vec<TriePlacement>,
    pub expected_naive_checks: f64,
    pub expected_sample_checks: f64,
    pub estimated_speedup: f64,
}

impl SamplePlan {
    pub fn sample_size(&self) -> usize {
        self.offsets.len()
    }

    pub fn description(&self) -> String {
        let [x, y, z] = self.placement_box.lengths;
        let offsets = self
            .offsets
            .iter()
            .map(|offset| format!("({},{},{})", offset.x, offset.y, offset.z))
            .collect::<Vec<_>>()
            .join(",");
        let leaves = self
            .nodes
            .iter()
            .filter(|node| node.output_count != 0)
            .count();
        format!(
            "sample-trie q={}, offsets=[{}], placements={}x{}x{} (V={}), trie={} nodes/{} leaves/{} outputs, expected texture checks {:.3}->{:.3} ({:.2}x)",
            self.sample_size(),
            offsets,
            x,
            y,
            z,
            self.placement_box.volume(),
            self.nodes.len(),
            leaves,
            self.placements.len(),
            self.expected_naive_checks,
            self.expected_sample_checks,
            self.estimated_speedup
        )
    }
}

/// Lattice anchors needed to cover one candidate tile.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AnchorGrid {
    pub start: Int3,
    pub counts: [u32; 3],
    pub steps: [u32; 3],
}

#[derive(Clone, Copy)]
struct EligibleConstraint {
    index: u8,
    symbol: u8,
}

#[derive(Clone)]
struct PlacementCandidate {
    origin: Int3,
    witness_indices: [u8; MAX_SAMPLE_SIZE],
    symbols: [u8; MAX_SAMPLE_SIZE],
}

struct MutableTrieNode {
    children: [u32; 4],
    outputs: Vec<usize>,
}

impl MutableTrieNode {
    fn new() -> Self {
        Self {
            children: [MISSING_TRIE_CHILD; 4],
            outputs: Vec::new(),
        }
    }
}

/// Chooses a compact-template plan according to the requested search mode.
pub fn select_sample_plan(
    filter: &[CompiledRotation],
    algorithm: TextureAlgorithm,
    forced_errors: i32,
    error_tolerance: i32,
    mode: SearchMode,
) -> Result<Option<SamplePlan>, String> {
    if mode == SearchMode::Naive || forced_errors > error_tolerance {
        return Ok(None);
    }
    if error_tolerance != 0 || forced_errors != 0 {
        return match mode {
            SearchMode::SampleTrie => Err(
                "searchMode=sample-trie currently requires errorTolerance=0 and no forced errors"
                    .to_owned(),
            ),
            _ => Ok(None),
        };
    }

    let plan = build_sample_plan(filter, algorithm);
    match (mode, plan) {
        (SearchMode::SampleTrie, None) => Err(
            "searchMode=sample-trie could not find a canonical four-way compact template"
                .to_owned(),
        ),
        (SearchMode::SampleTrie, Some(plan)) => Ok(Some(plan)),
        (SearchMode::Auto, Some(plan)) if plan.estimated_speedup >= AUTO_MIN_SPEEDUP => {
            Ok(Some(plan))
        }
        _ => Ok(None),
    }
}

/// Builds the lowest estimated texture-work plan, without applying the auto
/// mode's profitability threshold.
pub fn build_sample_plan(
    filter: &[CompiledRotation],
    algorithm: TextureAlgorithm,
) -> Option<SamplePlan> {
    let eligible = eligible_constraints(filter, algorithm);
    if eligible.is_empty() {
        return None;
    }
    let expected_naive_checks = expected_checks(filter, &[]);
    let mut best: Option<SamplePlan> = None;

    for sample_size in 1..=MAX_SAMPLE_SIZE {
        for cell_mask in 1_u16..=u8::MAX as u16 {
            if cell_mask & 1 == 0 || cell_mask.count_ones() as usize != sample_size {
                continue;
            }
            let shape = shape_from_mask(cell_mask);
            let candidates = find_placements(&shape, &eligible);
            let Some((placement_box, box_candidates)) = largest_solid_box(&candidates) else {
                continue;
            };

            for permutation in permutations(sample_size) {
                let plan = make_plan(
                    filter,
                    expected_naive_checks,
                    &shape,
                    &permutation,
                    placement_box,
                    &box_candidates,
                );
                let replace = best.as_ref().is_none_or(|current| {
                    plan.expected_sample_checks < current.expected_sample_checks
                        || (plan.expected_sample_checks == current.expected_sample_checks
                            && plan.nodes.len() < current.nodes.len())
                });
                if replace {
                    best = Some(plan);
                }
            }
        }
    }
    best
}

/// Converts a 16-way model index to the deterministic four-way trie symbol.
#[inline(always)]
pub fn visible_four_way_symbol(algorithm: TextureAlgorithm, index: u8) -> u8 {
    match algorithm {
        TextureAlgorithm::Vanilla3 => index >> 2,
        _ => index & 3,
    }
}

/// Computes the world-anchor grid for a tile. `None` asks the GPU backend to
/// use its naive kernel for an integer-boundary tile whose halo is not
/// representable as signed 32-bit anchor coordinates.
pub fn anchor_grid(plan: &SamplePlan, item: &WorkItem) -> Option<AnchorGrid> {
    let maximum = plan.placement_box.max();
    let x = axis_grid(
        item.start.x,
        item.end.x,
        plan.placement_box.min.x,
        maximum.x,
        plan.placement_box.lengths[0],
    )?;
    let y = axis_grid(
        item.start.y,
        item.end.y,
        plan.placement_box.min.y,
        maximum.y,
        plan.placement_box.lengths[1],
    )?;
    let z = axis_grid(
        item.start.z,
        item.end.z,
        plan.placement_box.min.z,
        maximum.z,
        plan.placement_box.lengths[2],
    )?;
    Some(AnchorGrid {
        start: Int3 {
            x: x.0,
            y: y.0,
            z: z.0,
        },
        counts: [x.1, y.1, z.1],
        steps: plan.placement_box.lengths,
    })
}

fn axis_grid(
    candidate_start: i32,
    candidate_end: i32,
    placement_min: i32,
    placement_max: i32,
    period: u32,
) -> Option<(i32, u32)> {
    let period = i64::from(period);
    let minimum = i64::from(candidate_start) + i64::from(placement_min);
    let maximum = i64::from(candidate_end) - 1 + i64::from(placement_max);
    let quotient = minimum.div_euclid(period);
    let first = if minimum.rem_euclid(period) == 0 {
        quotient * period
    } else {
        (quotient + 1) * period
    };
    if first > maximum {
        return None;
    }
    let count = u32::try_from((maximum - first) / period + 1).ok()?;
    let last = first + i64::from(count - 1) * period;
    Some((i32::try_from(first).ok()?, count)).filter(|_| i32::try_from(last).is_ok())
}

fn eligible_constraints(
    filter: &[CompiledRotation],
    algorithm: TextureAlgorithm,
) -> HashMap<(i32, i32, i32), EligibleConstraint> {
    filter
        .iter()
        .enumerate()
        .filter_map(|(index, constraint)| {
            canonical_symbol(constraint.accepted_indices, algorithm).map(|symbol| {
                (
                    (
                        i32::from(constraint.x),
                        i32::from(constraint.y),
                        i32::from(constraint.z),
                    ),
                    EligibleConstraint {
                        index: index as u8,
                        symbol,
                    },
                )
            })
        })
        .collect()
}

fn canonical_symbol(mask: u16, algorithm: TextureAlgorithm) -> Option<u8> {
    (0..4_u8).find(|&symbol| canonical_mask(algorithm, symbol) == mask)
}

fn canonical_mask(algorithm: TextureAlgorithm, symbol: u8) -> u16 {
    let mut mask = 0_u16;
    for index in 0..16_u8 {
        if visible_four_way_symbol(algorithm, index) == symbol {
            mask |= 1 << index;
        }
    }
    mask
}

fn shape_from_mask(mask: u16) -> Vec<Int3> {
    (0..8)
        .filter(|bit| mask & (1 << bit) != 0)
        .map(|bit| Int3 {
            x: (bit >> 2) & 1,
            y: (bit >> 1) & 1,
            z: bit & 1,
        })
        .collect()
}

fn find_placements(
    shape: &[Int3],
    eligible: &HashMap<(i32, i32, i32), EligibleConstraint>,
) -> Vec<PlacementCandidate> {
    let mut placements = Vec::new();
    for &(x, y, z) in eligible.keys() {
        let mut witness_indices = [0; MAX_SAMPLE_SIZE];
        let mut symbols = [0; MAX_SAMPLE_SIZE];
        let mut valid = true;
        for (slot, offset) in shape.iter().enumerate() {
            let Some(constraint) = eligible.get(&(x + offset.x, y + offset.y, z + offset.z)) else {
                valid = false;
                break;
            };
            witness_indices[slot] = constraint.index;
            symbols[slot] = constraint.symbol;
        }
        if valid {
            placements.push(PlacementCandidate {
                origin: Int3 { x, y, z },
                witness_indices,
                symbols,
            });
        }
    }
    placements
        .sort_by_key(|placement| (placement.origin.x, placement.origin.y, placement.origin.z));
    placements
}

fn largest_solid_box(
    candidates: &[PlacementCandidate],
) -> Option<(PlacementBox, Vec<PlacementCandidate>)> {
    let origins: HashSet<_> = candidates
        .iter()
        .map(|candidate| (candidate.origin.x, candidate.origin.y, candidate.origin.z))
        .collect();
    let mut boxes = Vec::new();
    for lower in candidates {
        for upper in candidates {
            if upper.origin.x < lower.origin.x
                || upper.origin.y < lower.origin.y
                || upper.origin.z < lower.origin.z
            {
                continue;
            }
            let lengths = [
                (upper.origin.x - lower.origin.x + 1) as u32,
                (upper.origin.y - lower.origin.y + 1) as u32,
                (upper.origin.z - lower.origin.z + 1) as u32,
            ];
            let volume: u32 = lengths.into_iter().product();
            if volume as usize <= candidates.len() {
                boxes.push((volume, lower.origin, upper.origin, lengths));
            }
        }
    }
    boxes.sort_unstable_by(|left, right| {
        right
            .0
            .cmp(&left.0)
            .then_with(|| point_key(left.1).cmp(&point_key(right.1)))
            .then_with(|| point_key(left.2).cmp(&point_key(right.2)))
    });

    for (_, lower, upper, lengths) in boxes {
        let mut solid = true;
        'coordinates: for x in lower.x..=upper.x {
            for y in lower.y..=upper.y {
                for z in lower.z..=upper.z {
                    if !origins.contains(&(x, y, z)) {
                        solid = false;
                        break 'coordinates;
                    }
                }
            }
        }
        if solid {
            let members = candidates
                .iter()
                .filter(|candidate| {
                    let origin = candidate.origin;
                    origin.x >= lower.x
                        && origin.x <= upper.x
                        && origin.y >= lower.y
                        && origin.y <= upper.y
                        && origin.z >= lower.z
                        && origin.z <= upper.z
                })
                .cloned()
                .collect();
            return Some((
                PlacementBox {
                    min: lower,
                    lengths,
                },
                members,
            ));
        }
    }
    None
}

fn point_key(point: Int3) -> (i32, i32, i32) {
    (point.x, point.y, point.z)
}

fn permutations(length: usize) -> Vec<Vec<usize>> {
    fn visit(prefix: &mut Vec<usize>, used: &mut [bool], output: &mut Vec<Vec<usize>>) {
        if prefix.len() == used.len() {
            output.push(prefix.clone());
            return;
        }
        for index in 0..used.len() {
            if used[index] {
                continue;
            }
            used[index] = true;
            prefix.push(index);
            visit(prefix, used, output);
            prefix.pop();
            used[index] = false;
        }
    }

    let mut output = Vec::new();
    visit(&mut Vec::new(), &mut vec![false; length], &mut output);
    output
}

fn make_plan(
    filter: &[CompiledRotation],
    expected_naive_checks: f64,
    shape: &[Int3],
    permutation: &[usize],
    placement_box: PlacementBox,
    candidates: &[PlacementCandidate],
) -> SamplePlan {
    let sample_size = shape.len();
    let mut mutable_nodes = vec![MutableTrieNode::new()];
    let mut node_depths = vec![0_usize];

    for (candidate_index, candidate) in candidates.iter().enumerate() {
        let mut node_index = 0_usize;
        for &shape_index in permutation {
            let symbol = candidate.symbols[shape_index] as usize;
            let child = mutable_nodes[node_index].children[symbol];
            node_index = if child == MISSING_TRIE_CHILD {
                let child = mutable_nodes.len();
                mutable_nodes.push(MutableTrieNode::new());
                node_depths.push(node_depths[node_index] + 1);
                mutable_nodes[node_index].children[symbol] = child as u32;
                child
            } else {
                child as usize
            };
        }
        mutable_nodes[node_index].outputs.push(candidate_index);
    }

    let mut placements = Vec::with_capacity(candidates.len());
    let mut nodes = Vec::with_capacity(mutable_nodes.len());
    for node in mutable_nodes {
        let output_start = placements.len() as u32;
        for candidate_index in node.outputs {
            let candidate = &candidates[candidate_index];
            placements.push(TriePlacement {
                origin: candidate.origin,
                witness_indices: candidate.witness_indices,
            });
        }
        nodes.push(FlatTrieNode {
            children: node.children,
            output_start,
            output_count: placements.len() as u32 - output_start,
        });
    }

    let mut nodes_by_depth = vec![0_u32; sample_size + 1];
    for depth in node_depths {
        nodes_by_depth[depth] += 1;
    }
    let expected_anchor_checks = (0..sample_size)
        .map(|depth| f64::from(nodes_by_depth[depth]) / 4_f64.powi(depth as i32))
        .sum::<f64>();
    let expected_verification_checks = candidates
        .iter()
        .map(|candidate| expected_checks(filter, &candidate.witness_indices[..sample_size]))
        .sum::<f64>()
        / candidates.len() as f64;
    let positive_probability = 4_f64.powi(-(sample_size as i32));
    let expected_sample_checks = expected_anchor_checks / f64::from(placement_box.volume())
        + positive_probability * expected_verification_checks;

    SamplePlan {
        offsets: permutation.iter().map(|&index| shape[index]).collect(),
        placement_box,
        nodes,
        placements,
        expected_naive_checks,
        expected_sample_checks,
        estimated_speedup: expected_naive_checks / expected_sample_checks,
    }
}

fn expected_checks(filter: &[CompiledRotation], skipped: &[u8]) -> f64 {
    let mut probability = 1.0;
    let mut checks = 0.0;
    for (index, constraint) in filter.iter().enumerate() {
        if skipped.contains(&(index as u8)) {
            continue;
        }
        checks += probability;
        probability *= f64::from(constraint.accepted_indices.count_ones()) / 16.0;
    }
    checks
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dense_filter(width: i32, depth: i32) -> Vec<CompiledRotation> {
        let mut filter = Vec::new();
        for x in 0..width {
            for z in 0..depth {
                filter.push(CompiledRotation {
                    x: x as i8,
                    y: 0,
                    z: z as i8,
                    accepted_indices: canonical_mask(
                        TextureAlgorithm::Vanilla3,
                        ((x * 3 + z) & 3) as u8,
                    ),
                });
            }
        }
        filter
    }

    #[test]
    fn finds_the_expected_dense_plan() {
        let plan = build_sample_plan(&dense_filter(7, 6), TextureAlgorithm::Vanilla3).unwrap();
        assert!((3..=4).contains(&plan.sample_size()));
        assert_eq!(plan.placement_box.volume(), 30);
        assert!(plan.estimated_speedup > 10.0);
        assert_eq!(plan.placements.len(), 30);
    }

    #[test]
    fn placement_box_covers_every_candidate_once() {
        let plan = build_sample_plan(&dense_filter(7, 6), TextureAlgorithm::Vanilla3).unwrap();
        let [period_x, period_y, period_z] = plan.placement_box.lengths.map(i64::from);
        for candidate_x in -13_i64..=13 {
            for candidate_y in -3_i64..=3 {
                for candidate_z in -13_i64..=13 {
                    let witnesses = plan
                        .placements
                        .iter()
                        .filter(|placement| {
                            (candidate_x + i64::from(placement.origin.x)).rem_euclid(period_x) == 0
                                && (candidate_y + i64::from(placement.origin.y))
                                    .rem_euclid(period_y)
                                    == 0
                                && (candidate_z + i64::from(placement.origin.z))
                                    .rem_euclid(period_z)
                                    == 0
                        })
                        .count();
                    assert_eq!(witnesses, 1);
                }
            }
        }
    }

    #[test]
    fn trie_leaf_contains_the_pattern_placement() {
        let filter = dense_filter(7, 6);
        let plan = build_sample_plan(&filter, TextureAlgorithm::Vanilla3).unwrap();
        let by_offset: HashMap<_, _> = filter
            .iter()
            .map(|constraint| {
                (
                    (
                        i32::from(constraint.x),
                        i32::from(constraint.y),
                        i32::from(constraint.z),
                    ),
                    canonical_symbol(constraint.accepted_indices, TextureAlgorithm::Vanilla3)
                        .unwrap(),
                )
            })
            .collect();
        let expected = plan.placements[plan.placements.len() / 2].origin;
        let mut node = 0_usize;
        for offset in &plan.offsets {
            let symbol = by_offset[&(
                expected.x + offset.x,
                expected.y + offset.y,
                expected.z + offset.z,
            )];
            node = plan.nodes[node].children[symbol as usize] as usize;
        }
        let leaf = plan.nodes[node];
        let outputs = &plan.placements
            [leaf.output_start as usize..(leaf.output_start + leaf.output_count) as usize];
        assert!(outputs.iter().any(|placement| placement.origin == expected));
    }

    #[test]
    fn anchor_grid_includes_negative_lattice_points() {
        let plan = build_sample_plan(&dense_filter(7, 6), TextureAlgorithm::Vanilla3).unwrap();
        let item = WorkItem {
            start: Int3 {
                x: -7,
                y: -2,
                z: -8,
            },
            end: Int3 { x: 4, y: 3, z: 5 },
            direction_index: 0,
            direction: 0,
        };
        let grid = anchor_grid(&plan, &item).unwrap();
        assert_eq!(grid.start.x.rem_euclid(grid.steps[0] as i32), 0);
        assert_eq!(grid.start.y.rem_euclid(grid.steps[1] as i32), 0);
        assert_eq!(grid.start.z.rem_euclid(grid.steps[2] as i32), 0);
        assert!(grid.counts.into_iter().all(|count| count > 0));
    }

    #[test]
    fn anchor_grid_declines_an_unrepresentable_integer_halo() {
        let plan = build_sample_plan(&dense_filter(7, 6), TextureAlgorithm::Vanilla3).unwrap();
        let item = WorkItem {
            start: Int3 {
                x: i32::MAX - 1,
                y: 0,
                z: i32::MAX - 1,
            },
            end: Int3 {
                x: i32::MAX,
                y: 1,
                z: i32::MAX,
            },
            direction_index: 0,
            direction: 0,
        };
        assert!(anchor_grid(&plan, &item).is_none());
    }

    #[test]
    fn forced_sample_mode_rejects_mismatch_searches() {
        let error = select_sample_plan(
            &dense_filter(3, 3),
            TextureAlgorithm::Vanilla3,
            0,
            1,
            SearchMode::SampleTrie,
        )
        .unwrap_err();
        assert!(error.contains("errorTolerance=0"));
    }

    #[test]
    fn auto_mode_keeps_a_wide_profitability_margin() {
        assert!(
            select_sample_plan(
                &dense_filter(7, 6),
                TextureAlgorithm::Vanilla3,
                0,
                0,
                SearchMode::Auto,
            )
            .unwrap()
            .is_some()
        );
        assert!(
            select_sample_plan(
                &dense_filter(2, 1),
                TextureAlgorithm::Vanilla3,
                0,
                0,
                SearchMode::Auto,
            )
            .unwrap()
            .is_none()
        );
    }
}
