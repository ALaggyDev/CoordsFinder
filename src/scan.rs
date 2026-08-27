//! Backend-independent scan planning and filter rotation.
//!
//! Plans split the X/Z search area into half-open tiles. Each tile is repeated
//! for every requested direction; CPU and GPU backends consume the same plan.

use crate::config::{ScanConfig, ScanOrder, TileSize, rotate_xz};
use crate::types::{Int3, RotationInfo};

/// A half-open 3D tile paired with one structure direction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WorkItem {
    pub start: Int3,
    pub end: Int3,
    pub direction_index: usize,
    pub direction: i32,
}

/// Maps work numbers to ordered work items without storing the item sequence.
#[derive(Clone, Debug)]
pub struct ScanPlan<'a> {
    x_start: i32,
    x_end: i32,
    y_start: i32,
    y_end: i32,
    z_start: i32,
    z_end: i32,
    tile_x: u64,
    tile_z: u64,
    x_tiles: u64,
    z_tiles: u64,
    directions: &'a [i32],
    scan_order: ScanOrder,
    center_x: i64,
    center_z: i64,
    max_radius: i64,
    total_items: usize,
    pub total_candidates: u64,
    pub total_candidates_saturated: bool,
}

/// Sequential view over an immutable [`ScanPlan`].
#[derive(Clone, Debug)]
pub struct WorkItems<'plan, 'config> {
    plan: &'plan ScanPlan<'config>,
    next: usize,
}

/// Rotates a filter into world orientation and prioritizes selective samples.
///
/// Four-way top/bottom faces rotate with the structure. Side faces retain
/// their two-state variant because a quarter turn does not change that state.
pub fn directional_filter(filter: &[RotationInfo], direction: i32) -> Vec<RotationInfo> {
    let mut result: Vec<_> = filter
        .iter()
        .copied()
        .map(|mut item| {
            let (x, z) = rotate_xz(i32::from(item.x), i32::from(item.z), direction);
            item.x = x as i8;
            item.z = z as i8;
            if item.visible_mask == 3 {
                item.rotation = (item.rotation + (direction / 90) as u8) % 4;
            }
            item
        })
        .collect();
    // Four-state samples reject random candidates more often, so test them first.
    result.sort_by_key(|item| std::cmp::Reverse(item.visible_mask));
    result
}

fn span(start: i32, end: i32) -> u64 {
    (i64::from(end) - i64::from(start)) as u64
}

/// Returns the number of coordinates in a work item and whether it overflowed.
pub fn candidate_count(item: &WorkItem) -> (u64, bool) {
    let mut count = 1_u64;
    for dimension in [
        span(item.start.x, item.end.x),
        span(item.start.y, item.end.y),
        span(item.start.z, item.end.z),
    ] {
        match count.checked_mul(dimension) {
            Some(value) => count = value,
            None => return (u64::MAX, true),
        }
    }
    (count, false)
}

/// Builds a lazy tile sequence for a scan configuration.
///
/// Candidate totals saturate at [`u64::MAX`]. The iterator itself uses constant
/// memory regardless of the number of work items.
pub fn make_plan(config: &ScanConfig, tile_size: TileSize) -> Result<ScanPlan<'_>, String> {
    if tile_size.x <= 0 || tile_size.z <= 0 {
        return Err("tile dimensions must be positive".to_owned());
    }
    let x_span = span(config.x_range.start, config.x_range.end);
    let z_span = span(config.z_range.start, config.z_range.end);
    let tile_x = tile_size.x as u64;
    let tile_z = tile_size.z as u64;
    let x_tiles = x_span.div_ceil(tile_x);
    let z_tiles = z_span.div_ceil(tile_z);
    let work_count = x_tiles
        .checked_mul(z_tiles)
        .and_then(|count| count.checked_mul(config.directions.len() as u64))
        .ok_or_else(|| "scan contains too many work items".to_owned())?;
    let total_items = usize::try_from(work_count)
        .map_err(|_| "scan contains too many work items for this build".to_owned())?;

    let center_x = (x_tiles as i64 - 1) / 2;
    let center_z = (z_tiles as i64 - 1) / 2;
    let max_radius = [
        center_x,
        center_z,
        x_tiles as i64 - 1 - center_x,
        z_tiles as i64 - 1 - center_z,
    ]
    .into_iter()
    .max()
    .unwrap_or(0);
    let dimensions = [
        x_span,
        span(config.y_range.start, config.y_range.end),
        z_span,
        config.directions.len() as u64,
    ];
    let total_candidates = dimensions.into_iter().try_fold(1_u64, u64::checked_mul);

    Ok(ScanPlan {
        x_start: config.x_range.start,
        x_end: config.x_range.end,
        y_start: config.y_range.start,
        y_end: config.y_range.end,
        z_start: config.z_range.start,
        z_end: config.z_range.end,
        tile_x,
        tile_z,
        x_tiles,
        z_tiles,
        directions: &config.directions,
        scan_order: config.scan_order,
        center_x,
        center_z,
        max_radius,
        total_items,
        total_candidates: total_candidates.unwrap_or(u64::MAX),
        total_candidates_saturated: total_candidates.is_none(),
    })
}

impl<'a> ScanPlan<'a> {
    /// Returns the total number of work items.
    pub fn total_items(&self) -> usize {
        self.total_items
    }

    /// Converts a work number to its tile and direction.
    pub fn work_item(&self, work_num: usize) -> Option<WorkItem> {
        if work_num >= self.total_items {
            return None;
        }
        let direction_index = work_num % self.directions.len();
        let tile_num = u64::try_from(work_num / self.directions.len()).ok()?;
        let (tile_index_x, tile_index_z) = match self.scan_order {
            ScanOrder::Linear => (tile_num / self.z_tiles, tile_num % self.z_tiles),
            ScanOrder::Spiral => self.spiral_tile(tile_num),
        };

        let x_start = i64::from(self.x_start) + (tile_index_x * self.tile_x) as i64;
        let z_start = i64::from(self.z_start) + (tile_index_z * self.tile_z) as i64;
        Some(WorkItem {
            start: Int3 {
                x: x_start as i32,
                y: self.y_start,
                z: z_start as i32,
            },
            end: Int3 {
                x: (x_start + self.tile_x as i64).min(i64::from(self.x_end)) as i32,
                y: self.y_end,
                z: (z_start + self.tile_z as i64).min(i64::from(self.z_end)) as i32,
            },
            direction_index,
            direction: self.directions[direction_index],
        })
    }

    /// Returns a sequential iterator over all work items.
    pub fn iter(&self) -> WorkItems<'_, 'a> {
        WorkItems {
            plan: self,
            next: 0,
        }
    }

    fn covered_tiles(&self, radius: i64) -> u64 {
        let x_min = (self.center_x - radius).max(0);
        let x_max = (self.center_x + radius).min(self.x_tiles as i64 - 1);
        let z_min = (self.center_z - radius).max(0);
        let z_max = (self.center_z + radius).min(self.z_tiles as i64 - 1);
        ((x_max - x_min + 1) as u64) * ((z_max - z_min + 1) as u64)
    }

    fn spiral_tile(&self, tile_num: u64) -> (u64, u64) {
        let mut low = 0;
        let mut high = self.max_radius;
        while low < high {
            let radius = low + (high - low) / 2;
            if tile_num < self.covered_tiles(radius) {
                high = radius;
            } else {
                low = radius + 1;
            }
        }
        let radius = low;
        if radius == 0 {
            return (self.center_x as u64, self.center_z as u64);
        }

        let mut offset = tile_num - self.covered_tiles(radius - 1);
        let x_max = self.x_tiles as i64 - 1;
        let z_max = self.z_tiles as i64 - 1;
        for edge in 0..4 {
            // Each corner belongs to exactly one edge, matching the original
            // clockwise spiral order: right, top, left, then bottom.
            let (fixed, start, end, step, fixed_x) = match edge {
                0 => (
                    self.center_x + radius,
                    self.center_z - radius + 1,
                    self.center_z + radius,
                    1,
                    true,
                ),
                1 => (
                    self.center_z + radius,
                    self.center_x + radius - 1,
                    self.center_x - radius,
                    -1,
                    false,
                ),
                2 => (
                    self.center_x - radius,
                    self.center_z + radius - 1,
                    self.center_z - radius,
                    -1,
                    true,
                ),
                3 => (
                    self.center_z - radius,
                    self.center_x - radius + 1,
                    self.center_x + radius,
                    1,
                    false,
                ),
                _ => unreachable!(),
            };
            let fixed_max = if fixed_x { x_max } else { z_max };
            if !(0..=fixed_max).contains(&fixed) {
                continue;
            }
            let variable_max = if fixed_x { z_max } else { x_max };
            let start = if step > 0 {
                start.max(0)
            } else {
                start.min(variable_max)
            };
            let end = if step > 0 {
                end.min(variable_max)
            } else {
                end.max(0)
            };
            if (step > 0 && start > end) || (step < 0 && start < end) {
                continue;
            }
            let edge_len = (end - start).unsigned_abs() + 1;
            if offset < edge_len {
                let position = start + step * offset as i64;
                let tile = if fixed_x {
                    (fixed, position)
                } else {
                    (position, fixed)
                };
                return (tile.0 as u64, tile.1 as u64);
            }
            offset -= edge_len;
        }
        unreachable!("spiral ring offset must map to a clipped edge")
    }
}

impl Iterator for WorkItems<'_, '_> {
    type Item = WorkItem;

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.plan.work_item(self.next)?;
        self.next += 1;
        Some(item)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.plan.total_items - self.next;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for WorkItems<'_, '_> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{IntRange, ScanConfig};
    use crate::types::TextureAlgorithm;
    use std::collections::HashSet;

    fn config() -> ScanConfig {
        ScanConfig {
            algorithm: TextureAlgorithm::Vanilla3,
            directions: vec![0, 90],
            x_range: IntRange { start: -2, end: 3 },
            y_range: IntRange { start: 0, end: 1 },
            z_range: IntRange { start: -2, end: 3 },
            cpu_tile_size: TileSize { x: 1, z: 1 },
            gpu_tile_size: TileSize { x: 1, z: 1 },
            filter: vec![RotationInfo::new(0, 0, 0, 0, false)],
            ..ScanConfig::default()
        }
    }

    #[test]
    fn rotates_and_orders_filters() {
        let filter = [
            RotationInfo::new(2, 3, -4, 1, false),
            RotationInfo::new(-1, 0, 5, 1, true),
        ];
        let rotated = directional_filter(&filter, 90);
        assert_eq!((rotated[0].x, rotated[0].z, rotated[0].rotation), (4, 2, 2));
        assert_eq!(
            (rotated[1].x, rotated[1].z, rotated[1].rotation),
            (-5, -1, 1)
        );
    }

    #[test]
    fn builds_linear_and_spiral_plans() {
        let mut config = config();
        config.scan_order = ScanOrder::Spiral;
        let spiral = make_plan(&config, config.gpu_tile_size).unwrap();
        assert_eq!(spiral.total_items(), 50);
        let spiral_items: Vec<_> = spiral.iter().collect();
        assert_eq!((spiral_items[0].start.x, spiral_items[0].start.z), (0, 0));
        let first_ring: Vec<_> = spiral_items
            .iter()
            .step_by(config.directions.len())
            .take(9)
            .map(|item| (item.start.x, item.start.z))
            .collect();
        assert_eq!(
            first_ring,
            [
                (0, 0),
                (1, 0),
                (1, 1),
                (0, 1),
                (-1, 1),
                (-1, 0),
                (-1, -1),
                (0, -1),
                (1, -1),
            ]
        );
        let visited: HashSet<_> = spiral_items
            .iter()
            .map(|item| (item.start.x, item.start.z, item.direction))
            .collect();
        assert_eq!(visited.len(), spiral_items.len());

        config.scan_order = ScanOrder::Linear;
        let linear = make_plan(&config, config.gpu_tile_size).unwrap();
        let first = linear.work_item(0).unwrap();
        assert_eq!((first.start.x, first.start.z), (-2, -2));
    }

    #[test]
    fn spiral_covers_narrow_rectangles() {
        let mut config = config();
        config.directions = vec![0];
        config.scan_order = ScanOrder::Spiral;
        config.x_range = IntRange { start: 0, end: 10 };
        config.z_range = IntRange { start: -1, end: 1 };
        let plan = make_plan(&config, TileSize { x: 3, z: 1 }).unwrap();
        assert_eq!(plan.total_items(), 8);
        let items: Vec<_> = plan.iter().collect();
        let visited: HashSet<_> = items
            .iter()
            .map(|item| (item.start.x, item.start.z))
            .collect();
        assert_eq!(visited.len(), 8);
    }

    #[test]
    fn indexed_spiral_matches_reference_for_rectangles() {
        fn reference(x_tiles: i64, z_tiles: i64) -> Vec<(i32, i32)> {
            let center_x = (x_tiles - 1) / 2;
            let center_z = (z_tiles - 1) / 2;
            let max_radius = [
                center_x,
                center_z,
                x_tiles - 1 - center_x,
                z_tiles - 1 - center_z,
            ]
            .into_iter()
            .max()
            .unwrap();
            let mut tiles = Vec::new();
            let mut emit = |x: i64, z: i64| {
                if x >= 0 && z >= 0 && x < x_tiles && z < z_tiles {
                    tiles.push((x as i32, z as i32));
                }
            };

            emit(center_x, center_z);
            for radius in 1..=max_radius {
                for z in center_z - radius + 1..=center_z + radius {
                    emit(center_x + radius, z);
                }
                for x in (center_x - radius..=center_x + radius - 1).rev() {
                    emit(x, center_z + radius);
                }
                for z in (center_z - radius..=center_z + radius - 1).rev() {
                    emit(center_x - radius, z);
                }
                for x in center_x - radius + 1..=center_x + radius {
                    emit(x, center_z - radius);
                }
            }
            tiles
        }

        let mut config = config();
        config.directions = vec![0];
        config.scan_order = ScanOrder::Spiral;
        for x_tiles in 1..=9 {
            for z_tiles in 1..=9 {
                config.x_range = IntRange {
                    start: 0,
                    end: x_tiles,
                };
                config.z_range = IntRange {
                    start: 0,
                    end: z_tiles,
                };
                let plan = make_plan(&config, TileSize { x: 1, z: 1 }).unwrap();
                let actual: Vec<_> = plan
                    .iter()
                    .map(|item| (item.start.x, item.start.z))
                    .collect();
                assert_eq!(actual, reference(x_tiles.into(), z_tiles.into()));
            }
        }
    }
}
