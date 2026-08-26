use crate::config::{ScanConfig, ScanOrder, TileSize, rotate_xz};
use crate::types::{Int3, RotationInfo};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WorkItem {
    pub start: Int3,
    pub end: Int3,
    pub direction_index: usize,
    pub direction: i32,
}

#[derive(Clone, Debug, Default)]
pub struct ScanPlan {
    pub items: Vec<WorkItem>,
    pub total_candidates: u64,
    pub total_candidates_saturated: bool,
}

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
    result.sort_by_key(|item| std::cmp::Reverse(item.visible_mask));
    result
}

fn span(start: i32, end: i32) -> u64 {
    (i64::from(end) - i64::from(start)) as u64
}

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

pub fn make_plan(config: &ScanConfig, tile_size: TileSize) -> Result<ScanPlan, String> {
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
    let capacity = usize::try_from(work_count)
        .map_err(|_| "scan contains too many work items for this build".to_owned())?;
    let mut plan = ScanPlan {
        items: Vec::with_capacity(capacity),
        ..ScanPlan::default()
    };

    let mut add_tile = |tile_index_x: u64, tile_index_z: u64, direction_index: usize| {
        let x_start = i64::from(config.x_range.start) + (tile_index_x * tile_x) as i64;
        let z_start = i64::from(config.z_range.start) + (tile_index_z * tile_z) as i64;
        let item = WorkItem {
            start: Int3 {
                x: x_start as i32,
                y: config.y_range.start,
                z: z_start as i32,
            },
            end: Int3 {
                x: (x_start + tile_x as i64).min(i64::from(config.x_range.end)) as i32,
                y: config.y_range.end,
                z: (z_start + tile_z as i64).min(i64::from(config.z_range.end)) as i32,
            },
            direction_index,
            direction: config.directions[direction_index],
        };
        let (candidates, saturated) = candidate_count(&item);
        match plan.total_candidates.checked_add(candidates) {
            Some(total) if !saturated => plan.total_candidates = total,
            _ => {
                plan.total_candidates = u64::MAX;
                plan.total_candidates_saturated = true;
            }
        }
        plan.items.push(item);
    };

    let mut emit = |x: i64, z: i64| {
        if x >= 0 && z >= 0 && x < x_tiles as i64 && z < z_tiles as i64 {
            for direction in 0..config.directions.len() {
                add_tile(x as u64, z as u64, direction);
            }
        }
    };

    match config.scan_order {
        ScanOrder::Linear => {
            for x in 0..x_tiles {
                for z in 0..z_tiles {
                    emit(x as i64, z as i64);
                }
            }
        }
        ScanOrder::Spiral => {
            let center_x = ((x_span - 1) / 2 / tile_x) as i64;
            let center_z = ((z_span - 1) / 2 / tile_z) as i64;
            let max_radius = [
                center_x,
                center_z,
                x_tiles as i64 - 1 - center_x,
                z_tiles as i64 - 1 - center_z,
            ]
            .into_iter()
            .max()
            .unwrap_or(0);
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
        }
    }
    if plan.items.len() != capacity {
        return Err("internal error: scan order did not cover every tile exactly once".to_owned());
    }
    Ok(plan)
}

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
        assert_eq!(spiral.items.len(), 50);
        assert_eq!((spiral.items[0].start.x, spiral.items[0].start.z), (0, 0));
        let visited: HashSet<_> = spiral
            .items
            .iter()
            .map(|item| (item.start.x, item.start.z, item.direction))
            .collect();
        assert_eq!(visited.len(), spiral.items.len());

        config.scan_order = ScanOrder::Linear;
        let linear = make_plan(&config, config.gpu_tile_size).unwrap();
        assert_eq!((linear.items[0].start.x, linear.items[0].start.z), (-2, -2));
    }

    #[test]
    fn spiral_covers_narrow_rectangles() {
        let mut config = config();
        config.directions = vec![0];
        config.scan_order = ScanOrder::Spiral;
        config.x_range = IntRange { start: 0, end: 10 };
        config.z_range = IntRange { start: -1, end: 1 };
        let plan = make_plan(&config, TileSize { x: 3, z: 1 }).unwrap();
        let visited: HashSet<_> = plan
            .items
            .iter()
            .map(|item| (item.start.x, item.start.z))
            .collect();
        assert_eq!(plan.items.len(), 8);
        assert_eq!(visited.len(), 8);
    }
}
