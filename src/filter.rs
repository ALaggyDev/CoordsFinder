//! Semantic filter transformation and compilation into block constraints.

use std::collections::HashMap;

use crate::types::{
    CompiledRotation, Face, Int3, MAX_FILTER_COUNT, RotationInfo, RotationKind, TextureAlgorithm,
};

pub(crate) fn rotate_xz(x: i32, z: i32, direction: i32) -> (i32, i32) {
    match direction / 90 {
        1 => (-z, x),
        2 => (-x, -z),
        3 => (z, -x),
        _ => (x, z),
    }
}

/// Netherrack's visible raw-PNG rotations for model indices 0 through 15.
/// Face order is up, down, north, south, east, west.
const NETHERRACK_FACE_ROTATIONS: [[u8; 6]; 16] = [
    [0, 0, 0, 0, 0, 0],
    [0, 2, 2, 0, 1, 3],
    [0, 0, 2, 2, 2, 2],
    [2, 0, 2, 0, 3, 1],
    [1, 3, 0, 0, 0, 0],
    [1, 1, 3, 1, 2, 0],
    [1, 3, 2, 2, 2, 2],
    [3, 3, 1, 3, 2, 0],
    [2, 2, 0, 0, 0, 0],
    [2, 0, 0, 2, 3, 1],
    [2, 2, 2, 2, 2, 2],
    [0, 2, 0, 2, 1, 3],
    [3, 1, 0, 0, 0, 0],
    [3, 3, 1, 3, 0, 2],
    [3, 1, 2, 2, 2, 2],
    [1, 1, 3, 1, 0, 2],
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SelectorFamily {
    Standard,
    Netherrack,
}

/// One direction-specific, fully compiled runtime filter.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedDirection {
    pub constraints: Vec<CompiledRotation>,
    pub forced_errors: i32,
}

/// Compiled filters for every configured direction, plus recoverable conflicts.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedFilters {
    pub directions: Vec<PreparedDirection>,
    pub conflicts_by_direction: Vec<(i32, Vec<Int3>)>,
}

impl PreparedFilters {
    /// Describes masks that can never match and consume the error budget.
    pub fn warning(&self) -> Option<String> {
        if self.conflicts_by_direction.is_empty() {
            return None;
        }
        let details = self
            .conflicts_by_direction
            .iter()
            .map(|(direction, offsets)| {
                let offsets = offsets
                    .iter()
                    .map(|offset| format!("({}, {}, {})", offset.x, offset.y, offset.z))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("direction {direction}: {offsets}")
            })
            .collect::<Vec<_>>()
            .join("; ");
        Some(format!(
            "combined observations create forced block errors ({details})"
        ))
    }
}

fn family(kind: RotationKind) -> SelectorFamily {
    match kind {
        RotationKind::StandardFourWay | RotationKind::StandardSide => SelectorFamily::Standard,
        RotationKind::Netherrack(_) => SelectorFamily::Netherrack,
    }
}

fn face_index(face: Face) -> usize {
    match face {
        Face::Up => 0,
        Face::Down => 1,
        Face::North => 2,
        Face::South => 3,
        Face::East => 4,
        Face::West => 5,
    }
}

fn visible_four_way(algorithm: TextureAlgorithm, index: u8) -> u8 {
    match algorithm {
        TextureAlgorithm::Vanilla3 => index >> 2,
        _ => index & 3,
    }
}

fn acceptance_mask(observation: RotationInfo, algorithm: TextureAlgorithm) -> u16 {
    let mut mask = 0_u16;
    for index in 0..16_u8 {
        let accepted = match observation.kind {
            RotationKind::StandardFourWay => {
                visible_four_way(algorithm, index) == observation.rotation
            }
            RotationKind::StandardSide => {
                visible_four_way(algorithm, index) & 1 == observation.rotation
            }
            RotationKind::Netherrack(face) => {
                NETHERRACK_FACE_ROTATIONS[index as usize][face_index(face)] == observation.rotation
            }
        };
        if accepted {
            mask |= 1 << index;
        }
    }
    mask
}

fn rotate_observation(mut observation: RotationInfo, direction: i32) -> RotationInfo {
    let turns = (direction / 90) as u8;
    let (x, z) = rotate_xz(
        i32::from(observation.x),
        i32::from(observation.z),
        direction,
    );
    observation.x = x as i8;
    observation.z = z as i8;
    match observation.kind {
        RotationKind::StandardFourWay => {
            observation.rotation = (observation.rotation + turns) % 4;
        }
        RotationKind::StandardSide => {}
        RotationKind::Netherrack(face) => {
            observation.kind = RotationKind::Netherrack(face.rotate_y(turns));
            observation.rotation = match face {
                Face::Up => (observation.rotation + turns) % 4,
                Face::Down => (observation.rotation + 4 - turns) % 4,
                _ => observation.rotation,
            };
        }
    }
    observation
}

fn compile_direction(
    observations: &[RotationInfo],
    algorithm: TextureAlgorithm,
    direction: i32,
) -> Result<(PreparedDirection, Vec<Int3>), String> {
    let mut groups: HashMap<(i8, i8, i8), (SelectorFamily, u16)> = HashMap::new();
    for &raw in observations {
        let observation = rotate_observation(raw, direction);
        let key = (observation.x, observation.y, observation.z);
        let observation_family = family(observation.kind);
        let mask = acceptance_mask(observation, algorithm);
        match groups.get_mut(&key) {
            Some((existing_family, combined_mask)) => {
                if *existing_family != observation_family {
                    return Err(format!(
                        "block offset ({}, {}, {}) mixes ordinary and netherrack observations",
                        raw.x, raw.y, raw.z
                    ));
                }
                *combined_mask &= mask;
            }
            None => {
                groups.insert(key, (observation_family, mask));
            }
        }
    }

    let mut conflicts = Vec::new();
    let mut constraints = Vec::new();
    for ((x, y, z), (_, accepted_indices)) in groups {
        if accepted_indices == 0 {
            conflicts.push(Int3 {
                x: i32::from(x),
                y: i32::from(y),
                z: i32::from(z),
            });
        } else {
            constraints.push(CompiledRotation {
                x,
                y,
                z,
                accepted_indices,
            });
        }
    }
    constraints.sort_by_key(|item| item.accepted_indices.count_ones());
    conflicts.sort_by_key(|item| (item.x, item.y, item.z));
    Ok((
        PreparedDirection {
            constraints,
            forced_errors: conflicts.len() as i32,
        },
        conflicts,
    ))
}

/// Compiles semantic face observations into one acceptance mask per block.
pub fn prepare_filters(
    observations: &[RotationInfo],
    algorithm: TextureAlgorithm,
    directions: &[i32],
    error_tolerance: i32,
) -> Result<PreparedFilters, String> {
    let mut prepared = Vec::with_capacity(directions.len());
    let mut conflicts_by_direction = Vec::new();
    let mut viable_directions = 0;
    for &direction in directions {
        let (compiled, conflicts) = compile_direction(observations, algorithm, direction)?;
        if compiled.forced_errors <= error_tolerance {
            viable_directions += 1;
        }
        if compiled.constraints.len() > MAX_FILTER_COUNT {
            return Err(format!(
                "filter contains more than {MAX_FILTER_COUNT} unique usable block offsets"
            ));
        }
        if compiled.constraints.is_empty() && compiled.forced_errors <= error_tolerance {
            return Err(
                "filter has no usable block constraints after combining observations".to_owned(),
            );
        }
        if !conflicts.is_empty() {
            conflicts_by_direction.push((direction, conflicts));
        }
        prepared.push(compiled);
    }
    if viable_directions == 0 {
        let details = conflicts_by_direction
            .iter()
            .map(|(direction, offsets)| {
                format!(
                    "direction {direction} has {} forced error(s)",
                    offsets.len()
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        return Err(format!(
            "combined observations exceed errorTolerance {error_tolerance} in every requested direction ({details})"
        ));
    }
    Ok(PreparedFilters {
        directions: prepared,
        conflicts_by_direction,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::texture::get_texture;

    #[test]
    fn legacy_masks_reproduce_bound_four_sampling() {
        for algorithm in [
            TextureAlgorithm::Vanilla1,
            TextureAlgorithm::Vanilla2,
            TextureAlgorithm::Vanilla3,
            TextureAlgorithm::Sodium1,
            TextureAlgorithm::Sodium2,
        ] {
            for x in -12..=12 {
                for y in -3..=3 {
                    let z = x * 17 - y * 31;
                    let expected = get_texture(algorithm, x, y, z, 4);
                    let index = get_texture(algorithm, x, y, z, 16);
                    let full = RotationInfo::new(0, 0, 0, expected, false);
                    let side = RotationInfo::new(0, 0, 0, expected & 1, true);
                    assert_ne!(acceptance_mask(full, algorithm) & (1 << index), 0);
                    assert_ne!(acceptance_mask(side, algorithm) & (1 << index), 0);
                }
            }
        }
    }

    #[test]
    fn all_netherrack_faces_follow_yaw_direction_rules() {
        let faces = [
            Face::Up,
            Face::Down,
            Face::North,
            Face::South,
            Face::East,
            Face::West,
        ];
        for (index, rotations) in NETHERRACK_FACE_ROTATIONS.iter().enumerate() {
            let x_rotation = index % 4;
            let y_rotation = index / 4;
            for (face_index_value, face) in faces.into_iter().enumerate() {
                let rotation = rotations[face_index_value];
                for turns in 0..4_usize {
                    let observation = RotationInfo::netherrack(0, 0, 0, rotation, face);
                    let rotated = rotate_observation(observation, (turns * 90) as i32);
                    let rotated_index = ((y_rotation + turns) % 4) * 4 + x_rotation;
                    assert_ne!(
                        acceptance_mask(rotated, TextureAlgorithm::Vanilla3) & (1 << rotated_index),
                        0,
                        "index={index}, face={face:?}, turns={turns}"
                    );
                }
            }
        }
    }

    #[test]
    fn combines_correlated_netherrack_faces() {
        let observations = [
            RotationInfo::netherrack(0, 0, 0, 1, Face::Up),
            RotationInfo::netherrack(0, 0, 0, 3, Face::North),
            RotationInfo::netherrack(0, 0, 0, 2, Face::East),
        ];
        let prepared = prepare_filters(&observations, TextureAlgorithm::Vanilla3, &[0], 0).unwrap();
        assert_eq!(prepared.directions[0].constraints.len(), 1);
        assert_eq!(
            prepared.directions[0].constraints[0].accepted_indices,
            1 << 5
        );
    }

    #[test]
    fn removes_recoverable_forced_errors() {
        let observations = [
            RotationInfo::netherrack(0, 0, 0, 0, Face::Up),
            RotationInfo::netherrack(0, 0, 0, 1, Face::Up),
            RotationInfo::new(1, 0, 0, 2, false),
        ];
        let prepared = prepare_filters(&observations, TextureAlgorithm::Vanilla3, &[0], 1).unwrap();
        assert_eq!(prepared.directions[0].forced_errors, 1);
        assert_eq!(prepared.directions[0].constraints.len(), 1);
        assert!(prepared.warning().is_some());
        assert!(prepare_filters(&observations, TextureAlgorithm::Vanilla3, &[0], 0).is_err());
    }

    #[test]
    fn keeps_viable_directions_when_grouped_faces_rule_one_out() {
        let observations = [
            RotationInfo::new(0, 0, 0, 0, false),
            RotationInfo::new(0, 0, 0, 0, true),
        ];
        let prepared =
            prepare_filters(&observations, TextureAlgorithm::Vanilla1, &[0, 90], 0).unwrap();
        assert_eq!(prepared.directions[0].forced_errors, 0);
        assert_eq!(prepared.directions[1].forced_errors, 1);
        assert!(prepared.warning().is_some());
    }

    #[test]
    fn rejects_mixed_selector_families_at_one_block() {
        let observations = [
            RotationInfo::new(0, 0, 0, 0, false),
            RotationInfo::netherrack(0, 0, 0, 0, Face::Up),
        ];
        assert!(
            prepare_filters(&observations, TextureAlgorithm::Vanilla3, &[0], 0)
                .unwrap_err()
                .contains("mixes ordinary and netherrack")
        );
    }

    #[test]
    fn rotates_netherrack_faces_and_rotations() {
        let observations = [
            RotationInfo::netherrack(2, 0, -3, 0, Face::Up),
            RotationInfo::netherrack(2, 0, -3, 0, Face::Down),
            RotationInfo::netherrack(2, 0, -3, 0, Face::North),
        ];
        let (prepared, _) =
            compile_direction(&observations, TextureAlgorithm::Vanilla3, 90).unwrap();
        assert_eq!(prepared.constraints.len(), 1);
        // These transformed faces are jointly satisfied by model index 4.
        assert_ne!(prepared.constraints[0].accepted_indices & (1 << 4), 0);
        assert_eq!(
            (prepared.constraints[0].x, prepared.constraints[0].z),
            (3, 2)
        );
    }
}
