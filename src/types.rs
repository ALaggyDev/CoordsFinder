//! Shared value types used by configuration, planning, and scan backends.

use std::fmt;
use std::str::FromStr;

/// Maximum number of compiled block constraints supported by the GPU layout.
pub const MAX_FILTER_COUNT: usize = 256;

/// Exact-search implementation selected for a scan.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SearchMode {
    /// Use the sample trie when its static cost model predicts a useful gain.
    #[default]
    Auto,
    /// Always use the existing candidate-at-a-time loop.
    Naive,
    /// Require the compact-template trie implementation.
    SampleTrie,
}

impl fmt::Display for SearchMode {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        output.write_str(match self {
            Self::Auto => "auto",
            Self::Naive => "naive",
            Self::SampleTrie => "sample-trie",
        })
    }
}

impl FromStr for SearchMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "naive" => Ok(Self::Naive),
            "sample-trie" | "sampletrie" | "trie" => Ok(Self::SampleTrie),
            _ => Err(format!("unknown search mode '{value}'")),
        }
    }
}

/// Texture randomization implementation to reproduce during a search.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum TextureAlgorithm {
    Vanilla1,
    Vanilla2,
    Vanilla3,
    Sodium1,
    Sodium2,
}

impl fmt::Display for TextureAlgorithm {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Vanilla1 => "Vanilla-1",
            Self::Vanilla2 => "Vanilla-2",
            Self::Vanilla3 => "Vanilla-3",
            Self::Sodium1 => "Sodium-1",
            Self::Sodium2 => "Sodium-2",
        };
        output.write_str(name)
    }
}

impl FromStr for TextureAlgorithm {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "vanilla-1" => Ok(Self::Vanilla1),
            "vanilla-2" => Ok(Self::Vanilla2),
            "vanilla-3" => Ok(Self::Vanilla3),
            "sodium-1" => Ok(Self::Sodium1),
            "sodium-2" => Ok(Self::Sodium2),
            _ => Err(format!("unknown texture algorithm '{value}'")),
        }
    }
}

/// A signed three-dimensional coordinate or half-open bound.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Int3 {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

/// One of Minecraft's six world-facing block faces.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u8)]
pub enum Face {
    Up,
    Down,
    North,
    South,
    East,
    West,
}

impl Face {
    /// Rotates a horizontal face clockwise around Y in world coordinates.
    pub fn rotate_y(self, quarter_turns: u8) -> Self {
        let mut face = self;
        for _ in 0..quarter_turns % 4 {
            face = match face {
                Self::North => Self::East,
                Self::East => Self::South,
                Self::South => Self::West,
                Self::West => Self::North,
                vertical => vertical,
            };
        }
        face
    }
}

/// How a configured texture observation is interpreted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RotationKind {
    StandardFourWay,
    StandardSide,
    Netherrack(Face),
}

/// One texture observation at an offset from a candidate origin.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RotationInfo {
    pub x: i8,
    pub y: i8,
    pub z: i8,
    pub rotation: u8,
    pub kind: RotationKind,
}

impl RotationInfo {
    /// Creates a normalized observation for a four-state face or two-state side.
    pub fn new(x: i8, y: i8, z: i8, rotation: u8, side: bool) -> Self {
        let kind = if side {
            RotationKind::StandardSide
        } else {
            RotationKind::StandardFourWay
        };
        Self {
            x,
            y,
            z,
            rotation: rotation % if side { 2 } else { 4 },
            kind,
        }
    }

    /// Creates a netherrack face observation.
    pub fn netherrack(x: i8, y: i8, z: i8, rotation: u8, face: Face) -> Self {
        Self {
            x,
            y,
            z,
            rotation: rotation % 4,
            kind: RotationKind::Netherrack(face),
        }
    }
}

/// One preprocessed block constraint consumed by both scan backends.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompiledRotation {
    pub x: i8,
    pub y: i8,
    pub z: i8,
    /// Bit `i` is set when 16-way model index `i` satisfies this block.
    pub accepted_indices: u16,
}

/// A candidate coordinate accepted by a scan backend.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Match {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub mismatches: i32,
    pub direction: i32,
}
