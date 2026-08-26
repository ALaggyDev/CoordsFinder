use std::fmt;
use std::str::FromStr;

pub const MAX_FILTER_COUNT: usize = 256;

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

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Int3 {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RotationInfo {
    pub x: i8,
    pub y: i8,
    pub z: i8,
    pub rotation: u8,
    /// `3` compares four variants; `1` folds side faces to two states.
    pub visible_mask: u8,
}

impl RotationInfo {
    pub fn new(x: i8, y: i8, z: i8, rotation: u8, side: bool) -> Self {
        let visible_mask = if side { 1 } else { 3 };
        Self {
            x,
            y,
            z,
            rotation: rotation % (visible_mask + 1),
            visible_mask,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Match {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub mismatches: i32,
    pub direction: i32,
}
