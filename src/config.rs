use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use crate::types::{MAX_FILTER_COUNT, RotationInfo, TextureAlgorithm};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IntRange {
    pub start: i32,
    pub end: i32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TileSize {
    pub x: i32,
    pub z: i32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScanOrder {
    Linear,
    Spiral,
}

#[derive(Clone, Debug)]
pub struct ScanConfig {
    pub algorithm: TextureAlgorithm,
    pub scan_order: ScanOrder,
    pub directions: Vec<i32>,
    pub x_range: IntRange,
    pub y_range: IntRange,
    pub z_range: IntRange,
    pub error_tolerance: i32,
    pub cpu_tile_size: TileSize,
    pub gpu_tile_size: TileSize,
    pub verbose: bool,
    pub filter: Vec<RotationInfo>,
    pub source_path: PathBuf,
}

impl Default for ScanConfig {
    fn default() -> Self {
        Self {
            algorithm: TextureAlgorithm::Vanilla3,
            scan_order: ScanOrder::Linear,
            directions: vec![0],
            x_range: IntRange { start: 0, end: 0 },
            y_range: IntRange { start: 0, end: 0 },
            z_range: IntRange { start: 0, end: 0 },
            error_tolerance: 0,
            cpu_tile_size: TileSize { x: 1024, z: 1024 },
            gpu_tile_size: TileSize {
                x: 16_384,
                z: 16_384,
            },
            verbose: false,
            filter: Vec::new(),
            source_path: PathBuf::new(),
        }
    }
}

fn compact_name(value: &str) -> String {
    value
        .chars()
        .filter(|character| !matches!(character, ' ' | '_' | '-' | '.'))
        .flat_map(char::to_lowercase)
        .collect()
}

fn parse_pair(value: &str) -> Result<(i32, i32), String> {
    let value = value.trim();
    let contents = value
        .strip_prefix('(')
        .and_then(|text| text.strip_suffix(')'))
        .ok_or_else(|| format!("expected '(first, second)', got '{value}'"))?;
    let (first, second) = contents
        .split_once(',')
        .ok_or_else(|| format!("expected two comma-separated integers, got '{value}'"))?;
    let first = first
        .trim()
        .parse::<i32>()
        .map_err(|_| format!("invalid integer '{}'", first.trim()))?;
    let second = second
        .trim()
        .parse::<i32>()
        .map_err(|_| format!("invalid integer '{}'", second.trim()))?;
    Ok((first, second))
}

fn parse_directions(value: &str) -> Result<Vec<i32>, String> {
    let contents = value
        .trim()
        .strip_prefix('[')
        .and_then(|text| text.strip_suffix(']'))
        .ok_or_else(|| "directions must use [0, 90, ...] syntax".to_owned())?;
    let mut directions = Vec::new();
    for item in contents.split(',') {
        let direction = item
            .trim()
            .parse::<i32>()
            .map_err(|_| format!("invalid direction '{}'", item.trim()))?;
        if !matches!(direction, 0 | 90 | 180 | 270) || directions.contains(&direction) {
            return Err(format!("direction {direction} is invalid or duplicated"));
        }
        directions.push(direction);
    }
    if directions.is_empty() {
        return Err("directions must not be empty".to_owned());
    }
    Ok(directions)
}

fn parse_bool(value: &str) -> Result<bool, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "true" | "yes" | "on" | "1" => Ok(true),
        "false" | "no" | "off" | "0" => Ok(false),
        _ => Err(format!("invalid boolean '{value}'")),
    }
}

fn parse_filter(value: &str) -> Result<RotationInfo, String> {
    let (coordinates, variant) = value
        .split_once('|')
        .ok_or_else(|| "filter rows must be: x y z | variant [side]".to_owned())?;
    let coordinates = coordinates
        .split_whitespace()
        .map(str::parse::<i8>)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| "filter offsets must fit in int8 range [-128, 127]".to_owned())?;
    if coordinates.len() != 3 {
        return Err("filter rows must contain three coordinates".to_owned());
    }

    let mut variant = variant.split_whitespace();
    let rotation = variant
        .next()
        .ok_or_else(|| "filter row is missing a variant".to_owned())?
        .parse::<u8>()
        .map_err(|_| "filter variant must be a non-negative integer".to_owned())?;
    let side = match variant.next().map(str::to_ascii_lowercase).as_deref() {
        None | Some("normal" | "false" | "0") => false,
        Some("side" | "true" | "1") => true,
        Some(marker) => return Err(format!("invalid side marker '{marker}'")),
    };
    if variant.next().is_some() {
        return Err("unexpected extra token in filter row".to_owned());
    }
    let maximum = if side { 1 } else { 3 };
    if rotation > maximum {
        return Err(format!("variant {rotation} exceeds maximum {maximum}"));
    }
    Ok(RotationInfo::new(
        coordinates[0],
        coordinates[1],
        coordinates[2],
        rotation,
        side,
    ))
}

fn line_error(path: &Path, line: usize, message: impl std::fmt::Display) -> String {
    format!("{}:{line}: {message}", path.display())
}

pub fn load(path: impl AsRef<Path>) -> Result<ScanConfig, String> {
    let path = path.as_ref();
    let contents = fs::read_to_string(path)
        .map_err(|error| format!("could not read {}: {error}", path.display()))?;
    let mut config = ScanConfig {
        source_path: path.to_owned(),
        ..ScanConfig::default()
    };
    let mut section = String::new();
    let mut seen = HashSet::new();

    for (index, original) in contents.lines().enumerate() {
        let line_number = index + 1;
        let line = original.split('#').next().unwrap_or_default().trim();
        if line.is_empty() {
            continue;
        }
        if let Some(name) = line
            .strip_prefix('[')
            .and_then(|text| text.strip_suffix(']'))
        {
            section = compact_name(name);
            if section != "filter" && section != "scan" && section != "settings" {
                return Err(line_error(
                    path,
                    line_number,
                    format!("unknown section '{name}'"),
                ));
            }
            continue;
        }
        if section == "filter" {
            config
                .filter
                .push(parse_filter(line).map_err(|error| line_error(path, line_number, error))?);
            continue;
        }

        let (key, value) = line
            .split_once('=')
            .ok_or_else(|| line_error(path, line_number, "expected key=value setting"))?;
        let key_name = compact_name(key);
        if !seen.insert(key_name.clone()) {
            return Err(line_error(
                path,
                line_number,
                format!("duplicate setting '{}'", key.trim()),
            ));
        }
        let value = value.trim();
        let result = match key_name.as_str() {
            "algorithm" => {
                TextureAlgorithm::from_str(value).map(|parsed| config.algorithm = parsed)
            }
            "scanorder" => match value.to_ascii_lowercase().as_str() {
                "linear" | "native" => Ok(config.scan_order = ScanOrder::Linear),
                "spiral" => Ok(config.scan_order = ScanOrder::Spiral),
                _ => Err(format!("invalid scan order '{value}'")),
            },
            "directions" => parse_directions(value).map(|parsed| config.directions = parsed),
            "xrange" | "yrange" | "zrange" => parse_pair(value).map(|(start, end)| {
                let range = IntRange { start, end };
                match key_name.as_str() {
                    "xrange" => config.x_range = range,
                    "yrange" => config.y_range = range,
                    _ => config.z_range = range,
                }
            }),
            "cputilesize" => {
                parse_pair(value).map(|(x, z)| config.cpu_tile_size = TileSize { x, z })
            }
            "cudatilesize" | "gputilesize" => {
                parse_pair(value).map(|(x, z)| config.gpu_tile_size = TileSize { x, z })
            }
            "errortolerance" => value
                .parse::<i32>()
                .map(|parsed| config.error_tolerance = parsed)
                .map_err(|_| format!("invalid error tolerance '{value}'")),
            "verbose" => parse_bool(value).map(|parsed| config.verbose = parsed),
            _ => Err(format!("unknown setting '{}'", key.trim())),
        };
        result.map_err(|error| line_error(path, line_number, error))?;
    }

    validate(&config).map_err(|error| format!("{}: {error}", path.display()))?;
    Ok(config)
}

fn validate(config: &ScanConfig) -> Result<(), String> {
    if config.x_range.start >= config.x_range.end
        || config.y_range.start >= config.y_range.end
        || config.z_range.start >= config.z_range.end
    {
        return Err("scan range starts must be less than their ends".to_owned());
    }
    if config.cpu_tile_size.x <= 0
        || config.cpu_tile_size.z <= 0
        || config.gpu_tile_size.x <= 0
        || config.gpu_tile_size.z <= 0
    {
        return Err("tile size dimensions must be positive".to_owned());
    }
    if config.error_tolerance < 0 {
        return Err("errorTolerance must be non-negative".to_owned());
    }
    if config.filter.is_empty() || config.filter.len() > MAX_FILTER_COUNT {
        return Err(format!(
            "filter must contain between 1 and {MAX_FILTER_COUNT} rows"
        ));
    }
    for &direction in &config.directions {
        for filter in &config.filter {
            let (x, z) = rotate_xz(i32::from(filter.x), i32::from(filter.z), direction);
            if i8::try_from(x).is_err() || i8::try_from(z).is_err() {
                return Err(format!(
                    "direction {direction} rotates a filter offset outside int8 range"
                ));
            }
        }
    }
    Ok(())
}

pub(crate) fn rotate_xz(x: i32, z: i32, direction: i32) -> (i32, i32) {
    match direction / 90 {
        1 => (-z, x),
        2 => (-x, -z),
        3 => (z, -x),
        _ => (x, z),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_existing_configuration() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let config = load(root.join("tests/modern.conf")).unwrap();
        assert_eq!(config.cpu_tile_size, TileSize { x: 7, z: 9 });
        assert_eq!(config.gpu_tile_size, TileSize { x: 70, z: 90 });
        assert_eq!(config.error_tolerance, 2);
        assert_eq!(config.scan_order, ScanOrder::Spiral);
    }

    #[test]
    fn reports_invalid_existing_configurations() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        assert!(
            load(root.join("tests/invalid_duplicate.conf"))
                .unwrap_err()
                .contains("duplicate setting")
        );
        assert!(
            load(root.join("tests/invalid_empty_range.conf"))
                .unwrap_err()
                .contains("starts must be less than their ends")
        );
    }
}
