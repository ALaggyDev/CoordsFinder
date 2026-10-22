//! Configuration-file parsing and validation.
//!
//! Ranges use half-open `[start, end)` bounds throughout the scanner. Parsing
//! accepts the legacy spelling of several settings, but produces one normalized
//! [`ScanConfig`] for both backends.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use crate::filter::{prepare_filters, rotate_xz};
use crate::types::{Face, Int3, OffsetInfo, RotationInfo, TextureAlgorithm};

/// A half-open integer range used for one scan axis.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IntRange {
    pub start: i32,
    pub end: i32,
}

/// Horizontal dimensions of one independently scheduled scan tile.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TileSize {
    pub x: i32,
    pub z: i32,
}

/// The order in which scan tiles are visited.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScanOrder {
    Linear,
    Spiral,
    ReverseSpiral,
}

/// Fully parsed and validated settings for a coordinate search.
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
    pub offsets: Vec<OffsetInfo>,
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
            gpu_tile_size: TileSize { x: 8192, z: 8192 },
            verbose: false,
            filter: Vec::new(),
            offsets: Vec::new(),
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
    let marker = variant.next().map(str::to_ascii_lowercase);
    if variant.next().is_some() {
        return Err("unexpected extra token in filter row".to_owned());
    }
    let (side, netherrack_face) = match marker.as_deref() {
        None | Some("normal" | "false" | "0") => (false, None),
        Some("side" | "true" | "1") => (true, None),
        Some("netherrack-up") => (false, Some(Face::Up)),
        Some("netherrack-down") => (false, Some(Face::Down)),
        Some("netherrack-north") => (false, Some(Face::North)),
        Some("netherrack-south") => (false, Some(Face::South)),
        Some("netherrack-east") => (false, Some(Face::East)),
        Some("netherrack-west") => (false, Some(Face::West)),
        Some(marker) => return Err(format!("invalid filter marker '{marker}'")),
    };
    let maximum = if side { 1 } else { 3 };
    if rotation > maximum {
        return Err(format!("variant {rotation} exceeds maximum {maximum}"));
    }
    Ok(match netherrack_face {
        Some(face) => RotationInfo::netherrack(
            coordinates[0],
            coordinates[1],
            coordinates[2],
            rotation,
            face,
        ),
        None => RotationInfo::new(
            coordinates[0],
            coordinates[1],
            coordinates[2],
            rotation,
            side,
        ),
    })
}

fn line_error(path: &Path, line: usize, message: impl std::fmt::Display) -> String {
    format!("{}:{line}: {message}", path.display())
}

fn parse_offset(value: &str) -> Result<OffsetInfo, String> {
    let (position, nibbles) = value
        .split_once('|')
        .ok_or("offset rows must be: x y z | offx offy offz [dripstone]")?;
    let position = position
        .split_whitespace()
        .map(str::parse::<i32>)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| "offset positions must fit in signed 32-bit integers")?;
    if position.len() != 3 {
        return Err("offset rows must contain three block coordinates".to_owned());
    }
    let tokens: Vec<_> = nibbles.split_whitespace().collect();
    let dripstone = tokens
        .get(3)
        .is_some_and(|s| s.eq_ignore_ascii_case("dripstone"));
    if tokens.len() != 3 && !(tokens.len() == 4 && dripstone) {
        return Err("expected offx offy offz [dripstone]; use * for an unknown axis".to_owned());
    }
    let mut nibbles = [None; 3];
    for (axis, token) in tokens[..3].iter().enumerate() {
        if *token != "*" {
            nibbles[axis] = Some(
                token
                    .parse::<u8>()
                    .ok()
                    .filter(|n| *n < 16)
                    .ok_or("offset nibbles must be 0..15 or *")?,
            );
        }
    }
    if nibbles.iter().all(Option::is_none) {
        return Err("offset row must constrain at least one axis".to_owned());
    }
    if dripstone && nibbles[1].is_some() {
        return Err("dripstone has no vertical offset; use * for offy".to_owned());
    }
    Ok(OffsetInfo {
        position: Int3 {
            x: position[0],
            y: position[1],
            z: position[2],
        },
        nibbles,
        dripstone,
    })
}

/// Loads, parses, and validates a scan configuration file.
pub fn load(path: impl AsRef<Path>) -> Result<ScanConfig, String> {
    let path = path.as_ref();
    let contents = fs::read_to_string(path)
        .map_err(|error| format!("could not read {}: {error}", path.display()))?;
    parse(&contents, path)
}

fn parse(contents: &str, path: &Path) -> Result<ScanConfig, String> {
    let mut config = ScanConfig {
        source_path: path.to_owned(),
        ..ScanConfig::default()
    };
    let mut section = String::new();
    let mut seen = HashSet::new();

    for (index, original) in contents.lines().enumerate() {
        let line_number = index + 1;
        // Comments are deliberately stripped before section and value parsing.
        let line = original.split('#').next().unwrap_or_default().trim();
        if line.is_empty() {
            continue;
        }
        if let Some(name) = line
            .strip_prefix('[')
            .and_then(|text| text.strip_suffix(']'))
        {
            section = compact_name(name);
            if !matches!(section.as_str(), "filter" | "offsets" | "scan" | "settings") {
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
        if section == "offsets" {
            config
                .offsets
                .push(parse_offset(line).map_err(|error| line_error(path, line_number, error))?);
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
                "linear" | "native" => {
                    config.scan_order = ScanOrder::Linear;
                    Ok(())
                }
                "spiral" => {
                    config.scan_order = ScanOrder::Spiral;
                    Ok(())
                }
                "reverse-spiral" => {
                    config.scan_order = ScanOrder::ReverseSpiral;
                    Ok(())
                }
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

    let offsets_only = config.filter.is_empty() && !config.offsets.is_empty();
    // Offsets cannot resolve height, so report each X/Z once when Y is omitted.
    if offsets_only && !seen.contains("yrange") {
        config.y_range = IntRange { start: 0, end: 1 };
    }
    for required in ["algorithm", "xrange", "yrange", "zrange", "errortolerance"] {
        if offsets_only && matches!(required, "algorithm" | "yrange" | "errortolerance") {
            continue;
        }
        if !seen.contains(required) {
            return Err(format!(
                "{}: missing required setting {required}",
                path.display()
            ));
        }
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
    if config.filter.is_empty() && config.offsets.is_empty() {
        return Err("filter or offsets must contain at least one row".to_owned());
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
    prepare_filters(
        &config.filter,
        config.algorithm,
        &config.directions,
        config.error_tolerance,
    )?;
    if !config.offsets.is_empty() {
        crate::offset::prepare(config)?;
    }
    Ok(())
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
    fn parses_reverse_spiral_scan_order() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let config = load(root.join("tests/reverse_spiral.conf")).unwrap();
        assert_eq!(config.scan_order, ScanOrder::ReverseSpiral);
    }

    #[test]
    fn parses_netherrack_face_markers() {
        let parsed = parse_filter("1 -2 3 | 3 netherrack-west").unwrap();
        assert_eq!(parsed, RotationInfo::netherrack(1, -2, 3, 3, Face::West));
        assert!(parse_filter("0 0 0 | 4 netherrack-up").is_err());
        assert!(parse_filter("0 0 0 | 0 netherrack").is_err());
    }

    #[test]
    fn parses_partial_xyz_and_dripstone_offsets() {
        let config =
            load(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/offsets.conf")).unwrap();
        assert!(config.filter.is_empty());
        assert_eq!(config.y_range, IntRange { start: 0, end: 1 });
        assert_eq!(config.error_tolerance, 0);
        assert_eq!(config.offsets.len(), 5);
        assert_eq!(config.offsets[0].nibbles, [Some(11), Some(11), Some(5)]);
        let partial = parse_offset("300 -9 -400 | 3 * 12 dripstone").unwrap();
        assert_eq!(
            partial.position,
            Int3 {
                x: 300,
                y: -9,
                z: -400
            }
        );
        assert_eq!(partial.nibbles, [Some(3), None, Some(12)]);
        assert!(partial.dripstone);
        for row in [
            "0 0 | 1 * 2",
            "0 0 0 | 16 * 2",
            "0 0 0 | -1 * 2",
            "0 0 0 | * * *",
            "0 0 0 | 1 2 3 dripstone",
            "0 0 0 | 1 * 2 flower",
            "0 0 0 | 1 *",
            "0 0 0 | 1 * 2 dripstone extra",
        ] {
            assert!(parse_offset(row).is_err(), "{row}");
        }
    }

    #[test]
    fn offset_only_settings_remain_optional_and_overridable() {
        let settings = "xRange = (0, 1)\nzRange = (0, 1)\n";
        let offsets = "[offsets]\n0 0 0 | 0 * 0\n";
        let path = Path::new("test.conf");
        let config = parse(&format!("{settings}{offsets}"), path).unwrap();
        assert_eq!(config.algorithm, TextureAlgorithm::Vanilla3);
        let config = parse(
            &format!(
                "{settings}algorithm = Sodium-2\nyRange = (-4, -2)\nerrorTolerance = 2\n{offsets}"
            ),
            path,
        )
        .unwrap();
        assert_eq!(config.algorithm, TextureAlgorithm::Sodium2);
        assert_eq!(config.y_range, IntRange { start: -4, end: -2 });
        assert_eq!(config.error_tolerance, 2);

        // Texture-only and mixed searches still require all texture settings.
        for observations in [
            "[filter]\n0 0 0 | 0\n".to_owned(),
            format!("{offsets}[filter]\n0 0 0 | 0\n"),
        ] {
            for (setting, missing) in [
                ("yRange = (0, 1)\nerrorTolerance = 0\n", "algorithm"),
                ("algorithm = Vanilla-3\nerrorTolerance = 0\n", "yrange"),
                ("algorithm = Vanilla-3\nyRange = (0, 1)\n", "errortolerance"),
            ] {
                assert!(
                    parse(&format!("{settings}{setting}{observations}"), path)
                        .unwrap_err()
                        .contains(&format!("missing required setting {missing}"))
                );
            }
        }
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
