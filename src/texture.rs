//! Minecraft texture variant algorithms.
//!
//! The structure follows TextureRotations' small Java implementations. Rust's
//! explicit wrapping methods preserve Java's defined two's-complement overflow.

use crate::types::TextureAlgorithm;

const JAVA_MULTIPLIER: u64 = 0x5deece66d;
const JAVA_MASK: u64 = (1 << 48) - 1;
const SODIUM_PHI: u64 = 0x9e3779b97f4a7c15;

#[inline(always)]
fn coordinate_random_raw(x: i32, y: i32, z: i32) -> i64 {
    let seed = (x.wrapping_mul(3_129_871) as i64) ^ (z as i64).wrapping_mul(116_129_781) ^ y as i64;
    seed.wrapping_mul(seed)
        .wrapping_mul(42_317_861)
        .wrapping_add(seed.wrapping_mul(11))
}

#[inline(always)]
fn coordinate_random_legacy(x: i32, y: i32, z: i32) -> i32 {
    (coordinate_random_raw(x, y, z) as i32) >> 16
}

#[inline(always)]
fn coordinate_random(x: i32, y: i32, z: i32) -> i64 {
    coordinate_random_raw(x, y, z) >> 16
}

#[inline(always)]
fn absolute_modulo(value: i32, modulus: u8) -> u8 {
    (value.unsigned_abs() % u32::from(modulus)) as u8
}

#[inline(always)]
fn stafford_mix13(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

#[inline(always)]
fn random_vanilla2(seed: i64) -> i32 {
    let seed = ((seed as u64) ^ JAVA_MULTIPLIER) & JAVA_MASK;
    (seed.wrapping_mul(0xbb20b4600a69).wrapping_add(0x40942de6ba) >> 16) as i32
}

#[inline(always)]
fn legacy_next_bits(seed: &mut u64, bits: u32) -> u32 {
    *seed = seed.wrapping_mul(JAVA_MULTIPLIER).wrapping_add(11) & JAVA_MASK;
    (*seed >> (48 - bits)) as u32
}

#[inline(always)]
fn legacy_next_int(seed: i64, bound: u8) -> u8 {
    let mut seed = ((seed as u64) ^ JAVA_MULTIPLIER) & JAVA_MASK;
    let bound = u32::from(bound);
    if bound.is_power_of_two() {
        return ((u64::from(bound) * u64::from(legacy_next_bits(&mut seed, 31))) >> 31) as u8;
    }

    loop {
        let bits = legacy_next_bits(&mut seed, 31);
        let value = bits % bound;
        if bits.wrapping_sub(value).wrapping_add(bound - 1) as i32 >= 0 {
            return value as u8;
        }
    }
}

#[inline(always)]
fn random_sodium1(mut seed: u64) -> i32 {
    seed ^= seed >> 33;
    seed = seed.wrapping_mul(0xff51afd7ed558ccd);
    seed ^= seed >> 33;
    seed = seed.wrapping_mul(0xc4ceb9fe1a85ec53);
    seed ^= seed >> 33;
    let first = stafford_mix13(seed.wrapping_add(SODIUM_PHI));
    let second = stafford_mix13(seed.wrapping_add(SODIUM_PHI).wrapping_add(SODIUM_PHI));
    first.wrapping_add(second) as i32
}

#[inline(always)]
fn random_sodium2(seed: u64) -> i32 {
    let low = stafford_mix13(seed ^ 7_640_891_576_956_012_809);
    let high = stafford_mix13(
        (seed ^ 7_640_891_576_956_012_809).wrapping_add_signed(-7_046_029_254_386_353_131),
    );
    low.wrapping_add(high).rotate_left(17).wrapping_add(low) as i32
}

/// A compile-time texture sampler used by performance-sensitive scan loops.
///
/// Each algorithm is represented by a zero-sized type. Making the CPU scanner
/// generic over this trait lets LLVM inline the selected algorithm rather than
/// branching once for every filter sample.
pub trait TextureSampler: Send + Sync {
    fn sample(x: i32, y: i32, z: i32, variants: u8) -> u8;
}

macro_rules! sampler {
    ($name:ident, $body:expr) => {
        #[derive(Clone, Copy, Debug)]
        pub struct $name;

        impl TextureSampler for $name {
            #[inline(always)]
            fn sample(x: i32, y: i32, z: i32, variants: u8) -> u8 {
                debug_assert!(variants > 0);
                $body(x, y, z, variants)
            }
        }
    };
}

sampler!(Vanilla1, |x, y, z, variants| absolute_modulo(
    coordinate_random_legacy(x, y, z),
    variants
));
sampler!(Vanilla2, |x, y, z, variants| absolute_modulo(
    random_vanilla2(coordinate_random(x, y, z)),
    variants
));
sampler!(Vanilla3, |x, y, z, variants| legacy_next_int(
    coordinate_random(x, y, z),
    variants
));
sampler!(Sodium1, |x, y, z, variants| absolute_modulo(
    random_sodium1(coordinate_random(x, y, z) as u64),
    variants
));
sampler!(Sodium2, |x, y, z, variants| absolute_modulo(
    random_sodium2(coordinate_random(x, y, z) as u64),
    variants
));

/// Samples an algorithm selected at runtime.
///
/// Scanners should dispatch once and use [`TextureSampler`] instead. This
/// convenience function is useful for tests and non-hot-path callers.
pub fn get_texture(algorithm: TextureAlgorithm, x: i32, y: i32, z: i32, variants: u8) -> u8 {
    assert!(variants > 0, "variant count must be positive");
    match algorithm {
        TextureAlgorithm::Vanilla1 => Vanilla1::sample(x, y, z, variants),
        TextureAlgorithm::Vanilla2 => Vanilla2::sample(x, y, z, variants),
        TextureAlgorithm::Vanilla3 => Vanilla3::sample(x, y, z, variants),
        TextureAlgorithm::Sodium1 => Sodium1::sample(x, y, z, variants),
        TextureAlgorithm::Sodium2 => Sodium2::sample(x, y, z, variants),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_texture_rotations_reference_vectors() {
        let algorithms = [
            TextureAlgorithm::Vanilla1,
            TextureAlgorithm::Vanilla2,
            TextureAlgorithm::Vanilla3,
            TextureAlgorithm::Sodium1,
            TextureAlgorithm::Sodium2,
        ];
        let vectors = [
            (0, 0, 0, [0, 0, 2, 3, 2]),
            (1, 2, 3, [0, 2, 3, 2, 0]),
            (-1, -2, -3, [3, 3, 0, 1, 0]),
            (353, -60, -53, [2, 2, 0, 1, 3]),
            (-29_999_984, -64, 29_999_983, [1, 2, 1, 3, 0]),
            (29_999_999, 319, -29_999_999, [3, 3, 2, 3, 3]),
            (-538, 67, -575, [3, 3, 1, 0, 2]),
            (17, -4, -31, [3, 0, 3, 0, 3]),
            (1_000_000, 319, -1_000_000, [0, 0, 2, 0, 0]),
            (-30_000_000, -64, 30_000_000, [0, 2, 0, 0, 2]),
            (1_234_567, 72, -7_654_321, [0, 1, 2, 2, 1]),
            (-16_777_216, 255, 16_777_215, [3, 2, 3, 1, 1]),
            (31, 63, 127, [1, 1, 1, 2, 2]),
            (-32, -64, -128, [1, 2, 0, 0, 0]),
            (4096, 0, 4096, [3, 0, 3, 2, 3]),
            (-4096, 1, -4096, [1, 1, 0, 2, 1]),
        ];

        for (x, y, z, expected) in vectors {
            for (index, algorithm) in algorithms.into_iter().enumerate() {
                assert_eq!(get_texture(algorithm, x, y, z, 4), expected[index]);
            }
        }
    }
}
