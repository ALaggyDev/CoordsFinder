//! Minecraft texture variant algorithms.
//!
//! The structure follows TextureRotations' small Java implementations. Rust's
//! explicit wrapping methods preserve Java's defined two's-complement overflow.

use crate::types::TextureAlgorithm;

const JAVA_MULTIPLIER: u64 = 0x5deece66d;
const JAVA_MASK: u64 = (1 << 48) - 1;
const SODIUM_PHI: u64 = 0x9e3779b97f4a7c15;

#[inline(always)]
#[lower::apply(wrapping)]
fn coord_seed_legacy(x: i32, y: i32, z: i32) -> i32 {
    let seed = (x * 3_129_871) ^ (z * 116_129_781) ^ y;
    (seed * seed * 42_317_861 + seed * 11) >> 16
}

#[inline(always)]
#[lower::apply(wrapping)]
fn coord_seed(x: i32, y: i32, z: i32) -> i64 {
    let seed = ((x * 3_129_871) as i64) ^ ((z as i64) * 116_129_781) ^ (y as i64);
    (seed * seed * 42_317_861 + seed * 11) >> 16
}

#[inline(always)]
fn absolute_modulo(value: i32, modulus: u8) -> u8 {
    (value.unsigned_abs() % u32::from(modulus)) as u8
}

#[inline(always)]
#[lower::apply(wrapping)]
fn java_next_long(seed: i64) -> i32 {
    let seed = ((seed as u64) ^ JAVA_MULTIPLIER) & JAVA_MASK;
    ((seed * 0xbb20b4600a69 + 0x40942de6ba) >> 16) as i32
}

#[inline(always)]
#[lower::apply(wrapping)]
fn java_next_int(seed: i64, bound: u32) -> u32 {
    let mut seed = ((seed as u64) ^ JAVA_MULTIPLIER) & JAVA_MASK;

    if bound.is_power_of_two() {
        seed = ((seed * JAVA_MULTIPLIER) + 11) & JAVA_MASK;
        return (seed >> (48u32 - bound.ilog2())) as u32;
    }

    loop {
        seed = ((seed * JAVA_MULTIPLIER) + 11) & JAVA_MASK;
        let bits = (seed >> 17) as u32;
        let value = bits % bound;
        if (bits - value + bound - 1) as i32 >= 0 {
            return value;
        }
    }
}

#[inline(always)]
#[lower::apply(wrapping)]
fn stafford_mix13(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9;
    value = (value ^ (value >> 27)) * 0x94d049bb133111eb;
    value ^ (value >> 31)
}

#[inline(always)]
#[lower::apply(wrapping)]
fn murmurhash3(seed: i64) -> i32 {
    let mut seed = seed as u64;
    seed ^= seed >> 33;
    seed = seed * 0xff51afd7ed558ccd;
    seed ^= seed >> 33;
    seed = seed * 0xc4ceb9fe1a85ec53;
    seed ^= seed >> 33;
    let first = stafford_mix13(seed + SODIUM_PHI);
    let second = stafford_mix13(seed + SODIUM_PHI * 2);
    (first + second) as i32
}

#[inline(always)]
#[lower::apply(wrapping)]
fn xoroshiro(seed: i64) -> i32 {
    let seed = seed as u64;
    let low = stafford_mix13(seed ^ 7_640_891_576_956_012_809);
    let high = stafford_mix13((seed ^ 7_640_891_576_956_012_809) - 7_046_029_254_386_353_131);
    ((low + high).rotate_left(17) + low) as i32
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

sampler!(Vanilla1, |x, y, z, variants| {
    let seed = coord_seed_legacy(x, y, z);
    absolute_modulo(seed, variants)
});
sampler!(Vanilla2, |x, y, z, variants| {
    let seed = coord_seed(x, y, z);
    absolute_modulo(java_next_long(seed), variants)
});
sampler!(Vanilla3, |x, y, z, variants| {
    let seed = coord_seed(x, y, z);
    java_next_int(seed, variants as u32) as u8
});
sampler!(Sodium1, |x, y, z, variants| {
    let seed = coord_seed(x, y, z);
    absolute_modulo(murmurhash3(seed), variants)
});
sampler!(Sodium2, |x, y, z, variants| {
    let seed = coord_seed(x, y, z);
    absolute_modulo(xoroshiro(seed), variants)
});

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
