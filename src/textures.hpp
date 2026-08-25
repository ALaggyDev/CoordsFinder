#pragma once

#include <cstdint>

#include "types.hpp"

#if defined(__CUDACC__)
#define CF_HOST_DEVICE __host__ __device__
#define CF_FORCE_INLINE __forceinline__
#elif defined(__HIP__)
// hipcc (clang-based) never defines __CUDACC__, so HIP needs its own branch.
#define CF_HOST_DEVICE __host__ __device__
#define CF_FORCE_INLINE inline __attribute__((always_inline))
#elif defined(_MSC_VER)
#define CF_HOST_DEVICE
#define CF_FORCE_INLINE __forceinline
#else
#define CF_HOST_DEVICE
#define CF_FORCE_INLINE inline __attribute__((always_inline))
#endif

// This file is ... written in a complex way. The reasons for this complexity are mainly due to:
// - Signed integer overflow is considered as an undefined behaviour in C/C++ (which I completely disagree with).
//     As a result, most of the math operations are done in unsigned instead of signed integers.
//     Special sign extend and wrap add functions are needed to circumvent this issue.
//
// - static_cast<...>(...) is very verbose. I would very much appreciate the classic C cast, but
//     unfortunately this is C++ and the "recommended way to write C++" is apparently to use static_cast.
//
// - If it wasn't obvious enough, a majority of the code is written by AIs.
//     AIs apparently prefer absolutely-correct code over clean code.
//
// In short, blame C/C++ and AIs.
//
// Reference: https://github.com/19MisterX98/TextureRotations/tree/master/src/main/java/texture

namespace texture_detail {
constexpr std::uint64_t JavaMultiplier = 0x5DEECE66DULL;
constexpr std::uint64_t JavaMask = (1ULL << 48) - 1;
constexpr std::uint64_t SodiumPhi = 0x9E3779B97F4A7C15ULL;

CF_HOST_DEVICE CF_FORCE_INLINE std::uint64_t signExtend32(std::uint32_t bits)
{
    return static_cast<std::uint64_t>(
        static_cast<std::int64_t>(static_cast<std::int32_t>(bits)));
}

CF_HOST_DEVICE CF_FORCE_INLINE std::uint8_t absoluteModulo(std::int32_t value, std::uint8_t mod)
{
    // Widen before negating so INT32_MIN remains representable.
    const std::uint32_t magnitude = static_cast<std::uint32_t>(
        value < 0 ? -static_cast<std::int64_t>(value) : value);
    return static_cast<std::uint8_t>(magnitude % mod);
}

CF_HOST_DEVICE CF_FORCE_INLINE std::uint64_t rotateLeft64(std::uint64_t bits, int distance)
{
    return (bits << distance) | (bits >> (64 - distance));
}

CF_HOST_DEVICE CF_FORCE_INLINE std::uint64_t staffordMix13(std::uint64_t bits)
{
    bits = (bits ^ (bits >> 30)) * 0xBF58476D1CE4E5B9ULL;
    bits = (bits ^ (bits >> 27)) * 0x94D049BB133111EBULL;
    return bits ^ (bits >> 31);
}

CF_HOST_DEVICE CF_FORCE_INLINE std::uint64_t coordinateRandomRaw(std::int32_t x, std::int32_t y, std::int32_t z)
{
    // Minecraft multiplies X as a 32-bit Java int before promoting it to long.
    const std::uint32_t xProduct = static_cast<std::uint32_t>(x) * 3129871U;
    std::uint64_t seed = signExtend32(xProduct)
        ^ (signExtend32(static_cast<std::uint32_t>(z)) * 116129781ULL)
        ^ signExtend32(static_cast<std::uint32_t>(y));

    seed = seed * seed * 42317861ULL + seed * 11ULL;
    return seed;
}

CF_HOST_DEVICE CF_FORCE_INLINE std::uint32_t coordinateRandomLegacy(std::int32_t x, std::int32_t y, std::int32_t z)
{
    return static_cast<std::uint32_t>(
        static_cast<std::int32_t>(coordinateRandomRaw(x, y, z)) >> 16);
}

CF_HOST_DEVICE CF_FORCE_INLINE std::uint64_t coordinateRandom(std::int32_t x, std::int32_t y, std::int32_t z)
{
    return static_cast<std::uint64_t>(
        static_cast<std::int64_t>(coordinateRandomRaw(x, y, z)) >> 16);
}

CF_HOST_DEVICE CF_FORCE_INLINE std::uint32_t randomVanilla2(std::uint64_t seed)
{
    seed = (seed ^ JavaMultiplier) & JavaMask;
    return static_cast<std::uint32_t>((seed * 0xBB20B4600A69ULL + 0x40942DE6BAULL) >> 16);
}

CF_HOST_DEVICE CF_FORCE_INLINE std::uint32_t legacyNextBits(std::uint64_t& seed, int bits)
{
    seed = (seed * JavaMultiplier + 11ULL) & JavaMask;
    return static_cast<std::uint32_t>(seed >> (48 - bits));
}

CF_HOST_DEVICE CF_FORCE_INLINE std::uint8_t legacyNextInt(std::uint64_t seed, std::uint8_t bound)
{
    seed = (seed ^ JavaMultiplier) & JavaMask;
    const std::uint32_t intBound = bound;

    if ((intBound & (0U - intBound)) == intBound) {
        // Power-of-two bound: the multiply-high form is exact, and for the
        // hot variants==4 call site the shift count folds to a constant.
        return static_cast<std::uint8_t>((static_cast<std::uint64_t>(intBound) * legacyNextBits(seed, 31)) >> 31);
    }

    std::uint32_t bits = legacyNextBits(seed, 31);
    std::uint32_t value = bits % intBound;
    while (static_cast<std::int32_t>(bits - value + intBound - 1) < 0) {
        bits = legacyNextBits(seed, 31);
        value = bits % intBound;
    }
    return static_cast<std::uint8_t>(value);
}

CF_HOST_DEVICE CF_FORCE_INLINE std::uint32_t randomSodium1(std::uint64_t seed)
{
    seed ^= seed >> 33;
    seed *= 0xff51afd7ed558ccdULL;
    seed ^= seed >> 33;
    seed *= 0xc4ceb9fe1a85ec53ULL;
    seed ^= seed >> 33;

    const std::uint64_t rand1 = staffordMix13(seed + SodiumPhi);
    const std::uint64_t rand2 = staffordMix13(seed + SodiumPhi + SodiumPhi);
    return static_cast<std::uint32_t>(rand1 + rand2);
}

CF_HOST_DEVICE CF_FORCE_INLINE std::uint32_t randomSodium2(std::uint64_t seed)
{
    std::uint64_t low = seed ^ 7640891576956012809ULL;
    std::uint64_t high = low - 7046029254386353131ULL;
    low = staffordMix13(low);
    high = staffordMix13(high);
    return static_cast<std::uint32_t>(rotateLeft64(low + high, 17) + low);
}
}

CF_HOST_DEVICE CF_FORCE_INLINE std::int32_t wrapAdd(std::int32_t value, std::int8_t offset)
{
    // Candidate-plus-filter coordinates follow Java int overflow at world limits.
    return static_cast<std::int32_t>(
        static_cast<std::uint32_t>(value) + static_cast<std::uint32_t>(offset));
}

template <TextureAlgorithm Mode>
struct TextureSampler;

template <>
struct TextureSampler<TextureAlgorithm::Vanilla1> {
    CF_HOST_DEVICE CF_FORCE_INLINE static std::uint8_t sample(std::int32_t x, std::int32_t y, std::int32_t z, std::uint8_t variants)
    {
        // Note Vanilla-1 deliberately narrows to int32 BEFORE shifting (legacy
        // behaviour: only seed bits 16..31 reach the modulo), while every later
        // algorithm arithmetic-shifts the full 64-bit value.
        return texture_detail::absoluteModulo(
            static_cast<std::int32_t>(texture_detail::coordinateRandomLegacy(x, y, z)), variants);
    }
};

template <>
struct TextureSampler<TextureAlgorithm::Vanilla2> {
    CF_HOST_DEVICE CF_FORCE_INLINE static std::uint8_t sample(std::int32_t x, std::int32_t y, std::int32_t z, std::uint8_t variants)
    {
        return texture_detail::absoluteModulo(
            static_cast<std::int32_t>(texture_detail::randomVanilla2(texture_detail::coordinateRandom(x, y, z))), variants);
    }
};

template <>
struct TextureSampler<TextureAlgorithm::Vanilla3> {
    CF_HOST_DEVICE CF_FORCE_INLINE static std::uint8_t sample(std::int32_t x, std::int32_t y, std::int32_t z, std::uint8_t variants)
    {
        return texture_detail::legacyNextInt(texture_detail::coordinateRandom(x, y, z), variants);
    }
};

template <>
struct TextureSampler<TextureAlgorithm::Sodium1> {
    CF_HOST_DEVICE CF_FORCE_INLINE static std::uint8_t sample(std::int32_t x, std::int32_t y, std::int32_t z, std::uint8_t variants)
    {
        return texture_detail::absoluteModulo(
            static_cast<std::int32_t>(texture_detail::randomSodium1(texture_detail::coordinateRandom(x, y, z))), variants);
    }
};

template <>
struct TextureSampler<TextureAlgorithm::Sodium2> {
    CF_HOST_DEVICE CF_FORCE_INLINE static std::uint8_t sample(std::int32_t x, std::int32_t y, std::int32_t z, std::uint8_t variants)
    {
        return texture_detail::absoluteModulo(
            static_cast<std::int32_t>(texture_detail::randomSodium2(texture_detail::coordinateRandom(x, y, z))), variants);
    }
};

template <TextureAlgorithm Mode>
CF_HOST_DEVICE CF_FORCE_INLINE std::uint8_t getTextureForMode(
    std::int32_t x,
    std::int32_t y,
    std::int32_t z,
    std::uint8_t variants)
{
    return TextureSampler<Mode>::sample(x, y, z, variants);
}

inline std::uint8_t getTexture(
    TextureAlgorithm mode,
    std::int32_t x,
    std::int32_t y,
    std::int32_t z,
    std::uint8_t variants)
{
    switch (mode) {
    case TextureAlgorithm::Vanilla1:
        return getTextureForMode<TextureAlgorithm::Vanilla1>(x, y, z, variants);
    case TextureAlgorithm::Vanilla2:
        return getTextureForMode<TextureAlgorithm::Vanilla2>(x, y, z, variants);
    case TextureAlgorithm::Vanilla3:
        return getTextureForMode<TextureAlgorithm::Vanilla3>(x, y, z, variants);
    case TextureAlgorithm::Sodium1:
        return getTextureForMode<TextureAlgorithm::Sodium1>(x, y, z, variants);
    case TextureAlgorithm::Sodium2:
    default:
        return getTextureForMode<TextureAlgorithm::Sodium2>(x, y, z, variants);
    }
}

#undef CF_HOST_DEVICE
#undef CF_FORCE_INLINE
