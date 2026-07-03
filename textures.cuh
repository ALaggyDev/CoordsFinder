#pragma once

#include <stdint.h>

#include "cuda_runtime.h"

enum TextureMode {
    TextureModeVanilla1,
    TextureModeVanilla2,
    TextureModeVanilla3,
    TextureModeSodium1,
    TextureModeSodium2
};

namespace texture_detail {
constexpr int64_t JavaMultiplier = 0x5DEECE66DLL;
constexpr int64_t JavaMask = (1LL << 48) - 1;
constexpr int64_t SodiumPhi = 0x9E3779B97F4A7C15LL;

__host__ __device__ __forceinline__ int64_t unsignedShiftRight(int64_t value, int distance)
{
    return static_cast<int64_t>(static_cast<uint64_t>(value) >> distance);
}

__host__ __device__ __forceinline__ int64_t rotateLeft64(int64_t value, int distance)
{
    const uint64_t bits = static_cast<uint64_t>(value);
    return static_cast<int64_t>((bits << distance) | (bits >> (64 - distance)));
}

__host__ __device__ __forceinline__ int positiveModulo(int value, char mod)
{
    const int64_t wide = value;
    const int64_t positive = wide < 0 ? -wide : wide;
    return static_cast<int>(positive % mod);
}

__host__ __device__ __forceinline__ int64_t staffordMix13(int64_t z)
{
    z = (z ^ unsignedShiftRight(z, 30)) * 0xBF58476D1CE4E5B9LL;
    z = (z ^ unsignedShiftRight(z, 27)) * 0x94D049BB133111EBLL;
    return z ^ unsignedShiftRight(z, 31);
}

__host__ __device__ __forceinline__ int64_t coordinateRandomRaw(int x, int y, int z)
{
    int64_t seed = static_cast<int64_t>(x * 3129871) ^ (static_cast<int64_t>(z) * 116129781LL) ^ static_cast<int64_t>(y);

    seed = seed * seed * 42317861LL + seed * 11LL;
    return seed;
}

__host__ __device__ __forceinline__ int coordinateRandomLegacy(int x, int y, int z)
{
    return static_cast<int>(coordinateRandomRaw(x, y, z)) >> 16;
}

__host__ __device__ __forceinline__ int64_t coordinateRandom(int x, int y, int z)
{
    return coordinateRandomRaw(x, y, z) >> 16;
}

__host__ __device__ __forceinline__ int randomVanilla2(int64_t seed)
{
    seed = (seed ^ JavaMultiplier) & JavaMask;
    return static_cast<int>(unsignedShiftRight(seed * 0xBB20B4600A69LL + 0x40942DE6BALL, 16));
}

__host__ __device__ __forceinline__ int64_t legacyScrambleSeed(int64_t seed)
{
    return (seed ^ JavaMultiplier) & JavaMask;
}

__host__ __device__ __forceinline__ int legacyNextBits(int64_t& seed, int bits)
{
    seed = (seed * JavaMultiplier + 11LL) & JavaMask;
    return static_cast<int>(unsignedShiftRight(seed, 48 - bits));
}

__host__ __device__ __forceinline__ int legacyNextInt(int64_t seed, char bound)
{
    seed = legacyScrambleSeed(seed);

    const int intBound = static_cast<int>(bound);
    if ((intBound & -intBound) == intBound) {
        return static_cast<int>((static_cast<int64_t>(intBound) * legacyNextBits(seed, 31)) >> 31);
    }

    int bits = legacyNextBits(seed, 31);
    int value = bits % intBound;
    while (static_cast<int64_t>(bits) - value + (intBound - 1) < 0) {
        bits = legacyNextBits(seed, 31);
        value = bits % intBound;
    }
    return value;
}

__host__ __device__ __forceinline__ int randomSodium1(int64_t seed)
{
    seed ^= unsignedShiftRight(seed, 33);
    seed *= 0xff51afd7ed558ccdLL;
    seed ^= unsignedShiftRight(seed, 33);
    seed *= 0xc4ceb9fe1a85ec53LL;
    seed ^= unsignedShiftRight(seed, 33);

    const int64_t rand1 = staffordMix13(seed + SodiumPhi);
    const int64_t rand2 = staffordMix13(seed + SodiumPhi + SodiumPhi);
    return static_cast<int>(rand1 + rand2);
}

__host__ __device__ __forceinline__ int randomSodium2(int64_t seed)
{
    int64_t low = seed ^ 7640891576956012809LL;
    int64_t high = low - 7046029254386353131LL;

    low = staffordMix13(low);
    high = staffordMix13(high);

    return static_cast<int>(rotateLeft64(low + high, 17) + low);
}
}

__host__ __device__ __forceinline__ char getTextureVanilla1(int x, int y, int z, char variantCount)
{
    return static_cast<char>(texture_detail::positiveModulo(texture_detail::coordinateRandomLegacy(x, y, z), variantCount));
}

__host__ __device__ __forceinline__ char getTextureVanilla2(int x, int y, int z, char variantCount)
{
    return static_cast<char>(texture_detail::positiveModulo(texture_detail::randomVanilla2(texture_detail::coordinateRandom(x, y, z)), variantCount));
}

__host__ __device__ __forceinline__ char getTextureVanilla3(int x, int y, int z, char variantCount)
{
    return static_cast<char>(texture_detail::legacyNextInt(texture_detail::coordinateRandom(x, y, z), variantCount));
}

__host__ __device__ __forceinline__ char getTextureSodium1(int x, int y, int z, char variantCount)
{
    return static_cast<char>(texture_detail::positiveModulo(texture_detail::randomSodium1(texture_detail::coordinateRandom(x, y, z)), variantCount));
}

__host__ __device__ __forceinline__ char getTextureSodium2(int x, int y, int z, char variantCount)
{
    return static_cast<char>(texture_detail::positiveModulo(texture_detail::randomSodium2(texture_detail::coordinateRandom(x, y, z)), variantCount));
}

template <TextureMode Mode>
struct TextureSampler;

template <>
struct TextureSampler<TextureModeVanilla1> {
    __device__ __forceinline__ static char sample(int x, int y, int z, char variantCount)
    {
        return getTextureVanilla1(x, y, z, variantCount);
    }
};

template <>
struct TextureSampler<TextureModeVanilla2> {
    __device__ __forceinline__ static char sample(int x, int y, int z, char variantCount)
    {
        return getTextureVanilla2(x, y, z, variantCount);
    }
};

template <>
struct TextureSampler<TextureModeVanilla3> {
    __device__ __forceinline__ static char sample(int x, int y, int z, char variantCount)
    {
        return getTextureVanilla3(x, y, z, variantCount);
    }
};

template <>
struct TextureSampler<TextureModeSodium1> {
    __device__ __forceinline__ static char sample(int x, int y, int z, char variantCount)
    {
        return getTextureSodium1(x, y, z, variantCount);
    }
};

template <>
struct TextureSampler<TextureModeSodium2> {
    __device__ __forceinline__ static char sample(int x, int y, int z, char variantCount)
    {
        return getTextureSodium2(x, y, z, variantCount);
    }
};

template <TextureMode Mode>
__device__ __forceinline__ char getTextureForMode(int x, int y, int z, char variantCount)
{
    return TextureSampler<Mode>::sample(x, y, z, variantCount);
}

__host__ __device__ __forceinline__ char getTexture(TextureMode mode, int x, int y, int z, char variantCount)
{
    switch (mode) {
    case TextureModeVanilla1:
        return getTextureVanilla1(x, y, z, variantCount);
    case TextureModeVanilla2:
        return getTextureVanilla2(x, y, z, variantCount);
    case TextureModeVanilla3:
        return getTextureVanilla3(x, y, z, variantCount);
    case TextureModeSodium1:
        return getTextureSodium1(x, y, z, variantCount);
    case TextureModeSodium2:
    default:
        return getTextureSodium2(x, y, z, variantCount);
    }
}
