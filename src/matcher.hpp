#pragma once

#include <cstddef>
#include <cstdint>

#include "textures.hpp"

#if defined(__CUDACC__)
#define CF_MATCHER_HOST_DEVICE __host__ __device__
#define CF_MATCHER_INLINE __forceinline__
#elif defined(_MSC_VER)
#define CF_MATCHER_HOST_DEVICE
#define CF_MATCHER_INLINE __forceinline
#else
#define CF_MATCHER_HOST_DEVICE
#define CF_MATCHER_INLINE inline __attribute__((always_inline))
#endif

template <TextureMode Mode>
CF_MATCHER_HOST_DEVICE CF_MATCHER_INLINE int countBadBlocks(
    std::int32_t x,
    std::int32_t y,
    std::int32_t z,
    const RotationInfo* filter,
    std::size_t filterCount,
    int maxBadBlocks)
{
    int badBlocks = 0;
    // CPU and CUDA share this early-exit rule, keeping tolerance semantics identical.
    for (std::size_t i = 0; i < filterCount; ++i) {
        const RotationInfo info = filter[i];
        const std::uint8_t variant = getTextureForMode<Mode>(
            wrapAdd(x, info.x),
            wrapAdd(y, info.y),
            wrapAdd(z, info.z),
            4);
        if ((variant & info.visibleMask) != info.rotation && ++badBlocks > maxBadBlocks) {
            break;
        }
    }
    return badBlocks;
}

#undef CF_MATCHER_HOST_DEVICE
#undef CF_MATCHER_INLINE
