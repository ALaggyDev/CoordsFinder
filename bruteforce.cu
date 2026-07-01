#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "bruteforce.cuh"
#include "textures.cuh"

const RotationInfo host_filter[] = {
    // (353, -60, -53)
    RotationInfo(-1, 0, 0, 3),
    RotationInfo(-2, 0, 0, 1),
    RotationInfo(-3, 0, 0, 2),
    RotationInfo(-4, 0, 0, 0),
    RotationInfo(-1, 0, 1, 0),
    RotationInfo(-2, 0, 1, 0),
    RotationInfo(-3, 0, 1, 1),
    RotationInfo(-4, 0, 1, 0),
    RotationInfo(0, 0, 0, 0, true),
    RotationInfo(1, 0, 0, 0, true),
    RotationInfo(2, 0, 0, 0, true),
    RotationInfo(3, 0, 0, 1, true),
    RotationInfo(0, 1, 0, 0, true),
    RotationInfo(1, 1, 0, 1, true),
    RotationInfo(2, 1, 0, 0, true),
    RotationInfo(3, 1, 0, 0, true),
    RotationInfo(0, 2, 0, 0, true),
    RotationInfo(1, 2, 0, 1, true),
    RotationInfo(2, 2, 0, 0, true),
    RotationInfo(3, 2, 0, 1, true),
    RotationInfo(0, 3, 0, 0, true),
    RotationInfo(1, 3, 0, 1, true),
    RotationInfo(2, 3, 0, 1, true),
    RotationInfo(3, 3, 0, 1, true),

    // (339, -57, -55)
    // RotationInfo(0, 0, 0, 1),
    // RotationInfo(1, 0, 0, 2),
    // RotationInfo(2, 0, 0, 0),
    // RotationInfo(0, 0, 1, 3),
    // RotationInfo(1, 0, 1, 0),
    // RotationInfo(2, 0, 1, 3),
    // RotationInfo(0, 0, 2, 0),
    // RotationInfo(1, 0, 2, 1),
    // RotationInfo(2, 0, 2, 1),
};

constexpr int FilterCount = sizeof(host_filter) / sizeof(RotationInfo);

__device__ __constant__ RotationInfo dev_filter[FilterCount];

cudaError_t initFilter()
{
    return cudaMemcpyToSymbol(dev_filter, host_filter, sizeof(host_filter));
}

template <TextureMode Mode>
__global__ void bruteForceKernel(Int3 start, Int3 endInclusive, int maxBadBlocks)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x + start.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y + start.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z + start.z;

    if (x > endInclusive.x || y > endInclusive.y || z > endInclusive.z) {
        return;
    }

    int badBlocks = 0;

#pragma unroll
    for (int i = 0; i < FilterCount; i++) {
        const RotationInfo info = dev_filter[i];
        const char textureVariant = getTextureForMode<Mode>(x + info.x, y + info.y, z + info.z, 4);
        const char texture = textureVariant & info.visibleMask;

        if (texture != info.rotation) {
            badBlocks++;
            if (badBlocks > maxBadBlocks) {
                return;
            }
        }
    }

    printf("Found with %d bad block(s)! (%d, %d, %d)\n", badBlocks, x, y, z);
}

cudaError_t launchBruteForce(TextureMode mode, Int3 start, Int3 endInclusive, int maxBadBlocks, dim3 grid, dim3 block)
{
    switch (mode) {
    case TextureModeVanilla12:
        bruteForceKernel<TextureModeVanilla12><<<grid, block>>>(start, endInclusive, maxBadBlocks);
        break;
    case TextureModeVanilla:
        bruteForceKernel<TextureModeVanilla><<<grid, block>>>(start, endInclusive, maxBadBlocks);
        break;
    case TextureModeVanilla21_1:
        bruteForceKernel<TextureModeVanilla21_1><<<grid, block>>>(start, endInclusive, maxBadBlocks);
        break;
    case TextureModeSodium:
        bruteForceKernel<TextureModeSodium><<<grid, block>>>(start, endInclusive, maxBadBlocks);
        break;
    case TextureModeSodium19:
    default:
        bruteForceKernel<TextureModeSodium19><<<grid, block>>>(start, endInclusive, maxBadBlocks);
        break;
    }

    return cudaGetLastError();
}
