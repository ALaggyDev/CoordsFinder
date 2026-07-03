#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "bruteforce.cuh"
#include "textures.cuh"

__device__ __constant__ RotationInfo dev_filter[MaxFilterCount];

cudaError_t initFilter(const RotationInfo* hostFilter, int filterCount)
{
    if (filterCount < 0 || filterCount > MaxFilterCount) {
        return cudaErrorInvalidValue;
    }

    return cudaMemcpyToSymbol(dev_filter, hostFilter, sizeof(RotationInfo) * filterCount);
}

template <TextureMode Mode>
__global__ void bruteForceKernel(Int3 start, Int3 endInclusive, int maxBadBlocks, int filterCount)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x + start.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y + start.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z + start.z;

    if (x > endInclusive.x || y > endInclusive.y || z > endInclusive.z) {
        return;
    }

    int badBlocks = 0;

    for (int i = 0; i < filterCount; i++) {
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

cudaError_t launchBruteForce(
    TextureMode mode,
    Int3 start,
    Int3 endInclusive,
    int maxBadBlocks,
    int filterCount,
    dim3 grid,
    dim3 block)
{
    switch (mode) {
    case TextureModeVanilla12:
        bruteForceKernel<TextureModeVanilla12><<<grid, block>>>(start, endInclusive, maxBadBlocks, filterCount);
        break;
    case TextureModeVanilla:
        bruteForceKernel<TextureModeVanilla><<<grid, block>>>(start, endInclusive, maxBadBlocks, filterCount);
        break;
    case TextureModeVanilla21_1:
        bruteForceKernel<TextureModeVanilla21_1><<<grid, block>>>(start, endInclusive, maxBadBlocks, filterCount);
        break;
    case TextureModeSodium:
        bruteForceKernel<TextureModeSodium><<<grid, block>>>(start, endInclusive, maxBadBlocks, filterCount);
        break;
    case TextureModeSodium19:
    default:
        bruteForceKernel<TextureModeSodium19><<<grid, block>>>(start, endInclusive, maxBadBlocks, filterCount);
        break;
    }

    return cudaGetLastError();
}
