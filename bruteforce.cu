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

__global__ void bruteForce(TextureMode mode, Int3 start, Int3 endInclusive, int maxBadBlocks)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x + start.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y + start.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z + start.z;

    if (x > endInclusive.x || y > endInclusive.y || z > endInclusive.z) {
        return;
    }

    int badBlocks = 0;

    for (int i = 0; i < FilterCount; i++) {
        const RotationInfo info = dev_filter[i];
        const char textureVariant = getTexture(mode, x + info.x, y + info.y, z + info.z, 4);
        const char texture = info.isSide ? textureVariant % 2 : textureVariant;

        if (texture != info.rotation) {
            badBlocks++;
            if (badBlocks > maxBadBlocks) {
                return;
            }
        }
    }

    printf("Found with %d bad block(s)! (%d, %d, %d)\n", badBlocks, x, y, z);
}
