#include <stdint.h>
#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "bruteforce.cuh"
#include "textures.cuh"

namespace {
constexpr int ThreadsPerAxis = 8;

struct ScanConfig {
    TextureMode mode = TextureModeVanilla;

    int xStart = -100000;
    int xEnd = 100000;
    int yStart = -60;
    int yEnd = 0;
    int zStart = -5000;
    int zEnd = 5000;

    int chunkBlocksX = 16384;
    int chunkBlocksZ = 64;
    int maxBadBlocks = 0;
    bool printChunks = true;
};

int ceilDiv(int value, int divisor)
{
    return (value + divisor - 1) / divisor;
}

int minInt(int lhs, int rhs)
{
    return lhs < rhs ? lhs : rhs;
}

const char* textureModeName(TextureMode mode)
{
    switch (mode) {
    case TextureModeVanilla12:
        return "Vanilla 1.12 legacy";
    case TextureModeVanilla:
        return "Vanilla";
    case TextureModeVanilla21_1:
        return "Vanilla 1.21.1";
    case TextureModeSodium:
        return "Sodium";
    case TextureModeSodium19:
    default:
        return "Sodium 1.19";
    }
}

cudaError_t launchChunk(const ScanConfig& config, Int3 start, Int3 end)
{
    const int xCount = end.x - start.x + 1;
    const int yCount = end.y - start.y + 1;
    const int zCount = end.z - start.z + 1;

    const dim3 grid(
        ceilDiv(xCount, ThreadsPerAxis),
        ceilDiv(yCount, ThreadsPerAxis),
        ceilDiv(zCount, ThreadsPerAxis));
    const dim3 block(ThreadsPerAxis, ThreadsPerAxis, ThreadsPerAxis);

    cudaError_t err = launchBruteForce(config.mode, start, end, config.maxBadBlocks, grid, block);
    if (err != cudaSuccess) {
        return err;
    }

    return cudaDeviceSynchronize();
}
}

int main()
{
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);

    const ScanConfig config;

    cudaError_t err = initFilter();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyToSymbol failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Scanning %s rotations from (%d, %d, %d) to (%d, %d, %d).\n",
        textureModeName(config.mode),
        config.xStart,
        config.yStart,
        config.zStart,
        config.xEnd,
        config.yEnd,
        config.zEnd);

    const int chunkSizeX = config.chunkBlocksX * ThreadsPerAxis;
    const int chunkSizeZ = config.chunkBlocksZ * ThreadsPerAxis;

    for (int z = config.zStart; z <= config.zEnd; z += chunkSizeZ) {
        const int zEnd = minInt(z + chunkSizeZ - 1, config.zEnd);

        for (int x = config.xStart; x <= config.xEnd; x += chunkSizeX) {
            const int xEnd = minInt(x + chunkSizeX - 1, config.xEnd);

            const Int3 start = { x, config.yStart, z };
            const Int3 end = { xEnd, config.yEnd, zEnd };

            if (config.printChunks) {
                printf("Scanning chunk from (%d, %d, %d) to (%d, %d, %d).\n",
                    start.x,
                    start.y,
                    start.z,
                    end.x,
                    end.y,
                    end.z);
            }

            err = launchChunk(config, start, end);
            if (err != cudaSuccess) {
                fprintf(stderr, "CUDA scan failed: %s\n", cudaGetErrorString(err));
                return 1;
            }
        }
    }

    err = cudaDeviceReset();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("All done!\n");
    return 0;
}
