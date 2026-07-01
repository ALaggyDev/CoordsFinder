#pragma once

#include "cuda_runtime.h"

#include "textures.cuh"

struct Int3 {
    int x;
    int y;
    int z;
};

struct RotationInfo {
    char x;
    char y;
    char z;
    char rotation;
    char visibleMask;

    constexpr RotationInfo(
        char vx = 0,
        char vy = 0,
        char vz = 0,
        char vrotation = 0,
        bool visSide = false)
        : x(vx),
          y(vy),
          z(vz),
          rotation(static_cast<char>(vrotation % (visSide ? 2 : 4))),
          visibleMask(static_cast<char>(visSide ? 1 : 3))
    {
    }
};

cudaError_t initFilter();

cudaError_t launchBruteForce(TextureMode mode, Int3 start, Int3 endInclusive, int maxBadBlocks, dim3 grid, dim3 block);
