#pragma once

#include "cuda_runtime.h"

#include "textures.cuh"

struct Int3 {
    int x;
    int y;
    int z;
};

struct Int2 {
    int x;
    int z;
};

enum XzRotation {
    XzRotation0 = 0,
    XzRotation90 = 1,
    XzRotation180 = 2,
    XzRotation270 = 3,
    XzRotationCount = 4
};

constexpr unsigned int XzRotationMask0 = 1u << XzRotation0;
constexpr unsigned int XzRotationMask90 = 1u << XzRotation90;
constexpr unsigned int XzRotationMask180 = 1u << XzRotation180;
constexpr unsigned int XzRotationMask270 = 1u << XzRotation270;
constexpr unsigned int XzRotationMaskAll =
    XzRotationMask0 | XzRotationMask90 | XzRotationMask180 | XzRotationMask270;

constexpr int MaxFilterCount = 256;

// XZ rotations are clockwise on a top-down Minecraft map where +X is east/right
// and +Z is south/down. For example, 90 degrees maps local +X to world +Z.
inline Int2 rotateXzOffset(int x, int z, int rotation)
{
    switch (rotation & 3) {
    case XzRotation90:
        return { -z, x };
    case XzRotation180:
        return { -x, -z };
    case XzRotation270:
        return { z, -x };
    case XzRotation0:
    default:
        return { x, z };
    }
}

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

cudaError_t initFilter(int xzRotation, const RotationInfo* hostFilter, int filterCount);

cudaError_t launchBruteForce(
    TextureMode mode,
    Int3 start,
    Int3 endInclusive,
    int maxBadBlocks,
    int filterCount,
    dim3 grid,
    dim3 block);
