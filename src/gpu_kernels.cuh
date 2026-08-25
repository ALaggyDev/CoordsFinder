#pragma once

// Shared GPU device code for the CUDA and HIP backends.
//
// HIP is deliberately source-compatible with CUDA at the device-code level:
// __global__/__device__/__constant__, dim3, atomicAdd/atomicExch, __ffs and
// the <<<grid, block>>> launch syntax all exist in both. The kernels below
// avoid every warp/wave-size-dependent feature, so AMD wave64 vs NVIDIA
// warp32 does not affect correctness.
//
// The including translation unit must define COORDSFINDER_GPU_HIP when
// building for HIP; the default is the CUDA runtime header set.

#include <cstddef>

#if defined(COORDSFINDER_GPU_HIP)
#include "hip/hip_runtime.h"
#else
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#endif

#include "textures.hpp"
#include "types.hpp"

// The device symbols and kernels further below live in an anonymous namespace
// so every translation unit that includes this header gets its own internal
// copies. That keeps the CUDA and HIP backends free to link into one binary
// without ODR collisions.
namespace coordsfinder_gpu {

constexpr unsigned int ThreadsX = 16;
constexpr unsigned int ThreadsZ = 16;
// Vertical candidates per thread. 32 packs the survivor mask into a single
// register; larger values (64) were measured slower due to mask ALU cost.
constexpr unsigned int CandidatesPerThreadY = 32;
constexpr unsigned int ResultCapacity = 65536;

namespace {

// All transformed direction filters fit comfortably in constant memory.
__device__ __constant__ RotationInfo deviceFilters[MaxDirectionCount][MaxFilterCount];
__device__ __constant__ int deviceFilterCounts[MaxDirectionCount];

// Scalar reference kernel: candidate-major with early exits. Exact for every
// errorTolerance; retained as the fallback for errorTolerance > 3 where the
// staged kernel's packed saturating counters would lose exactness.
template <TextureAlgorithm Mode>
__global__ void bruteForceKernel(
    Int3 start,
    Int3 end,
    int errorTolerance,
    std::size_t directionIndex,
    int direction,
    Match* results,
    unsigned int* resultCount,
    int* resultOverflow)
{
    // Tiles are limited to 32-bit spans.
    // A thread owns a short vertical column of at most CandidatesPerThreadY blocks.
    const std::uint32_t xOffset = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t yBaseOffset = blockIdx.y * CandidatesPerThreadY;
    const std::uint32_t zOffset = blockIdx.z * blockDim.z + threadIdx.z;

    const std::int32_t x = start.x + static_cast<std::int32_t>(xOffset);
    std::int32_t y = start.y + static_cast<std::int32_t>(yBaseOffset);
    const std::int32_t z = start.z + static_cast<std::int32_t>(zOffset);

    if (x >= end.x || z >= end.z) {
        return;
    }

    const std::int32_t yEnd = y + static_cast<std::int32_t>(CandidatesPerThreadY) < end.y
        ? y + static_cast<std::int32_t>(CandidatesPerThreadY)
        : end.y;

    const int filterCount = deviceFilterCounts[directionIndex];

    int filterIndex = 0;
    int mismatches = 0;
    while (y < yEnd) {
        const RotationInfo& info = deviceFilters[directionIndex][filterIndex];
        const std::uint8_t variant = getTextureForMode<Mode>(
            wrapAdd(x, info.x),
            wrapAdd(y, info.y),
            wrapAdd(z, info.z),
            4);

        if ((variant & info.visibleMask) != info.rotation) {
            ++mismatches;
        }

        ++filterIndex;

        if (mismatches > errorTolerance) {
            ++y;
            filterIndex = 0;
            mismatches = 0;
        }

        if (filterIndex == filterCount) {
            // Never silently lose matches: record capacity overflow for the host to report.
            const unsigned int index = atomicAdd(resultCount, 1U);
            if (index < ResultCapacity) {
                results[index] = { x, y, z, mismatches, direction };
            }
            else {
                atomicExch(resultOverflow, 1);
            }
            ++y;
            filterIndex = 0;
            mismatches = 0;
        }
    }
}

// Staged (filter-major) kernel: each thread owns CandidatesPerThreadY vertical
// candidates packed into a survivor bitmask. Filter entry j is evaluated for
// every live candidate before advancing to entry j+1. Compared to the scalar
// kernel this greatly reduces warp divergence, and it changes where the time
// goes: filterIndex is warp-uniform (constant-memory fetches broadcast instead
// of serializing), and the inner-loop samples are independent work rather than
// one long serial dependency chain. Both kernels issue a similar number of
// texture samples; the staged arrangement simply executes them far more
// efficiently.
//
// Mismatch counts live in 2-bit saturating fields packed into a u64 (one field
// per candidate). A saturated field means the count exceeds any supported
// tolerance <= 3, so the candidate dies; surviving candidates therefore always
// hold their true mismatch count. errorTolerance == 0 additionally compiles to
// a specialization with no mismatch bookkeeping at all.
template <TextureAlgorithm Mode, bool FastTol0>
__global__ void stagedBruteForceKernel(
    Int3 start,
    Int3 end,
    int errorTolerance,
    std::size_t directionIndex,
    int direction,
    Match* results,
    unsigned int* resultCount,
    int* resultOverflow)
{
    static_assert(CandidatesPerThreadY == 32, "survivor mask packing assumes 32 candidates");

    const std::uint32_t xOffset = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t zOffset = blockIdx.z * blockDim.z + threadIdx.z;
    const std::uint32_t yBaseOffset = blockIdx.y * CandidatesPerThreadY;

    const std::int32_t x = start.x + static_cast<std::int32_t>(xOffset);
    const std::int32_t z = start.z + static_cast<std::int32_t>(zOffset);
    if (x >= end.x || z >= end.z) {
        return;
    }

    const std::int32_t yBase = start.y + static_cast<std::int32_t>(yBaseOffset);
    if (yBase >= end.y) {
        return;
    }
    const int liveCount = static_cast<int>(
        min(static_cast<std::int32_t>(CandidatesPerThreadY), end.y - yBase));

    std::uint32_t alive = liveCount >= 32
        ? 0xFFFFFFFFu
        : ((1u << liveCount) - 1u);
    std::uint64_t mismatches = 0;

    const int filterCount = deviceFilterCounts[directionIndex];
    for (int filterIndex = 0; filterIndex < filterCount && alive != 0; ++filterIndex) {
        const RotationInfo info = deviceFilters[directionIndex][filterIndex];
        const std::int32_t sx = wrapAdd(x, info.x);
        const std::int32_t sz = wrapAdd(z, info.z);

        std::uint32_t live = alive;
        while (live != 0) {
            const int bit = __ffs(live) - 1;
            live &= live - 1;
            const std::int32_t sy = wrapAdd(yBase + bit, info.y);
            const std::uint8_t variant = getTextureForMode<Mode>(sx, sy, sz, 4);
            if ((variant & info.visibleMask) == info.rotation) {
                continue;
            }
            if (FastTol0) {
                alive &= ~(1u << bit);
                continue;
            }
            const unsigned shift = 2u * static_cast<unsigned>(bit);
            if (((mismatches >> shift) & 3ULL) == 3ULL) {
                alive &= ~(1u << bit); // saturated: exceeds every tolerance <= 3
                continue;
            }
            mismatches += (1ULL << shift);
            if (((mismatches >> shift) & 3ULL) > static_cast<std::uint64_t>(errorTolerance)) {
                alive &= ~(1u << bit);
            }
        }
    }

    while (alive != 0) {
        const int bit = __ffs(alive) - 1;
        alive &= alive - 1;
        // Never silently lose matches: record capacity overflow for the host to report.
        const unsigned int index = atomicAdd(resultCount, 1U);
        if (index < ResultCapacity) {
            results[index] = {
                x,
                yBase + bit,
                z,
                FastTol0 ? 0 : static_cast<std::int32_t>((mismatches >> (2u * static_cast<unsigned>(bit))) & 3ULL),
                direction };
        }
        else {
            atomicExch(resultOverflow, 1);
        }
    }
}

// Launch helpers: dispatch on algorithm/tolerance, shared by both backends so
// the kernel-selection policy exists exactly once. Returns the backend error
// code from cudaGetLastError()/hipGetLastError().
#if defined(COORDSFINDER_GPU_HIP)
inline auto gpuGetLastError() { return hipGetLastError(); }
#else
inline auto gpuGetLastError() { return cudaGetLastError(); }
#endif

template <TextureAlgorithm Mode>
auto launchMode(
    const WorkItem& item,
    int errorTolerance,
    dim3 grid,
    dim3 block,
    Match* results,
    unsigned int* resultCount,
    int* resultOverflow)
{
    if (errorTolerance == 0) {
        stagedBruteForceKernel<Mode, true><<<grid, block>>>(
            item.start,
            item.end,
            errorTolerance,
            item.directionIndex,
            item.direction,
            results,
            resultCount,
            resultOverflow);
    }
    else if (errorTolerance >= 1 && errorTolerance <= 3) {
        stagedBruteForceKernel<Mode, false><<<grid, block>>>(
            item.start,
            item.end,
            errorTolerance,
            item.directionIndex,
            item.direction,
            results,
            resultCount,
            resultOverflow);
    }
    else {
        // Packed saturating counters cannot represent tolerances above 3, and
        // negative values would wrap under the uint64 comparison; route both to
        // the exact scalar kernel.
        bruteForceKernel<Mode><<<grid, block>>>(
            item.start,
            item.end,
            errorTolerance,
            item.directionIndex,
            item.direction,
            results,
            resultCount,
            resultOverflow);
    }
    return gpuGetLastError();
}

inline auto launch(
    TextureAlgorithm mode,
    const WorkItem& item,
    int errorTolerance,
    dim3 grid,
    dim3 block,
    Match* results,
    unsigned int* resultCount,
    int* resultOverflow)
{
    switch (mode) {
    case TextureAlgorithm::Vanilla1:
        return launchMode<TextureAlgorithm::Vanilla1>(item, errorTolerance, grid, block, results, resultCount, resultOverflow);
    case TextureAlgorithm::Vanilla2:
        return launchMode<TextureAlgorithm::Vanilla2>(item, errorTolerance, grid, block, results, resultCount, resultOverflow);
    case TextureAlgorithm::Vanilla3:
        return launchMode<TextureAlgorithm::Vanilla3>(item, errorTolerance, grid, block, results, resultCount, resultOverflow);
    case TextureAlgorithm::Sodium1:
        return launchMode<TextureAlgorithm::Sodium1>(item, errorTolerance, grid, block, results, resultCount, resultOverflow);
    case TextureAlgorithm::Sodium2:
    default:
        return launchMode<TextureAlgorithm::Sodium2>(item, errorTolerance, grid, block, results, resultCount, resultOverflow);
    }
}

}
}
