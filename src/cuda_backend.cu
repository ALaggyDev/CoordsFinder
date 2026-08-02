#include "runner.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <string>
#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "textures.hpp"

namespace {
constexpr unsigned int ThreadsX = 16;
constexpr unsigned int ThreadsZ = 16;
constexpr unsigned int CandidatesPerThreadY = 16;
constexpr unsigned int ResultCapacity = 65536;

// All transformed direction filters fit comfortably in CUDA constant memory.
__device__ __constant__ RotationInfo deviceFilters[MaxDirectionCount][MaxFilterCount];
__device__ __constant__ int deviceFilterCounts[MaxDirectionCount];

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

    if (x > end.x || z > end.z) {
        return;
    }

    const std::int32_t yEnd = y + static_cast<std::int32_t>(CandidatesPerThreadY) < end.y + 1
        ? y + static_cast<std::int32_t>(CandidatesPerThreadY)
        : end.y + 1;

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

const char* cudaMessage(cudaError_t error)
{
    return cudaGetErrorString(error);
}

bool check(cudaError_t result, const char* operation, std::string* error)
{
    if (result == cudaSuccess) {
        return true;
    }
    if (error) {
        *error = std::string(operation) + ": " + cudaMessage(result);
    }
    return false;
}

template <TextureAlgorithm Mode>
cudaError_t launchMode(
    const WorkItem& item,
    int errorTolerance,
    dim3 grid,
    dim3 block,
    Match* results,
    unsigned int* resultCount,
    int* resultOverflow)
{
    bruteForceKernel<Mode><<<grid, block>>>(
        item.start,
        item.end,
        errorTolerance,
        item.directionIndex,
        item.direction,
        results,
        resultCount,
        resultOverflow);
    return cudaGetLastError();
}

cudaError_t launch(
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

bool cudaAvailable(std::string* reason)
{
    int deviceCount = 0;
    const cudaError_t result = cudaGetDeviceCount(&deviceCount);
    if (result != cudaSuccess) {
        if (reason) {
            *reason = cudaMessage(result);
        }
        cudaGetLastError();
        return false;
    }
    if (deviceCount == 0) {
        if (reason) {
            *reason = "no CUDA device was found";
        }
        return false;
    }
    return true;
}

bool runCudaScan(
    const ScanConfig& config,
    const ScanPlan& plan,
    ScanState* state,
    const MatchSink& sink,
    std::string* error)
{
    std::array<std::array<RotationInfo, MaxFilterCount>, MaxDirectionCount> filters = {};
    std::array<int, MaxDirectionCount> filterCounts = {};
    for (std::size_t direction = 0; direction < config.directions.size(); ++direction) {
        const std::vector<RotationInfo> transformed = makeDirectionalFilter(
            config.filter,
            config.directions[direction]);
        std::copy(transformed.begin(), transformed.end(), filters[direction].begin());
        filterCounts[direction] = static_cast<int>(transformed.size());
    }

    // Upload every direction once; spiral order can then switch directions per tile cheaply.
    if (!check(cudaMemcpyToSymbol(deviceFilters, filters.data(), sizeof(filters)), "upload filters", error)
        || !check(cudaMemcpyToSymbol(deviceFilterCounts, filterCounts.data(), sizeof(filterCounts)), "upload filter counts", error)) {
        return false;
    }

    Match* deviceResults = nullptr;
    unsigned int* deviceResultCount = nullptr;
    int* deviceResultOverflow = nullptr;
    if (!check(cudaMalloc(&deviceResults, sizeof(Match) * ResultCapacity), "allocate result buffer", error)
        || !check(cudaMalloc(&deviceResultCount, sizeof(unsigned int)), "allocate result counter", error)
        || !check(cudaMalloc(&deviceResultOverflow, sizeof(int)), "allocate result overflow flag", error)) {
        cudaFree(deviceResults);
        cudaFree(deviceResultCount);
        cudaFree(deviceResultOverflow);
        return false;
    }

    std::vector<Match> results(ResultCapacity);
    cudaDeviceProp properties = {};
    int device = 0;
    if (!check(cudaGetDevice(&device), "get CUDA device", error)
        || !check(cudaGetDeviceProperties(&properties, device), "get CUDA device properties", error)) {
        cudaFree(deviceResults);
        cudaFree(deviceResultCount);
        cudaFree(deviceResultOverflow);
        return false;
    }
    std::fprintf(stderr, "CUDA device: %s.\n", properties.name);

    bool succeeded = true;
    for (const WorkItem& item : plan.items) {
        if (state->cancelRequested.load(std::memory_order_relaxed)) {
            break;
        }
        if (config.verbose) {
            std::fprintf(stderr,
                "Scanning tile (%d, %d, %d) to (%d, %d, %d), direction %d.\n",
                item.start.x,
                item.start.y,
                item.start.z,
                item.end.x,
                item.end.y,
                item.end.z,
                item.direction);
        }

        const std::uint32_t xCount = static_cast<std::uint32_t>(item.end.x - item.start.x) + 1U;
        const std::uint32_t yCount = static_cast<std::uint32_t>(item.end.y - item.start.y) + 1U;
        const std::uint32_t zCount = static_cast<std::uint32_t>(item.end.z - item.start.z) + 1U;
        if (xCount == 0 || yCount == 0 || zCount == 0) {
            if (error) {
                *error = "a CUDA tile spans the full 32-bit coordinate range; reduce cudaTileSize or the coordinate range";
            }
            succeeded = false;
            break;
        }
        const dim3 block(ThreadsX, 1, ThreadsZ);
        const dim3 grid(
            1U + (xCount - 1U) / ThreadsX,
            1U + (yCount - 1U) / CandidatesPerThreadY,
            1U + (zCount - 1U) / ThreadsZ);
        if (grid.x > static_cast<unsigned int>(properties.maxGridSize[0])
            || grid.y > static_cast<unsigned int>(properties.maxGridSize[1])
            || grid.z > static_cast<unsigned int>(properties.maxGridSize[2])) {
            if (error) {
                *error = "tile dimensions exceed this CUDA device's grid limits; reduce cudaTileSize";
            }
            succeeded = false;
            break;
        }

        // One bounded buffer is reused and drained after each synchronized tile.
        if (!check(cudaMemset(deviceResultCount, 0, sizeof(unsigned int)), "reset result counter", error)
            || !check(cudaMemset(deviceResultOverflow, 0, sizeof(int)), "reset result overflow flag", error)
            || !check(launch(config.algorithm, item, config.errorTolerance, grid, block, deviceResults, deviceResultCount, deviceResultOverflow), "launch CUDA scan", error)
            || !check(cudaDeviceSynchronize(), "run CUDA scan", error)) {
            succeeded = false;
            break;
        }

        unsigned int resultCount = 0;
        int resultOverflow = 0;
        if (!check(cudaMemcpy(&resultCount, deviceResultCount, sizeof(resultCount), cudaMemcpyDeviceToHost), "read result count", error)
            || !check(cudaMemcpy(&resultOverflow, deviceResultOverflow, sizeof(resultOverflow), cudaMemcpyDeviceToHost), "read result overflow flag", error)) {
            succeeded = false;
            break;
        }
        if (resultOverflow) {
            if (error) {
                *error = "a tile produced more than 65536 matches; reduce cudaTileSize or tighten the filter";
            }
            succeeded = false;
            break;
        }
        if (resultCount > 0) {
            results.resize(resultCount);
            if (!check(cudaMemcpy(results.data(), deviceResults, sizeof(Match) * resultCount, cudaMemcpyDeviceToHost), "download results", error)) {
                succeeded = false;
                break;
            }
            sink(results);
            state->matches.fetch_add(resultCount, std::memory_order_relaxed);
        }
        state->candidates.fetch_add(workItemCandidateCount(item), std::memory_order_relaxed);
        state->completedItems.fetch_add(1, std::memory_order_relaxed);
    }

    cudaFree(deviceResults);
    cudaFree(deviceResultCount);
    cudaFree(deviceResultOverflow);
    return succeeded;
}
