#include "runner.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <string>
#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "matcher.hpp"

namespace {
constexpr unsigned int ThreadsPerAxis = 8;
constexpr unsigned int ResultCapacity = 65536;

// All transformed direction filters fit comfortably in CUDA constant memory.
__device__ __constant__ RotationInfo deviceFilters[MaxDirectionCount][MaxFilterCount];
__device__ __constant__ int deviceFilterCounts[MaxDirectionCount];

template <TextureMode Mode>
__global__ void bruteForceKernel(
    Int3 start,
    Int3 end,
    int maxBadBlocks,
    std::size_t directionIndex,
    int direction,
    Match* results,
    unsigned int* resultCount,
    int* resultOverflow)
{
    // Offsets use 64 bits so indexing remains safe near INT32_MIN and INT32_MAX.
    const std::uint64_t xOffset = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t yOffset = static_cast<std::uint64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const std::uint64_t zOffset = static_cast<std::uint64_t>(blockIdx.z) * blockDim.z + threadIdx.z;
    const std::uint64_t xCount = static_cast<std::uint64_t>(static_cast<std::int64_t>(end.x) - start.x) + 1;
    const std::uint64_t yCount = static_cast<std::uint64_t>(static_cast<std::int64_t>(end.y) - start.y) + 1;
    const std::uint64_t zCount = static_cast<std::uint64_t>(static_cast<std::int64_t>(end.z) - start.z) + 1;
    if (xOffset >= xCount || yOffset >= yCount || zOffset >= zCount) {
        return;
    }

    const std::int32_t x = static_cast<std::int32_t>(static_cast<std::int64_t>(start.x) + xOffset);
    const std::int32_t y = static_cast<std::int32_t>(static_cast<std::int64_t>(start.y) + yOffset);
    const std::int32_t z = static_cast<std::int32_t>(static_cast<std::int64_t>(start.z) + zOffset);
    const int badBlocks = countBadBlocks<Mode>(
        x,
        y,
        z,
        deviceFilters[directionIndex],
        static_cast<std::size_t>(deviceFilterCounts[directionIndex]),
        maxBadBlocks);
    if (badBlocks > maxBadBlocks) {
        return;
    }

    // Never silently lose matches: record capacity overflow for the host to report.
    const unsigned int index = atomicAdd(resultCount, 1U);
    if (index < ResultCapacity) {
        results[index] = { x, y, z, badBlocks, direction };
    }
    else {
        atomicExch(resultOverflow, 1);
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

template <TextureMode Mode>
cudaError_t launchMode(
    const WorkItem& item,
    int maxBadBlocks,
    dim3 grid,
    dim3 block,
    Match* results,
    unsigned int* resultCount,
    int* resultOverflow)
{
    bruteForceKernel<Mode><<<grid, block>>>(
        item.start,
        item.end,
        maxBadBlocks,
        item.directionIndex,
        item.direction,
        results,
        resultCount,
        resultOverflow);
    return cudaGetLastError();
}

cudaError_t launch(
    TextureMode mode,
    const WorkItem& item,
    int maxBadBlocks,
    dim3 grid,
    dim3 block,
    Match* results,
    unsigned int* resultCount,
    int* resultOverflow)
{
    switch (mode) {
    case TextureMode::Vanilla1:
        return launchMode<TextureMode::Vanilla1>(item, maxBadBlocks, grid, block, results, resultCount, resultOverflow);
    case TextureMode::Vanilla2:
        return launchMode<TextureMode::Vanilla2>(item, maxBadBlocks, grid, block, results, resultCount, resultOverflow);
    case TextureMode::Vanilla3:
        return launchMode<TextureMode::Vanilla3>(item, maxBadBlocks, grid, block, results, resultCount, resultOverflow);
    case TextureMode::Sodium1:
        return launchMode<TextureMode::Sodium1>(item, maxBadBlocks, grid, block, results, resultCount, resultOverflow);
    case TextureMode::Sodium2:
    default:
        return launchMode<TextureMode::Sodium2>(item, maxBadBlocks, grid, block, results, resultCount, resultOverflow);
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
        if (config.printChunks) {
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

        const std::uint64_t xCount = static_cast<std::uint64_t>(static_cast<std::int64_t>(item.end.x) - item.start.x) + 1;
        const std::uint64_t yCount = static_cast<std::uint64_t>(static_cast<std::int64_t>(item.end.y) - item.start.y) + 1;
        const std::uint64_t zCount = static_cast<std::uint64_t>(static_cast<std::int64_t>(item.end.z) - item.start.z) + 1;
        const dim3 block(ThreadsPerAxis, ThreadsPerAxis, ThreadsPerAxis);
        const dim3 grid(
            static_cast<unsigned int>((xCount + ThreadsPerAxis - 1) / ThreadsPerAxis),
            static_cast<unsigned int>((yCount + ThreadsPerAxis - 1) / ThreadsPerAxis),
            static_cast<unsigned int>((zCount + ThreadsPerAxis - 1) / ThreadsPerAxis));
        if (grid.x > static_cast<unsigned int>(properties.maxGridSize[0])
            || grid.y > static_cast<unsigned int>(properties.maxGridSize[1])
            || grid.z > static_cast<unsigned int>(properties.maxGridSize[2])) {
            if (error) {
                *error = "tile dimensions exceed this CUDA device's grid limits; reduce tileSizeX or tileSizeZ";
            }
            succeeded = false;
            break;
        }

        // One bounded buffer is reused and drained after each synchronized tile.
        if (!check(cudaMemset(deviceResultCount, 0, sizeof(unsigned int)), "reset result counter", error)
            || !check(cudaMemset(deviceResultOverflow, 0, sizeof(int)), "reset result overflow flag", error)
            || !check(launch(config.mode, item, config.maxBadBlocks, grid, block, deviceResults, deviceResultCount, deviceResultOverflow), "launch CUDA scan", error)
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
                *error = "a tile produced more than 65536 matches; reduce tileSizeX/tileSizeZ or tighten the filter";
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
