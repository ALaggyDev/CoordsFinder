#pragma once

// Shared host-side scan loop for the CUDA and HIP backends.
//
// HIP is source-compatible with CUDA at the host level too, so the entire scan
// loop exists exactly once here; a Backend trait adapts the small API
// differences between the runtimes (symbol copies, error handling, property
// layout). The trait must be defined in the backend's own translation unit,
// which also includes gpu_kernels.cuh for the device symbols.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

#include "runner.hpp"
#include "types.hpp"

namespace coordsfinder_gpu {

// Normalized subset of cudaDeviceProp/hipDeviceProp_t used by the scan loop.
struct DeviceInfo {
    const char* name;
    unsigned int maxGridSize[3];
};

template <typename Backend>
bool checkGpuResult(typename Backend::Error result, const char* operation, std::string* error)
{
    if (result == Backend::success) {
        return true;
    }
    if (error) {
        *error = std::string(operation) + ": " + Backend::errorString(result);
    }
    return false;
}

template <typename Backend>
bool gpuAvailable(std::string* reason)
{
    int deviceCount = 0;
    if (!checkGpuResult<Backend>(Backend::getDeviceCount(&deviceCount), "enumerate GPU devices", reason)) {
        Backend::drainError();
        return false;
    }
    if (deviceCount == 0) {
        if (reason) {
            *reason = std::string("no ") + Backend::name + " device was found";
        }
        return false;
    }
    return true;
}

template <typename Backend>
void freeGpuBuffers(Match* results, unsigned int* resultCount, int* resultOverflow)
{
    // Teardown failures have no meaningful handler; discard explicitly.
    (void)Backend::free(results);
    (void)Backend::free(resultCount);
    (void)Backend::free(resultOverflow);
}

template <typename Backend>
bool runGpuScan(
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
    if (!checkGpuResult<Backend>(Backend::uploadFilters(filters, filterCounts), "upload filters", error)) {
        return false;
    }

    Match* deviceResults = nullptr;
    unsigned int* deviceResultCount = nullptr;
    int* deviceResultOverflow = nullptr;
    if (!checkGpuResult<Backend>(Backend::malloc(reinterpret_cast<void**>(&deviceResults), sizeof(Match) * ResultCapacity), "allocate result buffer", error)
        || !checkGpuResult<Backend>(Backend::malloc(reinterpret_cast<void**>(&deviceResultCount), sizeof(unsigned int)), "allocate result counter", error)
        || !checkGpuResult<Backend>(Backend::malloc(reinterpret_cast<void**>(&deviceResultOverflow), sizeof(int)), "allocate result overflow flag", error)) {
        freeGpuBuffers<Backend>(deviceResults, deviceResultCount, deviceResultOverflow);
        return false;
    }

    DeviceInfo info = {};
    if (!checkGpuResult<Backend>(Backend::getDeviceInfo(&info), "get GPU device properties", error)) {
        freeGpuBuffers<Backend>(deviceResults, deviceResultCount, deviceResultOverflow);
        return false;
    }
    std::fprintf(stderr, "%s device: %s.\n", Backend::name, info.name);

    std::vector<Match> results(ResultCapacity);
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

        const std::uint32_t xCount = static_cast<std::uint32_t>(item.end.x - item.start.x);
        const std::uint32_t yCount = static_cast<std::uint32_t>(item.end.y - item.start.y);
        const std::uint32_t zCount = static_cast<std::uint32_t>(item.end.z - item.start.z);
        if (xCount == 0 || yCount == 0 || zCount == 0) {
            if (error) {
                *error = "a " + std::string(Backend::name)
                    + " tile spans the full 32-bit coordinate range; reduce cudaTileSize or the coordinate range";
            }
            succeeded = false;
            break;
        }
        const dim3 block(ThreadsX, 1, ThreadsZ);
        const dim3 grid(
            1U + (xCount - 1U) / ThreadsX,
            1U + (yCount - 1U) / CandidatesPerThreadY,
            1U + (zCount - 1U) / ThreadsZ);
        if (grid.x > info.maxGridSize[0] || grid.y > info.maxGridSize[1] || grid.z > info.maxGridSize[2]) {
            if (error) {
                *error = "tile dimensions exceed this " + std::string(Backend::name)
                    + " device's grid limits; reduce cudaTileSize";
            }
            succeeded = false;
            break;
        }

        // One bounded buffer is reused and drained after each synchronized tile.
        if (!checkGpuResult<Backend>(Backend::memset(deviceResultCount, 0, sizeof(unsigned int)), "reset result counter", error)
            || !checkGpuResult<Backend>(Backend::memset(deviceResultOverflow, 0, sizeof(int)), "reset result overflow flag", error)
            || !checkGpuResult<Backend>(launch(config.algorithm, item, config.errorTolerance, grid, block, deviceResults, deviceResultCount, deviceResultOverflow), "launch GPU scan", error)
            || !checkGpuResult<Backend>(Backend::synchronize(), "run GPU scan", error)) {
            succeeded = false;
            break;
        }

        unsigned int resultCount = 0;
        int resultOverflow = 0;
        if (!checkGpuResult<Backend>(Backend::memcpyToHost(&resultCount, deviceResultCount, sizeof(resultCount)), "read result count", error)
            || !checkGpuResult<Backend>(Backend::memcpyToHost(&resultOverflow, deviceResultOverflow, sizeof(resultOverflow)), "read result overflow flag", error)) {
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
            if (!checkGpuResult<Backend>(Backend::memcpyToHost(results.data(), deviceResults, sizeof(Match) * resultCount), "download results", error)) {
                succeeded = false;
                break;
            }
            sink(results);
            state->matches.fetch_add(resultCount, std::memory_order_relaxed);
        }
        state->candidates.fetch_add(workItemCandidateCount(item), std::memory_order_relaxed);
        state->completedItems.fetch_add(1, std::memory_order_relaxed);
    }

    freeGpuBuffers<Backend>(deviceResults, deviceResultCount, deviceResultOverflow);
    return succeeded;
}

}
