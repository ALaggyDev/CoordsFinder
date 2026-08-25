#include "runner.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <string>
#include <vector>

#define COORDSFINDER_GPU_HIP
#include "gpu_kernels.cuh"

namespace {

using coordsfinder_gpu::CandidatesPerThreadY;
using coordsfinder_gpu::ResultCapacity;
using coordsfinder_gpu::ThreadsX;
using coordsfinder_gpu::ThreadsZ;

const char* hipMessage(hipError_t error)
{
    return hipGetErrorString(error);
}

// HIP marks cleanup/error-drain calls [[nodiscard]]; failures during teardown
// have no meaningful handler, so discard explicitly.
inline void hipDiscard(hipError_t result)
{
    (void)result;
}

bool check(hipError_t result, const char* operation, std::string* error)
{
    if (result == hipSuccess) {
        return true;
    }
    if (error) {
        *error = std::string(operation) + ": " + hipMessage(result);
    }
    return false;
}

}

bool hipAvailable(std::string* reason)
{
    int deviceCount = 0;
    const hipError_t result = hipGetDeviceCount(&deviceCount);
    if (result != hipSuccess) {
        if (reason) {
            *reason = hipMessage(result);
        }
        hipDiscard(hipGetLastError());
        return false;
    }
    if (deviceCount == 0) {
        if (reason) {
            *reason = "no HIP device was found";
        }
        return false;
    }
    return true;
}

bool runHipScan(
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
    // HIP_SYMBOL wraps the device symbol as required by the HIP runtime API.
    if (!check(hipMemcpyToSymbol(HIP_SYMBOL(coordsfinder_gpu::deviceFilters), filters.data(), sizeof(filters)), "upload filters", error)
        || !check(hipMemcpyToSymbol(HIP_SYMBOL(coordsfinder_gpu::deviceFilterCounts), filterCounts.data(), sizeof(filterCounts)), "upload filter counts", error)) {
        return false;
    }

    Match* deviceResults = nullptr;
    unsigned int* deviceResultCount = nullptr;
    int* deviceResultOverflow = nullptr;
    if (!check(hipMalloc(&deviceResults, sizeof(Match) * ResultCapacity), "allocate result buffer", error)
        || !check(hipMalloc(&deviceResultCount, sizeof(unsigned int)), "allocate result counter", error)
        || !check(hipMalloc(&deviceResultOverflow, sizeof(int)), "allocate result overflow flag", error)) {
        hipDiscard(hipFree(deviceResults));
        hipDiscard(hipFree(deviceResultCount));
        hipDiscard(hipFree(deviceResultOverflow));
        return false;
    }

    std::vector<Match> results(ResultCapacity);
    hipDeviceProp_t properties = {};
    int device = 0;
    if (!check(hipGetDevice(&device), "get HIP device", error)
        || !check(hipGetDeviceProperties(&properties, device), "get HIP device properties", error)) {
        hipDiscard(hipFree(deviceResults));
        hipDiscard(hipFree(deviceResultCount));
        hipDiscard(hipFree(deviceResultOverflow));
        return false;
    }
    std::fprintf(stderr, "HIP device: %s.\n", properties.name);

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
                *error = "a HIP tile spans the full 32-bit coordinate range; reduce cudaTileSize or the coordinate range";
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
                *error = "tile dimensions exceed this HIP device's grid limits; reduce cudaTileSize";
            }
            succeeded = false;
            break;
        }

        // One bounded buffer is reused and drained after each synchronized tile.
        if (!check(hipMemset(deviceResultCount, 0, sizeof(unsigned int)), "reset result counter", error)
            || !check(hipMemset(deviceResultOverflow, 0, sizeof(int)), "reset result overflow flag", error)
            || !check(coordsfinder_gpu::launch(config.algorithm, item, config.errorTolerance, grid, block, deviceResults, deviceResultCount, deviceResultOverflow), "launch HIP scan", error)
            || !check(hipDeviceSynchronize(), "run HIP scan", error)) {
            succeeded = false;
            break;
        }

        unsigned int resultCount = 0;
        int resultOverflow = 0;
        if (!check(hipMemcpy(&resultCount, deviceResultCount, sizeof(resultCount), hipMemcpyDeviceToHost), "read result count", error)
            || !check(hipMemcpy(&resultOverflow, deviceResultOverflow, sizeof(resultOverflow), hipMemcpyDeviceToHost), "read result overflow flag", error)) {
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
            if (!check(hipMemcpy(results.data(), deviceResults, sizeof(Match) * resultCount, hipMemcpyDeviceToHost), "download results", error)) {
                succeeded = false;
                break;
            }
            sink(results);
            state->matches.fetch_add(resultCount, std::memory_order_relaxed);
        }
        state->candidates.fetch_add(workItemCandidateCount(item), std::memory_order_relaxed);
        state->completedItems.fetch_add(1, std::memory_order_relaxed);
    }

    hipDiscard(hipFree(deviceResults));
    hipDiscard(hipFree(deviceResultCount));
    hipDiscard(hipFree(deviceResultOverflow));
    return succeeded;
}
