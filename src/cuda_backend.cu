#include "runner.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <memory>
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

// __shared__ variables must be trivially default-constructible; RotationInfo's
// default member initializers disqualify it, so stage through this layout-identical POD.
struct SharedRotationInfo {
    std::int8_t x;
    std::int8_t y;
    std::int8_t z;
    std::uint8_t rotation;
    std::uint8_t visibleMask;
};
static_assert(sizeof(SharedRotationInfo) == sizeof(RotationInfo), "shared staging layout mismatch");

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
    __shared__ SharedRotationInfo sharedFilters[MaxFilterCount];
    const unsigned int threadCount = blockDim.x * blockDim.z;
    const unsigned int lane = threadIdx.x + threadIdx.z * blockDim.x;

    const int filterCount = deviceFilterCounts[directionIndex];
    // Stage the direction filter in shared memory: warp divergence on filterIndex
    // would otherwise serialize every __constant__ broadcast fetch.
    for (unsigned int i = lane; i < static_cast<unsigned int>(filterCount); i += threadCount) {
        const RotationInfo& src = deviceFilters[directionIndex][i];
        sharedFilters[i] = { src.x, src.y, src.z, src.rotation, src.visibleMask };
    }
    __syncthreads();

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

    int filterIndex = 0;
    int mismatches = 0;
    while (y < yEnd) {
        const SharedRotationInfo& info = sharedFilters[filterIndex];
        const std::uint8_t variant = getTextureForMode<Mode>(
            wrapAdd(x, info.x),
            wrapAdd(y, info.y),
            wrapAdd(z, info.z),
            4);

        if ((variant & info.visibleMask) != info.rotation) {
            ++mismatches;
        }

        ++filterIndex;

        // Single advance point: rejection wins ties against completion, exactly
        // matching the previous two-check sequencing.
        const bool rejected = mismatches > errorTolerance;
        if (rejected || filterIndex == filterCount) {
            if (!rejected) {
                // Never silently lose matches: record capacity overflow for the host to report.
                const unsigned int index = atomicAdd(resultCount, 1U);
                if (index < ResultCapacity) {
                    results[index] = { x, y, z, mismatches, direction };
                }
                else {
                    atomicExch(resultOverflow, 1);
                }
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

    // Double-buffered pipeline: while tile N runs, tile N-1's downloads complete and
    // its matches drain on the host. Results are still emitted strictly in plan order.
    constexpr std::size_t PipelineDepth = 2;

    cudaStream_t stream = nullptr;
    cudaEvent_t completion[PipelineDepth] = {};
    Match* deviceResults[PipelineDepth] = {};
    unsigned int* deviceResultCount[PipelineDepth] = {};
    int* deviceResultOverflow[PipelineDepth] = {};
    Match* stagedResults[PipelineDepth] = {};
    auto stagedCounts = std::unique_ptr<unsigned int[]>(new unsigned int[PipelineDepth]());
    auto stagedOverflows = std::unique_ptr<int[]>(new int[PipelineDepth]());

    bool allocationFailed = false;
    if (!check(cudaStreamCreate(&stream), "create CUDA stream", error)) {
        allocationFailed = true;
    }
    for (std::size_t i = 0; i < PipelineDepth && !allocationFailed; ++i) {
        if (!check(cudaEventCreateWithFlags(&completion[i], cudaEventDisableTiming), "create CUDA event", error)
            || !check(cudaMalloc(&deviceResults[i], sizeof(Match) * ResultCapacity), "allocate result buffer", error)
            || !check(cudaMalloc(&deviceResultCount[i], sizeof(unsigned int)), "allocate result counter", error)
            || !check(cudaMalloc(&deviceResultOverflow[i], sizeof(int)), "allocate result overflow flag", error)
            || !check(cudaMallocHost(&stagedResults[i], sizeof(Match) * ResultCapacity), "allocate pinned result staging", error)) {
            allocationFailed = true;
        }
    }

    cudaDeviceProp properties = {};
    int device = 0;
    if (!allocationFailed
        && (!check(cudaGetDevice(&device), "get CUDA device", error)
            || !check(cudaGetDeviceProperties(&properties, device), "get CUDA device properties", error))) {
        allocationFailed = true;
    }

    if (!allocationFailed) {
        std::fprintf(stderr, "CUDA device: %s.\n", properties.name);
        // Record each event once so early synchronizations always observe a real signal.
        for (std::size_t i = 0; i < PipelineDepth; ++i) {
            cudaEventRecord(completion[i], stream);
        }
    }

    bool succeeded = !allocationFailed;
    bool fatal = allocationFailed;
    bool invalidTile = false;
    std::size_t submitted = 0;
    std::size_t drained = 0;
    std::vector<Match> results;
    results.reserve(ResultCapacity);

    // Submissions and drains run PipelineDepth-1 tiles apart; each buffer is only
    // reused after its previous drain synchronized past the result download.
    constexpr int SubmitOk = 0;
    constexpr int SubmitInvalidTile = 1;
    constexpr int SubmitRuntimeFailure = 2;
    auto submitTile = [&](std::size_t index) -> int {

        const WorkItem& item = plan.items[index];
        const std::size_t b = index % PipelineDepth;

        const std::uint32_t xCount = static_cast<std::uint32_t>(item.end.x - item.start.x);
        const std::uint32_t yCount = static_cast<std::uint32_t>(item.end.y - item.start.y);
        const std::uint32_t zCount = static_cast<std::uint32_t>(item.end.z - item.start.z);
        if (xCount == 0 || yCount == 0 || zCount == 0) {
            if (error) {
                *error = "a CUDA tile spans the full 32-bit coordinate range; reduce cudaTileSize or the coordinate range";
            }
            return SubmitInvalidTile;
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
            return SubmitInvalidTile;
        }

        const bool enqueued = check(cudaMemsetAsync(deviceResultCount[b], 0, sizeof(unsigned int), stream), "reset result counter", error)
            && check(cudaMemsetAsync(deviceResultOverflow[b], 0, sizeof(int), stream), "reset result overflow flag", error)
            && check(launch(config.algorithm, item, config.errorTolerance, grid, block, deviceResults[b], deviceResultCount[b], deviceResultOverflow[b]), "launch CUDA scan", error)
            && check(cudaMemcpyAsync(&stagedCounts[b], deviceResultCount[b], sizeof(unsigned int), cudaMemcpyDeviceToHost, stream), "read result count", error)
            && check(cudaMemcpyAsync(&stagedOverflows[b], deviceResultOverflow[b], sizeof(int), cudaMemcpyDeviceToHost, stream), "read result overflow flag", error)
            && check(cudaEventRecord(completion[b], stream), "record CUDA tile event", error);
        return enqueued ? SubmitOk : SubmitRuntimeFailure;
    };

    auto drainTile = [&](std::size_t index) -> bool {
        const std::size_t b = index % PipelineDepth;
        if (!check(cudaEventSynchronize(completion[b]), "await CUDA tile", error)) {
            return false;
        }
        if (stagedOverflows[b]) {
            if (error) {
                *error = "a tile produced more than 65536 matches; reduce cudaTileSize or tighten the filter";
            }
            return false;
        }
        const unsigned int resultCount = stagedCounts[b];
        if (resultCount > 0) {
            if (!check(cudaMemcpyAsync(stagedResults[b], deviceResults[b], sizeof(Match) * resultCount, cudaMemcpyDeviceToHost, stream), "download results", error)
                || !check(cudaEventRecord(completion[b], stream), "record CUDA download event", error)
                || !check(cudaEventSynchronize(completion[b]), "finish CUDA download", error)) {
                return false;
            }
            results.assign(stagedResults[b], stagedResults[b] + resultCount);
            sink(results);
            state->matches.fetch_add(resultCount, std::memory_order_relaxed);
        }
        state->candidates.fetch_add(workItemCandidateCount(plan.items[index]), std::memory_order_relaxed);
        state->completedItems.fetch_add(1, std::memory_order_relaxed);
        return true;
    };

    for (; submitted < plan.items.size() && !fatal; ++submitted) {
        if (state->cancelRequested.load(std::memory_order_relaxed)) {
            break;
        }
        if (config.verbose) {
            const WorkItem& item = plan.items[submitted];
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
        // Drain the tile that previously occupied this buffer before reusing it:
        // submitting first would overwrite its device buffer, staged count, and
        // completion event before its results were ever downloaded.
        if (submitted >= PipelineDepth) {
            if (!drainTile(submitted - PipelineDepth)) {
                fatal = true;
                break;
            }
            ++drained;
        }
        const int submitResult = submitTile(submitted);
        if (submitResult == SubmitInvalidTile) {
            // The rejected tile never reached the GPU; everything already submitted
            // stays valid and is flushed below before reporting the failure.
            invalidTile = true;
            break;
        }
        if (submitResult != SubmitOk) {
            fatal = true;
            break;
        }
    }
    if (!fatal) {
        // Flush every tile that reached the GPU, including after cancellation or a
        // later invalid tile: their results are exact and dropping them would
        // silently lose work.
        while (drained < submitted && drainTile(drained)) {
            ++drained;
        }
        if (drained < submitted) {
            fatal = true;
        }
    }
    succeeded = !fatal && !invalidTile;

    cudaStreamSynchronize(stream);
    for (std::size_t i = 0; i < PipelineDepth; ++i) {
        cudaEventDestroy(completion[i]);
        cudaFree(deviceResults[i]);
        cudaFree(deviceResultCount[i]);
        cudaFree(deviceResultOverflow[i]);
        cudaFreeHost(stagedResults[i]);
    }
    cudaStreamDestroy(stream);
    return succeeded;
}
