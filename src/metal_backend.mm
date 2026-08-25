#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "runner.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {
constexpr std::uint64_t CandidatesPerThreadY = 128;
constexpr std::uint64_t DefaultBatchWorkItems = 1U << 20;
constexpr std::uint32_t ResultCapacity = 65536;

struct MetalFilter {
    std::int32_t x;
    std::int32_t y;
    std::int32_t z;
    std::uint32_t rotation;
    std::uint32_t visibleMask;
};

struct MetalParameters {
    std::uint32_t workOffsetLow;
    std::uint32_t workOffsetHigh;
    std::int32_t xStart;
    std::int32_t yStart;
    std::int32_t zStart;
    std::uint32_t yCount;
    std::uint32_t zCount;
    std::uint32_t yBlockCount;
    std::uint32_t filterBase;
    std::uint32_t filterCount;
    std::uint32_t errorTolerance;
    std::uint32_t resultCapacity;
    std::int32_t direction;
};

static_assert(sizeof(MetalFilter) == 20, "Metal filter layout must match MSL");
static_assert(sizeof(MetalParameters) == 52, "Metal parameter layout must match MSL");
static_assert(sizeof(Match) == 20, "Metal result layout must match Match");

constexpr char MetalSource[] = R"METAL(
#include <metal_stdlib>
using namespace metal;

constant uint kTextureMode [[function_constant(0)]];

constant uint candidatesPerThreadY = 128u;
constant ulong javaMultiplier = 0x5DEECE66Dul;
constant ulong javaMask = (1ul << 48) - 1ul;
constant ulong sodiumPhi = 0x9E3779B97F4A7C15ul;

struct ScanParameters {
    uint workOffsetLow;
    uint workOffsetHigh;
    int xStart;
    int yStart;
    int zStart;
    uint yCount;
    uint zCount;
    uint yBlockCount;
    uint filterBase;
    uint filterCount;
    uint errorTolerance;
    uint resultCapacity;
    int direction;
};

struct Filter {
    int x;
    int y;
    int z;
    uint rotation;
    uint visibleMask;
};

struct Match {
    int x;
    int y;
    int z;
    int mismatches;
    int direction;
};

inline ulong make_ulong(uint low, uint high) {
    return ulong(low) | (ulong(high) << 32);
}

inline int wrap_add(int value, int offset) {
    return as_type<int>(as_type<uint>(value) + as_type<uint>(offset));
}

inline uint absolute_modulo(int value, uint modulus) {
    long wide = long(value);
    ulong magnitude = ulong(wide < 0 ? -wide : wide);
    return uint(magnitude % ulong(modulus));
}

inline ulong rotate_left_64(ulong value, uint distance) {
    return (value << distance) | (value >> (64u - distance));
}

inline ulong stafford_mix_13(ulong value) {
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ul;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBul;
    return value ^ (value >> 31);
}

inline ulong coordinate_random_raw(int x, int y, int z) {
    uint xProductBits = as_type<uint>(x) * 3129871u;
    int xProduct = as_type<int>(xProductBits);
    ulong seed = ulong(long(xProduct));
    seed ^= ulong(long(z)) * 116129781ul;
    seed ^= ulong(long(y));
    return seed * seed * 42317861ul + seed * 11ul;
}

inline int coordinate_random_legacy(int x, int y, int z) {
    int low = as_type<int>(uint(coordinate_random_raw(x, y, z)));
    return low >> 16;
}

inline ulong coordinate_random(int x, int y, int z) {
    long shifted = as_type<long>(coordinate_random_raw(x, y, z)) >> 16;
    return as_type<ulong>(shifted);
}

inline int random_vanilla_2(ulong seed) {
    seed = (seed ^ javaMultiplier) & javaMask;
    ulong mixed = seed * 0xBB20B4600A69ul + 0x40942DE6BAul;
    return as_type<int>(uint(mixed >> 16));
}

inline uint legacy_next_int_4(ulong seed) {
    seed = (seed ^ javaMultiplier) & javaMask;
    seed = (seed * javaMultiplier + 11ul) & javaMask;
    uint bits = uint(seed >> 17);
    return uint((4ul * ulong(bits)) >> 31);
}

inline int random_sodium_1(ulong seed) {
    seed ^= seed >> 33;
    seed *= 0xff51afd7ed558ccdul;
    seed ^= seed >> 33;
    seed *= 0xc4ceb9fe1a85ec53ul;
    seed ^= seed >> 33;

    ulong first = stafford_mix_13(seed + sodiumPhi);
    ulong second = stafford_mix_13(seed + sodiumPhi + sodiumPhi);
    return as_type<int>(uint(first + second));
}

inline int random_sodium_2(ulong seed) {
    ulong low = seed ^ 7640891576956012809ul;
    ulong high = low - 7046029254386353131ul;
    low = stafford_mix_13(low);
    high = stafford_mix_13(high);
    return as_type<int>(uint(rotate_left_64(low + high, 17) + low));
}

inline uint texture_variant(int x, int y, int z) {
    switch (kTextureMode) {
    case 0:
        return absolute_modulo(coordinate_random_legacy(x, y, z), 4u);
    case 1:
        return absolute_modulo(random_vanilla_2(coordinate_random(x, y, z)), 4u);
    case 2:
        return legacy_next_int_4(coordinate_random(x, y, z));
    case 3:
        return absolute_modulo(random_sodium_1(coordinate_random(x, y, z)), 4u);
    case 4:
    default:
        return absolute_modulo(random_sodium_2(coordinate_random(x, y, z)), 4u);
    }
}

kernel void search_coordinates(
    constant ScanParameters& parameters [[buffer(0)]],
    constant Filter* filters [[buffer(1)]],
    device atomic_uint* resultCount [[buffer(2)]],
    device atomic_uint* resultOverflow [[buffer(3)]],
    device Match* results [[buffer(4)]],
    uint threadIndex [[thread_position_in_grid]])
{
    ulong workOffset = make_ulong(parameters.workOffsetLow, parameters.workOffsetHigh);
    ulong workIndex = workOffset + ulong(threadIndex);
    ulong columnIndex = workIndex / ulong(parameters.yBlockCount);
    uint yBlockIndex = uint(workIndex % ulong(parameters.yBlockCount));
    ulong xOffset = columnIndex / ulong(parameters.zCount);
    ulong zOffset = columnIndex % ulong(parameters.zCount);

    int x = as_type<int>(as_type<uint>(parameters.xStart) + uint(xOffset));
    int z = as_type<int>(as_type<uint>(parameters.zStart) + uint(zOffset));
    ulong yBase = ulong(yBlockIndex) * ulong(candidatesPerThreadY);
    uint yBatchCount = uint(min(
        ulong(candidatesPerThreadY),
        ulong(parameters.yCount) - yBase));

    for (uint yOffset = 0; yOffset < yBatchCount; ++yOffset) {
        int y = as_type<int>(
            as_type<uint>(parameters.yStart) + uint(yBase) + yOffset);
        uint mismatches = 0u;
        for (uint filterIndex = 0; filterIndex < parameters.filterCount; ++filterIndex) {
            Filter filter = filters[parameters.filterBase + filterIndex];
            uint visibleVariant = texture_variant(
                wrap_add(x, filter.x),
                wrap_add(y, filter.y),
                wrap_add(z, filter.z)) & filter.visibleMask;
            if (visibleVariant != filter.rotation) {
                ++mismatches;
                if (mismatches > parameters.errorTolerance) {
                    break;
                }
            }
        }

        if (mismatches <= parameters.errorTolerance) {
            uint resultIndex = atomic_fetch_add_explicit(
                resultCount,
                1u,
                memory_order_relaxed);
            if (resultIndex < parameters.resultCapacity) {
                results[resultIndex] = {
                    x,
                    y,
                    z,
                    int(mismatches),
                    parameters.direction,
                };
            }
            else {
                atomic_store_explicit(resultOverflow, 1u, memory_order_relaxed);
            }
        }
    }
}
)METAL";

void setError(std::string* error, const char* operation, NSError* detail = nil)
{
    if (!error) {
        return;
    }
    *error = operation;
    if (detail) {
        const char* description = detail.localizedDescription.UTF8String;
        if (description) {
            *error += std::string(": ") + description;
        }
    }
}

bool checkedMultiply(std::uint64_t lhs, std::uint64_t rhs, std::uint64_t* result)
{
    if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs) {
        return false;
    }
    *result = lhs * rhs;
    return true;
}

std::uint64_t span(std::int32_t start, std::int32_t end)
{
    return static_cast<std::uint64_t>(static_cast<std::int64_t>(end) - start);
}

std::uint64_t candidatePrefix(
    std::uint64_t workItems,
    std::uint64_t yBlockCount,
    std::uint64_t yCount)
{
    const std::uint64_t completeColumns = workItems / yBlockCount;
    const std::uint64_t partialBlocks = workItems % yBlockCount;
    return completeColumns * yCount
        + std::min(partialBlocks * CandidatesPerThreadY, yCount);
}

bool makePipeline(
    id<MTLDevice> device,
    TextureAlgorithm algorithm,
    id<MTLComputePipelineState>* pipeline,
    std::string* error)
{
    NSString* source = [NSString stringWithUTF8String:MetalSource];
    NSError* detail = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&detail];
    if (!library) {
        setError(error, "compile Metal library", detail);
        return false;
    }

    MTLFunctionConstantValues* constants = [MTLFunctionConstantValues new];
    std::uint32_t mode = static_cast<std::uint32_t>(algorithm);
    [constants setConstantValue:&mode type:MTLDataTypeUInt atIndex:0];
    id<MTLFunction> function = [library newFunctionWithName:@"search_coordinates"
                                            constantValues:constants
                                                     error:&detail];
    if (!function) {
        setError(error, "specialize Metal kernel", detail);
        return false;
    }

    *pipeline = [device newComputePipelineStateWithFunction:function error:&detail];
    if (!*pipeline) {
        setError(error, "create Metal compute pipeline", detail);
        return false;
    }
    return true;
}

std::vector<MetalFilter> makeMetalFilters(const ScanConfig& config)
{
    std::vector<MetalFilter> filters;
    filters.reserve(config.filter.size() * config.directions.size());
    for (int direction : config.directions) {
        const std::vector<RotationInfo> transformed = makeDirectionalFilter(
            config.filter,
            direction);
        for (const RotationInfo& filter : transformed) {
            filters.push_back({
                filter.x,
                filter.y,
                filter.z,
                filter.rotation,
                filter.visibleMask,
            });
        }
    }
    return filters;
}
}

bool metalAvailable(std::string* reason)
{
#if !defined(__arm64__) && !defined(__aarch64__)
    if (reason) {
        *reason = "the Metal backend requires an Apple-silicon Mac";
    }
    return false;
#else
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            if (reason) {
                *reason = "no Metal device was found";
            }
            return false;
        }
        if (!device.hasUnifiedMemory) {
            if (reason) {
                *reason = "the default Metal device does not expose unified memory";
            }
            return false;
        }
        return true;
    }
#endif
}

bool runMetalScan(
    const ScanConfig& config,
    const ScanPlan& plan,
    ScanState* state,
    const MatchSink& sink,
    std::string* error)
{
    if (!state) {
        if (error) {
            *error = "missing scan state";
        }
        return false;
    }

    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device || !device.hasUnifiedMemory) {
            if (error) {
                *error = "no supported Apple-silicon Metal device is available";
            }
            return false;
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) {
            setError(error, "create Metal command queue");
            return false;
        }

        id<MTLComputePipelineState> pipeline = nil;
        if (!makePipeline(device, config.algorithm, &pipeline, error)) {
            return false;
        }

        const std::vector<MetalFilter> filters = makeMetalFilters(config);
        id<MTLBuffer> filterBuffer = [device newBufferWithBytes:filters.data()
                                                        length:filters.size() * sizeof(MetalFilter)
                                                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> resultCountBuffer = [device newBufferWithLength:sizeof(std::uint32_t)
                                                              options:MTLResourceStorageModeShared];
        id<MTLBuffer> resultOverflowBuffer = [device newBufferWithLength:sizeof(std::uint32_t)
                                                                 options:MTLResourceStorageModeShared];
        id<MTLBuffer> resultBuffer = [device newBufferWithLength:sizeof(Match) * ResultCapacity
                                                         options:MTLResourceStorageModeShared];
        if (!filterBuffer || !resultCountBuffer || !resultOverflowBuffer || !resultBuffer) {
            setError(error, "allocate Metal buffers");
            return false;
        }

        const NSUInteger executionWidth = pipeline.threadExecutionWidth;
        const NSUInteger preferredWidth = std::min<NSUInteger>(
            256,
            pipeline.maxTotalThreadsPerThreadgroup);
        const NSUInteger threadgroupWidth = std::max<NSUInteger>(
            executionWidth,
            preferredWidth - preferredWidth % executionWidth);
        std::fprintf(stderr, "Metal device: %s (%lu threads per threadgroup).\n",
            device.name.UTF8String,
            static_cast<unsigned long>(threadgroupWidth));

        std::vector<Match> matches;
        matches.reserve(ResultCapacity);
        std::uint64_t adaptiveBatchSize = DefaultBatchWorkItems;

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

            const std::uint64_t xCount = span(item.start.x, item.end.x);
            const std::uint64_t yCount = span(item.start.y, item.end.y);
            const std::uint64_t zCount = span(item.start.z, item.end.z);
            const std::uint64_t yBlockCount = (yCount + CandidatesPerThreadY - 1)
                / CandidatesPerThreadY;
            std::uint64_t columnCount = 0;
            std::uint64_t totalWorkItems = 0;
            bool candidatesSaturated = false;
            workItemCandidateCount(item, &candidatesSaturated);
            if (candidatesSaturated
                || !checkedMultiply(xCount, zCount, &columnCount)
                || !checkedMultiply(columnCount, yBlockCount, &totalWorkItems)
                || yCount > std::numeric_limits<std::uint32_t>::max()
                || zCount > std::numeric_limits<std::uint32_t>::max()
                || yBlockCount > std::numeric_limits<std::uint32_t>::max()) {
                if (error) {
                    *error = "a Metal tile is too large; reduce metalTileSize or the coordinate range";
                }
                return false;
            }

            MetalParameters parameters = {};
            parameters.xStart = item.start.x;
            parameters.yStart = item.start.y;
            parameters.zStart = item.start.z;
            parameters.yCount = static_cast<std::uint32_t>(yCount);
            parameters.zCount = static_cast<std::uint32_t>(zCount);
            parameters.yBlockCount = static_cast<std::uint32_t>(yBlockCount);
            parameters.filterBase = static_cast<std::uint32_t>(
                item.directionIndex * config.filter.size());
            parameters.filterCount = static_cast<std::uint32_t>(config.filter.size());
            parameters.errorTolerance = static_cast<std::uint32_t>(config.errorTolerance);
            parameters.resultCapacity = ResultCapacity;
            parameters.direction = item.direction;

            std::uint64_t workOffset = 0;
            while (workOffset < totalWorkItems) {
                if (state->cancelRequested.load(std::memory_order_relaxed)) {
                    break;
                }
                const std::uint64_t batchCount = std::min(
                    adaptiveBatchSize,
                    totalWorkItems - workOffset);
                parameters.workOffsetLow = static_cast<std::uint32_t>(workOffset);
                parameters.workOffsetHigh = static_cast<std::uint32_t>(workOffset >> 32);
                std::memset(resultCountBuffer.contents, 0, sizeof(std::uint32_t));
                std::memset(resultOverflowBuffer.contents, 0, sizeof(std::uint32_t));

                id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
                if (!commandBuffer || !encoder) {
                    setError(error, "create Metal compute command");
                    return false;
                }
                [encoder setComputePipelineState:pipeline];
                [encoder setBytes:&parameters length:sizeof(parameters) atIndex:0];
                [encoder setBuffer:filterBuffer offset:0 atIndex:1];
                [encoder setBuffer:resultCountBuffer offset:0 atIndex:2];
                [encoder setBuffer:resultOverflowBuffer offset:0 atIndex:3];
                [encoder setBuffer:resultBuffer offset:0 atIndex:4];
                [encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(batchCount), 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(
                         static_cast<NSUInteger>(std::min<std::uint64_t>(
                             threadgroupWidth,
                             batchCount)),
                         1,
                         1)];
                [encoder endEncoding];
                [commandBuffer commit];
                [commandBuffer waitUntilCompleted];

                if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
                    setError(error, "run Metal scan", commandBuffer.error);
                    return false;
                }

                const std::uint32_t resultCount = *static_cast<std::uint32_t*>(
                    resultCountBuffer.contents);
                const std::uint32_t resultOverflow = *static_cast<std::uint32_t*>(
                    resultOverflowBuffer.contents);
                if (resultOverflow != 0 || resultCount > ResultCapacity) {
                    if (batchCount == 1) {
                        if (error) {
                            *error = "one Metal work item exceeded the result buffer capacity";
                        }
                        return false;
                    }
                    adaptiveBatchSize = std::max<std::uint64_t>(1, batchCount / 2);
                    continue;
                }

                if (resultCount > 0) {
                    const Match* deviceMatches = static_cast<const Match*>(resultBuffer.contents);
                    matches.assign(deviceMatches, deviceMatches + resultCount);
                    sink(matches);
                    state->matches.fetch_add(resultCount, std::memory_order_relaxed);
                }

                const std::uint64_t nextOffset = workOffset + batchCount;
                const std::uint64_t processed = candidatePrefix(
                    nextOffset,
                    yBlockCount,
                    yCount) - candidatePrefix(workOffset, yBlockCount, yCount);
                state->candidates.fetch_add(processed, std::memory_order_relaxed);
                workOffset = nextOffset;
            }

            if (!state->cancelRequested.load(std::memory_order_relaxed)) {
                state->completedItems.fetch_add(1, std::memory_order_relaxed);
            }
        }
        return true;
    }
}
