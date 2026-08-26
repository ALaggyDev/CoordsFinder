#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "runner.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <utility>
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

struct MetalLatticeGateOffset {
    std::int32_t x;
    std::int32_t y;
    std::int32_t z;
    std::uint32_t filterIndex;
};

struct MetalLatticeParameters {
    std::uint32_t workOffsetLow;
    std::uint32_t workOffsetHigh;
    std::int32_t xStart;
    std::int32_t yStart;
    std::int32_t zStart;
    std::int32_t xEnd;
    std::int32_t zEnd;
    std::uint32_t yCount;
    std::uint32_t zCount;
    std::uint32_t yBlockCount;
    std::uint32_t filterBase;
    std::uint32_t filterCount;
    std::uint32_t resultCapacity;
    std::int32_t sampleXStart;
    std::int32_t sampleZStart;
    std::uint32_t sampleZCount;
    std::int32_t gateYOffset;
    std::uint32_t gateRotation;
    std::int32_t direction;
};

struct LatticeGatePlan {
    std::uint32_t rotation;
    std::int32_t yOffset;
    std::array<MetalLatticeGateOffset, 4> offsets;
    std::int32_t minimumX;
    std::int32_t maximumX;
    std::int32_t minimumZ;
    std::int32_t maximumZ;
};

static_assert(sizeof(MetalFilter) == 20, "Metal filter layout must match MSL");
static_assert(sizeof(MetalParameters) == 52, "Metal parameter layout must match MSL");
static_assert(sizeof(MetalLatticeGateOffset) == 16, "Metal lattice offset layout must match MSL");
static_assert(sizeof(MetalLatticeParameters) == 76, "Metal lattice parameters must match MSL");
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

struct LatticeGateOffset {
    int x;
    int y;
    int z;
    uint filterIndex;
};

struct LatticeScanParameters {
    uint workOffsetLow;
    uint workOffsetHigh;
    int xStart;
    int yStart;
    int zStart;
    int xEnd;
    int zEnd;
    uint yCount;
    uint zCount;
    uint yBlockCount;
    uint filterBase;
    uint filterCount;
    uint resultCapacity;
    int sampleXStart;
    int sampleZStart;
    uint sampleZCount;
    int gateYOffset;
    uint gateRotation;
    int direction;
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

inline ulong coordinate_seed_xz(int x, int z) {
    uint xProductBits = as_type<uint>(x) * 3129871u;
    int xProduct = as_type<int>(xProductBits);
    ulong seed = ulong(long(xProduct));
    seed ^= ulong(long(z)) * 116129781ul;
    return seed;
}

inline ulong coordinate_random_raw_from_xz_seed(ulong xzSeed, int y) {
    ulong seed = xzSeed;
    seed ^= ulong(long(y));
    return seed * seed * 42317861ul + seed * 11ul;
}

inline ulong coordinate_random_raw(int x, int y, int z) {
    return coordinate_random_raw_from_xz_seed(coordinate_seed_xz(x, z), y);
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
    ulong raw = coordinate_random_raw(x, y, z);
    switch (kTextureMode) {
    case 0: {
        int low = as_type<int>(uint(raw));
        return absolute_modulo(low >> 16, 4u);
    }
    case 1: {
        long shifted = as_type<long>(raw) >> 16;
        return absolute_modulo(random_vanilla_2(as_type<ulong>(shifted)), 4u);
    }
    case 2: {
        long shifted = as_type<long>(raw) >> 16;
        return legacy_next_int_4(as_type<ulong>(shifted));
    }
    case 3: {
        long shifted = as_type<long>(raw) >> 16;
        return absolute_modulo(random_sodium_1(as_type<ulong>(shifted)), 4u);
    }
    case 4:
    default: {
        long shifted = as_type<long>(raw) >> 16;
        return absolute_modulo(random_sodium_2(as_type<ulong>(shifted)), 4u);
    }
    }
}

inline uint texture_variant_from_xz_seed(ulong xzSeed, int y) {
    ulong raw = coordinate_random_raw_from_xz_seed(xzSeed, y);
    switch (kTextureMode) {
    case 0: {
        int low = as_type<int>(uint(raw));
        return absolute_modulo(low >> 16, 4u);
    }
    case 1: {
        long shifted = as_type<long>(raw) >> 16;
        return absolute_modulo(random_vanilla_2(as_type<ulong>(shifted)), 4u);
    }
    case 2: {
        long shifted = as_type<long>(raw) >> 16;
        return legacy_next_int_4(as_type<ulong>(shifted));
    }
    case 3: {
        long shifted = as_type<long>(raw) >> 16;
        return absolute_modulo(random_sodium_1(as_type<ulong>(shifted)), 4u);
    }
    case 4:
    default: {
        long shifted = as_type<long>(raw) >> 16;
        return absolute_modulo(random_sodium_2(as_type<ulong>(shifted)), 4u);
    }
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

inline void verify_lattice_hit(
    constant LatticeScanParameters& parameters,
    constant Filter* filters,
    constant LatticeGateOffset* gateOffsets,
    device atomic_uint* resultCount,
    device atomic_uint* resultOverflow,
    device Match* results,
    int sampleX,
    int sampleZ,
    uint candidateYOffset)
{
    int candidateY = as_type<int>(
        as_type<uint>(parameters.yStart) + candidateYOffset);

    for (uint gateIndex = 0u; gateIndex < 4u; ++gateIndex) {
        LatticeGateOffset gate = gateOffsets[gateIndex];
        int candidateX = wrap_add(sampleX, -gate.x);
        int candidateZ = wrap_add(sampleZ, -gate.z);
        if (candidateX < parameters.xStart || candidateX >= parameters.xEnd
            || candidateZ < parameters.zStart || candidateZ >= parameters.zEnd) {
            continue;
        }

        bool matched = true;
        for (uint filterIndex = 0u; filterIndex < parameters.filterCount; ++filterIndex) {
            if (filterIndex == gate.filterIndex) {
                continue;
            }
            Filter filter = filters[parameters.filterBase + filterIndex];
            uint visibleVariant = texture_variant(
                wrap_add(candidateX, filter.x),
                wrap_add(candidateY, filter.y),
                wrap_add(candidateZ, filter.z)) & filter.visibleMask;
            if (visibleVariant != filter.rotation) {
                matched = false;
                break;
            }
        }

        if (!matched) {
            continue;
        }

        uint resultIndex = atomic_fetch_add_explicit(
            resultCount,
            1u,
            memory_order_relaxed);
        if (resultIndex < parameters.resultCapacity) {
            results[resultIndex] = {
                candidateX,
                candidateY,
                candidateZ,
                0,
                parameters.direction,
            };
        }
        else {
            atomic_store_explicit(resultOverflow, 1u, memory_order_relaxed);
        }
    }
}

kernel void search_coordinates_lattice(
    constant LatticeScanParameters& parameters [[buffer(0)]],
    constant Filter* filters [[buffer(1)]],
    constant LatticeGateOffset* gateOffsets [[buffer(2)]],
    device atomic_uint* resultCount [[buffer(3)]],
    device atomic_uint* resultOverflow [[buffer(4)]],
    device Match* results [[buffer(5)]],
    uint threadIndex [[thread_position_in_grid]])
{
    ulong workOffset = make_ulong(parameters.workOffsetLow, parameters.workOffsetHigh);
    ulong workIndex = workOffset + ulong(threadIndex);
    ulong sampleIndex = workIndex / ulong(parameters.yBlockCount);
    uint yBlockIndex = uint(workIndex % ulong(parameters.yBlockCount));
    ulong sampleXIndex = sampleIndex / ulong(parameters.sampleZCount);
    ulong sampleZIndex = sampleIndex % ulong(parameters.sampleZCount);
    int sampleX = int(long(parameters.sampleXStart) + long(sampleXIndex) * 2l);
    int sampleZ = int(long(parameters.sampleZStart) + long(sampleZIndex) * 2l);
    ulong gateXZSeed = coordinate_seed_xz(sampleX, sampleZ);
    ulong yBase = ulong(yBlockIndex) * ulong(candidatesPerThreadY);
    uint yBatchCount = uint(min(
        ulong(candidatesPerThreadY),
        ulong(parameters.yCount) - yBase));

    // Compact the successful gate Ys so SIMD lanes enter full verification
    // together instead of diverging immediately after the first hash.
    ulong lowHits = 0ul;
    ulong highHits = 0ul;
    for (uint localY = 0u; localY < yBatchCount; ++localY) {
        uint candidateYOffset = uint(yBase) + localY;
        int candidateY = as_type<int>(
            as_type<uint>(parameters.yStart) + candidateYOffset);
        uint visibleVariant = texture_variant_from_xz_seed(
            gateXZSeed,
            wrap_add(candidateY, parameters.gateYOffset));
        if (visibleVariant == parameters.gateRotation) {
            if (localY < 64u) {
                lowHits |= 1ul << localY;
            }
            else {
                highHits |= 1ul << (localY - 64u);
            }
        }
    }

    while (lowHits != 0ul) {
        uint localY = uint(ctz(lowHits));
        lowHits &= lowHits - 1ul;
        verify_lattice_hit(
            parameters,
            filters,
            gateOffsets,
            resultCount,
            resultOverflow,
            results,
            sampleX,
            sampleZ,
            uint(yBase) + localY);
    }
    while (highHits != 0ul) {
        uint localY = uint(ctz(highHits)) + 64u;
        highHits &= highHits - 1ul;
        verify_lattice_hit(
            parameters,
            filters,
            gateOffsets,
            resultCount,
            resultOverflow,
            results,
            sampleX,
            sampleZ,
            uint(yBase) + localY);
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

bool makePipelines(
    id<MTLDevice> device,
    TextureAlgorithm algorithm,
    bool includeLatticeGate,
    id<MTLComputePipelineState>* pipeline,
    id<MTLComputePipelineState>* latticePipeline,
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

    *latticePipeline = nil;
    if (includeLatticeGate) {
        id<MTLFunction> latticeFunction = [library newFunctionWithName:@"search_coordinates_lattice"
                                                        constantValues:constants
                                                                 error:&detail];
        if (!latticeFunction) {
            setError(error, "specialize Metal lattice kernel", detail);
            return false;
        }
        *latticePipeline = [device newComputePipelineStateWithFunction:latticeFunction error:&detail];
        if (!*latticePipeline) {
            setError(error, "create Metal lattice pipeline", detail);
            return false;
        }
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

int positiveModulo2(std::int32_t value)
{
    const int remainder = value % 2;
    return remainder < 0 ? remainder + 2 : remainder;
}

std::int64_t firstEvenAtLeast(std::int64_t value)
{
    const std::int64_t remainder = value % 2;
    if (remainder == 0) {
        return value;
    }
    return remainder > 0 ? value + 1 : value - remainder;
}

std::optional<LatticeGatePlan> makeLatticeGatePlan(
    const ScanConfig& config,
    const std::vector<MetalFilter>& directionalFilters)
{
    if (config.errorTolerance != 0
        || config.scanOrder != ScanOrder::Linear
        || config.directions.size() != 1
        || directionalFilters.size() != config.filter.size()) {
        return std::nullopt;
    }

    const auto [minimumXOffset, maximumXOffset] = std::minmax_element(
        directionalFilters.begin(),
        directionalFilters.end(),
        [](const MetalFilter& lhs, const MetalFilter& rhs) { return lhs.x < rhs.x; });
    const auto [minimumZOffset, maximumZOffset] = std::minmax_element(
        directionalFilters.begin(),
        directionalFilters.end(),
        [](const MetalFilter& lhs, const MetalFilter& rhs) { return lhs.z < rhs.z; });
    if (directionalFilters.empty()
        || static_cast<std::int64_t>(config.xRange.start) + minimumXOffset->x < std::numeric_limits<std::int32_t>::min()
        || static_cast<std::int64_t>(config.xRange.end) - 1 + maximumXOffset->x > std::numeric_limits<std::int32_t>::max()
        || static_cast<std::int64_t>(config.zRange.start) + minimumZOffset->z < std::numeric_limits<std::int32_t>::min()
        || static_cast<std::int64_t>(config.zRange.end) - 1 + maximumZOffset->z > std::numeric_limits<std::int32_t>::max()) {
        return std::nullopt;
    }

    using GroupKey = std::pair<std::int32_t, std::uint32_t>;
    using ResidueGroup = std::array<std::optional<MetalLatticeGateOffset>, 4>;
    std::map<GroupKey, ResidueGroup> groups;
    for (std::size_t index = 0; index < directionalFilters.size(); ++index) {
        const MetalFilter& filter = directionalFilters[index];
        if (filter.visibleMask != 3) {
            continue;
        }
        const int residue = positiveModulo2(filter.x) * 2 + positiveModulo2(filter.z);
        std::optional<MetalLatticeGateOffset>& slot = groups[{ filter.y, filter.rotation }][residue];
        if (!slot) {
            slot = MetalLatticeGateOffset {
                filter.x,
                filter.y,
                filter.z,
                static_cast<std::uint32_t>(index),
            };
        }
    }

    for (const auto& [key, residues] : groups) {
        if (!std::all_of(residues.begin(), residues.end(), [](const auto& value) {
                return value.has_value();
            })) {
            continue;
        }

        LatticeGatePlan plan = {};
        plan.yOffset = key.first;
        plan.rotation = key.second;
        for (std::size_t index = 0; index < residues.size(); ++index) {
            plan.offsets[index] = *residues[index];
        }
        const auto [minimumX, maximumX] = std::minmax_element(
            plan.offsets.begin(),
            plan.offsets.end(),
            [](const auto& lhs, const auto& rhs) { return lhs.x < rhs.x; });
        const auto [minimumZ, maximumZ] = std::minmax_element(
            plan.offsets.begin(),
            plan.offsets.end(),
            [](const auto& lhs, const auto& rhs) { return lhs.z < rhs.z; });
        plan.minimumX = minimumX->x;
        plan.maximumX = maximumX->x;
        plan.minimumZ = minimumZ->z;
        plan.maximumZ = maximumZ->z;
        return plan;
    }
    return std::nullopt;
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

bool metalUsesLatticeGate(const ScanConfig& config)
{
    return makeLatticeGatePlan(config, makeMetalFilters(config)).has_value();
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

        const std::vector<MetalFilter> filters = makeMetalFilters(config);
        const std::optional<LatticeGatePlan> latticeGate = makeLatticeGatePlan(
            config,
            filters);
        id<MTLComputePipelineState> pipeline = nil;
        id<MTLComputePipelineState> latticePipeline = nil;
        if (!makePipelines(
                device,
                config.algorithm,
                latticeGate.has_value(),
                &pipeline,
                &latticePipeline,
                error)) {
            return false;
        }

        id<MTLBuffer> filterBuffer = [device newBufferWithBytes:filters.data()
                                                        length:filters.size() * sizeof(MetalFilter)
                                                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> latticeOffsetBuffer = nil;
        if (latticeGate) {
            latticeOffsetBuffer = [device newBufferWithBytes:latticeGate->offsets.data()
                                                        length:sizeof(latticeGate->offsets)
                                                       options:MTLResourceStorageModeShared];
        }
        id<MTLBuffer> resultCountBuffer = [device newBufferWithLength:sizeof(std::uint32_t)
                                                              options:MTLResourceStorageModeShared];
        id<MTLBuffer> resultOverflowBuffer = [device newBufferWithLength:sizeof(std::uint32_t)
                                                                 options:MTLResourceStorageModeShared];
        id<MTLBuffer> resultBuffer = [device newBufferWithLength:sizeof(Match) * ResultCapacity
                                                         options:MTLResourceStorageModeShared];
        if (!filterBuffer
            || (latticeGate && !latticeOffsetBuffer)
            || !resultCountBuffer
            || !resultOverflowBuffer
            || !resultBuffer) {
            setError(error, "allocate Metal buffers");
            return false;
        }

        id<MTLComputePipelineState> activePipeline = latticePipeline
            ? latticePipeline
            : pipeline;
        const NSUInteger executionWidth = activePipeline.threadExecutionWidth;
        const NSUInteger preferredWidth = std::min<NSUInteger>(
            256,
            activePipeline.maxTotalThreadsPerThreadgroup);
        const NSUInteger threadgroupWidth = std::max<NSUInteger>(
            executionWidth,
            preferredWidth - preferredWidth % executionWidth);
        std::fprintf(stderr, "Metal device: %s (%lu threads per threadgroup, %s).\n",
            device.name.UTF8String,
            static_cast<unsigned long>(threadgroupWidth),
            latticeGate ? "2x2 lattice gate" : "baseline");

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
            const bool useLatticeGate = latticeGate.has_value();
            std::uint64_t columnCount = 0;
            std::uint64_t totalWorkItems = 0;
            bool candidatesSaturated = false;
            const std::uint64_t itemCandidates = workItemCandidateCount(
                item,
                &candidatesSaturated);
            if (candidatesSaturated
                || yCount > std::numeric_limits<std::uint32_t>::max()
                || zCount > std::numeric_limits<std::uint32_t>::max()
                || yBlockCount > std::numeric_limits<std::uint32_t>::max()) {
                if (error) {
                    *error = "a Metal tile is too large; reduce metalTileSize or the coordinate range";
                }
                return false;
            }

            MetalParameters parameters = {};
            MetalLatticeParameters latticeParameters = {};
            if (useLatticeGate) {
                const std::int64_t sampleXLow = static_cast<std::int64_t>(item.start.x)
                    + latticeGate->minimumX;
                const std::int64_t sampleXHigh = static_cast<std::int64_t>(item.end.x) - 1
                    + latticeGate->maximumX;
                const std::int64_t sampleZLow = static_cast<std::int64_t>(item.start.z)
                    + latticeGate->minimumZ;
                const std::int64_t sampleZHigh = static_cast<std::int64_t>(item.end.z) - 1
                    + latticeGate->maximumZ;
                const std::int64_t sampleXStart = firstEvenAtLeast(sampleXLow);
                const std::int64_t sampleZStart = firstEvenAtLeast(sampleZLow);
                const std::uint64_t sampleXCount = static_cast<std::uint64_t>(
                    (sampleXHigh - sampleXStart) / 2 + 1);
                const std::uint64_t sampleZCount = static_cast<std::uint64_t>(
                    (sampleZHigh - sampleZStart) / 2 + 1);
                if (sampleXCount > std::numeric_limits<std::uint32_t>::max()
                    || sampleZCount > std::numeric_limits<std::uint32_t>::max()
                    || !checkedMultiply(sampleXCount, sampleZCount, &columnCount)
                    || !checkedMultiply(columnCount, yBlockCount, &totalWorkItems)) {
                    if (error) {
                        *error = "a Metal lattice tile is too large; reduce metalTileSize";
                    }
                    return false;
                }

                latticeParameters.xStart = item.start.x;
                latticeParameters.yStart = item.start.y;
                latticeParameters.zStart = item.start.z;
                latticeParameters.xEnd = item.end.x;
                latticeParameters.zEnd = item.end.z;
                latticeParameters.yCount = static_cast<std::uint32_t>(yCount);
                latticeParameters.zCount = static_cast<std::uint32_t>(zCount);
                latticeParameters.yBlockCount = static_cast<std::uint32_t>(yBlockCount);
                latticeParameters.filterBase = static_cast<std::uint32_t>(
                    item.directionIndex * config.filter.size());
                latticeParameters.filterCount = static_cast<std::uint32_t>(config.filter.size());
                latticeParameters.resultCapacity = ResultCapacity;
                latticeParameters.sampleXStart = static_cast<std::int32_t>(sampleXStart);
                latticeParameters.sampleZStart = static_cast<std::int32_t>(sampleZStart);
                latticeParameters.sampleZCount = static_cast<std::uint32_t>(sampleZCount);
                latticeParameters.gateYOffset = latticeGate->yOffset;
                latticeParameters.gateRotation = latticeGate->rotation;
                latticeParameters.direction = item.direction;
            }
            else {
                if (!checkedMultiply(xCount, zCount, &columnCount)
                    || !checkedMultiply(columnCount, yBlockCount, &totalWorkItems)) {
                    if (error) {
                        *error = "a Metal tile is too large; reduce metalTileSize or the coordinate range";
                    }
                    return false;
                }
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
            }

            std::uint64_t workOffset = 0;
            std::uint64_t creditedCandidates = 0;
            while (workOffset < totalWorkItems) {
                if (state->cancelRequested.load(std::memory_order_relaxed)) {
                    break;
                }
                const std::uint64_t batchCount = std::min(
                    adaptiveBatchSize,
                    totalWorkItems - workOffset);
                if (useLatticeGate) {
                    latticeParameters.workOffsetLow = static_cast<std::uint32_t>(workOffset);
                    latticeParameters.workOffsetHigh = static_cast<std::uint32_t>(workOffset >> 32);
                }
                else {
                    parameters.workOffsetLow = static_cast<std::uint32_t>(workOffset);
                    parameters.workOffsetHigh = static_cast<std::uint32_t>(workOffset >> 32);
                }
                std::memset(resultCountBuffer.contents, 0, sizeof(std::uint32_t));
                std::memset(resultOverflowBuffer.contents, 0, sizeof(std::uint32_t));

                id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
                if (!commandBuffer || !encoder) {
                    setError(error, "create Metal compute command");
                    return false;
                }
                if (useLatticeGate) {
                    [encoder setComputePipelineState:latticePipeline];
                    [encoder setBytes:&latticeParameters length:sizeof(latticeParameters) atIndex:0];
                    [encoder setBuffer:filterBuffer offset:0 atIndex:1];
                    [encoder setBuffer:latticeOffsetBuffer offset:0 atIndex:2];
                    [encoder setBuffer:resultCountBuffer offset:0 atIndex:3];
                    [encoder setBuffer:resultOverflowBuffer offset:0 atIndex:4];
                    [encoder setBuffer:resultBuffer offset:0 atIndex:5];
                }
                else {
                    [encoder setComputePipelineState:pipeline];
                    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:0];
                    [encoder setBuffer:filterBuffer offset:0 atIndex:1];
                    [encoder setBuffer:resultCountBuffer offset:0 atIndex:2];
                    [encoder setBuffer:resultOverflowBuffer offset:0 atIndex:3];
                    [encoder setBuffer:resultBuffer offset:0 atIndex:4];
                }
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
                const std::uint64_t nextCreditedCandidates = useLatticeGate
                    ? nextOffset == totalWorkItems
                        ? itemCandidates
                        : static_cast<std::uint64_t>(
                            static_cast<long double>(itemCandidates)
                            * static_cast<long double>(nextOffset)
                            / static_cast<long double>(totalWorkItems))
                    : creditedCandidates + candidatePrefix(
                        nextOffset,
                        yBlockCount,
                        yCount) - candidatePrefix(workOffset, yBlockCount, yCount);
                state->candidates.fetch_add(
                    nextCreditedCandidates - creditedCandidates,
                    std::memory_order_relaxed);
                creditedCandidates = nextCreditedCandidates;
                workOffset = nextOffset;
            }

            if (!state->cancelRequested.load(std::memory_order_relaxed)) {
                state->completedItems.fetch_add(1, std::memory_order_relaxed);
            }
        }
        return true;
    }
}
