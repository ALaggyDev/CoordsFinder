#include "scan.hpp"

#include <algorithm>
#include <limits>
#include <new>

namespace {
struct Int2 {
    int x;
    int z;
};

Int2 rotateXzOffset(int x, int z, int direction)
{
    switch (direction / 90) {
    case 1:
        return { -z, x };
    case 2:
        return { -x, -z };
    case 3:
        return { z, -x };
    case 0:
    default:
        return { x, z };
    }
}

std::uint64_t span(int start, int end)
{
    return static_cast<std::uint64_t>(static_cast<std::int64_t>(end) - start);
}

std::uint64_t saturatedMultiply(std::uint64_t lhs, std::uint64_t rhs, bool* saturated)
{
    if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs) {
        if (saturated) {
            *saturated = true;
        }
        return std::numeric_limits<std::uint64_t>::max();
    }
    return lhs * rhs;
}

std::uint64_t saturatedAdd(std::uint64_t lhs, std::uint64_t rhs, bool* saturated)
{
    if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) {
        if (saturated) {
            *saturated = true;
        }
        return std::numeric_limits<std::uint64_t>::max();
    }
    return lhs + rhs;
}
}

std::vector<RotationInfo> makeDirectionalFilter(
    const std::vector<RotationInfo>& filter,
    int direction)
{
    std::vector<RotationInfo> result;
    result.reserve(filter.size());

    const int quarterTurns = direction / 90;
    for (RotationInfo info : filter) {
        const Int2 rotated = rotateXzOffset(info.x, info.z, direction);
        info.x = static_cast<std::int8_t>(rotated.x);
        info.z = static_cast<std::int8_t>(rotated.z);
        if (info.visibleMask == 3) {
            info.rotation = static_cast<std::uint8_t>((info.rotation + quarterTurns) % 4);
        }
        result.push_back(info);
    }
    // Four-state samples reject three quarters of candidates, ahead of two-state side samples.
    // Preserve the config order within each group so equivalent filters remain deterministic.
    std::stable_sort(result.begin(), result.end(), [](const RotationInfo& lhs, const RotationInfo& rhs) {
        return lhs.visibleMask > rhs.visibleMask;
    });
    return result;
}

std::uint64_t workItemCandidateCount(const WorkItem& item, bool* saturated)
{
    bool overflow = false;
    const std::uint64_t xy = saturatedMultiply(
        span(item.start.x, item.end.x),
        span(item.start.y, item.end.y),
        &overflow);
    const std::uint64_t xyz = saturatedMultiply(xy, span(item.start.z, item.end.z), &overflow);
    if (saturated) {
        *saturated = overflow;
    }
    return xyz;
}

bool makeScanPlan(
    const ScanConfig& config,
    TileSize tileSize,
    ScanPlan* plan,
    std::string* error)
{
    if (!plan) {
        if (error) {
            *error = "missing scan plan output";
        }
        return false;
    }

    if (tileSize.x <= 0 || tileSize.z <= 0) {
        if (error) {
            *error = "tile dimensions must be positive";
        }
        return false;
    }

    const std::uint64_t xSpan = span(config.xRange.start, config.xRange.end);
    const std::uint64_t zSpan = span(config.zRange.start, config.zRange.end);
    const std::uint64_t tileX = static_cast<std::uint64_t>(tileSize.x);
    const std::uint64_t tileZ = static_cast<std::uint64_t>(tileSize.z);
    const std::uint64_t xTiles = 1 + (xSpan - 1) / tileX;
    const std::uint64_t zTiles = 1 + (zSpan - 1) / tileZ;

    bool workCountOverflow = false;
    const std::uint64_t tileCount = saturatedMultiply(xTiles, zTiles, &workCountOverflow);
    const std::uint64_t workCount = saturatedMultiply(
        tileCount,
        static_cast<std::uint64_t>(config.directions.size()),
        &workCountOverflow);

    if (workCountOverflow || workCount > plan->items.max_size()) {
        if (error) {
            *error = "scan contains too many work items for this build";
        }
        return false;
    }

    ScanPlan built;
    try {
        built.items.reserve(static_cast<std::size_t>(workCount));
    }
    catch (const std::bad_alloc&) {
        if (error) {
            *error = "not enough memory to build the scan plan";
        }
        return false;
    }

    // Tiles are anchored at the minimum bounds; edge tiles retain exclusive ends.
    auto addTile = [&](std::uint64_t tileIndexX, std::uint64_t tileIndexZ, std::size_t directionIndex) {
        const std::int64_t xStart = static_cast<std::int64_t>(config.xRange.start)
            + static_cast<std::int64_t>(tileIndexX * tileX);
        const std::int64_t zStart = static_cast<std::int64_t>(config.zRange.start)
            + static_cast<std::int64_t>(tileIndexZ * tileZ);
        const std::int64_t xEnd = std::min<std::int64_t>(
            xStart + tileSize.x,
            config.xRange.end);
        const std::int64_t zEnd = std::min<std::int64_t>(
            zStart + tileSize.z,
            config.zRange.end);

        WorkItem item = {
            { static_cast<int>(xStart), config.yRange.start, static_cast<int>(zStart) },
            { static_cast<int>(xEnd), config.yRange.end, static_cast<int>(zEnd) },
            directionIndex,
            config.directions[directionIndex],
        };
        bool itemSaturated = false;
        const std::uint64_t candidates = workItemCandidateCount(item, &itemSaturated);
        built.totalCandidates = saturatedAdd(
            built.totalCandidates,
            candidates,
            &built.totalCandidatesSaturated);
        built.totalCandidatesSaturated = built.totalCandidatesSaturated || itemSaturated;
        built.items.push_back(item);
    };

    if (config.scanOrder == ScanOrder::Linear) {
        // X -> Z -> Direction ordering.
        for (std::uint64_t x = 0; x < xTiles; ++x) {
            for (std::uint64_t z = 0; z < zTiles; ++z) {
                for (std::size_t direction = 0; direction < config.directions.size(); ++direction) {
                    addTile(x, z, direction);
                }
            }
        }
    }
    else {
        const std::int64_t centerX = static_cast<std::int64_t>((xSpan - 1) / 2 / tileX);
        const std::int64_t centerZ = static_cast<std::int64_t>((zSpan - 1) / 2 / tileZ);
        const std::int64_t maxRadius = std::max({
            centerX,
            centerZ,
            static_cast<std::int64_t>(xTiles - 1) - centerX,
            static_cast<std::int64_t>(zTiles - 1) - centerZ,
        });

        // Spiral mode is tile-major: check every direction before moving farther out.
        auto emit = [&](std::int64_t x, std::int64_t z) {
            if (x < 0 || z < 0 || x >= static_cast<std::int64_t>(xTiles) || z >= static_cast<std::int64_t>(zTiles)) {
                return;
            }
            for (std::size_t direction = 0; direction < config.directions.size(); ++direction) {
                addTile(static_cast<std::uint64_t>(x), static_cast<std::uint64_t>(z), direction);
            }
        };

        emit(centerX, centerZ);
        for (std::int64_t radius = 1; radius <= maxRadius; ++radius) {
            // Walk clockwise around each ring. Each side is clipped before iteration,
            // avoiding quadratic work for very long, narrow search rectangles.
            const std::int64_t right = centerX + radius;
            if (right >= 0 && right < static_cast<std::int64_t>(xTiles)) {
                const std::int64_t from = std::max<std::int64_t>(0, centerZ - radius + 1);
                const std::int64_t to = std::min<std::int64_t>(static_cast<std::int64_t>(zTiles) - 1, centerZ + radius);
                for (std::int64_t z = from; z <= to; ++z) {
                    emit(right, z);
                }
            }

            const std::int64_t bottom = centerZ + radius;
            if (bottom >= 0 && bottom < static_cast<std::int64_t>(zTiles)) {
                const std::int64_t from = std::min<std::int64_t>(static_cast<std::int64_t>(xTiles) - 1, centerX + radius - 1);
                const std::int64_t to = std::max<std::int64_t>(0, centerX - radius);
                for (std::int64_t x = from; x >= to; --x) {
                    emit(x, bottom);
                }
            }

            const std::int64_t left = centerX - radius;
            if (left >= 0 && left < static_cast<std::int64_t>(xTiles)) {
                const std::int64_t from = std::min<std::int64_t>(static_cast<std::int64_t>(zTiles) - 1, centerZ + radius - 1);
                const std::int64_t to = std::max<std::int64_t>(0, centerZ - radius);
                for (std::int64_t z = from; z >= to; --z) {
                    emit(left, z);
                }
            }

            const std::int64_t top = centerZ - radius;
            if (top >= 0 && top < static_cast<std::int64_t>(zTiles)) {
                const std::int64_t from = std::max<std::int64_t>(0, centerX - radius + 1);
                const std::int64_t to = std::min<std::int64_t>(static_cast<std::int64_t>(xTiles) - 1, centerX + radius);
                for (std::int64_t x = from; x <= to; ++x) {
                    emit(x, top);
                }
            }
        }
    }

    if (built.items.size() != static_cast<std::size_t>(workCount)) {
        if (error) {
            *error = "internal error: scan order did not cover every tile exactly once";
        }
        return false;
    }

    *plan = std::move(built);
    return true;
}
