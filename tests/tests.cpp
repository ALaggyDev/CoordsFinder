#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "runner.hpp"
#include "textures.hpp"

namespace {
int failures = 0;

void expect(bool condition, const char* message)
{
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}

ScanConfig baseConfig()
{
    ScanConfig config;
    config.algorithm = TextureAlgorithm::Vanilla3;
    config.directions = { 0 };
    config.xRange = { -2, 3 };
    config.yRange = { 0, 1 };
    config.zRange = { -2, 3 };
    config.cpuTileSize = { 1, 1 };
    config.cudaTileSize = { 1, 1 };
    config.errorTolerance = 0;
    config.filter = { RotationInfo(0, 0, 0, 0) };
    return config;
}

void testDirectionalFilter()
{
    const std::vector<RotationInfo> input = {
        RotationInfo(2, 3, -4, 1),
        RotationInfo(-1, 0, 5, 1, true),
    };
    const std::vector<RotationInfo> rotated = makeDirectionalFilter(input, 90);
    expect(rotated[0].x == 4 && rotated[0].z == 2, "90-degree XZ rotation");
    expect(rotated[0].rotation == 2, "top-face variant rotation");
    expect(rotated[1].x == -5 && rotated[1].z == -1, "side XZ rotation");
    expect(rotated[1].rotation == 1, "side variant remains unchanged");

    const std::vector<RotationInfo> ordered = makeDirectionalFilter({
        RotationInfo(1, 0, 0, 0, true),
        RotationInfo(2, 0, 0, 0),
        RotationInfo(3, 0, 0, 1, true),
        RotationInfo(4, 0, 0, 1),
    }, 0);
    expect(ordered[0].x == 2 && ordered[1].x == 4, "four-state filters are ordered first");
    expect(ordered[2].x == 1 && ordered[3].x == 3, "filter ordering remains stable within rejection groups");
}

void testTextureAlgorithms()
{
    struct Vector {
        int x;
        int y;
        int z;
        std::uint8_t expected[5];
    };
    // Vanilla1 | Vanilla2 | Vanilla3 | Sodium1 | Sodium2. Each algorithm uses 4 variants.
    const Vector vectors[] = {
        { 0, 0, 0, { 0, 0, 2, 3, 2 } },
        { 1, 2, 3, { 0, 2, 3, 2, 0 } },
        { -1, -2, -3, { 3, 3, 0, 1, 0 } },
        { 353, -60, -53, { 2, 2, 0, 1, 3 } },
        { -29999984, -64, 29999983, { 1, 2, 1, 3, 0 } },
        { 29999999, 319, -29999999, { 3, 3, 2, 3, 3 } },
        { -538, 67, -575, { 3, 3, 1, 0, 2 } },
        { 17, -4, -31, { 3, 0, 3, 0, 3 } },
        { 1000000, 319, -1000000, { 0, 0, 2, 0, 0 } },
        { -30000000, -64, 30000000, { 0, 2, 0, 0, 2 } },
        { 1234567, 72, -7654321, { 0, 1, 2, 2, 1 } },
        { -16777216, 255, 16777215, { 3, 2, 3, 1, 1 } },
        { 31, 63, 127, { 1, 1, 1, 2, 2 } },
        { -32, -64, -128, { 1, 2, 0, 0, 0 } },
        { 4096, 0, 4096, { 3, 0, 3, 2, 3 } },
        { -4096, 1, -4096, { 1, 1, 0, 2, 1 } },
        { 7355608, 64, -8355608, { 2, 3, 1, 1, 2 } },
        { -1200345, 15, 9090909, { 2, 3, 0, 0, 3 } },
        { 8675309, -59, -3141592, { 0, 0, 1, 1, 2 } },
        { -2718281, 118, 1618033, { 2, 1, 2, 3, 1 } },
    };
    const TextureAlgorithm modes[] = {
        TextureAlgorithm::Vanilla1,
        TextureAlgorithm::Vanilla2,
        TextureAlgorithm::Vanilla3,
        TextureAlgorithm::Sodium1,
        TextureAlgorithm::Sodium2,
    };

    for (const Vector& vector : vectors) {
        for (std::size_t mode = 0; mode < 5; ++mode) {
            expect(
                getTexture(modes[mode], vector.x, vector.y, vector.z, 4) == vector.expected[mode],
                "texture sampler matches TextureRotations reference vector");
        }
    }
    expect(wrapAdd(std::numeric_limits<int>::max(), 1) == std::numeric_limits<int>::min(), "coordinate addition wraps at INT_MAX");
    expect(wrapAdd(std::numeric_limits<int>::min(), -1) == std::numeric_limits<int>::max(), "coordinate addition wraps at INT_MIN");
}

void testConfigParsing()
{
    ScanConfig modern;
    std::string error;
    const std::string root = COORDSFINDER_SOURCE_DIR;
    expect(loadScanConfig((root + "/tests/modern.conf").c_str(), &modern, &error), "load backend-specific tile config");
    expect(modern.cpuTileSize.x == 7 && modern.cpuTileSize.z == 9, "load CPU tile sizes");
    expect(modern.cudaTileSize.x == 70 && modern.cudaTileSize.z == 90, "load CUDA tile sizes");
    expect(modern.metalTileSize.x == 700 && modern.metalTileSize.z == 900, "load Metal tile sizes");
    expect(modern.errorTolerance == 2, "load error tolerance");
    expect(!modern.verbose, "load verbose setting");
    expect(modern.scanOrder == ScanOrder::Spiral, "load spiral scan order");
    ScanConfig example;
    expect(loadScanConfig((root + "/example.conf").c_str(), &example, &error), "load example config");
    expect(example.scanOrder == ScanOrder::Linear, "example config defaults to linear scanning");
    expect(!loadScanConfig((root + "/tests/invalid_duplicate.conf").c_str(), &modern, &error), "reject duplicate settings");
    expect(error.find("duplicate setting") != std::string::npos, "report duplicate setting clearly");
    expect(!loadScanConfig((root + "/tests/invalid_empty_range.conf").c_str(), &modern, &error), "reject empty scan ranges");
    expect(error.find("starts must be less than their ends") != std::string::npos, "report empty scan ranges clearly");
}

void testScanOrders()
{
    ScanConfig config = baseConfig();
    config.directions = { 0, 90 };
    std::string error;

    config.scanOrder = ScanOrder::Spiral;
    ScanPlan spiral;
    expect(makeScanPlan(config, config.cpuTileSize, &spiral, &error), "build spiral plan");
    expect(spiral.items.size() == 50, "spiral work-item count");
    expect(spiral.items[0].start.x == 0 && spiral.items[0].start.z == 0, "spiral starts at center");
    expect(spiral.items[0].direction == 0 && spiral.items[1].direction == 90, "spiral is tile-major across directions");

    std::set<std::tuple<int, int, int>> visited;
    for (const WorkItem& item : spiral.items) {
        visited.emplace(item.start.x, item.start.z, item.direction);
    }
    expect(visited.size() == spiral.items.size(), "spiral visits each tile and direction once");

    config.scanOrder = ScanOrder::Linear;
    ScanPlan linear;
    expect(makeScanPlan(config, config.cpuTileSize, &linear, &error), "build linear plan");
    expect(linear.items[0].start.x == -2 && linear.items[0].start.z == -2, "linear starts at minimum bounds");
    expect(linear.items[0].direction == 0 && linear.items[1].direction == 90, "linear is tile-major across directions");

    config.directions = { 0 };
    config.scanOrder = ScanOrder::Spiral;
    config.xRange = { 0, 10 };
    config.zRange = { -1, 1 };
    config.cpuTileSize = { 3, 1 };
    ScanPlan rectangle;
    expect(makeScanPlan(config, config.cpuTileSize, &rectangle, &error), "build non-square spiral plan");
    visited.clear();
    for (const WorkItem& item : rectangle.items) {
        visited.emplace(item.start.x, item.start.z, item.direction);
    }
    expect(rectangle.items.size() == 8 && visited.size() == 8, "non-square spiral covers every clipped tile once");
}

void testCpuScan()
{
    ScanConfig config = baseConfig();
    config.xRange = { 17, 18 };
    config.yRange = { -4, -3 };
    config.zRange = { -31, -30 };
    config.filter[0].rotation = getTexture(config.algorithm, 17, -4, -31, 4);

    ScanPlan plan;
    std::string error;
    expect(makeScanPlan(config, config.cpuTileSize, &plan, &error), "build CPU test plan");
    ScanState state;
    std::vector<Match> matches;
    const bool succeeded = runCpuScan(
        config,
        plan,
        2,
        &state,
        [&](const std::vector<Match>& batch) { matches.insert(matches.end(), batch.begin(), batch.end()); },
        &error);
    expect(succeeded, "run CPU scan");
    expect(matches.size() == 1, "CPU scan finds the expected coordinate");
    expect(state.candidates.load() == 1, "CPU scan candidate count");

    config = baseConfig();
    config.scanOrder = ScanOrder::Spiral;
    ScanPlan threadedPlan;
    expect(makeScanPlan(config, config.cpuTileSize, &threadedPlan, &error), "build threaded CPU plan");
    ScanState threadedState;
    matches.clear();
    expect(runCpuScan(
        config,
        threadedPlan,
        4,
        &threadedState,
        [&](const std::vector<Match>& batch) { matches.insert(matches.end(), batch.begin(), batch.end()); },
        &error), "run multithreaded CPU scan");
    std::set<std::tuple<int, int, int>> expected;
    std::set<std::tuple<int, int, int>> actual;
    for (int z = config.zRange.start; z < config.zRange.end; ++z) {
        for (int x = config.xRange.start; x < config.xRange.end; ++x) {
            if (getTexture(config.algorithm, x, 0, z, 4) == 0) {
                expected.emplace(x, 0, z);
            }
        }
    }
    for (const Match& match : matches) {
        actual.emplace(match.x, match.y, match.z);
    }
    expect(actual == expected, "multithreaded CPU scan matches scalar results");
    expect(threadedState.candidates.load() == 25, "multithreaded CPU candidate count");
}

#if defined(COORDSFINDER_HAS_METAL)
using MatchKey = std::tuple<int, int, int, int, int>;

std::multiset<MatchKey> matchKeys(const std::vector<Match>& matches)
{
    std::multiset<MatchKey> keys;
    for (const Match& match : matches) {
        keys.emplace(
            match.x,
            match.y,
            match.z,
            match.mismatches,
            match.direction);
    }
    return keys;
}

void testMetalScan()
{
    std::string reason;
    if (!metalAvailable(&reason)) {
        std::printf("Skipping Metal runtime tests: %s.\n", reason.c_str());
        return;
    }

    const TextureAlgorithm modes[] = {
        TextureAlgorithm::Vanilla1,
        TextureAlgorithm::Vanilla2,
        TextureAlgorithm::Vanilla3,
        TextureAlgorithm::Sodium1,
        TextureAlgorithm::Sodium2,
    };
    for (TextureAlgorithm mode : modes) {
        ScanConfig config = baseConfig();
        config.algorithm = mode;
        config.directions = { 0, 90 };
        config.xRange = { -2, 2 };
        // Cross both the old 16-Y and optimized 128-Y chunk boundaries.
        config.yRange = { -64, 66 };
        config.zRange = { -1, 2 };
        config.cpuTileSize = { 4, 3 };
        config.metalTileSize = { 4, 3 };
        config.filter = {
            RotationInfo(1, 0, -1, 2),
            RotationInfo(0, 1, 0, 1, true),
        };

        std::string error;
        ScanPlan plan;
        expect(makeScanPlan(config, config.metalTileSize, &plan, &error), "build Metal parity plan");

        ScanState cpuState;
        std::vector<Match> cpuMatches;
        expect(runCpuScan(
            config,
            plan,
            2,
            &cpuState,
            [&](const std::vector<Match>& batch) {
                cpuMatches.insert(cpuMatches.end(), batch.begin(), batch.end());
            },
            &error), "run CPU parity reference");

        ScanState metalState;
        std::vector<Match> metalMatches;
        expect(runMetalScan(
            config,
            plan,
            &metalState,
            [&](const std::vector<Match>& batch) {
                metalMatches.insert(metalMatches.end(), batch.begin(), batch.end());
            },
            &error), "run Metal parity scan");

        expect(matchKeys(metalMatches) == matchKeys(cpuMatches), "Metal matches CPU for every texture algorithm");
        expect(metalState.candidates.load() == cpuState.candidates.load(), "Metal parity candidate count");
        expect(metalState.completedItems.load() == plan.items.size(), "Metal parity work-item count");
    }

    for (TextureAlgorithm mode : modes) {
        ScanConfig lattice = baseConfig();
        lattice.algorithm = mode;
        lattice.scanOrder = ScanOrder::Linear;
        lattice.directions = { 90 };
        lattice.xRange = { -8, 9 };
        lattice.yRange = { -2, 3 };
        lattice.zRange = { -8, 9 };
        lattice.cpuTileSize = { 7, 6 };
        lattice.metalTileSize = { 7, 6 };
        lattice.filter = {
            RotationInfo(-2, -1, -4, 3),
            RotationInfo(-1, -1, -4, 3),
            RotationInfo(-3, -1, -3, 3),
            RotationInfo(-2, -1, -3, 3),
        };
        expect(metalUsesLatticeGate(lattice), "detect compatible 2x2 lattice gate");

        std::string error;
        ScanPlan plan;
        expect(makeScanPlan(lattice, lattice.metalTileSize, &plan, &error), "build lattice parity plan");
        ScanState cpuState;
        std::vector<Match> cpuMatches;
        expect(runCpuScan(
            lattice,
            plan,
            2,
            &cpuState,
            [&](const std::vector<Match>& batch) {
                cpuMatches.insert(cpuMatches.end(), batch.begin(), batch.end());
            },
            &error), "run lattice CPU reference");
        ScanState metalState;
        std::vector<Match> metalMatches;
        expect(runMetalScan(
            lattice,
            plan,
            &metalState,
            [&](const std::vector<Match>& batch) {
                metalMatches.insert(metalMatches.end(), batch.begin(), batch.end());
            },
            &error), "run optimized Metal lattice scan");
        expect(matchKeys(metalMatches) == matchKeys(cpuMatches), "lattice gate matches CPU across texture modes and tile boundaries");
        expect(metalState.candidates.load() == cpuState.candidates.load(), "lattice gate preserves logical candidate count");
        expect(metalState.completedItems.load() == plan.items.size(), "lattice gate completes every tile");
    }

    ScanConfig fallback = baseConfig();
    fallback.filter = {
        RotationInfo(0, 0, 0, 2),
        RotationInfo(1, 0, 0, 2),
        RotationInfo(0, 0, 1, 2),
        RotationInfo(1, 0, 1, 2),
    };
    fallback.scanOrder = ScanOrder::Spiral;
    expect(!metalUsesLatticeGate(fallback), "spiral scan uses the baseline Metal kernel");
    fallback.scanOrder = ScanOrder::Linear;
    fallback.directions = { 0, 90 };
    expect(!metalUsesLatticeGate(fallback), "multi-direction scan uses the baseline Metal kernel");
    fallback.directions = { 0 };
    fallback.errorTolerance = 1;
    expect(!metalUsesLatticeGate(fallback), "tolerant scan uses the baseline Metal kernel");

    ScanConfig overflow = baseConfig();
    overflow.xRange = { 0, 600 };
    overflow.yRange = { 0, 128 };
    overflow.zRange = { 0, 1 };
    overflow.metalTileSize = { 600, 1 };
    overflow.errorTolerance = 1;
    ScanPlan overflowPlan;
    std::string error;
    expect(makeScanPlan(overflow, overflow.metalTileSize, &overflowPlan, &error), "build Metal overflow plan");
    ScanState overflowState;
    std::uint64_t emitted = 0;
    expect(runMetalScan(
        overflow,
        overflowPlan,
        &overflowState,
        [&](const std::vector<Match>& batch) { emitted += batch.size(); },
        &error), "retry overflowing Metal result batches");
    expect(emitted == 76800, "Metal overflow retry preserves every match");
    expect(overflowState.candidates.load() == 76800, "Metal overflow retry counts candidates once");
}
#endif
}

int main()
{
    testTextureAlgorithms();
    testDirectionalFilter();
    testConfigParsing();
    testScanOrders();
    testCpuScan();
#if defined(COORDSFINDER_HAS_METAL)
    testMetalScan();
#endif
    if (failures != 0) {
        std::fprintf(stderr, "%d test(s) failed.\n", failures);
        return EXIT_FAILURE;
    }
    std::printf("All tests passed.\n");
    return EXIT_SUCCESS;
}
