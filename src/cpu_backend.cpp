#include "runner.hpp"

#include <algorithm>
#include <cstdio>
#include <mutex>
#include <thread>

#include "matcher.hpp"

namespace {
constexpr std::size_t ResultBatchSize = 256;

template <TextureAlgorithm Mode>
void scanWorkItem(
    const ScanConfig& config,
    const WorkItem& item,
    const std::vector<RotationInfo>& filter,
    ScanState* state,
    std::vector<Match>* matches,
    const std::function<void()>& flush)
{
    for (std::int64_t x = item.start.x; x <= item.end.x; ++x) {
        for (std::int64_t z = item.start.z; z <= item.end.z; ++z) {
            if (state->cancelRequested.load(std::memory_order_relaxed)) {
                return;
            }

            for (std::int64_t y = item.start.y; y <= item.end.y; ++y) {
                const int mismatches = countMismatches<Mode>(
                    static_cast<std::int32_t>(x),
                    static_cast<std::int32_t>(y),
                    static_cast<std::int32_t>(z),
                    filter.data(),
                    filter.size(),
                    config.errorTolerance);

                if (mismatches <= config.errorTolerance) {
                    matches->push_back({
                        static_cast<std::int32_t>(x),
                        static_cast<std::int32_t>(y),
                        static_cast<std::int32_t>(z),
                        mismatches,
                        item.direction,
                    });
                    if (matches->size() == ResultBatchSize) {
                        flush();
                    }
                }
            }
        }
    }
}

template <TextureAlgorithm Mode>
bool runMode(
    const ScanConfig& config,
    const ScanPlan& plan,
    unsigned int threadCount,
    ScanState* state,
    const MatchSink& sink)
{
    std::vector<std::vector<RotationInfo>> filters;
    filters.reserve(config.directions.size());
    for (int direction : config.directions) {
        filters.push_back(makeDirectionalFilter(config.filter, direction));
    }

    std::atomic<std::size_t> nextItem { 0 };
    std::mutex sinkMutex;
    threadCount = static_cast<unsigned int>(std::max<std::size_t>(
        1,
        std::min<std::size_t>(threadCount, plan.items.size())));

    auto worker = [&] {
        std::vector<Match> matches;
        matches.reserve(ResultBatchSize);

        // Batch rare matches so worker threads do not lock stdout per coordinate.
        auto flush = [&] {
            if (matches.empty()) {
                return;
            }
            std::lock_guard<std::mutex> lock(sinkMutex);
            sink(matches);
            state->matches.fetch_add(matches.size(), std::memory_order_relaxed);
            matches.clear();
        };

        // An atomic cursor gives dynamic load balancing while retaining plan priority.
        while (!state->cancelRequested.load(std::memory_order_relaxed)) {
            const std::size_t index = nextItem.fetch_add(1, std::memory_order_relaxed);
            if (index >= plan.items.size()) {
                break;
            }

            const WorkItem& item = plan.items[index];
            if (config.verbose) {
                std::lock_guard<std::mutex> lock(sinkMutex);
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

            scanWorkItem<Mode>(config, item, filters[item.directionIndex], state, &matches, flush);
            flush();
            if (!state->cancelRequested.load(std::memory_order_relaxed)) {
                state->candidates.fetch_add(workItemCandidateCount(item), std::memory_order_relaxed);
                state->completedItems.fetch_add(1, std::memory_order_relaxed);
            }
        }
        flush();
    };

    std::vector<std::thread> workers;
    workers.reserve(threadCount);
    for (unsigned int i = 0; i < threadCount; ++i) {
        workers.emplace_back(worker);
    }
    for (std::thread& thread : workers) {
        thread.join();
    }
    return true;
}
}

bool runCpuScan(
    const ScanConfig& config,
    const ScanPlan& plan,
    unsigned int threadCount,
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

    switch (config.algorithm) {
    // Dispatch once here instead of branching on the texture algorithm for every sample.
    case TextureAlgorithm::Vanilla1:
        return runMode<TextureAlgorithm::Vanilla1>(config, plan, threadCount, state, sink);
    case TextureAlgorithm::Vanilla2:
        return runMode<TextureAlgorithm::Vanilla2>(config, plan, threadCount, state, sink);
    case TextureAlgorithm::Vanilla3:
        return runMode<TextureAlgorithm::Vanilla3>(config, plan, threadCount, state, sink);
    case TextureAlgorithm::Sodium1:
        return runMode<TextureAlgorithm::Sodium1>(config, plan, threadCount, state, sink);
    case TextureAlgorithm::Sodium2:
    default:
        return runMode<TextureAlgorithm::Sodium2>(config, plan, threadCount, state, sink);
    }
}
