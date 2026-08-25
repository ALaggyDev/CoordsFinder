#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "scan.hpp"

struct ScanState {
    std::atomic<bool> cancelRequested { false };
    // Backends update progress counters at work-item boundaries for low overhead.
    std::atomic<std::uint64_t> candidates { 0 };
    std::atomic<std::uint64_t> matches { 0 };
    std::atomic<std::uint64_t> completedItems { 0 };
};

using MatchSink = std::function<void(const std::vector<Match>&)>;

bool runCpuScan(
    const ScanConfig& config,
    const ScanPlan& plan,
    unsigned int threadCount,
    ScanState* state,
    const MatchSink& sink,
    std::string* error);

#if defined(COORDSFINDER_HAS_CUDA)
bool cudaAvailable(std::string* reason);

bool runCudaScan(
    const ScanConfig& config,
    const ScanPlan& plan,
    ScanState* state,
    const MatchSink& sink,
    std::string* error);
#endif

#if defined(COORDSFINDER_HAS_METAL)
bool metalAvailable(std::string* reason);

bool runMetalScan(
    const ScanConfig& config,
    const ScanPlan& plan,
    ScanState* state,
    const MatchSink& sink,
    std::string* error);
#endif
