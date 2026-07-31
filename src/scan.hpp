#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "config.hpp"

struct WorkItem {
    Int3 start;
    Int3 end;
    // Index selects the cached transformed filter; direction is retained for output.
    std::size_t directionIndex;
    int direction;
};

struct ScanPlan {
    std::vector<WorkItem> items;
    std::uint64_t totalCandidates = 0;
    bool totalCandidatesSaturated = false;
};

std::vector<RotationInfo> makeDirectionalFilter(
    const std::vector<RotationInfo>& filter,
    int direction);

std::uint64_t workItemCandidateCount(const WorkItem& item, bool* saturated = nullptr);

bool makeScanPlan(const ScanConfig& config, ScanPlan* plan, std::string* error);
