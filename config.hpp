#pragma once

#include <string>
#include <vector>

#include "bruteforce.cuh"

struct ScanConfig {
    TextureMode mode;
    std::vector<int> directions = { 0 };

    int xStart;
    int xEnd;
    int yStart;
    int yEnd;
    int zStart;
    int zEnd;

    int chunkBlocksX;
    int chunkBlocksZ;
    int maxBadBlocks;
    bool printChunks;

    std::vector<RotationInfo> filter;
    std::string sourcePath;
};

std::vector<RotationInfo> makeDirectionalFilter(
    const std::vector<RotationInfo>& filter,
    int direction);

bool loadScanConfig(const char* requestedPath, ScanConfig* config, std::string* error);
