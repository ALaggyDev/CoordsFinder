#pragma once

#include <string>
#include <vector>

#include "bruteforce.cuh"

struct ScanConfig {
    TextureMode mode;

    int xStart;
    int xEnd;
    int yStart;
    int yEnd;
    int zStart;
    int zEnd;

    int chunkBlocksX;
    int chunkBlocksZ;
    int maxBadBlocks;
    unsigned int xzRotationMask;
    bool printChunks;

    std::vector<RotationInfo> filter;
    std::string sourcePath;
    bool usedFallback;
};

bool loadScanConfig(const char* requestedPath, ScanConfig* config, std::string* error);
