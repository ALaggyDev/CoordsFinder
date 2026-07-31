#pragma once

#include <string>
#include <vector>

#include "types.hpp"

struct ScanConfig {
    TextureMode mode = TextureMode::Vanilla3;
    ScanOrder scanOrder = ScanOrder::Linear;
    std::vector<int> directions = { 0 };

    int xStart;
    int xEnd;
    int yStart;
    int yEnd;
    int zStart;
    int zEnd;

    int tileSizeX = 1024;
    int tileSizeZ = 1024;
    int maxBadBlocks = 0;
    bool printChunks = false;

    std::vector<RotationInfo> filter;
    std::string sourcePath;
};

bool loadScanConfig(const char* requestedPath, ScanConfig* config, std::string* error);
