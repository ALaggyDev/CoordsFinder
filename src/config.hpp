#pragma once

#include <string>
#include <vector>

#include "types.hpp"

struct TileSize {
    int x;
    int z;
};

struct IntRange {
    int start;
    int end;
};

struct ScanConfig {
    TextureAlgorithm algorithm = TextureAlgorithm::Vanilla3;
    ScanOrder scanOrder = ScanOrder::Linear;
    std::vector<int> directions = { 0 };

    IntRange xRange;
    IntRange yRange;
    IntRange zRange;

    int errorTolerance = 0;

    TileSize cpuTileSize = { 1024, 1024 };
    TileSize cudaTileSize = { 16384, 16384 };
    TileSize metalTileSize = { 16384, 16384 };
    bool verbose = false;

    std::vector<RotationInfo> filter;
    std::string sourcePath;
};

bool loadScanConfig(const char* requestedPath, ScanConfig* config, std::string* error);
