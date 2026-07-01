#include "config.hpp"

#include <algorithm>
#include <climits>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>

#include "simple_ini.hpp"

namespace {

const char* DefaultConfigPath = "coordsfinder.conf";
const char* FallbackConfigPath = "coordsfinder.example.conf";

bool fileExists(const char* path)
{
    std::ifstream input(path);
    return static_cast<bool>(input);
}

std::string lowerCopy(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

std::string compactName(std::string value)
{
    value = lowerCopy(value);
    value.erase(std::remove_if(value.begin(), value.end(), [](char ch) {
        return ch == ' ' || ch == '_' || ch == '-' || ch == '.';
    }), value.end());
    return value;
}

std::string linePrefix(const char* path, const simple_ini::Line& line)
{
    return std::string(path) + ":" + std::to_string(line.number) + ": ";
}

bool parseInt(const std::string& text, int* value)
{
    char* end = nullptr;
    const long parsed = std::strtol(text.c_str(), &end, 10);
    if (end == text.c_str() || *end != '\0') {
        return false;
    }

    if (parsed < INT_MIN || parsed > INT_MAX) {
        return false;
    }

    *value = static_cast<int>(parsed);
    return true;
}

bool parseBool(const std::string& text, bool* value)
{
    const std::string lowered = lowerCopy(text);
    if (lowered == "true" || lowered == "yes" || lowered == "on" || lowered == "1") {
        *value = true;
        return true;
    }
    if (lowered == "false" || lowered == "no" || lowered == "off" || lowered == "0") {
        *value = false;
        return true;
    }
    return false;
}

bool parseTextureMode(const std::string& text, TextureMode* mode)
{
    const std::string name = compactName(text);
    if (name == "vanilla112" || name == "vanilla12") {
        *mode = TextureModeVanilla12;
        return true;
    }
    if (name == "vanilla") {
        *mode = TextureModeVanilla;
        return true;
    }
    if (name == "vanilla1211" || name == "vanilla211") {
        *mode = TextureModeVanilla21_1;
        return true;
    }
    if (name == "sodium") {
        *mode = TextureModeSodium;
        return true;
    }
    if (name == "sodium119" || name == "sodium19") {
        *mode = TextureModeSodium19;
        return true;
    }
    return false;
}

bool rotationTokenToMask(const std::string& token, unsigned int* mask)
{
    const std::string lowered = lowerCopy(token);
    if (lowered == "all") {
        *mask |= XzRotationMaskAll;
        return true;
    }
    if (lowered == "none") {
        *mask = 0;
        return true;
    }

    int degrees = 0;
    if (!parseInt(token, &degrees)) {
        return false;
    }

    switch (degrees) {
    case 0:
        *mask |= XzRotationMask0;
        return true;
    case 90:
        *mask |= XzRotationMask90;
        return true;
    case 180:
        *mask |= XzRotationMask180;
        return true;
    case 270:
        *mask |= XzRotationMask270;
        return true;
    default:
        return false;
    }
}

bool parseRotationMask(const std::string& text, unsigned int* mask)
{
    std::string normalized = text;
    std::replace(normalized.begin(), normalized.end(), ',', ' ');

    std::istringstream input(normalized);
    std::string token;
    unsigned int parsedMask = 0;
    bool sawToken = false;

    while (input >> token) {
        sawToken = true;
        if (!rotationTokenToMask(token, &parsedMask)) {
            return false;
        }
    }

    if (!sawToken) {
        return false;
    }

    *mask = parsedMask;
    return true;
}

bool fitsChar(int value)
{
    return value >= -128 && value <= 127;
}

struct SettingFlags {
    bool mode = false;
    bool xStart = false;
    bool xEnd = false;
    bool yStart = false;
    bool yEnd = false;
    bool zStart = false;
    bool zEnd = false;
    bool chunkBlocksX = false;
    bool chunkBlocksZ = false;
    bool maxBadBlocks = false;
    bool xzRotationMask = false;
    bool printChunks = false;
};

bool parseFilterLine(
    const char* path,
    const simple_ini::Line& line,
    std::vector<RotationInfo>* filter,
    std::string* error)
{
    std::istringstream input(line.text);
    int x = 0;
    int y = 0;
    int z = 0;
    int rotation = 0;
    std::string separator;

    if (!(input >> x >> y >> z >> separator >> rotation) || separator != "|") {
        if (error) {
            *error = linePrefix(path, line) + "filter rows must be: x y z | rotation [side]";
        }
        return false;
    }

    bool side = false;
    std::string sideText;
    if (input >> sideText) {
        const std::string sideName = lowerCopy(sideText);
        if (sideName == "side" || sideName == "true" || sideName == "1") {
            side = true;
        }
        else if (sideName == "normal" || sideName == "false" || sideName == "0") {
            side = false;
        }
        else {
            if (error) {
                *error = linePrefix(path, line) + "expected optional side marker, got '" + sideText + "'";
            }
            return false;
        }
    }

    std::string extra;
    if (input >> extra) {
        if (error) {
            *error = linePrefix(path, line) + "unexpected extra token '" + extra + "'";
        }
        return false;
    }

    if (!fitsChar(x) || !fitsChar(y) || !fitsChar(z)) {
        if (error) {
            *error = linePrefix(path, line) + "filter offsets must fit in signed char range [-128, 127]";
        }
        return false;
    }

    if (rotation < 0 || rotation > (side ? 1 : 3)) {
        if (error) {
            *error = linePrefix(path, line) + "rotation is out of range for this filter row";
        }
        return false;
    }

    filter->push_back(RotationInfo(
        static_cast<char>(x),
        static_cast<char>(y),
        static_cast<char>(z),
        static_cast<char>(rotation),
        side));
    return true;
}

bool applySetting(ScanConfig* config, SettingFlags* flags, const std::string& key, const std::string& value)
{
    const std::string name = compactName(key);

    if (name == "mode") {
        flags->mode = parseTextureMode(value, &config->mode);
        return flags->mode;
    }
    if (name == "xstart") {
        flags->xStart = parseInt(value, &config->xStart);
        return flags->xStart;
    }
    if (name == "xend") {
        flags->xEnd = parseInt(value, &config->xEnd);
        return flags->xEnd;
    }
    if (name == "ystart") {
        flags->yStart = parseInt(value, &config->yStart);
        return flags->yStart;
    }
    if (name == "yend") {
        flags->yEnd = parseInt(value, &config->yEnd);
        return flags->yEnd;
    }
    if (name == "zstart") {
        flags->zStart = parseInt(value, &config->zStart);
        return flags->zStart;
    }
    if (name == "zend") {
        flags->zEnd = parseInt(value, &config->zEnd);
        return flags->zEnd;
    }
    if (name == "chunkblocksx") {
        flags->chunkBlocksX = parseInt(value, &config->chunkBlocksX);
        return flags->chunkBlocksX;
    }
    if (name == "chunkblocksz") {
        flags->chunkBlocksZ = parseInt(value, &config->chunkBlocksZ);
        return flags->chunkBlocksZ;
    }
    if (name == "maxbadblocks") {
        flags->maxBadBlocks = parseInt(value, &config->maxBadBlocks);
        return flags->maxBadBlocks;
    }
    if (name == "xzrotations" || name == "xzrotationmask") {
        flags->xzRotationMask = parseRotationMask(value, &config->xzRotationMask);
        return flags->xzRotationMask;
    }
    if (name == "printchunks") {
        flags->printChunks = parseBool(value, &config->printChunks);
        return flags->printChunks;
    }

    return false;
}

bool validateRequiredSettings(const SettingFlags& flags, std::string* error)
{
    if (!flags.mode) {
        if (error) {
            *error = "missing required setting mode";
        }
        return false;
    }
    if (!flags.xStart || !flags.xEnd || !flags.yStart || !flags.yEnd || !flags.zStart || !flags.zEnd) {
        if (error) {
            *error = "missing one or more required scan bound settings";
        }
        return false;
    }
    if (!flags.chunkBlocksX || !flags.chunkBlocksZ) {
        if (error) {
            *error = "missing required chunkBlocksX or chunkBlocksZ setting";
        }
        return false;
    }
    if (!flags.maxBadBlocks) {
        if (error) {
            *error = "missing required setting maxBadBlocks";
        }
        return false;
    }
    if (!flags.xzRotationMask) {
        if (error) {
            *error = "missing required setting xzRotations";
        }
        return false;
    }
    if (!flags.printChunks) {
        if (error) {
            *error = "missing required setting printChunks";
        }
        return false;
    }
    return true;
}

bool validateConfig(const ScanConfig& config, const SettingFlags& flags, std::string* error)
{
    if (!validateRequiredSettings(flags, error)) {
        return false;
    }

    if (config.xStart > config.xEnd || config.yStart > config.yEnd || config.zStart > config.zEnd) {
        if (error) {
            *error = "scan start bounds must be less than or equal to end bounds";
        }
        return false;
    }

    if (config.chunkBlocksX <= 0 || config.chunkBlocksZ <= 0) {
        if (error) {
            *error = "chunkBlocksX and chunkBlocksZ must be positive";
        }
        return false;
    }

    if (config.maxBadBlocks < 0) {
        if (error) {
            *error = "maxBadBlocks must be non-negative";
        }
        return false;
    }

    if (config.filter.empty()) {
        if (error) {
            *error = "filter must contain at least one row";
        }
        return false;
    }

    if (config.filter.size() > MaxFilterCount) {
        if (error) {
            *error = "filter contains more rows than MaxFilterCount";
        }
        return false;
    }

    return true;
}

}

bool loadScanConfig(const char* requestedPath, ScanConfig* config, std::string* error)
{
    const bool hasRequestedPath = requestedPath && requestedPath[0] != '\0';
    const char* path = hasRequestedPath ? requestedPath : DefaultConfigPath;
    bool usedFallback = false;

    if (!hasRequestedPath && !fileExists(path) && fileExists(FallbackConfigPath)) {
        path = FallbackConfigPath;
        usedFallback = true;
    }

    std::vector<simple_ini::Line> lines;
    if (!simple_ini::readFile(path, &lines, error)) {
        return false;
    }

    ScanConfig parsed = {};
    SettingFlags flags = {};
    parsed.sourcePath = path;
    parsed.usedFallback = usedFallback;

    for (const simple_ini::Line& line : lines) {
        const std::string section = compactName(line.section);

        if (section == "filter") {
            if (line.isKeyValue) {
                if (error) {
                    *error = linePrefix(path, line) + "filter rows should not use key=value syntax";
                }
                return false;
            }
            if (!parseFilterLine(path, line, &parsed.filter, error)) {
                return false;
            }
            continue;
        }

        if (!section.empty() && section != "scan" && section != "settings") {
            if (error) {
                *error = linePrefix(path, line) + "unknown section '" + line.section + "'";
            }
            return false;
        }

        if (!line.isKeyValue) {
            if (error) {
                *error = linePrefix(path, line) + "expected key=value setting";
            }
            return false;
        }

        if (!applySetting(&parsed, &flags, line.key, line.value)) {
            if (error) {
                *error = linePrefix(path, line) + "invalid setting '" + line.key + "' or value '" + line.value + "'";
            }
            return false;
        }
    }

    if (!validateConfig(parsed, flags, error)) {
        if (error) {
            *error = std::string(path) + ": " + *error;
        }
        return false;
    }

    *config = parsed;
    return true;
}
