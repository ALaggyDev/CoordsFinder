#include "config.hpp"

#include <algorithm>
#include <climits>
#include <cctype>
#include <cstdlib>
#include <sstream>

#include "simple_ini.hpp"

namespace {

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
    const std::string name = lowerCopy(text);
    if (name == "vanilla-1") {
        *mode = TextureModeVanilla1;
        return true;
    }
    if (name == "vanilla-2") {
        *mode = TextureModeVanilla2;
        return true;
    }
    if (name == "vanilla-3") {
        *mode = TextureModeVanilla3;
        return true;
    }
    if (name == "sodium-1") {
        *mode = TextureModeSodium1;
        return true;
    }
    if (name == "sodium-2") {
        *mode = TextureModeSodium2;
        return true;
    }
    return false;
}

bool isDirection(int value)
{
    return value == 0 || value == 90 || value == 180 || value == 270;
}

bool parseDirections(const std::string& text, std::vector<int>* directions)
{
    if (text.size() < 2 || text.front() != '[' || text.back() != ']') {
        return false;
    }

    const std::string contents = simple_ini::trim(text.substr(1, text.size() - 2));
    if (contents.empty()) {
        return false;
    }

    std::vector<int> parsed;
    size_t start = 0;

    while (start <= contents.size()) {
        const size_t comma = contents.find(',', start);
        const std::string token = simple_ini::trim(contents.substr(
            start,
            comma == std::string::npos ? std::string::npos : comma - start));

        int direction = 0;
        if (token.empty() ||
            !parseInt(token, &direction) ||
            !isDirection(direction) ||
            std::find(parsed.begin(), parsed.end(), direction) != parsed.end()) {
            return false;
        }

        parsed.push_back(direction);

        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }

    *directions = parsed;
    return true;
}

bool fitsChar(int value)
{
    return value >= -128 && value <= 127;
}

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
    int variant = 0;
    std::string separator;

    if (!(input >> x >> y >> z >> separator >> variant) || separator != "|") {
        if (error) {
            *error = linePrefix(path, line) + "filter rows must be: x y z | variant [side]";
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

    if (variant < 0 || variant > (side ? 1 : 3)) {
        if (error) {
            *error = linePrefix(path, line) + "variant is out of range for this filter row";
        }
        return false;
    }

    filter->push_back(RotationInfo(
        static_cast<char>(x),
        static_cast<char>(y),
        static_cast<char>(z),
        static_cast<char>(variant),
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
    if (name == "directions") {
        return parseDirections(value, &config->directions);
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

    for (int direction : config.directions) {
        for (const RotationInfo& info : config.filter) {
            const Int2 rotated = rotateXzOffset(
                static_cast<signed char>(info.x),
                static_cast<signed char>(info.z),
                direction);
            if (!fitsChar(rotated.x) || !fitsChar(rotated.z)) {
                if (error) {
                    *error =
                        "direction " + std::to_string(direction) +
                        " rotates a filter offset outside signed char range [-128, 127]";
                }
                return false;
            }
        }
    }

    return true;
}

}

std::vector<RotationInfo> makeDirectionalFilter(
    const std::vector<RotationInfo>& filter,
    int direction)
{
    std::vector<RotationInfo> directionalFilter;
    directionalFilter.reserve(filter.size());

    const int quarterTurns = direction / 90;
    for (RotationInfo info : filter) {
        const Int2 rotated = rotateXzOffset(
            static_cast<signed char>(info.x),
            static_cast<signed char>(info.z),
            direction);
        info.x = static_cast<char>(rotated.x);
        info.z = static_cast<char>(rotated.z);

        if (info.visibleMask == 3) {
            info.rotation = static_cast<char>((info.rotation + quarterTurns) % 4);
        }

        directionalFilter.push_back(info);
    }

    return directionalFilter;
}

bool loadScanConfig(const char* requestedPath, ScanConfig* config, std::string* error)
{
    const bool hasRequestedPath = requestedPath && requestedPath[0] != '\0';
    if (!hasRequestedPath) {
        if (error) {
            *error = "missing config path";
        }
        return false;
    }

    const char* path = requestedPath;

    std::vector<simple_ini::Line> lines;
    if (!simple_ini::readFile(path, &lines, error)) {
        return false;
    }

    ScanConfig parsed = {};
    SettingFlags flags = {};
    parsed.sourcePath = path;

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
