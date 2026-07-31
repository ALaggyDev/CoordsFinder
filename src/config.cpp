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

bool parseIntPair(const std::string& text, int* first, int* second)
{
    std::istringstream input(text);
    char opening = '\0';
    char comma = '\0';
    char closing = '\0';
    int parsedFirst = 0;
    int parsedSecond = 0;
    std::string extra;
    if (!(input >> opening >> parsedFirst >> comma >> parsedSecond >> closing)
        || opening != '('
        || comma != ','
        || closing != ')'
        || (input >> extra)) {
        return false;
    }
    *first = parsedFirst;
    *second = parsedSecond;
    return true;
}

bool parseTileSize(const std::string& text, TileSize* value)
{
    return parseIntPair(text, &value->x, &value->z);
}

bool parseRange(const std::string& text, IntRange* value)
{
    return parseIntPair(text, &value->start, &value->end);
}

bool parseTextureAlgorithm(const std::string& text, TextureAlgorithm* algorithm)
{
    const std::string name = lowerCopy(text);
    if (name == "vanilla-1") {
        *algorithm = TextureAlgorithm::Vanilla1;
        return true;
    }
    if (name == "vanilla-2") {
        *algorithm = TextureAlgorithm::Vanilla2;
        return true;
    }
    if (name == "vanilla-3") {
        *algorithm = TextureAlgorithm::Vanilla3;
        return true;
    }
    if (name == "sodium-1") {
        *algorithm = TextureAlgorithm::Sodium1;
        return true;
    }
    if (name == "sodium-2") {
        *algorithm = TextureAlgorithm::Sodium2;
        return true;
    }
    return false;
}

bool parseScanOrder(const std::string& text, ScanOrder* order)
{
    const std::string name = lowerCopy(text);
    if (name == "linear" || name == "native") {
        *order = ScanOrder::Linear;
        return true;
    }
    if (name == "spiral") {
        *order = ScanOrder::Spiral;
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
    bool algorithm = false;
    bool xRange = false;
    bool yRange = false;
    bool zRange = false;
    bool errorTolerance = false;
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
            *error = linePrefix(path, line) + "filter offsets must fit in int8 range [-128, 127]";
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
        x,
        y,
        z,
        variant,
        side));
    return true;
}

bool applySetting(ScanConfig* config, SettingFlags* flags, const std::string& key, const std::string& value)
{
    const std::string name = compactName(key);

    if (name == "algorithm") {
        flags->algorithm = parseTextureAlgorithm(value, &config->algorithm);
        return flags->algorithm;
    }
    if (name == "directions") {
        return parseDirections(value, &config->directions);
    }
    if (name == "scanorder") {
        return parseScanOrder(value, &config->scanOrder);
    }
    if (name == "xrange") {
        flags->xRange = parseRange(value, &config->xRange);
        return flags->xRange;
    }
    if (name == "yrange") {
        flags->yRange = parseRange(value, &config->yRange);
        return flags->yRange;
    }
    if (name == "zrange") {
        flags->zRange = parseRange(value, &config->zRange);
        return flags->zRange;
    }
    if (name == "cputilesize") {
        return parseTileSize(value, &config->cpuTileSize);
    }
    if (name == "cudatilesize") {
        return parseTileSize(value, &config->cudaTileSize);
    }
    if (name == "errortolerance") {
        flags->errorTolerance = parseInt(value, &config->errorTolerance);
        return flags->errorTolerance;
    }
    if (name == "verbose") {
        return parseBool(value, &config->verbose);
    }

    return false;
}

bool validateRequiredSettings(const SettingFlags& flags, std::string* error)
{
    if (!flags.algorithm) {
        if (error) {
            *error = "missing required setting algorithm";
        }
        return false;
    }
    if (!flags.xRange || !flags.yRange || !flags.zRange) {
        if (error) {
            *error = "missing one or more required scan range settings";
        }
        return false;
    }
    if (!flags.errorTolerance) {
        if (error) {
            *error = "missing required setting errorTolerance";
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

    if (config.xRange.start > config.xRange.end
        || config.yRange.start > config.yRange.end
        || config.zRange.start > config.zRange.end) {
        if (error) {
            *error = "scan range starts must be less than or equal to their ends";
        }
        return false;
    }

    if (config.cpuTileSize.x <= 0 || config.cpuTileSize.z <= 0) {
        if (error) {
            *error = "cpuTileSize dimensions must be positive";
        }
        return false;
    }

    if (config.cudaTileSize.x <= 0 || config.cudaTileSize.z <= 0) {
        if (error) {
            *error = "cudaTileSize dimensions must be positive";
        }
        return false;
    }

    if (config.errorTolerance < 0) {
        if (error) {
            *error = "errorTolerance must be non-negative";
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
            *error = "filter contains more than 256 rows";
        }
        return false;
    }

    for (int direction : config.directions) {
        for (const RotationInfo& info : config.filter) {
            const Int2 rotated = rotateXzOffset(
                info.x,
                info.z,
                direction);
            if (!fitsChar(rotated.x) || !fitsChar(rotated.z)) {
                if (error) {
                    *error =
                        "direction " + std::to_string(direction) +
                        " rotates a filter offset outside int8 range [-128, 127]";
                }
                return false;
            }
        }
    }

    return true;
}

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
    std::vector<std::string> seenSettings;
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


        const std::string settingName = compactName(line.key);
        if (std::find(seenSettings.begin(), seenSettings.end(), settingName) != seenSettings.end()) {
            if (error) {
                *error = linePrefix(path, line) + "duplicate setting '" + line.key + "'";
            }
            return false;
        }
        seenSettings.push_back(settingName);

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
