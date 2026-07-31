#pragma once

#include <fstream>
#include <string>
#include <vector>

namespace simple_ini {

struct Line {
    int number = 0;
    std::string section;
    std::string key;
    std::string value;
    std::string text;
    bool isKeyValue = false;
};

inline bool isSpace(char ch)
{
    return ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n';
}

inline std::string trim(const std::string& value)
{
    size_t start = 0;
    while (start < value.size() && isSpace(value[start])) {
        start++;
    }

    size_t end = value.size();
    while (end > start && isSpace(value[end - 1])) {
        end--;
    }

    return value.substr(start, end - start);
}

inline std::string stripComment(const std::string& value)
{
    bool inSingleQuote = false;
    bool inDoubleQuote = false;

    for (size_t i = 0; i < value.size(); i++) {
        const char ch = value[i];
        if (ch == '\'' && !inDoubleQuote) {
            inSingleQuote = !inSingleQuote;
        }
        else if (ch == '"' && !inSingleQuote) {
            inDoubleQuote = !inDoubleQuote;
        }
        else if (!inSingleQuote && !inDoubleQuote && (ch == '#' || ch == ';')) {
            return value.substr(0, i);
        }
    }

    return value;
}

inline bool readFile(const char* path, std::vector<Line>* lines, std::string* error)
{
    std::ifstream input(path);
    if (!input) {
        if (error) {
            *error = std::string("could not open ") + path;
        }
        return false;
    }

    std::string section;
    std::string rawLine;
    int lineNumber = 0;

    while (std::getline(input, rawLine)) {
        lineNumber++;

        const std::string text = trim(stripComment(rawLine));
        if (text.empty()) {
            continue;
        }

        if (text.front() == '[') {
            if (text.back() != ']') {
                if (error) {
                    *error = std::string(path) + ":" + std::to_string(lineNumber) + ": malformed section header";
                }
                return false;
            }

            section = trim(text.substr(1, text.size() - 2));
            if (section.empty()) {
                if (error) {
                    *error = std::string(path) + ":" + std::to_string(lineNumber) + ": empty section name";
                }
                return false;
            }

            continue;
        }

        Line line;
        line.number = lineNumber;
        line.section = section;
        line.text = text;

        const size_t equals = text.find('=');
        if (equals != std::string::npos) {
            line.isKeyValue = true;
            line.key = trim(text.substr(0, equals));
            line.value = trim(text.substr(equals + 1));

            if (line.key.empty()) {
                if (error) {
                    *error = std::string(path) + ":" + std::to_string(lineNumber) + ": empty key";
                }
                return false;
            }
        }

        lines->push_back(line);
    }

    return true;
}

}
