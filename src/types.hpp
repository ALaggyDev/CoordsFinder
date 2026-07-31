#pragma once

#include <cstddef>
#include <cstdint>

enum class TextureMode : std::uint8_t {
    Vanilla1,
    Vanilla2,
    Vanilla3,
    Sodium1,
    Sodium2,
};

enum class ScanOrder : std::uint8_t {
    Linear,
    Spiral,
};

inline const char* textureModeName(TextureMode mode)
{
    switch (mode) {
    case TextureMode::Vanilla1:
        return "Vanilla-1";
    case TextureMode::Vanilla2:
        return "Vanilla-2";
    case TextureMode::Vanilla3:
        return "Vanilla-3";
    case TextureMode::Sodium1:
        return "Sodium-1";
    case TextureMode::Sodium2:
    default:
        return "Sodium-2";
    }
}

struct Int3 {
    std::int32_t x;
    std::int32_t y;
    std::int32_t z;
};

constexpr std::size_t MaxFilterCount = 256;
constexpr std::size_t MaxDirectionCount = 4;

struct RotationInfo {
    std::int8_t x = 0;
    std::int8_t y = 0;
    std::int8_t z = 0;
    std::uint8_t rotation = 0;
    // 0b11 compares all four variants; 0b01 folds side faces to two states.
    std::uint8_t visibleMask = 3;

    constexpr RotationInfo() = default;

    constexpr RotationInfo(int vx, int vy, int vz, int vrotation, bool visibleSide = false)
        : x(static_cast<std::int8_t>(vx)),
          y(static_cast<std::int8_t>(vy)),
          z(static_cast<std::int8_t>(vz)),
          rotation(static_cast<std::uint8_t>(vrotation % (visibleSide ? 2 : 4))),
          visibleMask(static_cast<std::uint8_t>(visibleSide ? 1 : 3))
    {
    }
};

static_assert(sizeof(RotationInfo) == 5, "RotationInfo must have a stable CPU/CUDA layout");

struct Match {
    std::int32_t x;
    std::int32_t y;
    std::int32_t z;
    std::int32_t badBlocks;
    std::int32_t direction;
};
