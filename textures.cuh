#pragma once

#include "cuda_runtime.h"

enum TextureMode {
    TextureModeVanilla12,
    TextureModeVanilla,
    TextureModeVanilla21_1,
    TextureModeSodium,
    TextureModeSodium19
};

__host__ __device__ char getTexture(TextureMode mode, int x, int y, int z, char variantCount);
