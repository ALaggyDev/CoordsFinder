# CoordsFinder

CUDA brute-force scanner for matching Minecraft block model texture variants at candidate coordinates.

The filter is defined in `bruteforce.cu` as a list of `RotationInfo` entries:

```cpp
RotationInfo(x, y, z, variant)
RotationInfo(x, y, z, variant, true) // side-face check
```

Normal checks compare the selected four-state model variant directly. Side checks still sample Minecraft's four model variants, then fold them to two visible side states.

## Build on Windows

Requirements:

- NVIDIA CUDA Toolkit with `nvcc`
- Visual Studio 2022 C++ build tools

Build:

```powershell
.\build_windows.ps1
```

The script produces `scanner-win.exe`.

## Run

Edit `ScanConfig` in `main.cu` to change the search bounds, texture mode, and maximum bad blocks. Then run:

```powershell
.\scanner-win.exe
```

`ScanConfig::xzRotationMask` controls which world XZ facings are searched:

```cpp
unsigned int xzRotationMask = XzRotationMask0 | XzRotationMask180;
unsigned int xzRotationMask = XzRotationMaskAll;
```

Each selected rotation prepares a rotated filter on the CPU, copies it to the GPU, then runs a separate full brute-force pass.

The XZ rotations are top-down Minecraft map rotations where +X is east/right and +Z is south/down:

- `0`: `(x, z) -> (x, z)`
- `90`: `(x, z) -> (-z, x)`
- `180`: `(x, z) -> (-x, -z)`
- `270`: `(x, z) -> (z, -x)`

Generated binaries and CUDA/MSVC build artifacts are ignored by Git.
