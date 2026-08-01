# CoordsFinder

Cross-platform brute-force scanner for matching Minecraft block-model texture variants at candidate coordinates.

CoordsFinder supports a multithreaded CPU backend and a CUDA backend. CUDA is an optional backend for NVIDIA GPUs; a CPU-only build does not need the CUDA Toolkit.

## Build with CMake

Requirements:

- CMake 3.24 or newer
- A C++20 compiler: Visual Studio 2022 on Windows, or GCC/Clang on Linux
- Optional: NVIDIA CUDA Toolkit for the CUDA backend

The repository includes two presets. Configure once, then build whenever the source changes.

### CPU-only

```powershell
cmake --preset cpu-release
cmake --build --preset cpu-release
ctest --preset cpu-release
```

### CPU + CUDA

```powershell
cmake --preset cuda-release
cmake --build --preset cuda-release
ctest --preset cuda-release
```

On Windows, the executable is under `build/<preset>/Release/coordsfinder.exe`. On single-config Linux generators, it is normally under `build/<preset>/coordsfinder`.

## Run

Run with automatic backend selection:

```powershell
.\build\cuda-release\Release\coordsfinder.exe .\coordsfinder.example.conf
```

Useful options:

```text
--backend auto|cpu|cuda  Select the execution backend
--threads N              Set CPU worker count
--validate               Validate and summarize without scanning
--help                   Show all options
```

`auto` selects CUDA when it was compiled in and a CUDA device is available; otherwise it selects CPU.

## Scan order

`linear` preserves the original direction-major, minimum-X then minimum-Z ordering:

```ini
scanOrder = linear
```

`spiral` begins at the X/Z tile containing the center of the bounds and scans rectangular rings outwards. It checks every selected direction for a tile before moving farther from the center:

```ini
scanOrder = spiral
```

Scan ranges are inclusive and use `(start, end)` pairs:

```ini
xRange = (-5000, 5000)
yRange = (-60, 0)
zRange = (-5000, 5000)
```

CPU and CUDA use independent tile sizes. CPU tiles are dynamically claimed worker
tasks, while each CUDA tile is one kernel launch and result-buffer drain:

```ini
cpuTileSize = (1024, 1024)
cudaTileSize = (16384, 16384)

# Accept up to this many non-matching filter rows at a candidate.
errorTolerance = 0

# Log each work item as it starts, in addition to normal periodic progress.
verbose = false
```

Large CUDA tiles reduce launch overhead but increase cancellation latency and may
overflow the 65,536-match per-tile result buffer when the filter is loose. Reduce
`cudaTileSize` if that occurs.

## Algorithms

Select the texture algorithm in the config:

```ini
algorithm = Vanilla-3
```

| Minecraft version | Algorithm |
| --- | --- |
| <= 1.12.2 | `Vanilla-1` |
| 1.13-1.21.1 | `Vanilla-2` |
| 1.21.2+ | `Vanilla-3` |

| Sodium version | Minecraft version | Algorithm |
| --- | --- | --- |
| 1.0-4.1 | 1.16-1.18.2 | `Sodium-1` |
| 4.2-4.8 | 1.19-1.19.3 | `Sodium-2` |
| 4.9+ | 1.19.3+ | Use the Minecraft mode |

Filter rows use one of these forms:

```text
x y z | variant
x y z | variant side
```

Normal checks compare the selected four-state model variant directly. Side checks sample the four model variants, then fold them to two visible-side states.

`directions` selects possible horizontal compass rotations:

```ini
directions = [0, 90, 180, 270]
```

For each direction, X/Z filter offsets are rotated clockwise on a top-down Minecraft map. Top and bottom variants advance by one state per quarter-turn; side variants remain unchanged.
