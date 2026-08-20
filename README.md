# CoordsFinder

CoordsFinder is the fastest Minecraft texture rotation cracker for cracking coordinates from a screenshot!

CoordsFinder supports a multithreaded CPU backend and a CUDA backend.

Check out [my video](TODO) for info :)

## Usage

### Google Colab

You can run CoordsFinder directly on Google Colab with [this notebook](TODO), without needing to install anything! (Nvidia Tesla T4 GPU is available for free!)

### Pre-built binaries

Pre-built CPU and CUDA binaries are available for Windows and Linux. Download the latest release from the [releases page](https://github.com/ALaggyDev/CoordsFinder/releases/latest).

Pre-built binaries have the following minimum versions:

-   Minimum CUDA version: 12.8
-   Minimum GPU compute capability: Turing (GTX 16, RTX 20 series)
-   Glibc: 2.35 (Linux)

### Build with CMake

Requirements:

-   CMake 3.24 or newer
-   A C++20 compiler: Visual Studio 2022 on Windows, or GCC/Clang on Linux
-   Optional: NVIDIA CUDA Toolkit for the CUDA backend

CPU-only:

```sh
cmake --preset cpu-release
cmake --build --preset cpu-release
# ctest --preset cpu-release
```

CPU + CUDA:

```sh
cmake --preset cuda-release
cmake --build --preset cuda-release
# ctest --preset cuda-release
```

On Windows, the executable is under `build\<preset>\Release\coordsfinder.exe`. On Linux, it is under `build/<preset>/coordsfinder`.

## Run

To run CoordsFinder, provide a search config file as the first argument:

```sh
<coordsfinder_exe> ./example.conf
```

CLI options:

```text
-b, --backend auto|cpu|cuda  Select the execution backend
-t, --threads N              Set CPU worker count
-e, --validate               Validate and summarize without scanning
-h, --help                   Show all options
-v, --version                Show the version
```

The `auto` backend selects CUDA when it was compiled in and a CUDA device is available; otherwise it selects CPU.

## Search config

An example search config is included in [example.conf](./example.conf). It is a simple INI-like file with the following sections:

```ini
# Comment starts with a hash. (#)

algorithm = Vanilla-3             # Vanilla-1, Vanilla-2, Vanilla-3, Sodium-1, Sodium-2
scanOrder = spiral                # linear, spiral
directions = [0]                  # 0, 90, 180, 270

xRange = (-5000, 5000)
yRange = (-60, 0)
zRange = (-5000, 5000)

errorTolerance = 0                # Maximum number of texture rotation errors accepted

cpuTileSize = (1024, 1024)
cudaTileSize = (16384, 16384)
verbose = false

[filter]
# x y z | variant [side]
-6 0 0 | 3
-5 0 0 | 3
-6 0 -1 | 0 side
-5 0 -1 | 1 side
```

More examples can be found in the [example](./examples) folder.

### Algorithm

Select the texture algorithm in the config. If unsure, use `Vanilla-3` as a safe default.

| Minecraft version | Algorithm   |
| ----------------- | ----------- |
| <= 1.12.2         | `Vanilla-1` |
| 1.13-1.21.1       | `Vanilla-2` |
| 1.21.2+           | `Vanilla-3` |

| Sodium version | Minecraft version | Algorithm              |
| -------------- | ----------------- | ---------------------- |
| 1.0-4.1        | 1.16-1.18.2       | `Sodium-1`             |
| 4.2-4.8        | 1.19-1.19.3       | `Sodium-2`             |
| 4.9+           | 1.19.3+           | Use the Minecraft mode |

### Scan order

Scan order determines the order in which tiles are scanned.
- `linear` starts from the minimum X/Z corner to the maximum X/Z corner.
- `spiral` begins at the center and scans in a clockwise spiral pattern.

### Directions

`directions` are **very important** if the cardinal direction of the screenshot is unknown. `directions` tells CoordsFinder if it should rotates the filter offsets and variants.

```ini
directions = [0, 90, 180, 270]
```

For example, if `directions = [0, 180]`, CoordsFinder will both scan the filter as-is and *also* scan the filter rotated 180 degrees horizontally.

If the screenshot direction is unknown, it is recommended to use `directions = [0, 90, 180, 270]` or `directions = [0, 180]`.

### Scan ranges

Self-explanatory. Range ends are exclusive.

### Error tolerance

Error tolerance dictates the maximum number of non-matching texture rotations allowed per candidate.

Note that error tolerance **severely impacts** performance. It is not recommended to use an error tolerance above 3.

### Tile sizes & verbosity (advanced)

The default settings for tile sizes and verbosity are reasonable and do not need to be changed typically.

```ini
cpuTileSize = (1024, 1024)
cudaTileSize = (16384, 16384)
verbose = false
```

Tile size determines the area scanned per work item. Verbose mode prints out the output for each work item.

### Filters

Filter rows use one of these forms:

```text
x y z | variant
x y z | variant side
```

The first 3 numbers are the relative block coordinates to an origin. The fourth number is the texture variant. The optional `side` keyword indicates that the filter is a side face.

## Speed

Benchmark setup:
- Search area: -225000 to 225000 in X/Z, -60 to 0 in Y (Donut SMP area)
- CPU: AMD Ryzen AI 9 365 (10 cores, 20 threads)
- GPU: NVIDIA RTX 5080 Laptop
- OS: Windows 11, on MSVC
- Error tolerance: 0

|                     | CPU (1 thread)  | CPU (20 threads) | GPU (CUDA)      |
| ------------------- | --------------- | ---------------- | --------------- |
| Peak position/sec   | 168M            | 1,860M           | **103,000M!**      |
| Estimated time      | 20 hours 5 mins | 1 hours 48 mins  | **2 mins 5 secs!** |

## FAQs

### Which blocks have texture rotations?

Here's a list of blocks that have "texture rotations", as of Minecraft 1.21.11. Note that I may have missed some blocks, and not all of them have been tested.
- Grass block
- Rooted Dirt
- Dirt
- Dirt path
- Stone & Infested stone, with side face variants
- Deepslate & Infested deepslate, with side face variants
- Bedrock, with side face variants
- Sculk, with side face variants
- Podzol
- Mycelium
- Sand
- Red sand
- All 16 colors of concrete powder
- Lily pad
- Sea pickle?
- Turtle egg?
- Netherrack (NOTE: not supported yet, since it has 16 variants)

Flower random offsets are not part of the texture rotation algorithm (block variant model) but are instead hard-coded into the game. I will be looking into it in the future.

### How does texture rotation cracking even work?

I will spare my words here and instead link to these amazing resources that explain the concept in detail:
- [Texture Rotation Reverser Java](https://github.com/19MisterX98/TextureRotations) by 19MisterX98
- [Texture Exploit Guide](https://gitea.com/ChromeCrusher/Texploit-Guide) by ChromeCrusher

### What is the difference between WebCoordsFinder and CoordsFinder?

WebCoordsFinder is a web-based app. It allows users to upload a screenshot, draw the grid, mark the texture rotations, and either perform the scan on the app or download a config file to use in CoordsFinder. It is a convenient way to generate a config file without having to painstakingly mark and write it by hand.

CoordsFinder is a command-line tool that performs the actual bruteforce search. It supports CUDA and is much faster than the built-in WebCoordsFinder scanner.

In short: Start with WebCoordsFinder, and either use the built-in scanner or use CoordsFinder.

### I don't have a Nvidia GPU but I want to scan faster! What should I do?

Use the free Google Colab notebook in [here](TODO)!

### How is CoordsFinder so fast?

It is fast because:
- It is written in C++, which is a compiled language.
- It uses GPU acceleration to massively parallelize the search.
- Careful optimizations such as reducing warp divergence and using constant memory are made to the CUDA backend.

In the future, I may look into alternative searching algorithms such as the [Boyer-Moore algorithm](https://en.wikipedia.org/wiki/Boyer%E2%80%93Moore_string-search_algorithm) to further improve performance. Optimization improvements and suggestions are very much appreciated!

### Why is texture rotation cracking relatively unknown to the Minecraft community?

Honestly I have no idea. Texture rotation cracking is certainly not a new concept, and there's a lot of information about it online (The earliest reference I can find was from 2019 by [hacker mann](https://www.youtube.com/watch?v=6__hO4cc1pA)!). However, most of the information just didn't reach the general Minecraft community somehow. Instead, bedrock cracking and "cloud cracking" have taken the spotlight instead of texture rotation cracking, which is a shame.

Now, WebCoordsFinder basically perfected texture rotation cracking and made it accessible to everyone. I hope that this project will help spread awareness of texture rotation cracking and its immense power!

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request!
