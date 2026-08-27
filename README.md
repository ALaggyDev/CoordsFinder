# CoordsFinder

CoordsFinder is the fastest Minecraft texture rotation cracker for cracking coordinates from a screenshot!

CoordsFinder is written in Rust and includes a portable multithreaded CPU backend and a [wgpu](https://wgpu.rs/) compute backend, so one build works across CPU-only systems, Vulkan, DirectX 12, and Metal.

Check out my video on YouTube!

Links:

- [YouTube Video](https://www.youtube.com/watch?v=gXTN9DD_Cp0)
- [WebCoordsFinder](https://github.com/ALaggyDev/WebCoordsFinder)
- [Colab Notebook](https://colab.research.google.com/drive/17qih1n6VpQx_77C2spIF-JOJp17y9Jt6?usp=sharing)

If you like this project, please star it on GitHub and share it with your friends!

If you want to include CoordsFinder in your own video/project, please credit me and my project as a gesture of kindness. Thank you!

## Usage

### Google Colab

You can run CoordsFinder directly on Google Colab with [this notebook](https://colab.research.google.com/drive/17qih1n6VpQx_77C2spIF-JOJp17y9Jt6?usp=sharing), without needing to install anything! An NVIDIA Tesla T4 GPU is available for free.

### Pre-built binaries

Pre-built binaries are available for Windows, Linux, and Apple-silicon Macs. Each binary includes both the CPU and GPU backends. Download the latest version from the [releases page](https://github.com/ALaggyDev/CoordsFinder/releases/latest).

The CPU backend has no additional runtime requirements. The optional GPU backend needs a driver with wgpu's `SHADER_INT64` feature. This is available through Vulkan, DirectX 12 with DXC, and Metal 2.3+ on supported hardware. In the default `auto` mode, CoordsFinder uses a compatible GPU when available and otherwise falls back to CPU.

### Build with Cargo

Requirements:

- Rust 1.87 or newer
- A supported GPU and driver if you want to use the optional GPU backend

Build and test:

```sh
cargo build --release
cargo test
```

The executable is `target/release/coordsfinder` (`coordsfinder.exe` on Windows).

## Run

To run CoordsFinder, provide a search config file as the first argument:

```sh
coordsfinder ./example.conf
```

CLI options:

```text
-b, --backend <auto|cpu|gpu>  Select the execution backend (default: auto)
-t, --threads <N>             Set the CPU worker count
-e, --validate                Validate and summarize without scanning
-h, --help                    Show all options
-v, --version                 Show the version
```

For example, to select the CPU backend explicitly and use eight worker threads:

```sh
coordsfinder --backend cpu --threads 8 ./example.conf
```

Only matches are written to standard output. Device information, progress, and completion status are written to standard error, so match output can be redirected safely. Press Ctrl+C to stop; CPU workers poll between X/Z columns and the GPU backend stops after its active tile.

## Search config

An example search config is included in [`example.conf`](./example.conf). It is a simple INI-like file with the following sections:

```ini
# Comments start with a hash. (#)

algorithm = Vanilla-3             # Vanilla-1, Vanilla-2, Vanilla-3, Sodium-1, Sodium-2
scanOrder = spiral                # linear, spiral
directions = [0]                  # 0, 90, 180, 270

xRange = (-5000, 5000)
yRange = (-60, 0)
zRange = (-5000, 5000)

errorTolerance = 0                # Maximum number of texture rotation errors accepted

cpuTileSize = (1024, 1024)
gpuTileSize = (16384, 16384)
verbose = false

[filter]
# x y z | variant [side]
-6 0 0 | 3
-5 0 0 | 3
-6 0 -1 | 0 side
-5 0 -1 | 1 side
```

More examples can be found in the [`examples`](./examples) folder.

### Algorithm

Select the texture algorithm in the config. If unsure, use `Vanilla-3` as a safe default.

| Minecraft version | Algorithm   |
| ----------------- | ----------- |
| <= 1.12.2         | `Vanilla-1` |
| 1.13-1.21.1       | `Vanilla-2` |
| 1.21.2+           | `Vanilla-3` |

| Sodium version | Minecraft version | Algorithm                     |
| -------------- | ----------------- | ----------------------------- |
| 1.0-4.1        | 1.16-1.18.2       | `Sodium-1`                    |
| 4.2-4.8        | 1.19-1.19.3       | `Sodium-2`                    |
| 4.9+           | 1.19.3+           | Use the matching Vanilla mode |

### Scan order

Scan order determines the order in which tiles are scanned.

- `linear` starts from the minimum X/Z corner and moves to the maximum X/Z corner.
- `spiral` begins at the center and scans in a clockwise spiral pattern.

### Directions

`directions` are **very important** if the cardinal direction of the screenshot is unknown. They tell CoordsFinder whether it should rotate the filter offsets and variants.

```ini
directions = [0, 90, 180, 270]
```

For example, if `directions = [0, 180]`, CoordsFinder scans the filter as-is and also scans it rotated 180 degrees horizontally. Four-state top-face samples are rotated with the direction; two-state `side` samples are not.

If the screenshot direction is unknown, it is recommended to use `directions = [0, 90, 180, 270]` or `directions = [0, 180]`.

### Scan ranges

The scan ranges define the candidate coordinate area. Range ends are exclusive.

### Error tolerance

Error tolerance dictates the maximum number of non-matching texture rotations allowed per candidate.

Note that error tolerance **severely impacts** performance. It is not recommended to use an error tolerance above 3.

### Tile sizes & verbosity (advanced)

The default tile sizes and verbosity are reasonable and typically do not need to be changed.

```ini
cpuTileSize = (1024, 1024)
gpuTileSize = (16384, 16384)
verbose = false
```

The selected backend uses its matching tile size, which determines the X/Z area scanned per work item. Reduce `gpuTileSize` if a tile could produce more than 262,144 matches or exceeds an adapter's dispatch limits. The legacy `cudaTileSize` name is accepted as an alias for `gpuTileSize`, so existing configs continue to work. Verbose mode prints progress for every work item.

### Filters

Filter rows use one of these forms:

```text
x y z | variant
x y z | variant side
```

The first three numbers are the relative block coordinates to an origin. The fourth number is the texture variant. The optional `side` keyword indicates that the filter is a side face.

## Speed

Benchmark setup:
- Benchmark config: [`benchmark.conf`](./examples/benchmark.conf)
- Search area: -225000 to 225000 in X/Z, -60 to 0 in Y (Donut SMP area)
- Error tolerance: 0
- CPU: AMD Ryzen AI 9 365 (10 cores, 20 threads)
- GPU: NVIDIA RTX 5080 Laptop
- OS: Windows 11, on MSVC

EDIT: Several performance improvements were made, benchmark results are now updated!

|                     | CPU (1 thread)  | CPU (20 threads) | GPU (Vulkan)    |
| ------------------- | --------------- | ---------------- | --------------- |
| Peak position/sec   | 188M            | 2,180M           | **155,000M!**      |
| Estimated time      | 17 hours 57 mins | 1 hours 33 mins  | **1 mins 18 secs!** |

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

CoordsFinder is a command-line tool that performs the actual brute-force search. It supports multithreaded CPU scanning and GPU acceleration through wgpu, and is much faster than the built-in WebCoordsFinder scanner on supported hardware.

In short: Start with WebCoordsFinder, and either use the built-in scanner or use CoordsFinder.

### I don't have a compatible GPU but I want to scan faster! What should I do?

Use the free [Google Colab notebook](https://colab.research.google.com/drive/17qih1n6VpQx_77C2spIF-JOJp17y9Jt6?usp=sharing)!

### How is CoordsFinder so fast?

It is fast because:

- It is written in Rust, which is a compiled language with predictable performance.
- It uses multithreading on the CPU and compute shaders on the GPU to massively parallelize the search.
- The search implementations avoid unnecessary work in their inner loops and share precomputed scan data across candidates.

Optimization improvements and suggestions are very much appreciated!

### Why is texture rotation cracking relatively unknown to the Minecraft community?

Honestly I have no idea. Texture rotation cracking is certainly not a new concept, and there's a lot of information about it online (the earliest reference I can find was from 2019 by [hacker mann](https://www.youtube.com/watch?v=6__hO4cc1pA)!). However, most of the information just didn't reach the general Minecraft community somehow. Instead, bedrock cracking and "cloud cracking" have taken the spotlight instead of texture rotation cracking, which is a shame.

Now, WebCoordsFinder basically perfected texture rotation cracking and made it accessible to everyone. I hope that this project will help spread awareness of texture rotation cracking and its immense power!

## Contributing

TLDR: The project creator (which is me, Laggy) is a person who **learned to code** well before the era of AIs. As such, he cares about code quality and code elegance well more than the "new-gen AI coders".

AI-generated PRs are allowed, with the following requirements:

- Please review the code and write the PR description yourself (as a **human**). You should understand your code and are responsible for your code.
- Keep the code change minimal and scoped as much as possible.
- Avoid optimizations that make the code hard or impossible to read.
- "Premature optimization is the root of all evil." I hope you understand what this phrase means.
- Please don't flood the repos with meaningless PRs.
