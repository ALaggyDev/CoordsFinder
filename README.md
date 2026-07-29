# CoordsFinder

CUDA brute-force scanner for matching Minecraft block model texture variants at candidate coordinates.

The scan bounds, texture mode, compass directions, and filter rows are loaded from a config file.
`coordsfinder.conf` is ignored by Git, and `coordsfinder.example.conf` is the tracked starting point.

## Texture Modes

| MC Version | Mode |
| --- | --- |
| <= 1.12.2 | `Vanilla-1` |
| 1.13-1.21.1 | `Vanilla-2` |
| 1.21.2+ | `Vanilla-3` |

| Sodium Version | MC Version | Mode |
| --- | --- | --- |
| 1.0-4.1 | 1.16-1.18.2 | `Sodium-1` |
| 4.2-4.8 | 1.19-1.19.3 | `Sodium-2` |
| 4.9+ | 1.19.3+ | Use the MC mode |

Filter rows use this format:

```text
x y z | variant
x y z | variant side
```

Normal checks compare the selected four-state model variant directly. Side checks still sample Minecraft's four model variants, then fold them to two visible side states.

`directions` selects the possible horizontal compass rotations:

```ini
directions = [0, 90, 180, 270]
```

If `directions` is omitted, the scanner defaults to `[0]`.

Each direction runs a separate scan. Before uploading the filter, the CPU rotates every X/Z offset clockwise on a top-down Minecraft map:

- `0`: `(x, z) -> (x, z)`
- `90`: `(x, z) -> (-z, x)`
- `180`: `(x, z) -> (-x, -z)`
- `270`: `(x, z) -> (z, -x)`

For top and bottom faces, the CPU also advances the variant by one state per quarter-turn: `(variant + direction / 90) % 4`. Side-face variants are left unchanged. The GPU receives an ordinary, already-transformed filter for every pass.

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

Copy the example config once, then edit your local ignored config:

```powershell
Copy-Item coordsfinder.example.conf coordsfinder.conf
```

Run with a config path:

```powershell
.\scanner-win.exe .\coordsfinder.conf
```

Generated binaries and CUDA/MSVC build artifacts are ignored by Git.
