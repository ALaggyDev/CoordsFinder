# CoordsFinder

CUDA brute-force scanner for matching Minecraft block model texture variants at candidate coordinates.

The scan bounds, texture mode, and filter rows are loaded from a config file.
`coordsfinder.conf` is ignored by Git, and `coordsfinder.example.conf` is the tracked starting point.

Filter rows use this format:

```text
x y z | variant
x y z | variant side
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

Copy the example config once, then edit your local ignored config:

```powershell
Copy-Item coordsfinder.example.conf coordsfinder.conf
```

Run with a config path:

```powershell
.\scanner-win.exe .\coordsfinder.conf
```

Generated binaries and CUDA/MSVC build artifacts are ignored by Git.
