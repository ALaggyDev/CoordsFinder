param(
    [string]$Arch = "sm_120",
    [string]$Out = "scanner-win.exe"
)

$ErrorActionPreference = "Stop"

$nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
if (-not $nvcc) {
    $cudaRoots = @(
        "$env:ProgramFiles\NVIDIA GPU Computing Toolkit\CUDA\v13.2",
        "$env:ProgramFiles\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
    )

    foreach ($root in $cudaRoots) {
        $candidate = Join-Path $root "bin\nvcc.exe"
        if (Test-Path $candidate) {
            $nvcc = @{ Source = $candidate }
            break
        }
    }
}

if (-not $nvcc) {
    throw "Could not find nvcc.exe. Install the CUDA Toolkit or add it to PATH."
}

$cl = Get-Command cl -ErrorAction SilentlyContinue
if ($cl) {
    $ccbin = Split-Path $cl.Source -Parent
}
else {
    $vsRoot = Join-Path ${env:ProgramFiles} "Microsoft Visual Studio\2022"
    $clPath = Get-ChildItem $vsRoot -Recurse -Filter cl.exe -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -like "*\bin\Hostx64\x64\cl.exe" } |
        Select-Object -First 1

    if (-not $clPath) {
        throw "Could not find the x64 MSVC compiler. Install Visual Studio C++ build tools."
    }

    $ccbin = Split-Path $clPath.FullName -Parent
}

$nvccPath = $nvcc.Source

& $nvccPath `
    -std=c++14 `
    "-arch=$Arch" `
    -rdc=true `
    -Xcompiler /utf-8 `
    -ccbin $ccbin `
    main.cu bruteforce.cu textures.cu `
    -o $Out

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "Built $Out for $Arch"
