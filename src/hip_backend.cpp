#include "runner.hpp"

#include <array>
#include <cstddef>
#include <string>
#include <vector>

#define COORDSFINDER_GPU_HIP
#include "gpu_kernels.cuh"
#include "gpu_backend.hpp"

namespace {

struct HipBackend {
    static constexpr const char* name = "HIP";
    using Error = hipError_t;
    static constexpr Error success = hipSuccess;

    static const char* errorString(Error error) { return hipGetErrorString(error); }
    // HIP marks cleanup/error-drain calls [[nodiscard]]; discard explicitly.
    static void drainError() { (void)hipGetLastError(); }
    static Error getDeviceCount(int* count) { return hipGetDeviceCount(count); }
    static Error getDeviceInfo(coordsfinder_gpu::DeviceInfo* info)
    {
        int device = 0;
        const Error result = hipGetDevice(&device);
        if (result != hipSuccess) {
            return result;
        }
        hipDeviceProp_t properties = {};
        const Error propertiesResult = hipGetDeviceProperties(&properties, device);
        if (propertiesResult != hipSuccess) {
            return propertiesResult;
        }
        info->name = properties.name;
        for (int axis = 0; axis < 3; ++axis) {
            info->maxGridSize[axis] = properties.maxGridSize[axis];
        }
        return hipSuccess;
    }
    static Error malloc(void** ptr, std::size_t bytes) { return hipMalloc(ptr, bytes); }
    static Error free(void* ptr) { return hipFree(ptr); }
    static Error memset(void* ptr, int value, std::size_t bytes) { return hipMemset(ptr, value, bytes); }
    static Error memcpyToHost(void* dst, const void* src, std::size_t bytes)
    {
        return hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost);
    }
    static Error synchronize() { return hipDeviceSynchronize(); }
    static Error uploadFilters(
        const std::array<std::array<RotationInfo, MaxFilterCount>, MaxDirectionCount>& filters,
        const std::array<int, MaxDirectionCount>& filterCounts)
    {
        // HIP_SYMBOL wraps the device symbol as required by the HIP runtime API.
        const Error result = hipMemcpyToSymbol(
            HIP_SYMBOL(coordsfinder_gpu::deviceFilters), filters.data(), sizeof(filters));
        if (result != hipSuccess) {
            return result;
        }
        return hipMemcpyToSymbol(
            HIP_SYMBOL(coordsfinder_gpu::deviceFilterCounts), filterCounts.data(), sizeof(filterCounts));
    }
};

}

bool hipAvailable(std::string* reason)
{
    return coordsfinder_gpu::gpuAvailable<HipBackend>(reason);
}

bool runHipScan(
    const ScanConfig& config,
    const ScanPlan& plan,
    ScanState* state,
    const MatchSink& sink,
    std::string* error)
{
    return coordsfinder_gpu::runGpuScan<HipBackend>(config, plan, state, sink, error);
}
