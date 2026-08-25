#include "runner.hpp"

#include <array>
#include <cstddef>
#include <string>
#include <vector>

#define COORDSFINDER_GPU_CUDA
#include "gpu_kernels.cuh"
#include "gpu_backend.hpp"

namespace {

struct CudaBackend {
    static constexpr const char* name = "CUDA";
    using Error = cudaError_t;
    static constexpr Error success = cudaSuccess;

    static const char* errorString(Error error) { return cudaGetErrorString(error); }
    static void drainError() { (void)cudaGetLastError(); }
    static Error getDeviceCount(int* count) { return cudaGetDeviceCount(count); }
    static Error getDeviceInfo(coordsfinder_gpu::DeviceInfo* info)
    {
        int device = 0;
        const Error result = cudaGetDevice(&device);
        if (result != cudaSuccess) {
            return result;
        }
        cudaDeviceProp properties = {};
        const Error propertiesResult = cudaGetDeviceProperties(&properties, device);
        if (propertiesResult != cudaSuccess) {
            return propertiesResult;
        }
        info->name = properties.name;
        for (int axis = 0; axis < 3; ++axis) {
            info->maxGridSize[axis] = properties.maxGridSize[axis];
        }
        return cudaSuccess;
    }
    static Error malloc(void** ptr, std::size_t bytes) { return cudaMalloc(ptr, bytes); }
    static Error free(void* ptr) { return cudaFree(ptr); }
    static Error memset(void* ptr, int value, std::size_t bytes) { return cudaMemset(ptr, value, bytes); }
    static Error memcpyToHost(void* dst, const void* src, std::size_t bytes)
    {
        return cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
    }
    static Error synchronize() { return cudaDeviceSynchronize(); }
    static Error uploadFilters(
        const std::array<std::array<RotationInfo, MaxFilterCount>, MaxDirectionCount>& filters,
        const std::array<int, MaxDirectionCount>& filterCounts)
    {
        const Error result = cudaMemcpyToSymbol(
            coordsfinder_gpu::deviceFilters, filters.data(), sizeof(filters));
        if (result != cudaSuccess) {
            return result;
        }
        return cudaMemcpyToSymbol(
            coordsfinder_gpu::deviceFilterCounts, filterCounts.data(), sizeof(filterCounts));
    }
};

}

bool cudaAvailable(std::string* reason)
{
    return coordsfinder_gpu::gpuAvailable<CudaBackend>(reason);
}

bool runCudaScan(
    const ScanConfig& config,
    const ScanPlan& plan,
    ScanState* state,
    const MatchSink& sink,
    std::string* error)
{
    return coordsfinder_gpu::runGpuScan<CudaBackend>(config, plan, state, sink, error);
}
