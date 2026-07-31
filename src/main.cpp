#include <atomic>
#include <chrono>
#include <csignal>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <mutex>

#include "config.hpp"
#include "runner.hpp"

#ifndef COORDSFINDER_VERSION
#define COORDSFINDER_VERSION "dev"
#endif

namespace {
enum class Backend {
    Auto,
    Cpu,
    Cuda,
};

ScanState* activeState = nullptr;

void handleInterrupt(int)
{
    if (activeState) {
        activeState->cancelRequested.store(true, std::memory_order_relaxed);
    }
}

void printUsage(const char* program)
{
    std::printf(
        "CoordsFinder %s\n"
        "Usage: %s [options] <config-file>\n\n"
        "Options:\n"
        "  --backend auto|cpu|cuda  Select the execution backend (default: auto)\n"
        "  --threads N              CPU worker count (default: hardware threads)\n"
        "  --validate               Validate and summarize without scanning\n"
        "  --help                   Show this help\n"
        "  --version                Show the version\n",
        COORDSFINDER_VERSION,
        program);
}

bool parseBackend(const char* text, Backend* backend)
{
    if (std::strcmp(text, "auto") == 0) {
        *backend = Backend::Auto;
        return true;
    }
    if (std::strcmp(text, "cpu") == 0) {
        *backend = Backend::Cpu;
        return true;
    }
    if (std::strcmp(text, "cuda") == 0) {
        *backend = Backend::Cuda;
        return true;
    }
    return false;
}

bool parseThreads(const char* text, unsigned int* threads)
{
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(text, &end, 10);
    if (end == text || *end != '\0' || parsed == 0 || parsed > 65535) {
        return false;
    }
    *threads = static_cast<unsigned int>(parsed);
    return true;
}

const char* scanOrderName(ScanOrder order)
{
    return order == ScanOrder::Spiral ? "spiral" : "linear";
}
}

int main(int argc, char** argv)
{
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);

    Backend requestedBackend = Backend::Auto;
    unsigned int threadCount = 0;
    bool validateOnly = false;
    const char* configPath = nullptr;

    for (int i = 1; i < argc; ++i) {
        const char* argument = argv[i];
        if (std::strcmp(argument, "--help") == 0 || std::strcmp(argument, "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        }
        if (std::strcmp(argument, "--version") == 0) {
            std::printf("CoordsFinder %s\n", COORDSFINDER_VERSION);
            return 0;
        }
        if (std::strcmp(argument, "--validate") == 0) {
            validateOnly = true;
            continue;
        }
        if (std::strcmp(argument, "--backend") == 0 || std::strcmp(argument, "--threads") == 0) {
            if (++i >= argc) {
                std::fprintf(stderr, "Missing value for %s.\n", argument);
                return 1;
            }
            if (std::strcmp(argument, "--backend") == 0) {
                if (!parseBackend(argv[i], &requestedBackend)) {
                    std::fprintf(stderr, "Invalid backend '%s'.\n", argv[i]);
                    return 1;
                }
            }
            else if (!parseThreads(argv[i], &threadCount)) {
                std::fprintf(stderr, "Invalid thread count '%s'.\n", argv[i]);
                return 1;
            }
            continue;
        }
        if (argument[0] == '-') {
            std::fprintf(stderr, "Unknown option '%s'.\n", argument);
            return 1;
        }
        if (configPath) {
            std::fprintf(stderr, "Only one config file may be supplied.\n");
            return 1;
        }
        configPath = argument;
    }

    if (!configPath) {
        printUsage(argv[0]);
        return 1;
    }

    ScanConfig config;
    std::string error;
    if (!loadScanConfig(configPath, &config, &error)) {
        std::fprintf(stderr, "Config error: %s\n", error.c_str());
        return 1;
    }

    ScanPlan plan;
    if (!makeScanPlan(config, &plan, &error)) {
        std::fprintf(stderr, "Scan plan error: %s\n", error.c_str());
        return 1;
    }

    FILE* summary = validateOnly ? stdout : stderr;
    std::fprintf(summary, "Loaded %s with %zu filters and %zu direction(s).\n",
        config.sourcePath.c_str(),
        config.filter.size(),
        config.directions.size());
    std::fprintf(summary, "Mode: %s; order: %s; work items: %zu; candidates: ",
        textureModeName(config.mode),
        scanOrderName(config.scanOrder),
        plan.items.size());
    if (plan.totalCandidatesSaturated) {
        std::fprintf(summary, ">= %llu (display saturated).\n",
            static_cast<unsigned long long>(plan.totalCandidates));
    }
    else {
        std::fprintf(summary, "%llu.\n", static_cast<unsigned long long>(plan.totalCandidates));
    }

    if (validateOnly) {
        std::printf("Config and scan plan are valid.\n");
        return 0;
    }

    Backend backend = requestedBackend;
#if defined(COORDSFINDER_HAS_CUDA)
    std::string cudaReason;
    const bool hasCuda = cudaAvailable(&cudaReason);
    // Auto gracefully falls back to CPU when the binary has no usable CUDA device.
    if (backend == Backend::Auto) {
        backend = hasCuda ? Backend::Cuda : Backend::Cpu;
    }
    if (backend == Backend::Cuda && !hasCuda) {
        std::fprintf(stderr, "CUDA is unavailable: %s\n", cudaReason.c_str());
        return 1;
    }
#else
    if (backend == Backend::Cuda) {
        std::fprintf(stderr, "This is a CPU-only build. Reconfigure with COORDSFINDER_ENABLE_CUDA=ON.\n");
        return 1;
    }
    backend = Backend::Cpu;
#endif

    std::fprintf(stderr, "Backend: %s", backend == Backend::Cuda ? "CUDA" : "CPU");
    if (backend == Backend::Cpu) {
        if (threadCount == 0) {
            const unsigned int available = std::thread::hardware_concurrency();
            threadCount = available == 0 ? 1 : available;

            std::fprintf(stderr, " (automatic threads: %u)", threadCount);
        }
        else {
            std::fprintf(stderr, " (%u threads)", threadCount);
        }
    }
    std::fprintf(stderr, ".\n");

    ScanState state;
    activeState = &state;
    // Backends poll this flag between inexpensive units of work.
    std::signal(SIGINT, handleInterrupt);

    // Only matches go to stdout; status and progress remain pipe-safe on stderr.
    const MatchSink sink = [](const std::vector<Match>& matches) {
        for (const Match& match : matches) {
            std::printf("Found with %d bad block(s)! (%d, %d, %d), direction %d\n",
                match.badBlocks,
                match.x,
                match.y,
                match.z,
                match.direction);
        }
    };

    const auto started = std::chrono::steady_clock::now();
    std::atomic<bool> running { true };
    std::mutex progressMutex;
    std::condition_variable progressWake;
    std::thread progress([&] {
        std::unique_lock<std::mutex> lock(progressMutex);
        // The condition variable lets short scans finish without a one-second join delay.
        while (!progressWake.wait_for(lock, std::chrono::seconds(1), [&] {
            return !running.load(std::memory_order_relaxed);
        })) {
            if (config.printChunks) {
                continue;
            }
            const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            const std::uint64_t candidates = state.candidates.load(std::memory_order_relaxed);
            const double rate = elapsed > 0.0 ? static_cast<double>(candidates) / elapsed : 0.0;
            std::fprintf(stderr,
                "Progress: %llu/%zu work items, %.3f M candidates/s, %llu match(es).\n",
                static_cast<unsigned long long>(state.completedItems.load(std::memory_order_relaxed)),
                plan.items.size(),
                rate / 1000000.0,
                static_cast<unsigned long long>(state.matches.load(std::memory_order_relaxed)));
        }
    });

    bool succeeded = false;
    if (backend == Backend::Cpu) {
        succeeded = runCpuScan(config, plan, threadCount, &state, sink, &error);
    }
#if defined(COORDSFINDER_HAS_CUDA)
    else {
        succeeded = runCudaScan(config, plan, &state, sink, &error);
    }
#endif

    running.store(false, std::memory_order_relaxed);
    progressWake.notify_all();
    progress.join();
    activeState = nullptr;

    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    const std::uint64_t candidates = state.candidates.load(std::memory_order_relaxed);
    if (!succeeded) {
        std::fprintf(stderr, "Scan failed: %s\n", error.c_str());
        return 1;
    }
    if (state.cancelRequested.load(std::memory_order_relaxed)) {
        std::fprintf(stderr, "Scan cancelled after %.2f seconds.\n", elapsed);
        return 130;
    }

    std::fprintf(stderr, "All done in %.2f seconds (%.3f M candidates/s, %llu match(es)).\n",
        elapsed,
        elapsed > 0.0 ? static_cast<double>(candidates) / elapsed / 1000000.0 : 0.0,
        static_cast<unsigned long long>(state.matches.load(std::memory_order_relaxed)));
    return 0;
}
