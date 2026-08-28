#pragma once

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>

namespace roofline {

// Simple wall-clock stopwatch.
class Timer {
public:
    void start() { t0_ = std::chrono::steady_clock::now(); }

    // Returns elapsed seconds since start().
    double stop() {
        auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(t1 - t0_).count();
    }

private:
    std::chrono::steady_clock::time_point t0_;
};

// Runs `fn` `reps` times (after `warmup` untimed warmup runs), returns the
// median wall-clock time in seconds. Median is used instead of mean because
// micro-benchmarks are prone to occasional OS-noise outliers, and we care
// about a stable, reproducible number more than capturing tail behaviour.
template <typename Fn>
double median_time_seconds(Fn&& fn, int warmup = 3, int reps = 10) {
    for (int i = 0; i < warmup; ++i) fn();

    std::vector<double> samples;
    samples.reserve(reps);
    Timer t;
    for (int i = 0; i < reps; ++i) {
        t.start();
        fn();
        samples.push_back(t.stop());
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

// Reporting helpers -----------------------------------------------------

struct Result {
    std::string name;
    double seconds;
    double bytes;   // total bytes moved (read+write), for bandwidth results
    double flops;   // total floating point ops, for compute results
};

inline double gb_per_sec(double bytes, double seconds) {
    return (bytes / 1e9) / seconds;
}

inline double gflop_per_sec(double flops, double seconds) {
    return (flops / 1e9) / seconds;
}

inline void print_result(const Result& r) {
    std::printf("%-28s  time=%.6f s", r.name.c_str(), r.seconds);
    if (r.bytes > 0) {
        std::printf("  bw=%.2f GB/s", gb_per_sec(r.bytes, r.seconds));
    }
    if (r.flops > 0) {
        std::printf("  compute=%.2f GFLOP/s", gflop_per_sec(r.flops, r.seconds));
    }
    std::printf("\n");
}

// Emits a single CSV row: name,elements,seconds,bytes,flops,gbps,gflops
// Header should be written once by the caller.
inline void print_csv_row(const std::string& name, size_t n, const Result& r) {
    double gbps = r.bytes > 0 ? gb_per_sec(r.bytes, r.seconds) : 0.0;
    double gflops = r.flops > 0 ? gflop_per_sec(r.flops, r.seconds) : 0.0;
    std::printf("%s,%zu,%.6f,%.0f,%.0f,%.4f,%.4f\n",
                name.c_str(), n, r.seconds, r.bytes, r.flops, gbps, gflops);
}

} // namespace roofline
