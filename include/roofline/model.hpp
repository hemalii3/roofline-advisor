#pragma once

#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

namespace roofline {

// Roofline parameters for one device, as measured by the compute_bound /
// bandwidth_bound micro-benchmarks in benchmarks/.
struct Device {
    std::string name;
    double peak_gflops;      // horizontal roofline ceiling
    double peak_bandwidth;   // GB/s, diagonal roofline slope
    double pcie_gbps;        // host<->device transfer bandwidth (0 for host/CPU)

    // Arithmetic intensity (FLOP/byte) at which the roofline "ridge point"
    // sits -- below this, kernels are bandwidth-bound; above, compute-bound.
    double ridge_point() const { return peak_gflops / peak_bandwidth; }

    // Achievable GFLOP/s for a kernel with the given arithmetic intensity,
    // per the standard roofline model: min(peak compute, AI * peak bw).
    double achievable_gflops(double arithmetic_intensity) const {
        return std::min(peak_gflops, arithmetic_intensity * peak_bandwidth);
    }
};

// Per-element cost profile of a parallel STL kernel (from pSTL-Bench).
struct KernelProfile {
    std::string name;
    double flops_per_elem;
    double bytes_per_elem;   // total read+write bytes touched per element

    double arithmetic_intensity() const { return flops_per_elem / bytes_per_elem; }
};

// Predicted wall-clock time (seconds) to run a kernel over n elements on a
// given device, NOT including any host<->device transfer.
inline double predict_compute_seconds(const KernelProfile& k, const Device& d, size_t n) {
    double ai = k.arithmetic_intensity();
    double gflops = d.achievable_gflops(ai);
    double total_flops = k.flops_per_elem * static_cast<double>(n);
    return (total_flops / 1e9) / gflops;
}

// Predicted host<->device transfer time (seconds) for n elements, assuming
// input must be copied to the device and output copied back. elem_bytes is
// the size of one element (e.g. 8 for double); this is a simplification --
// many kernels have different input/output sizes (e.g. reduce writes back
// only a scalar) -- so callers should pass the correct in+out byte total
// per element for their kernel, not just sizeof(T).
inline double predict_transfer_seconds(double transfer_bytes_total, double pcie_gbps) {
    if (pcie_gbps <= 0.0) return 0.0; // host device, no transfer
    return (transfer_bytes_total / 1e9) / pcie_gbps;
}

// Total predicted time including transfer, for offload decision-making.
// This is the PESSIMISTIC / naive model: it assumes a fresh full transfer
// on every call. Real USM-based implementations that cache device buffers
// across repeated calls (as pSTL-Bench does) pay the transfer cost once
// and reuse it -- see predict_total_seconds_amortized below.
inline double predict_total_seconds(const KernelProfile& k, const Device& d, size_t n,
                                     double transfer_bytes_total) {
    return predict_compute_seconds(k, d, n)
         + predict_transfer_seconds(transfer_bytes_total, d.pcie_gbps);
}

// Total predicted time including transfer, AMORTIZED across `reuse_count`
// calls on the same cached device buffer. This matches benchmark
// methodologies (like pSTL-Bench's) that transfer input once and reuse it
// across repetitions: the one-time transfer cost is divided by the number
// of times that transfer gets reused, so its effective per-call cost
// shrinks as reuse_count grows. reuse_count=1 is equivalent to the naive
// model above.
inline double predict_total_seconds_amortized(const KernelProfile& k, const Device& d, size_t n,
                                               double transfer_bytes_total, int reuse_count) {
    double transfer_secs = predict_transfer_seconds(transfer_bytes_total, d.pcie_gbps);
    double amortized_transfer = transfer_secs / std::max(1, reuse_count);
    return predict_compute_seconds(k, d, n) + amortized_transfer;
}

// Finds the smallest n (searched in [n_min, n_max], doubling) at which
// device `gpu` becomes faster than device `cpu` for kernel k, including
// gpu's transfer cost. Returns 0 if no crossover found in range (GPU never
// wins), or n_max+1 if GPU wins even at n_min (crossover below range).
struct CrossoverResult {
    bool found = false;
    size_t crossover_n = 0;
    bool gpu_wins_at_min = false;
};

inline CrossoverResult find_crossover(const KernelProfile& k,
                                       const Device& cpu, const Device& gpu,
                                       double gpu_transfer_bytes_per_elem,
                                       size_t n_min, size_t n_max) {
    CrossoverResult result;
    auto gpu_time = [&](size_t n) {
        return predict_total_seconds(k, gpu, n, gpu_transfer_bytes_per_elem * n);
    };
    auto cpu_time = [&](size_t n) {
        return predict_compute_seconds(k, cpu, n);
    };

    bool gpu_wins_at_min = gpu_time(n_min) < cpu_time(n_min);
    bool gpu_wins_at_max = gpu_time(n_max) < cpu_time(n_max);

    if (gpu_wins_at_min) {
        result.found = true;
        result.gpu_wins_at_min = true;
        result.crossover_n = n_min;
        return result;
    }
    if (!gpu_wins_at_max) {
        result.found = false; // GPU never wins in this range
        return result;
    }

    // Binary search (in log-space via doubling then bisection) for the
    // crossover point between n_min and n_max.
    size_t lo = n_min, hi = n_max;
    while (hi - lo > std::max<size_t>(1, lo / 100)) { // ~1% precision
        size_t mid = lo + (hi - lo) / 2;
        if (gpu_time(mid) < cpu_time(mid)) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    result.found = true;
    result.crossover_n = hi;
    return result;
}

// Same as find_crossover, but using the amortized-transfer model (transfer
// cost divided by reuse_count). Pass reuse_count=1 to reproduce the naive
// find_crossover result exactly.
inline CrossoverResult find_crossover_amortized(const KernelProfile& k,
                                                 const Device& cpu, const Device& gpu,
                                                 double gpu_transfer_bytes_per_elem,
                                                 int reuse_count,
                                                 size_t n_min, size_t n_max) {
    CrossoverResult result;
    auto gpu_time = [&](size_t n) {
        return predict_total_seconds_amortized(k, gpu, n, gpu_transfer_bytes_per_elem * n, reuse_count);
    };
    auto cpu_time = [&](size_t n) {
        return predict_compute_seconds(k, cpu, n);
    };

    bool gpu_wins_at_min = gpu_time(n_min) < cpu_time(n_min);
    bool gpu_wins_at_max = gpu_time(n_max) < cpu_time(n_max);

    if (gpu_wins_at_min) {
        result.found = true;
        result.gpu_wins_at_min = true;
        result.crossover_n = n_min;
        return result;
    }
    if (!gpu_wins_at_max) {
        result.found = false;
        return result;
    }

    size_t lo = n_min, hi = n_max;
    while (hi - lo > std::max<size_t>(1, lo / 100)) {
        size_t mid = lo + (hi - lo) / 2;
        if (gpu_time(mid) < cpu_time(mid)) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    result.found = true;
    result.crossover_n = hi;
    return result;
}

} // namespace roofline
