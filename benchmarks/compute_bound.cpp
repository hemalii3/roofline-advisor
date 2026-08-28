// compute_bound.cpp
//
// Measures achievable peak compute throughput (GFLOP/s) on the CPU.
// This gives the horizontal "roofline" ceiling: the compute bound that
// high-arithmetic-intensity kernels asymptote towards.
//
// Method: a tight FMA loop over a small buffer that fits in L1 cache, so
// memory bandwidth is not the bottleneck -- we want to isolate compute
// throughput. Multiple independent accumulator chains are used to expose
// instruction-level parallelism (a single dependent FMA chain will be
// latency-bound, not throughput-bound, and will badly under-report peak).
//
// Usage: ./compute_bound [threads]
//   threads: number of OpenMP threads to use (default: all available)

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <thread>

#include "roofline/timer.hpp"
#include "roofline/dnb.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

// Small enough to stay resident in L1 across the whole benchmark.
constexpr int kBufferElems = 1024;
constexpr int kIters = 20'000'000; // iterations over the buffer
constexpr int kChains = 8;         // independent FMA accumulator chains

// Each iteration does kChains independent FMAs -> kChains*2 FLOPs.
double run_single_thread(int iters) {
    alignas(64) float buf[kBufferElems];
    for (int i = 0; i < kBufferElems; ++i) buf[i] = 1.0f + 0.0001f * i;

    float acc[kChains];
    for (int c = 0; c < kChains; ++c) acc[c] = 1.0f + 0.01f * c;

    roofline::Timer t;
    t.start();
    for (int it = 0; it < iters; ++it) {
        int idx = it & (kBufferElems - 1);
        #pragma GCC unroll 8
        for (int c = 0; c < kChains; ++c) {
            acc[c] = acc[c] * buf[idx] + buf[(idx + c) & (kBufferElems - 1)];
        }
    }
    double secs = t.stop();

    for (int c = 0; c < kChains; ++c) roofline::do_not_optimize(acc[c]);
    return secs;
}

} // namespace

int main(int argc, char** argv) {
    int nthreads = std::thread::hardware_concurrency();
#ifdef _OPENMP
    if (argc > 1) nthreads = std::atoi(argv[1]);
    omp_set_num_threads(nthreads);
#else
    nthreads = 1; // no OpenMP available; single-thread result only
#endif

    std::printf("# compute_bound: threads=%d buffer_elems=%d iters=%d chains=%d\n",
                nthreads, kBufferElems, kIters, kChains);
    std::printf("name,threads,seconds,bytes,flops,gbps,gflops\n");

    double flops_per_thread = static_cast<double>(kIters) * kChains * 2.0; // FMA = 2 flops

#ifdef _OPENMP
    double secs = roofline::median_time_seconds([&]() {
        #pragma omp parallel num_threads(nthreads)
        {
            run_single_thread(kIters);
        }
    }, /*warmup=*/2, /*reps=*/7);
#else
    double secs = roofline::median_time_seconds([&]() {
        run_single_thread(kIters);
    }, 2, 7);
#endif

    roofline::Result r{"compute_bound_fma", secs, 0.0, flops_per_thread * nthreads};
    roofline::print_csv_row(r.name, static_cast<size_t>(nthreads), r);

    return 0;
}
