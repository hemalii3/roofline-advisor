// bandwidth_bound.cpp
//
// Measures achievable peak memory bandwidth (GB/s) via a STREAM-triad-style
// kernel: c[i] = a[i] + scalar * b[i]. This gives the diagonal roofline
// ceiling that low-arithmetic-intensity kernels (reduce, for_each, find,
// copy-like scan) are bound by.
//
// Buffers are sized well beyond LLC capacity so the measurement reflects
// DRAM bandwidth, not cache bandwidth. Run with a range of sizes via the
// sweep script to also see the cache-to-DRAM transition, which is useful
// context even though only the DRAM plateau feeds the roofline model.
//
// Usage: ./bandwidth_bound <n_elements> [threads]

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <thread>

#include "roofline/timer.hpp"
#include "roofline/dnb.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc, char** argv) {
    size_t n = (argc > 1) ? std::strtoull(argv[1], nullptr, 10) : (1ull << 26); // ~256MB/array default
    int nthreads = std::thread::hardware_concurrency();
#ifdef _OPENMP
    if (argc > 2) nthreads = std::atoi(argv[2]);
    omp_set_num_threads(nthreads);
#else
    nthreads = 1;
#endif

    std::vector<double> a(n), b(n), c(n);
    for (size_t i = 0; i < n; ++i) {
        a[i] = 1.0;
        b[i] = 2.0;
    }
    const double scalar = 3.0;

    auto triad = [&]() {
#ifdef _OPENMP
        #pragma omp parallel for num_threads(nthreads) schedule(static)
#endif
        for (size_t i = 0; i < n; ++i) {
            c[i] = a[i] + scalar * b[i];
        }
    };

    double secs = roofline::median_time_seconds(triad, /*warmup=*/2, /*reps=*/7);
    roofline::do_not_optimize(c[0]);
    roofline::do_not_optimize(c[n - 1]);

    // Triad touches: read a[i], read b[i], write c[i] -> 3 * 8 bytes/elem.
    double bytes = static_cast<double>(n) * 3.0 * sizeof(double);
    double flops = static_cast<double>(n) * 2.0; // one mul, one add per elem

    std::printf("# bandwidth_bound (STREAM triad): n=%zu threads=%d array_bytes_each=%.1fMB\n",
                n, nthreads, (n * sizeof(double)) / 1e6);
    std::printf("name,elements,seconds,bytes,flops,gbps,gflops\n");
    roofline::Result r{"bandwidth_bound_triad", secs, bytes, flops};
    roofline::print_csv_row(r.name, n, r);

    return 0;
}
