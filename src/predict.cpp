// predict.cpp
//
// Loads device roofline parameters (data/devices.csv) and kernel profiles
// (data/kernels.csv), then for each kernel:
//   - prints predicted achievable GFLOP/s on each device
//   - finds the predicted crossover input size where GPU (incl. transfer)
//     beats CPU
//   - emits a CSV (roofline_predictions.csv) that scripts/plot_roofline.py
//     turns into the headline figure.
//
// Usage: ./predict [devices.csv] [kernels.csv] [out.csv] [reuse_count]
//
// reuse_count (default 10, matching pSTL-Bench's repetition count) controls
// the amortized-transfer model: it's how many times a device buffer gets
// reused after being transferred once, before being freed. Pass 1 to make
// the amortized model identical to the naive one.

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>

#include "roofline/model.hpp"

namespace {

std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string field;
    while (std::getline(ss, field, ',')) fields.push_back(field);
    return fields;
}

bool is_comment_or_blank(const std::string& line) {
    for (char c : line) {
        if (c == ' ' || c == '\t' || c == '\r') continue;
        return c == '#' ? true : false;
    }
    return true; // all whitespace
}

std::vector<roofline::Device> load_devices(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<roofline::Device> devices;
    std::string line;
    while (std::getline(f, line)) {
        if (is_comment_or_blank(line)) continue;
        auto fields = split_csv_line(line);
        if (fields.size() < 4) continue;
        roofline::Device d;
        d.name = fields[0];
        d.peak_gflops = std::stod(fields[1]);
        d.peak_bandwidth = std::stod(fields[2]);
        d.pcie_gbps = std::stod(fields[3]);
        devices.push_back(d);
    }
    return devices;
}

std::vector<roofline::KernelProfile> load_kernels(const std::string& path,
                                                    std::vector<double>& transfer_bytes_out) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<roofline::KernelProfile> kernels;
    std::string line;
    while (std::getline(f, line)) {
        if (is_comment_or_blank(line)) continue;
        auto fields = split_csv_line(line);
        if (fields.size() < 4) continue;
        roofline::KernelProfile k;
        k.name = fields[0];
        k.flops_per_elem = std::stod(fields[1]);
        k.bytes_per_elem = std::stod(fields[2]);
        kernels.push_back(k);
        transfer_bytes_out.push_back(std::stod(fields[3]));
    }
    return kernels;
}

} // namespace

int main(int argc, char** argv) {
    std::string devices_path = argc > 1 ? argv[1] : "data/devices.csv";
    std::string kernels_path = argc > 2 ? argv[2] : "data/kernels.csv";
    std::string out_path = argc > 3 ? argv[3] : "roofline_predictions.csv";
    int reuse_count = argc > 4 ? std::atoi(argv[4]) : 10;

    auto devices = load_devices(devices_path);
    std::vector<double> transfer_bytes;
    auto kernels = load_kernels(kernels_path, transfer_bytes);

    // Expect exactly one device named "cpu" and one named "gpu" for the
    // crossover analysis; other devices are still shown in the roofline
    // table but skipped for crossover (extend this if you add more).
    const roofline::Device* cpu = nullptr;
    const roofline::Device* gpu = nullptr;
    for (auto& d : devices) {
        if (d.name == "cpu") cpu = &d;
        if (d.name == "gpu") gpu = &d;
    }

    std::printf("=== Roofline summary ===\n");
    for (auto& d : devices) {
        std::printf("%-6s peak=%.1f GFLOP/s  bw=%.1f GB/s  ridge=%.3f FLOP/byte  pcie=%.1f GB/s\n",
                    d.name.c_str(), d.peak_gflops, d.peak_bandwidth, d.ridge_point(), d.pcie_gbps);
    }
    std::printf("\n=== Kernel arithmetic intensity ===\n");
    for (auto& k : kernels) {
        std::printf("%-10s AI=%.4f FLOP/byte\n", k.name.c_str(), k.arithmetic_intensity());
    }

    std::ofstream out(out_path);
    out << "kernel,arithmetic_intensity,cpu_gflops,gpu_gflops,"
        << "naive_crossover_n,naive_status,amortized_crossover_n,amortized_status,reuse_count\n";

    std::printf("\n=== Predicted crossover points (reuse_count=%d) ===\n", reuse_count);
    std::printf("%-10s %-22s %-30s\n", "kernel", "naive (fresh transfer)", "amortized (cached buffer)");
    for (size_t i = 0; i < kernels.size(); ++i) {
        auto& k = kernels[i];
        double ai = k.arithmetic_intensity();
        double cpu_gflops = cpu ? cpu->achievable_gflops(ai) : 0.0;
        double gpu_gflops = gpu ? gpu->achievable_gflops(ai) : 0.0;

        std::string naive_status = "n/a", amortized_status = "n/a";
        size_t naive_n = 0, amortized_n = 0;

        if (cpu && gpu) {
            auto naive_res = roofline::find_crossover(k, *cpu, *gpu, transfer_bytes[i],
                                                        /*n_min=*/1'000, /*n_max=*/(1ull << 32));
            if (!naive_res.found) {
                naive_status = "never";
            } else if (naive_res.gpu_wins_at_min) {
                naive_status = "always";
                naive_n = naive_res.crossover_n;
            } else {
                naive_status = "crosses";
                naive_n = naive_res.crossover_n;
            }

            auto amortized_res = roofline::find_crossover_amortized(
                k, *cpu, *gpu, transfer_bytes[i], reuse_count,
                /*n_min=*/1'000, /*n_max=*/(1ull << 32));
            if (!amortized_res.found) {
                amortized_status = "never";
            } else if (amortized_res.gpu_wins_at_min) {
                amortized_status = "always";
                amortized_n = amortized_res.crossover_n;
            } else {
                amortized_status = "crosses";
                amortized_n = amortized_res.crossover_n;
            }

            std::string naive_disp = naive_status;
            if (naive_status == "crosses" || naive_status == "always") {
                naive_disp += " (n~" + std::to_string(naive_n) + ")";
            }
            std::string amortized_disp = amortized_status;
            if (amortized_status == "crosses" || amortized_status == "always") {
                amortized_disp += " (n~" + std::to_string(amortized_n) + ")";
            }
            std::printf("%-10s %-22s %-30s\n", k.name.c_str(), naive_disp.c_str(), amortized_disp.c_str());
        }

        out << k.name << "," << ai << "," << cpu_gflops << "," << gpu_gflops << ","
            << naive_n << "," << naive_status << ","
            << amortized_n << "," << amortized_status << "," << reuse_count << "\n";
    }

    std::printf("\nWrote %s (feed to scripts/plot_roofline.py)\n", out_path.c_str());
    return 0;
}
