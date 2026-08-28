#!/usr/bin/env python3
"""
build_actual_results.py

Turns master_results.csv (flattened pSTL-Bench JSON output) into
actual_results.csv (kernel,device,gflops), the format plot_roofline.py's
--actual flag expects, so predicted vs. measured can go on the same figure.

Design decisions baked in here (documented so they're easy to defend or
change later):
  - CPU baseline = GNU-TBB only (not "whichever backend is fastest per
    algorithm" -- a fixed baseline is a cleaner, more defensible comparison
    than cherry-picking per row).
  - GPU = Clang-SYCL (the GPU-targeted SYCL backend, not Clang-SYCL_CPU).
  - Largest tested input size per (algorithm, backend) is used, since the
    roofline model's predictions are asymptotic and most meaningful at
    large n.
  - master_results.csv's "gbps" column is actually raw bytes/sec (despite
    the misleading field name inherited from Google Benchmark's
    bytes_per_second counter) -- divided by 1e9 here to get real GB/s.
  - Achieved GFLOP/s is NOT measured directly (pSTL-Bench only logs
    bytes/sec, not FLOP counts). It's derived as
        achieved_gflops = achieved_gbps * kernel_arithmetic_intensity
    using the AI values from data/kernels.csv. This is a modeling choice,
    not a direct measurement -- state it explicitly in the writeup.

Usage:
    python3 scripts/build_actual_results.py \
        --master master_results.csv \
        --kernels data/kernels.csv \
        --out actual_results.csv
"""
import argparse
import csv


# master_results.csv algorithm names -> data/kernels.csv kernel names
ALGO_NAME_MAP = {
    "find": "find",
    "for_each": "for_each",
    "inclusive_scan": "inc_scan",
    "reduce": "reduce",
    "sort": "sort",
}

BACKEND_TO_DEVICE = {
    "GNU-TBB": "cpu",
    "Clang-SYCL": "gpu",
}


def load_kernel_ai(path):
    """Returns {kernel_name: arithmetic_intensity} from data/kernels.csv."""
    ai = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split(",")
            if len(fields) < 3:
                continue
            name, flops_per_elem, bytes_per_elem = fields[0], float(fields[1]), float(fields[2])
            ai[name] = flops_per_elem / bytes_per_elem
    return ai


def load_best_rows(master_path):
    """
    Returns {(algorithm, backend): row} keeping the row with the largest
    'size' for each (algorithm, backend) pair, restricted to the backends
    we care about (GNU-TBB, Clang-SYCL).
    """
    best = {}
    with open(master_path) as f:
        for row in csv.DictReader(f):
            if row["backend"] not in BACKEND_TO_DEVICE:
                continue
            if not row["size"] or not row["gbps"]:
                continue
            key = (row["algorithm"], row["backend"])
            size = int(row["size"])
            if key not in best or size > int(best[key]["size"]):
                best[key] = row
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default="master_results.csv")
    ap.add_argument("--kernels", default="data/kernels.csv")
    ap.add_argument("--out", default="actual_results.csv")
    args = ap.parse_args()

    ai_by_kernel = load_kernel_ai(args.kernels)
    best_rows = load_best_rows(args.master)

    out_rows = []
    skipped = []
    for (algorithm, backend), row in sorted(best_rows.items()):
        kernel = ALGO_NAME_MAP.get(algorithm)
        device = BACKEND_TO_DEVICE.get(backend)
        if kernel is None or device is None:
            skipped.append((algorithm, backend, "no name/device mapping"))
            continue
        if kernel not in ai_by_kernel:
            skipped.append((algorithm, backend, f"'{kernel}' not in {args.kernels} -- fill it in first"))
            continue

        raw_bytes_per_sec = float(row["gbps"])  # actually bytes/sec, see module docstring
        achieved_gbps = raw_bytes_per_sec / 1e9
        achieved_gflops = achieved_gbps * ai_by_kernel[kernel]

        out_rows.append((kernel, device, achieved_gflops, row["size"], achieved_gbps))

    with open(args.out, "w") as f:
        f.write("kernel,device,gflops\n")
        for kernel, device, gflops, size, gbps in out_rows:
            f.write(f"{kernel},{device},{gflops}\n")

    print(f"Wrote {len(out_rows)} rows to {args.out}")
    print()
    print(f"{'kernel':12s} {'device':6s} {'size':>12s} {'achieved GB/s':>14s} {'achieved GFLOP/s':>18s}")
    for kernel, device, gflops, size, gbps in out_rows:
        print(f"{kernel:12s} {device:6s} {size:>12s} {gbps:>14.2f} {gflops:>18.4f}")

    if skipped:
        print()
        print("Skipped (fix and re-run):")
        for algorithm, backend, reason in skipped:
            print(f"  {algorithm} / {backend}: {reason}")


if __name__ == "__main__":
    main()
