#!/usr/bin/env bash
# Sweeps bandwidth_bound across a range of sizes to find the DRAM plateau
# (once arrays are well beyond LLC size, GB/s should flatten out -- that
# plateau value is what you plug into data/devices.csv as peak_bandwidth).
#
# Usage: scripts/sweep_bandwidth.sh [build_dir] [max_threads]
set -euo pipefail

BUILD_DIR="${1:-build}"
MAX_THREADS="${2:-$(nproc)}"
BIN="${BUILD_DIR}/benchmarks/bandwidth_bound"

if [[ ! -x "$BIN" ]]; then
    echo "error: $BIN not found -- build the project first (see README)" >&2
    exit 1
fi

OUT="data/bandwidth_sweep.csv"
echo "name,elements,seconds,bytes,flops,gbps,gflops" > "$OUT"

# Sizes from well within L2 up through well beyond typical LLC (adjust
# upper bound if your machine has an unusually large cache).
SIZES=(65536 262144 1048576 4194304 16777216 67108864 268435456)

for n in "${SIZES[@]}"; do
    echo "n=$n threads=$MAX_THREADS" >&2
    "$BIN" "$n" "$MAX_THREADS" | tail -n1 >> "$OUT"
done

echo "Wrote $OUT"
