#!/usr/bin/env python3
"""
extract_pstlbench.py

Flattens a directory of pSTL-Bench Google-Benchmark JSON result files into
one master CSV, so you can eyeball / filter / pivot in a spreadsheet or
pandas rather than hand-copying numbers out of dozens of JSON files.
"""
import argparse
import glob
import json
import os
import statistics
import sys
from collections import defaultdict


def parse_run_name(name):
    parts = name.split("/")
    backend = parts[0] if len(parts) > 0 else ""
    algorithm = parts[1] if len(parts) > 1 else ""
    if "::" in algorithm:
        algorithm = algorithm.split("::", 1)[1]

    dtype = parts[2] if len(parts) > 2 else ""

    size = None
    tag_parts = []
    for p in parts[3:]:
        if p.isdigit() and size is None:
            size = int(p)
        elif p == "manual_time":
            continue
        else:
            tag_parts.append(p)

    tag = "/".join(tag_parts)
    return backend, algorithm, dtype, size, tag


def median(values):
    return statistics.median(values) if values else None


def process_file(path):
    with open(path) as f:
        data = json.load(f)

    benchmarks = data.get("benchmarks", [])

    groups = defaultdict(list)
    for b in benchmarks:
        run_name = b.get("run_name", b.get("name", ""))
        groups[run_name].append(b)

    rows = []
    for run_name, entries in groups.items():
        backend, algorithm, dtype, size, tag = parse_run_name(run_name)

        real_times = [e["real_time"] for e in entries if "real_time" in e]
        cpu_times = [e["cpu_time"] for e in entries if "cpu_time" in e]
        gbps_vals = [e["bytes_per_second"] for e in entries if "bytes_per_second" in e]

        rows.append({
            "file": os.path.basename(path),
            "backend": backend,
            "algorithm": algorithm,
            "dtype": dtype,
            "size": size,
            "tag": tag,
            "real_time_ns": median(real_times),
            "cpu_time_ns": median(cpu_times),
            "gbps": median(gbps_vals),
            "repetitions": len(entries),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", help="directory containing pSTL-Bench *.json result files")
    ap.add_argument("--out", default="master_results.csv")
    args = ap.parse_args()

    json_files = sorted(glob.glob(os.path.join(args.results_dir, "*.json")))
    if not json_files:
        print(f"error: no .json files found in {args.results_dir}", file=sys.stderr)
        sys.exit(1)

    all_rows = []
    skipped = []
    for path in json_files:
        try:
            all_rows.extend(process_file(path))
        except (json.JSONDecodeError, KeyError) as e:
            skipped.append((path, str(e)))

    if skipped:
        print(f"warning: skipped {len(skipped)} files that didn't parse as expected:", file=sys.stderr)
        for path, err in skipped:
            print(f"  {os.path.basename(path)}: {err}", file=sys.stderr)

    cols = ["file", "backend", "algorithm", "dtype", "size", "tag",
            "real_time_ns", "cpu_time_ns", "gbps", "repetitions"]
    with open(args.out, "w") as f:
        f.write(",".join(cols) + "\n")
        for row in all_rows:
            f.write(",".join(str(row[c]) if row[c] is not None else "" for c in cols) + "\n")

    print(f"Wrote {len(all_rows)} rows from {len(json_files)} files to {args.out}")
    print(f"({len(skipped)} files skipped -- see warnings above)")


if __name__ == "__main__":
    main()
