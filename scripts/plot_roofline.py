#!/usr/bin/env python3
"""
plot_roofline.py

Renders the headline figure: a log-log roofline plot (achievable GFLOP/s vs
arithmetic intensity) for CPU and GPU, with each of the five algorithms
plotted as a point at its arithmetic intensity, colored by predicted
crossover status. This is the plot to put front-and-center in the README.

Usage:
    python3 scripts/plot_roofline.py \
        --devices data/devices.csv \
        --predictions roofline_predictions.csv \
        --out docs/roofline.png

Also optionally overlays *actual* measured GFLOP/s (from your P1 pSTL-Bench
results) if you pass --actual actual_results.csv with columns
kernel,device,gflops -- this is what lets you show predicted-vs-actual on
the same figure, which is the validation story for the writeup.
"""
import argparse
import csv
import math
import sys

import matplotlib.pyplot as plt
import numpy as np


def load_devices(path):
    devices = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, peak_gflops, peak_bw, pcie = line.split(",")
            devices[name] = {
                "peak_gflops": float(peak_gflops),
                "peak_bw": float(peak_bw),
                "pcie_gbps": float(pcie),
            }
    return devices


def load_predictions(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_actual(path):
    rows = []
    if not path:
        return rows
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def roofline_curve(peak_gflops, peak_bw, ai_range):
    return [min(peak_gflops, ai * peak_bw) for ai in ai_range]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", default="data/devices.csv")
    ap.add_argument("--predictions", default="roofline_predictions.csv")
    ap.add_argument("--actual", default=None,
                     help="optional CSV with columns kernel,device,gflops "
                          "from real pSTL-Bench measurements")
    ap.add_argument("--out", default="docs/roofline.png")
    args = ap.parse_args()

    devices = load_devices(args.devices)
    predictions = load_predictions(args.predictions)
    actual = load_actual(args.actual)

    ai_min = min(0.01, min(float(p["arithmetic_intensity"]) for p in predictions) / 2)
    ai_min = max(ai_min, 1e-3)  # floor: some kernels (e.g. find) may have AI=0,
                                 # which breaks a log-scale axis -- clamp instead
                                 # of crashing, and note it in the plot's caption.
    ai_max = max(100.0, max(float(p["arithmetic_intensity"]) for p in predictions) * 2)
    ai_range = np.logspace(math.log10(ai_min), math.log10(ai_max), 200)

    fig, ax = plt.subplots(figsize=(9, 6))

    colors = {"cpu": "#2b6cb0", "gpu": "#c05621"}
    for name, params in devices.items():
        curve = roofline_curve(params["peak_gflops"], params["peak_bw"], ai_range)
        ax.plot(ai_range, curve, label=f"{name} roofline", color=colors.get(name, None), linewidth=2)
        ridge = params["peak_gflops"] / params["peak_bw"]
        ax.axvline(ridge, color=colors.get(name, "gray"), linestyle=":", alpha=0.4)

    # Predicted points
    for p in predictions:
        ai = float(p["arithmetic_intensity"])
        ai_plot = max(ai, ai_min)  # floor AI=0 kernels (e.g. find) so they're
                                    # visible on the log axis instead of vanishing
        label_suffix = " (AI≈0, see docs)" if ai == 0 else ""
        for dev_key, gflops_key, marker in [("cpu", "cpu_gflops", "o"), ("gpu", "gpu_gflops", "^")]:
            gflops = float(p[gflops_key])
            ax.scatter([ai_plot], [gflops], marker=marker, s=80,
                       color=colors.get(dev_key), edgecolor="black", zorder=5)
            ax.annotate(p["kernel"] + label_suffix, (ai_plot, gflops), textcoords="offset points",
                        xytext=(6, 6), fontsize=8)

    # Actual measured points, if provided -- hollow markers to distinguish
    # from the filled predicted markers.
    for a in actual:
        ai = None
        for p in predictions:
            if p["kernel"] == a["kernel"]:
                ai = float(p["arithmetic_intensity"])
                break
        if ai is None:
            continue
        gflops = float(a["gflops"])
        dev = a["device"]
        marker = "o" if dev == "cpu" else "^"
        ai_plot = max(ai, ai_min)  # same floor as predicted points, for AI=0 kernels
        ax.scatter([ai_plot], [gflops], marker=marker, s=100, facecolors="none",
                   edgecolors=colors.get(dev, "black"), linewidths=2, zorder=6)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("Achievable performance (GFLOP/s)")
    ax.set_title("Roofline: predicted vs actual, CPU vs GPU\n"
                  "(filled = predicted, hollow = measured)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        print("Run ./build/predict first to generate roofline_predictions.csv", file=sys.stderr)
        sys.exit(1)
