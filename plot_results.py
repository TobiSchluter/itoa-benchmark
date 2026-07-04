#!/usr/bin/env python3
"""Plot itoa-benchmark results.

Reads a CSV produced by the `itoa` driver with columns:
    Type,Function,Mode,Series,X,Time_ns
and writes one PNG per (mode, width-class, series) into an output directory.

Signed and unsigned of the same width are drawn side by side so the cost of the
sign branch is directly comparable. Function colours are stable across every
figure.

Usage:
    plot_results.py RESULT.csv [--outdir DIR] [--exclude a,b] [--only a,b]
"""
import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Group signed/unsigned of the same width so they render side by side.
WIDTH_CLASSES = [
    ("32-bit",  ["u32", "i32"]),
    ("64-bit",  ["u64", "i64"]),
    ("128-bit", ["u128", "i128"]),
]

MODE_TITLES = {
    "bylength":      "Throughput by digit count (all values same length)",
    "loguniform":    "Log-uniform mix (equal count per digit length)",
    "unpredictable": "Marginal cost under unpredictable-length noise",
    "admixture":     "Two-length admixture (branch-misprediction sweep)",
}


def load(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            r["X"] = float(r["X"])
            r["Time_ns"] = float(r["Time_ns"])
            rows.append(r)
    return rows


def color_map(functions):
    funcs = sorted(functions)
    # tab20 + tab20b give 40 distinguishable colours (we have ~21 functions).
    base = (matplotlib.colormaps["tab20"].colors
            + matplotlib.colormaps["tab20b"].colors)
    return {fn: base[i % len(base)] for i, fn in enumerate(funcs)}


def clean_style():
    plt.rcParams.update({
        "figure.dpi": 130,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
        "lines.markersize": 4,
    })


def series_for(rows, mode, types):
    s = set()
    for r in rows:
        if r["Mode"] == mode and r["Type"] in types:
            s.add(r["Series"])
    return sorted(s)


def subset(rows, mode, typ, series):
    """Return {function: (xs, ys)} sorted by x."""
    by_fn = defaultdict(list)
    for r in rows:
        if r["Mode"] == mode and r["Type"] == typ and r["Series"] == series:
            by_fn[r["Function"]].append((r["X"], r["Time_ns"]))
    out = {}
    for fn, pts in by_fn.items():
        pts.sort()
        out[fn] = ([p[0] for p in pts], [p[1] for p in pts])
    return out


def plot_line(rows, mode, wc_name, types, series, colors, outdir, logy):
    present = [t for t in types if any(
        r["Type"] == t and r["Mode"] == mode and r["Series"] == series for r in rows)]
    if not present:
        return
    fig, axes = plt.subplots(1, len(present), figsize=(6.2 * len(present), 4.4),
                             squeeze=False, sharey=True)
    axes = axes[0]
    xlabel = "decimal digits"
    if mode == "admixture":
        a = series.split("x")[0]
        xlabel = f"% of {a}-digit values (rest = {series.split('x')[1]}-digit)"

    used = set()
    for ax, typ in zip(axes, present):
        data = subset(rows, mode, typ, series)
        for fn in sorted(data):
            xs, ys = data[fn]
            ax.plot(xs, ys, marker="o", color=colors[fn], label=fn)
            used.add(fn)
        ax.set_title(typ)
        ax.set_xlabel(xlabel)
        if logy:
            ax.set_yscale("log")
    axes[0].set_ylabel("ns / conversion")

    handles = [plt.Line2D([], [], color=colors[fn], marker="o", label=fn)
               for fn in sorted(used)]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5),
               fontsize=8, ncol=1)
    suptitle = f"{MODE_TITLES.get(mode, mode)}  —  {wc_name}"
    if series and mode != "admixture":
        suptitle += f"  [{series}]"
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=(0, 0, 0.86, 1))

    tag = series.replace("x", "x").replace("-", "_") if series else ""
    name = f"{mode}_{wc_name.split('-')[0]}" + (f"_{tag}" if tag else "")
    path = os.path.join(outdir, name + ".png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def plot_bars(rows, mode, wc_name, types, series, colors, outdir):
    """loguniform: one bar per function (single point) per type."""
    present = [t for t in types if any(
        r["Type"] == t and r["Mode"] == mode and r["Series"] == series for r in rows)]
    if not present:
        return
    fig, axes = plt.subplots(1, len(present), figsize=(5.2 * len(present), 4.6),
                             squeeze=False, sharex=True)
    axes = axes[0]
    for ax, typ in zip(axes, present):
        data = subset(rows, mode, typ, series)
        items = sorted(data.items(), key=lambda kv: kv[1][1][0])  # by time
        names = [k for k, _ in items]
        vals = [v[1][0] for _, v in items]
        ax.barh(range(len(names)), vals, color=[colors[n] for n in names])
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_title(typ)
        ax.set_xlabel("ns / conversion")
        ax.invert_yaxis()
    fig.suptitle(f"{MODE_TITLES.get(mode, mode)}  —  {wc_name}  [{series}]",
                 fontsize=11)
    fig.tight_layout()
    name = f"{mode}_{wc_name.split('-')[0]}_{series.replace('-', '_')}"
    path = os.path.join(outdir, name + ".png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--exclude", default="sprintf",
                    help="comma-separated functions to drop everywhere (default: sprintf)")
    ap.add_argument("--only", default="", help="comma-separated functions to keep")
    ap.add_argument("--exclude-small", default="naive",
                    help="functions to drop from 32/64-bit plots only "
                         "(kept on 128-bit); default: naive")
    args = ap.parse_args()

    rows = load(args.csv)
    excl = {s for s in args.exclude.split(",") if s}
    only = {s for s in args.only.split(",") if s}
    if excl:
        rows = [r for r in rows if r["Function"] not in excl]
    if only:
        rows = [r for r in rows if r["Function"] in only]

    outdir = args.outdir or os.path.join(os.path.dirname(os.path.abspath(args.csv)), "plots")
    os.makedirs(outdir, exist_ok=True)

    excl_small = {s for s in args.exclude_small.split(",") if s}

    clean_style()
    colors = color_map({r["Function"] for r in rows})
    modes = sorted({r["Mode"] for r in rows})

    for mode in modes:
        for wc_name, types in WIDTH_CLASSES:
            # Drop the slow-and-scale-dominating functions from the small widths
            # (e.g. naive), but keep them on 128-bit where they are the baseline.
            wc_rows = rows
            if excl_small and not wc_name.startswith("128"):
                wc_rows = [r for r in rows if r["Function"] not in excl_small]
            for series in series_for(wc_rows, mode, types):
                if mode == "loguniform":
                    plot_bars(wc_rows, mode, wc_name, types, series, colors, outdir)
                else:
                    logy = mode in ("bylength", "unpredictable")
                    plot_line(wc_rows, mode, wc_name, types, series, colors, outdir, logy)

    print("\nDone. PNGs in", outdir)


if __name__ == "__main__":
    main()
