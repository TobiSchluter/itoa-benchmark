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
    "uniform":       "Equidistributed values (byte-printing regime)",
}


def load(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            r["X"] = float(r["X"])
            r["Time_ns"] = float(r["Time_ns"])
            rows.append(r)
    return rows


# Canonical function order: colours (and zmij markers) are assigned by NAME
# from this list, so every subset -- --only runs, the straddle drops, digest
# CSVs -- draws each function in the same colour as the full set. Functions
# not listed here get stable colours after the known ones, in sorted order.
KNOWN_FUNCTIONS = [
    "amartin", "branchlut", "branchlut2", "count", "countlut", "fmt",
    "jeaiii", "lut", "mwilson", "naive", "null", "sprintf", "sse2",
    "tmueller", "toy256", "unnamed", "unrolledlut", "yy",
    "zmij_scalar", "zmij_x64_v1", "zmij_x64_v2", "zmij_x64_v3",
    "zmij_x64_v4", "zmij_x64_native", "zmij_neon",
]


def _canonical(functions):
    known = [fn for fn in KNOWN_FUNCTIONS if fn in functions]
    return known + sorted(set(functions) - set(KNOWN_FUNCTIONS))


def color_map(functions):
    # tab20 + tab20b give 40 distinguishable colours (we have ~25 functions).
    base = (matplotlib.colormaps["tab20"].colors
            + matplotlib.colormaps["tab20b"].colors)
    unknown = sorted(set(functions) - set(KNOWN_FUNCTIONS))
    def idx(fn):
        if fn in KNOWN_FUNCTIONS:
            return KNOWN_FUNCTIONS.index(fn)
        return len(KNOWN_FUNCTIONS) + unknown.index(fn)
    return {fn: base[idx(fn) % len(base)] for fn in functions}


# Distinct filled marker shapes on top of colour. Only the zmij lines get
# markers, so they stand out against the competitor field (and stay tellable
# apart from each other in print / greyscale).
_MARKERS = ["o", "s", "^", "v", "D", "P", "X", "*", "<", ">", "p", "h", "H", "d", "8"]


def marker_map(functions):
    markers = {fn: None for fn in functions}
    zmij = [fn for fn in _canonical(functions) if fn.startswith("zmij")]
    for i, fn in enumerate(zmij):
        markers[fn] = _MARKERS[i % len(_MARKERS)]
    return markers


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


def plot_line(rows, mode, wc_name, types, series, colors, markers, outdir, logy, suffix=""):
    present = [t for t in types if any(
        r["Type"] == t and r["Mode"] == mode and r["Series"] == series for r in rows)]
    if not present:
        return
    fig, axes = plt.subplots(1, len(present), figsize=(6.2 * len(present), 4.4),
                             squeeze=False, sharey=True)
    axes = axes[0]
    # For admixture, plot magnitude increasing left->right: x = % of the LARGER
    # value, so the left edge is all-smaller and the right edge all-larger.
    # Stored X is % of the smaller value, so plot 100 - X.
    xlabel = "decimal digits"
    flip = mode == "admixture"
    if mode == "admixture":
        if "x" in series:
            a, b = series.split("x")
            xlabel = f"% of {b}-digit values (rest = {a}-digit)"
        elif series == "straddle64":
            xlabel = "% of values above 2^64 (rest below; u64↔u128 branch)"
        elif series == "straddle1e32":
            xlabel = "% of values above 1e32 (rest below; 1-peel↔2-peel branch)"

    # The straddle sweeps exist to expose zmij's u128 branch bumps; the fmt/naive
    # baselines (~250ns) would dwarf them, so drop them and let the axis zoom to
    # the zmij range.
    drop = {"fmt", "naive"} if series.startswith("straddle") else set()
    used = set()
    for ax, typ in zip(axes, present):
        data = subset(rows, mode, typ, series)
        for fn in sorted(data):
            if fn in drop:
                continue
            xs, ys = data[fn]
            if flip:
                xs = [100 - x for x in xs]
            ax.plot(xs, ys, marker=markers[fn], color=colors[fn], label=fn)
            used.add(fn)
        ax.set_title(typ)
        ax.set_xlabel(xlabel)
        if logy:
            ax.set_yscale("log")
    axes[0].set_ylabel("ns / conversion")

    handles = [plt.Line2D([], [], color=colors[fn], marker=markers[fn], label=fn)
               for fn in sorted(used)]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5),
               fontsize=8, ncol=1)
    suptitle = f"{MODE_TITLES.get(mode, mode)}  —  {wc_name}"
    if series and mode != "admixture":
        suptitle += f"  [{series}]"
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=(0, 0, 0.86, 1))

    tag = series.replace("x", "x").replace("-", "_") if series else ""
    name = f"{mode}_{wc_name.split('-')[0]}" + (f"_{tag}" if tag else "") + suffix
    path = os.path.join(outdir, name + ".png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def plot_bars(rows, mode, wc_name, types, series, colors, outdir, suffix=""):
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
    name = f"{mode}_{wc_name.split('-')[0]}_{series.replace('-', '_')}" + suffix
    path = os.path.join(outdir, name + ".png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--exclude", default="sprintf,zmij_x64_v4,zmij_x64_native",
                    help="comma-separated functions to drop everywhere; ignored "
                         "when --only is given (default: sprintf plus the v4/"
                         "native zmij tiers, which crowd the full plots)")
    ap.add_argument("--only", default="", help="comma-separated functions to keep")
    ap.add_argument("--suffix", default="",
                    help="appended to every output filename (e.g. _zmij for "
                         "an --only zmij subset written next to the full set)")
    ap.add_argument("--exclude-small", default="naive",
                    help="functions to drop from 32/64-bit plots only "
                         "(kept on 128-bit); default: naive")
    args = ap.parse_args()

    rows = load(args.csv)
    excl = {s for s in args.exclude.split(",") if s}
    only = {s for s in args.only.split(",") if s}
    # --only is authoritative: an explicit keep-list re-includes functions the
    # (default) exclude list would drop, e.g. the v4/native tiers in the
    # _zmij subset.
    if only:
        rows = [r for r in rows if r["Function"] in only]
    elif excl:
        rows = [r for r in rows if r["Function"] not in excl]

    outdir = args.outdir or os.path.join(os.path.dirname(os.path.abspath(args.csv)), "plots")
    os.makedirs(outdir, exist_ok=True)

    excl_small = {s for s in args.exclude_small.split(",") if s}

    clean_style()
    colors = color_map({r["Function"] for r in rows})
    markers = marker_map({r["Function"] for r in rows})
    modes = sorted({r["Mode"] for r in rows})

    for mode in modes:
        for wc_name, types in WIDTH_CLASSES:
            # Drop the slow-and-scale-dominating functions from the small widths
            # (e.g. naive), but keep them on 128-bit where they are the baseline.
            wc_rows = rows
            if excl_small and not wc_name.startswith("128"):
                wc_rows = [r for r in rows if r["Function"] not in excl_small]
            for series in series_for(wc_rows, mode, types):
                if mode in ("loguniform", "uniform"):
                    plot_bars(wc_rows, mode, wc_name, types, series, colors, outdir, args.suffix)
                else:
                    # bylength/unpredictable are linear at 32/64-bit. The full
                    # 128-bit set keeps log, where the naive baseline dominates
                    # the range. The zmij-only subset (--suffix) drops fmt/naive,
                    # so its range is tight -- linear shows the additive branch
                    # bump far better than log.
                    is_128 = wc_name.startswith("128")
                    logy = mode in ("bylength", "unpredictable") and is_128 and not args.suffix
                    plot_line(wc_rows, mode, wc_name, types, series, colors, markers, outdir, logy, args.suffix)

    print("\nDone. PNGs in", outdir)


if __name__ == "__main__":
    main()
