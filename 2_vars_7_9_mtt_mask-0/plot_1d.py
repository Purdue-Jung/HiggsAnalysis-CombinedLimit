#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uproot

WC_LIST = [
    "ctGRe","ctGIm","cQj18","cQj38","cQj11","cQj31",
    "ctu8","ctd8","ctj8","cQu8","cQd8","ctu1",
    "ctd1","ctj1","cQu1","cQd1"
]

def find_crossings(x, y, level):
    """
    Find approximate x positions where y crosses a given level.
    Uses linear interpolation between scan points.
    Returns a sorted list of crossing x values.
    """
    crossings = []
    for i in range(len(x) - 1):
        y1, y2 = y[i], y[i + 1]
        if (y1 - level) * (y2 - level) <= 0 and y1 != y2:
            # Linear interpolation
            x1, x2 = x[i], x[i + 1]
            xc = x1 + (level - y1) * (x2 - x1) / (y2 - y1)
            crossings.append(xc)
    return sorted(crossings)

def load_scan(poi, tag):
    """Load 1D likelihood scan for a single POI."""
    fname = f"higgsCombineScan1D_{tag}.{poi}.MultiDimFit.mH120.root"

    if not os.path.isfile(fname):
        print(f"[WARN] Missing scan file for {poi}: {fname}")
        return None

    tree = uproot.open(fname)["limit"]
    arr = tree.arrays(library="np")

    if poi not in arr:
        print(f"[WARN] POI {poi} not found in branches = {list(arr.keys())}")
        return None

    if "deltaNLL" not in arr:
        print(f"[WARN] deltaNLL not found in {fname}")
        return None

    df = pd.DataFrame({
        poi: arr[poi],
        "deltaNLL": arr["deltaNLL"],
    })

    # Sort & clean
    df = df.drop_duplicates().sort_values(poi).reset_index(drop=True)

    # IMPORTANT: re-zero likelihood
    df["twoNLL"] = 2 * (df["deltaNLL"] - df["deltaNLL"].min())

    return df


def plot_scan(df, poi, tag, outdir):
    """Make a 1D likelihood scan plot with automatic x-range."""
    x = df[poi].values
    y = df["twoNLL"].values

    # Find crossings
    x_1s = find_crossings(x, y, 1.00)
    x_2s = find_crossings(x, y, 4.0)

    # Determine x-range
    xs = []
    if len(x_1s) >= 2:
        xs.extend(x_1s[:2])
    if len(x_2s) >= 2:
        xs.extend(x_2s[:2])

    if len(xs) >= 2:
        xmin, xmax = min(xs), max(xs)
        margin = 0.25 * (xmax - xmin)
        xmin -= margin
        xmax += margin
    else:
        # Fallback: use central region of scan
        xmin, xmax = np.percentile(x, [10, 90])

    plt.figure(figsize=(6, 4))
    plt.plot(x, y, "-o", ms=3, lw=2)

    plt.ylim(0, 5)
    plt.xlim(xmin, xmax)
    plt.xlabel(poi)
    plt.ylabel(r"$-2\Delta\log(\mathcal{L})$")
    plt.title(f"{poi} ({tag})")

    plt.grid(True, alpha=0.4)

    # Wilks thresholds (1D)
    plt.axhline(1.00, color="red", ls="--", label=r"$1\sigma$")
    plt.axhline(4.0, color="green", ls="--", label="95% CL")
    plt.legend()

    out = os.path.join(outdir, f"Scan1D_{tag}_{poi}.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Plot 1D SMEFT likelihood scans")
    ap.add_argument(
        "--mode",
        choices=["asimov", "data"],
        required=True,
        help="Which scans to plot"
    )
    ap.add_argument(
        "--outdir",
        default="scan_plots_1d",
        help="Output directory for plots"
    )
    args = ap.parse_args()

    tag = "asimov" if args.mode == "asimov" else "data"
    outdir = os.path.join(args.outdir, tag.lower())
    os.makedirs(outdir, exist_ok=True)

    print(f"\n===== 1D SMEFT Likelihood Scans ({tag}) =====\n")

    for poi in WC_LIST:
        df = load_scan(poi, tag)
        if df is None:
            print(f"[SKIP] {poi}: missing or invalid scan")
            continue

        if df.empty:
            print(f"[SKIP] {poi}: empty scan")
            continue

        if df["twoNLL"].isna().all():
            print(f"[SKIP] {poi}: all Δ(2NLL) values NaN")
            continue

        if df["twoNLL"].nunique() == 1:
            print(f"[WARN] {poi}: flat likelihood → no sensitivity")
            continue

        print(f"[OK]  Loaded {poi} ({tag})")
        plot_scan(df, poi, tag, outdir)

    print(f"\nAll plots saved to: {outdir}/\n")


if __name__ == "__main__":
    main()
