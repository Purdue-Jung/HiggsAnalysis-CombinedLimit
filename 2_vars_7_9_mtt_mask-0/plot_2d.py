#!/usr/bin/env python3
import uproot
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import os
import argparse

# ===========================================
# Wilson Coefficients
# ===========================================
WC_LIST = [
    "ctGRe","ctGIm","cQj18","cQj38","cQj11","cQj31",
    "ctu8","ctd8","ctj8","cQu8","cQd8","ctu1",
    "ctd1","ctj1","cQu1","cQd1"
]


# ===========================================
# Load scan file (supports swapped POI order)
# ===========================================
def load_scan_file(p0, p1, tag):
    fname1 = f"higgsCombineScan2D_{tag}.{p0}.{p1}.MultiDimFit.mH120.root"
    fname2 = f"higgsCombineScan2D_{tag}.{p1}.{p0}.MultiDimFit.mH120.root"

    if os.path.isfile(fname1):
        return fname1
    if os.path.isfile(fname2):
        return fname2
    return None


# ===========================================
# Load Combine output into a DataFrame
# ===========================================
def load_scan_dataframe(filename, p0, p1):
    f = uproot.open(filename)["limit"]
    arr = f.arrays(library="np")

    branches = arr.keys()
    if (p0 not in branches) or (p1 not in branches):
        raise RuntimeError(
            f"Required branches missing for {p0}, {p1}. "
            f"Available = {branches}"
        )

    df = pd.DataFrame({
        p0: arr[p0],
        p1: arr[p1],
        "deltaNLL": arr["deltaNLL"],
    })

    # Sort & clean
    df = df.drop_duplicates(subset=[p0, p1]).sort_values([p0, p1]).reset_index(drop=True).reset_index(drop=True)

    # CRITICAL: re-zero likelihood
    df["twoNLL"] = 2 * (df["deltaNLL"] - df["deltaNLL"].min())

    # after df is built
    print("nx, ny:", df[p0].nunique(), df[p1].nunique())
    print("min/max", p0, df[p0].min(), df[p0].max(), "|", p1, df[p1].min(), df[p1].max())
    
    # check if 0 row/col exist
    print("has", p0, "=0 ?", np.any(np.isclose(df[p0].values, 0.0)))
    print("has", p1, "=0 ?", np.any(np.isclose(df[p1].values, 0.0)))
    
    # check missing cells after pivot
    Zdf = df.pivot(index=p1, columns=p0, values="twoNLL")
    print("pivot shape:", Zdf.shape, "NaN frac:", np.isnan(Zdf.values).mean())

    return df


# ===========================================
# Plot heatmap + contours
# ===========================================
def plot_2d(df, p0, p1, outdir, tag):
    os.makedirs(outdir, exist_ok=True)

    # Extract unique grid
    x = np.sort(df[p0].unique())
    y = np.sort(df[p1].unique())

    nx = len(x)
    ny = len(y)

    # Defensive checks
    if nx < 2 or ny < 2:
        print(f"[SKIP] Not enough grid points for ({p0}, {p1}) → nx={nx}, ny={ny}")
        return

    Z = df.pivot(index=p1, columns=p0, values="twoNLL").values

    if Z.size == 0:
        print(f"[SKIP] Empty Z for ({p0}, {p1})")
        return

    if np.all(~np.isfinite(Z)):
        print(f"[SKIP] Z all NaN for ({p0}, {p1})")
        return

    # Contour levels (2D Wilks)
    levels = [2.30, 5.99]   # 68%, 95%

    plt.figure(figsize=(7,6))

    # Heatmap
    plt.contourf(x, y, Z, levels=50, cmap="viridis")
    cbar = plt.colorbar()
    cbar.set_label(r"$\Delta(2\,\mathrm{NLL})$")

    # Contours
    cs = plt.contour(x, y, Z, levels=levels,
                     colors=["white", "red"], linewidths=2)
    fmt = {levels[0]: "68%", levels[1]: "95%"}
    plt.clabel(cs, fmt=fmt, inline=True, fontsize=10)

    # Labels
    plt.xlabel(p0)
    plt.ylabel(p1)
    plt.title(f"2D Likelihood Scan ({tag}): {p0} vs {p1}")

    plt.tight_layout()

    outname = f"{outdir}/Scan2D_{tag}_{p0}_{p1}.png"
    plt.savefig(outname, dpi=150)
    plt.close()

    print(f"[OK] Saved {outname}")


# ===========================================
# MAIN
# ===========================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Plot 2D SMEFT likelihood scans")
    ap.add_argument(
        "--mode",
        choices=["asimov", "data"],
        required=True,
        help="Which scans to plot"
    )
    ap.add_argument(
        "--outdir",
        default="scan_plots_2d",
        help="Base output directory"
    )
    args = ap.parse_args()

    tag = "Asimov" if args.mode == "asimov" else "Data"
    outdir = os.path.join(args.outdir, tag.lower())
    os.makedirs(outdir, exist_ok=True)

    print(f"\n===== 2D SMEFT Likelihood Scans ({tag}) =====\n")

    for p0, p1 in itertools.combinations(WC_LIST, 2):
        fname = load_scan_file(p0, p1, tag)
        if fname is None:
            print(f"[SKIP] Missing scan for ({p0}, {p1})")
            continue

        print(f"[RUN] Loading scan for ({p0}, {p1}) → {fname}")

        df = load_scan_dataframe(fname, p0, p1)
        plot_2d(df, p0, p1, outdir, tag)

    print(f"\nAll 2D plots saved in: {outdir}/\n")
