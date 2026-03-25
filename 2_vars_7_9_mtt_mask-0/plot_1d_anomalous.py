#!/usr/bin/env python3

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uproot
import mplhep as hep

from matplotlib.patches import Patch
from uncertainties import ufloat
from uncertainties import unumpy as unp


# ============================================================
# CMS style
# ============================================================
hep.style.use("CMS")
plt.rcParams.update({
    "font.size": 15,
    "axes.labelsize": 15,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
})


# ============================================================
# Wilson coefficients
# ============================================================
WC_LIST = [
    "ctGRe", "ctGIm", "cQj18", "cQj38", "cQj11", "cQj31",
    "ctu8", "ctd8", "ctj8", "cQu8", "cQd8", "ctu1",
    "ctd1", "ctj1", "cQu1", "cQd1"
]

# Optional display scaling in summary plot
WC_SCALE = {
    "ctGRe": 5.0,
}


# ============================================================
# Anomalous definitions
# ============================================================
ANOMALOUS_ORDER = [
    "cVV", "cVA", "cAV", "cAA",
    "c1", "c2", "c3", "c1_minus_c2_plus_c3",
    "mu_t", "d_t"
]

ANOMALOUS_LABELS = {
    "cVV": r"$c_{VV}$",
    "cVA": r"$c_{VA}$",
    "cAV": r"$c_{AV}$",
    "cAA": r"$c_{AA}$",
    "c1": r"$c_{1}$",
    "c2": r"$c_{2}$",
    "c3": r"$c_{3}$",
    "c1_minus_c2_plus_c3": r"$c_{1}-c_{2}+c_{3}$",
    "mu_t": r"$\mu_t$",
    "d_t": r"$d_t$",
}

# Use TeV consistently
MT = 0.1725
GS = 1.166

CMS_PUB = {
    "mu_t":  (-0.005, 0.005),
    "d_t":   (-0.004, 0.008),
    "c_mm":  (-0.017, 0.012),
    "c_pp":  (-0.002, 0.003),
    "cVV":   ( 0.016, 0.013),
    "cVA":   (-0.009, 0.018),
    "cAV":   (-0.001, 0.017),
    "cAA":   ( 0.000, 0.020),
    "c1":    ( 0.13,  0.11),
    "c3":    (-0.07,  0.14),
    "c1_minus_c2_plus_c3": (-0.01, 0.08),
}


# ============================================================
# Utilities
# ============================================================
def find_crossings(x, y, level):
    pts = []
    for i in range(len(x) - 1):
        x1, x2 = x[i], x[i + 1]
        y1, y2 = y[i], y[i + 1]

        if not (np.isfinite(x1) and np.isfinite(x2) and np.isfinite(y1) and np.isfinite(y2)):
            continue
        if y1 == y2:
            continue

        if (y1 - level) * (y2 - level) <= 0:
            xc = x1 + (level - y1) * (x2 - x1) / (y2 - y1)
            pts.append(xc)

    return sorted(set(np.round(pts, 12)))


def load_scan(poi, tag, profiled=False):
    suf = "_profiled" if profiled else ""
    fname = f"higgsCombineScan1D_{tag}.{poi}{suf}.MultiDimFit.mH120.root"

    if not os.path.isfile(fname):
        return None

    try:
        arr = uproot.open(fname)["limit"].arrays(library="np")
    except Exception as exc:
        print(f"[WARN] Failed to read {fname}: {exc}")
        return None

    if poi not in arr:
        print(f"[WARN] Branch '{poi}' not found in {fname}")
        return None
    if "deltaNLL" not in arr:
        print(f"[WARN] Branch 'deltaNLL' not found in {fname}")
        return None

    status = arr["status"] if "status" in arr else np.zeros_like(arr["deltaNLL"])

    df = pd.DataFrame({
        poi: arr[poi],
        "deltaNLL": arr["deltaNLL"],
        "status": status,
    })

    df = df[
        np.isfinite(df[poi]) &
        np.isfinite(df["deltaNLL"]) &
        (df["status"] == 0)
    ].copy()

    if df.empty:
        return None

    df = df.drop_duplicates(subset=[poi]).sort_values(poi).reset_index(drop=True)
    df["twoNLL"] = 2.0 * (df["deltaNLL"] - df["deltaNLL"].min())
    return df


def extract_intervals(df, poi):
    x = df[poi].to_numpy()
    y = df["twoNLL"].to_numpy()

    c1 = find_crossings(x, y, 1.0)
    c2 = find_crossings(x, y, 4.0)

    if len(c1) < 2:
        return None

    out = {
        "lo1": c1[0],
        "hi1": c1[-1],
        "lo2": np.nan,
        "hi2": np.nan,
    }

    if len(c2) >= 2:
        out["lo2"] = c2[0]
        out["hi2"] = c2[-1]

    return out


def choose_plot_range(df, poi):
    x = df[poi].to_numpy()
    y = df["twoNLL"].to_numpy()

    xs = []
    xs.extend(find_crossings(x, y, 1.0))
    xs.extend(find_crossings(x, y, 4.0))
    xs = [v for v in xs if np.isfinite(v)]

    if len(xs) >= 2:
        xmin = min(xs)
        xmax = max(xs)
        width = xmax - xmin
        if width <= 0:
            width = max(np.ptp(x), 1.0)
        margin = 0.25 * width
        return xmin - margin, xmax + margin

    q10, q90 = np.percentile(x, [10, 90])
    if np.isclose(q10, q90):
        xmin, xmax = np.min(x), np.max(x)
        if np.isclose(xmin, xmax):
            xmin -= 1.0
            xmax += 1.0
        return xmin, xmax

    margin = 0.25 * (q90 - q10)
    return q10 - margin, q90 + margin


def choose_fit_result(result_dict):
    if result_dict.get("profiled") is not None:
        return result_dict["profiled"], "profiled"
    if result_dict.get("conditional") is not None:
        return result_dict["conditional"], "conditional"
    return None, None


def require_both_or_none(xmin, xmax, name):
    if (xmin is None) ^ (xmax is None):
        raise ValueError(f"Please provide both --{name}-xmin and --{name}-xmax together.")


# ============================================================
# Plotting
# ============================================================
def plot_scan(df, poi, tag, profiled, outdir, xmin=None, xmax=None):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    intervals = extract_intervals(df, poi)
    if intervals is not None:
        if np.isfinite(intervals["lo2"]) and np.isfinite(intervals["hi2"]):
            ax.axvspan(intervals["lo2"], intervals["hi2"], color="#FFD700", alpha=0.55, zorder=0)
        ax.axvspan(intervals["lo1"], intervals["hi1"], color="#00C000", alpha=0.45, zorder=1)

    ax.plot(df[poi], df["twoNLL"], "-o", color="black", lw=2, ms=3, zorder=2)
    ax.axhline(1.0, ls="--", color="black", lw=1)
    ax.axhline(4.0, ls="--", color="black", lw=1)

    if xmin is None and xmax is None:
        xmin_use, xmax_use = choose_plot_range(df, poi)
    else:
        xmin_use, xmax_use = xmin, xmax

    ax.set_ylim(0, 5)
    ax.set_xlim(xmin_use, xmax_use)
    ax.set_xlabel(poi)
    ax.set_ylabel(r"$-2\Delta\log(\mathcal{L})$")

    handles = [Patch(color="#00C000", alpha=0.45, label="68% CL")]
    if intervals is not None and np.isfinite(intervals["lo2"]) and np.isfinite(intervals["hi2"]):
        handles.append(Patch(color="#FFD700", alpha=0.55, label="95% CL"))
    ax.legend(handles=handles, frameon=False, loc="upper right")

    pretty_tag = "Asimov" if tag == "asimov" else "Data"
    subtitle = "profiled" if profiled else "conditional"
    ax.set_title(f"{poi} ({pretty_tag}, {subtitle})")

    hep.cms.label("Work in Progress", lumi=138, data=(tag == "data"), ax=ax)

    os.makedirs(outdir, exist_ok=True)
    name = f"Scan1D_{tag}_{poi}{'_profiled' if profiled else ''}.png"
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, name), dpi=160)
    plt.close()


def make_summary_plot(results, tag, outname, xmin=None, xmax=None):
    fig, ax = plt.subplots(figsize=(7.8, 10.8))
    yvals = np.arange(len(WC_LIST))[::-1]

    for i, wc in enumerate(WC_LIST):
        y = yvals[i]
        scale = WC_SCALE.get(wc, 1.0)

        p = results[wc].get("profiled")
        c = results[wc].get("conditional")

        if p is not None:
            if np.isfinite(p["lo2"]) and np.isfinite(p["hi2"]):
                ax.plot([scale * p["lo2"], scale * p["hi2"]], [y, y], color="black", lw=1)
            ax.plot([scale * p["lo1"], scale * p["hi1"]], [y, y], color="black", lw=3)

        if c is not None:
            if np.isfinite(c["lo2"]) and np.isfinite(c["hi2"]):
                ax.plot([scale * c["lo2"], scale * c["hi2"]], [y, y], color="red", lw=1, ls=":")
            ax.plot([scale * c["lo1"], scale * c["hi1"]], [y, y], color="red", lw=3, ls=":")

    ax.axvline(0.0, color="gray", lw=1)

    labels = []
    for wc in WC_LIST:
        if wc in WC_SCALE and WC_SCALE[wc] != 1.0:
            labels.append(f"{wc} × {WC_SCALE[wc]}")
        else:
            labels.append(wc)

    ax.set_yticks(yvals)
    ax.set_yticklabels(labels)
    ax.set_xlabel(r"Wilson coefficient / $\Lambda^2$ [TeV$^{-2}$]")

    if xmin is None and xmax is None:
        xs = []
        for wc in WC_LIST:
            for kind in ["profiled", "conditional"]:
                r = results[wc].get(kind)
                if r is None:
                    continue
                for key in ["lo1", "hi1", "lo2", "hi2"]:
                    if key in r and np.isfinite(r[key]):
                        xs.append(WC_SCALE.get(wc, 1.0) * r[key])

        if xs:
            xmin_auto = min(xs)
            xmax_auto = max(xs)
            width = xmax_auto - xmin_auto
            if width <= 0:
                width = 1.0
            margin = 0.15 * width
            ax.set_xlim(xmin_auto - margin, xmax_auto + margin)
    else:
        ax.set_xlim(xmin, xmax)

    handles = [
        plt.Line2D([0], [0], color="black", lw=3, label="Profiled 68%"),
        plt.Line2D([0], [0], color="red", lw=3, ls=":", label="Conditional 68%"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right")

    hep.cms.label("Work in Progress", lumi=138, data=(tag == "data"), ax=ax)

    plt.tight_layout()
    plt.savefig(outname, dpi=200)
    plt.close()


def plot_anomalous_summary(fit_results, cms_results, order, outname,
                           xlabel="Anomalous coupling", lumi="138 fb$^{-1}$ (13 TeV)",
                           xmin=None, xmax=None):
    fig, ax = plt.subplots(figsize=(8.0, 10.0))
    y = np.arange(len(order))
    dy = 0.15

    for i, name in enumerate(order):
        yi = y[i]

        r = fit_results.get(name)
        if r is not None:
            ax.plot([r["lo95"], r["hi95"]], [yi - dy, yi - dy], color="black", lw=1)
            ax.plot([r["lo68"], r["hi68"]], [yi - dy, yi - dy], color="black", lw=3)
            ax.plot(r["best"], yi - dy, "o", color="black", ms=4)

        cms = cms_results.get(name)
        if cms is not None:
            best, err = cms
            ax.plot([best - 2 * err, best + 2 * err], [yi + dy, yi + dy], color="red", lw=1, ls=":")
            ax.plot([best - err, best + err], [yi + dy, yi + dy], color="red", lw=3, ls=":")
            ax.plot(best, yi + dy, "s", color="red", ms=4)

    ax.axvline(0.0, color="gray", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([ANOMALOUS_LABELS.get(x, x) for x in order])
    ax.set_xlabel(xlabel)
    ax.invert_yaxis()

    if xmin is not None and xmax is not None:
        ax.set_xlim(xmin, xmax)

    handles = [
        plt.Line2D([0], [0], color="black", lw=3, label="This work (68%)"),
        plt.Line2D([0], [0], color="red", lw=3, ls=":", label="CMS published (68%)"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right")

    hep.cms.label("Work in Progress", lumi=138, data=False, ax=ax)

    plt.tight_layout()
    plt.savefig(outname, dpi=200)
    plt.close()


# ============================================================
# SMEFT -> anomalous translation
# ============================================================
def translate_anomalous_exact(wc_fit):
    mt, gs = MT, GS
    norm = mt**2 / gs**2

    required = [
        "ctGRe", "ctGIm", "cQj18", "cQj38",
        "ctu8", "ctd8", "ctj8", "cQu8", "cQd8"
    ]
    missing = [k for k in required if k not in wc_fit]
    if missing:
        raise RuntimeError(
            "Cannot translate to anomalous couplings because these fitted WCs are missing: "
            + ", ".join(missing)
        )

    out = {}

    out["cVV"] = (
        (wc_fit["ctj8"] + wc_fit["cQj18"]) / 2
        + (wc_fit["ctu8"] + wc_fit["ctd8"] + wc_fit["cQu8"] + wc_fit["cQd8"]) / 4
    ) * norm

    out["cAA"] = (
        -(wc_fit["ctj8"] + wc_fit["cQj18"]) / 2
        + (wc_fit["ctu8"] + wc_fit["ctd8"] - wc_fit["cQu8"] - wc_fit["cQd8"]) / 4
    ) * norm

    out["cVA"] = (
        (wc_fit["ctj8"] - wc_fit["cQj18"]) / 2
        + (wc_fit["ctu8"] + wc_fit["ctd8"] - wc_fit["cQu8"] - wc_fit["cQd8"]) / 4
    ) * norm

    out["cAV"] = (
        -(wc_fit["ctj8"] + wc_fit["cQj18"]) / 2
        + (wc_fit["ctu8"] + wc_fit["ctd8"] + wc_fit["cQu8"] + wc_fit["cQd8"]) / 4
    ) * norm

    out["c1"] = (
        (wc_fit["ctu8"] - wc_fit["ctd8"]) / 2
        + (wc_fit["cQu8"] - wc_fit["cQd8"]) / 2
        + wc_fit["cQj38"]
    ) * norm

    out["c2"] = (
        (wc_fit["ctu8"] - wc_fit["ctd8"]) / 2
        - (wc_fit["cQu8"] - wc_fit["cQd8"]) / 2
        + wc_fit["cQj38"]
    ) * norm

    out["c3"] = (
        (wc_fit["ctu8"] - wc_fit["ctd8"]) / 2
        - (wc_fit["cQu8"] - wc_fit["cQd8"]) / 2
        - wc_fit["cQj38"]
    ) * norm

    out["c1_minus_c2_plus_c3"] = (
        (wc_fit["ctu8"] - wc_fit["ctd8"]) / 2
        + (wc_fit["cQu8"] - wc_fit["cQd8"]) / 2
        - wc_fit["cQj38"]
    ) * norm

    out["mu_t"] = 2 * mt**2 * wc_fit["ctGRe"]
    out["d_t"] = 2 * mt**2 * wc_fit["ctGIm"]

    return out


def anomalous_to_intervals(anom):
    out = {}
    for k, v in anom.items():
        mu = unp.nominal_values(v)
        sig = unp.std_devs(v)
        out[k] = {
            "best": mu,
            "lo68": mu - sig,
            "hi68": mu + sig,
            "lo95": mu - 2.0 * sig,
            "hi95": mu + 2.0 * sig,
        }
    return out


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="Plot 1D SMEFT scans and translate to anomalous couplings")
    ap.add_argument("--mode", choices=["asimov", "data"], required=True,
                    help="Choose input scan set")
    ap.add_argument("--outdir", default="scan_plots_1d_anomalous",
                    help="Output directory")

    ap.add_argument("--scan-xmin", type=float, default=None,
                    help="Manual xmin for all individual 1D scan plots")
    ap.add_argument("--scan-xmax", type=float, default=None,
                    help="Manual xmax for all individual 1D scan plots")

    ap.add_argument("--summary-xmin", type=float, default=None,
                    help="Manual xmin for the SMEFT summary plot")
    ap.add_argument("--summary-xmax", type=float, default=None,
                    help="Manual xmax for the SMEFT summary plot")

    ap.add_argument("--anom-xmin", type=float, default=None,
                    help="Manual xmin for the anomalous summary plot")
    ap.add_argument("--anom-xmax", type=float, default=None,
                    help="Manual xmax for the anomalous summary plot")

    args = ap.parse_args()

    require_both_or_none(args.scan_xmin, args.scan_xmax, "scan")
    require_both_or_none(args.summary_xmin, args.summary_xmax, "summary")
    require_both_or_none(args.anom_xmin, args.anom_xmax, "anom")

    tag = args.mode
    pretty_tag = "Asimov" if tag == "asimov" else "Data"

    os.makedirs(args.outdir, exist_ok=True)
    results = {wc: {} for wc in WC_LIST}

    for wc in WC_LIST:
        for prof in [False, True]:
            df = load_scan(wc, tag, profiled=prof)
            if df is None:
                continue

            subdir = os.path.join(args.outdir, f"{tag}_{'profiled' if prof else 'conditional'}")
            plot_scan(
                df, wc, tag, prof, subdir,
                xmin=args.scan_xmin, xmax=args.scan_xmax
            )

            ints = extract_intervals(df, wc)
            if ints is not None:
                results[wc]["profiled" if prof else "conditional"] = ints

    print(f"\n================ SMEFTsim Wilson coefficients ({pretty_tag}) =================")
    print("   WC        best-fit        68% CL            source")
    print("--------------------------------------------------------------------------")

    any_fit = False
    wc_fit = {}
    fit_sources = {}

    for wc in WC_LIST:
        r, label = choose_fit_result(results[wc])
        if r is None:
            print(f"{wc:8s}   not fitted")
            continue

        best = 0.5 * (r["lo1"] + r["hi1"])
        err = 0.5 * (r["hi1"] - r["lo1"])

        print(f"{wc:8s}   {best:+12.4e}   ± {err:10.4e}   [{label}]")

        wc_fit[wc] = ufloat(best, err)
        fit_sources[wc] = label
        any_fit = True

    make_summary_plot(
        results,
        tag,
        os.path.join(args.outdir, f"SMEFT_summary_{tag}.png"),
        xmin=args.summary_xmin,
        xmax=args.summary_xmax,
    )

    if not any_fit:
        print("\n[WARN] No valid SMEFT intervals found. Stop before anomalous translation.")
        return

    if any(src == "conditional" for src in fit_sources.values()):
        print("\n[WARN] Some anomalous inputs are using conditional SMEFT intervals because profiled scans were not found.")

    try:
        anom = anomalous_to_intervals(translate_anomalous_exact(wc_fit))
    except RuntimeError as exc:
        print(f"\n[WARN] {exc}")
        print("[WARN] Skipping anomalous summary plot.")
        return

    print(f"\n================ Anomalous couplings ({pretty_tag}) =================")
    print("   coupling                best-fit        68% CL")
    print("---------------------------------------------------------------")

    for name in ANOMALOUS_ORDER:
        if name not in anom:
            print(f"{name:22s}   not defined")
            continue

        r = anom[name]
        best = r["best"]
        err = 0.5 * (r["hi68"] - r["lo68"])
        print(f"{name:22s}   {best:+12.4e}   ± {err:10.4e}")

    plot_anomalous_summary(
        fit_results=anom,
        cms_results=CMS_PUB,
        order=ANOMALOUS_ORDER,
        outname=os.path.join(args.outdir, f"Anomalous_summary_{tag}.png"),
        xlabel="Anomalous coupling",
        lumi="138 fb$^{-1}$ (13 TeV)",
        xmin=args.anom_xmin,
        xmax=args.anom_xmax,
    )

    print(f"\n[OK] Outputs written under: {args.outdir}")


if __name__ == "__main__":
    main()