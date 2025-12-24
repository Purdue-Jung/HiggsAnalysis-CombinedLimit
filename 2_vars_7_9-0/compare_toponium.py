#!/usr/bin/env python3
import os
import json
import argparse
import yaml
import numpy as np
import uproot
import matplotlib.pyplot as plt
import mplhep as hep

hep.style.use("CMS")

# ------------------------------------------------------------
# Styling
# ------------------------------------------------------------
PROCESS_COLORS = {
    "ttbar_smeft":          "#cc0000",
    "top":                  "#ff00ff",
    "Z+jets":               "#3366ff",
    "toponium_dileptonic":  "#ffa500",
    "ttbar_smeft_other":    "#ffcc33",
    "ttbar+ZorW":           "#ff6666",
    "diboson":              "#dddddd",
    "W+jets":               "#33cc33",
    "data_obs":             "black",
}

STACK_WITH_TOPONIUM = [
    "W+jets",
    "diboson",
    "Z+jets",
    "top",
    "toponium_dileptonic",
    "ttbar+ZorW",
    "ttbar_smeft_other",
    "ttbar_smeft",
]

STACK_NO_TOPONIUM = [
    p for p in STACK_WITH_TOPONIUM if p != "toponium_dileptonic"
]

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def list_channels_processes(f):
    out = {}
    for k in f.keys():
        name = k.split(";")[0]
        if "/" not in name:
            continue
        ch, proc = name.split("/", 1)
        out.setdefault(ch, []).append(proc)
    return {k: sorted(v) for k, v in out.items()}

def read_th1(f, ch, proc):
    h = f[f"{ch}/{proc}"]
    vals = h.values(flow=False).astype(float)
    edges = h.axis().edges().astype(float)
    vars_ = h.variances(flow=False)
    errs = None if vars_ is None else np.sqrt(np.maximum(vars_, 0.0))
    return edges, vals, errs

def identify_variable(nb_flat, var_nbins):
    matched = [v for v, n in var_nbins.items() if n == nb_flat]
    if len(matched) != 1:
        raise RuntimeError(
            f"Cannot uniquely identify observable: nbins={nb_flat}, matches={matched}"
        )
    return matched[0]

def stack_mc(hists, order):
    tot = None
    var = None
    for p in order:
        if p not in hists:
            continue
        v, e = hists[p]
        if tot is None:
            tot = v.copy()
            var = (e**2 if e is not None else v.copy())
        else:
            tot += v
            var += (e**2 if e is not None else v)
    return tot, np.sqrt(np.maximum(var, 0.0))

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
def plot_toponium_comparison(outpath, xedges, data, data_err,
                             mc_with, mc_with_err,
                             mc_no, mc_no_err,
                             xlabel):

    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    fig, (ax, rax) = plt.subplots(
        2, 1, figsize=(8, 8),
        gridspec_kw={"height_ratios": (3, 1), "hspace": 0},
        sharex=True
    )

    hep.cms.label("Work In Progress", data=True, ax=ax)

    centers = 0.5 * (xedges[:-1] + xedges[1:])

    # Upper panel: with toponium
    ax.step(xedges[:-1], mc_with, where="post", label="MC (with toponium)", color="C0")
    ax.fill_between(
        centers, mc_with - mc_with_err, mc_with + mc_with_err,
        step="mid", alpha=0.3, color="C0"
    )

    ax.errorbar(
        centers, data, yerr=data_err,
        fmt="o", color="black", label="Data"
    )

    ax.set_ylabel("Events / bin")
    ax.legend()

    # Lower panel: ratio
    denom = np.where(mc_with == 0, 1.0, mc_with)
    ratio = mc_no / denom

    rax.axhline(1.0, color="black")
    rax.step(xedges[:-1], ratio, where="post", color="C3")
    rax.set_ylabel("No topo / With topo")
    rax.set_xlabel(xlabel)
    rax.set_ylim(0.8, 1.2)

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

    print(f"[OK] wrote {outpath}")

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("templates_root")
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--outdir", default="toponium_comparison")
    args = ap.parse_args()

    with open(args.yaml) as f:
        y = yaml.safe_load(f)

    var_edges = {v: np.asarray(y["variables"][v][0]) for v in y["variables"]}
    var_nbins = {v: len(var_edges[v]) - 1 for v in var_edges}

    froot = uproot.open(args.templates_root)
    ch_map = list_channels_processes(froot)

    for ch, procs in ch_map.items():
        # read one process to infer variable
        edges, vals, _ = read_th1(froot, ch, procs[0])
        vname = identify_variable(vals.size, var_nbins)
        xedges = var_edges[vname]

        mc_hists = {}
        data = data_err = None

        for p in procs:
            _, v, e = read_th1(froot, ch, p)
            if p == "data_obs":
                data = v
                data_err = e if e is not None else np.sqrt(np.maximum(v, 0.0))
            else:
                mc_hists[p] = (v, e)

        mc_with, mc_with_err = stack_mc(mc_hists, STACK_WITH_TOPONIUM)
        mc_no,   mc_no_err   = stack_mc(mc_hists, STACK_NO_TOPONIUM)

        out = os.path.join(
            args.outdir, ch, f"{ch}_{vname}_toponium_comparison.png"
        )

        plot_toponium_comparison(
            out, xedges,
            data, data_err,
            mc_with, mc_with_err,
            mc_no, mc_no_err,
            xlabel=vname
        )

if __name__ == "__main__":
    main()
