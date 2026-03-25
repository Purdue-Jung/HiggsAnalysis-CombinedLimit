#!/usr/bin/env python3
import os
import argparse
import yaml
import numpy as np
import uproot
import matplotlib.pyplot as plt
import mplhep as hep

from uncertainties import unumpy as unp

hep.style.use("CMS")

# ------------------------------------------------------------
# COLORS / STACK ORDER
# ------------------------------------------------------------
PROCESS_COLORS = {
    "ttbar_smeft":          "#cc0000",
    "top":                  "#ff00ff",
    "Z+jets":               "#3366ff",
    "toponium_dileptonic":  "#ffa500",
    "ttbar_smeft_other":    "#ffcc33",
    "ttbar+ZorW":           "#ff6666",
    "diboson":              "#ffffff",
    "W+jets":               "#33cc33",
    "data_obs":             "black",
}

DEFAULT_STACK_ORDER = [
    "W+jets",
    "diboson",
    "Z+jets",
    "top",
    "toponium_dileptonic",
    "ttbar+ZorW",
    "ttbar_smeft_other",
    "ttbar_smeft",
]

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def list_channels_processes(f):
    out = {}
    for k in f.keys():
        k = k.split(";")[0]
        if "/" not in k:
            continue
        ch, proc = k.split("/", 1)
        out.setdefault(ch, set()).add(proc)
    return {k: sorted(v) for k, v in out.items()}


def read_th1_uarray(f, ch, proc):
    h = f[f"{ch}/{proc}"]

    vals = h.values(flow=False).astype(float)
    edges = h.axis().edges().astype(float)

    vars_ = h.variances(flow=False)
    if vars_ is None:
        errs = np.sqrt(np.maximum(vals, 0.0))
    else:
        errs = np.sqrt(np.maximum(vars_.astype(float), 0.0))

    return edges, unp.uarray(vals, errs)


def infer_shape(nflat, dims):
    prod = 1
    for n in dims:
        prod *= n
    return tuple(dims) if prod == nflat else None


def unflatten(arr, shape):
    return np.asarray(arr).reshape(shape, order="C")


def project_axis(uarr_nd, axis_keep):
    """Sum over other axes with correct uncertainty propagation."""
    axes = tuple(i for i in range(uarr_nd.ndim) if i != axis_keep)

    nom = unp.nominal_values(uarr_nd)
    std = unp.std_devs(uarr_nd)

    nom_1d = np.sum(nom, axis=axes)
    std_1d = np.sqrt(np.sum(std**2, axis=axes))

    return unp.uarray(nom_1d, std_1d)

# ------------------------------------------------------------
# PLOTTING
# ------------------------------------------------------------
def make_stack_ratio_plot(outpath, xedges, data_u, mc_u_hists, mc_order, xlabel):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    widths = np.diff(xedges)
    centers = 0.5 * (xedges[:-1] + xedges[1:])

    # Convert to density ONLY here
    data_d = data_u / widths
    mc_d = {p: u / widths for p, u in mc_u_hists.items()}

    fig, (ax, rax) = plt.subplots(
        2, 1, figsize=(8, 8),
        gridspec_kw={"height_ratios": (3, 1), "hspace": 0},
        sharex=True,
    )

    fig.subplots_adjust(left=0.14)
    hep.cms.label("Work In Progress", data=True, ax=ax)

    # ---- STACK ----
    stack_vals, stack_labels, stack_colors = [], [], []
    for p in mc_order:
        if p not in mc_d:
            continue
        stack_vals.append(unp.nominal_values(mc_d[p]))
        stack_labels.append(p)
        stack_colors.append(PROCESS_COLORS.get(p, "#999999"))

    hep.histplot(
        stack_vals,
        xedges,
        stack=True,
        histtype="fill",
        color=stack_colors,
        label=stack_labels,
        edgecolor="black",
        linewidth=1.0,
        ax=ax,
    )

    # ---- MC UNCERTAINTY BAND (bin-aligned, hatched) ----
    mc_tot = sum(mc_d.values())
    y_lo = unp.nominal_values(mc_tot - unp.std_devs(mc_tot))
    y_hi = unp.nominal_values(mc_tot + unp.std_devs(mc_tot))

    ax.fill_between(
        xedges[:-1],
        y_lo,
        y_hi,
        step="post",
        facecolor="none",
        edgecolor="black",
        hatch="///",
        linewidth=0.0,
        label="Uncertainty",
    )

    # ---- DATA ----
    ax.errorbar(
        centers,
        unp.nominal_values(data_d),
        yerr=unp.std_devs(data_d),
        fmt="o",
        color="black",
        capsize=2,
        label="Data",
    )

    ax.set_ylabel("Events / bin width", labelpad=12)
    ax.legend(loc="upper left", ncol=2, frameon=False, fontsize=10)

    # ---- RATIO ----
    ratio = data_d / mc_tot
    rax.axhline(1.0, color="black", linewidth=1)

    rax.errorbar(
        centers,
        unp.nominal_values(ratio),
        yerr=unp.std_devs(ratio),
        fmt="o",
        color="black",
        capsize=2,
    )

    rax.set_ylabel("Data / Pred.")
    rax.set_xlabel(xlabel)
    rax.set_ylim(0.9, 1.09)
    rax.grid(axis="y", linestyle=":", alpha=0.5)

    plt.tight_layout()

    fig.savefig(outpath)
    plt.close(fig)
    print(f"[OK] wrote {outpath}")

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("templates_root")
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--outdir", default="closure_plots_physical")
    ap.add_argument("--channels", default="")
    args = ap.parse_args()

    with open(args.yaml) as f:
        y = yaml.safe_load(f)

    var_names = list(y["variables"].keys())
    var_edges = {k: np.asarray(y["variables"][k][0]) for k in var_names}
    var_nbins = {k: len(var_edges[k]) - 1 for k in var_names}

    f = uproot.open(args.templates_root)
    ch_map = list_channels_processes(f)

    channels = args.channels.split(",") if args.channels else sorted(ch_map)

    for ch in channels:
        procs = ch_map[ch]

        hists = {}
        for p in procs:
            _, uarr = read_th1_uarray(f, ch, p)
            hists[p] = uarr

        nb_flat = len(next(iter(hists.values())))

        shape2 = None
        if len(var_names) >= 2:
            shape2 = infer_shape(
                nb_flat, [var_nbins[var_names[0]], var_nbins[var_names[1]]]
            )

        shape1 = infer_shape(nb_flat, [var_nbins[var_names[0]]])

        if shape2 is not None:
            tasks = [
                (var_names[0], 0, shape2),
                (var_names[1], 1, shape2),
            ]
        elif shape1 is not None:
            tasks = [(var_names[0], 0, shape1)]
        else:
            raise RuntimeError("Cannot interpret flattened template shape")

        for vname, axis_keep, shape in tasks:
            mc_u = {}
            data_u = None

            for p, u in hists.items():
                u_nd = unflatten(u, shape)
                u_1d = project_axis(u_nd, axis_keep)

                if p == "data_obs":
                    data_u = u_1d
                else:
                    mc_u[p] = u_1d

            if data_u is None:
                data_u = sum(mc_u.values())

            out = os.path.join(args.outdir, ch, f"{ch}_{vname}.png")
            make_stack_ratio_plot(
                out,
                var_edges[vname],
                data_u,
                mc_u,
                DEFAULT_STACK_ORDER,
                vname,
            )

if __name__ == "__main__":
    main()
