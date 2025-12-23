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

# Match your framework colors (same idea as in your yields_and_plots_v2.py)
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

# Processes to stack (data is drawn separately)
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

def list_channels_processes(f):
    ch_map = {}
    for k in f.keys():
        name = k.split(";")[0]
        if "/" not in name:
            continue
        ch, proc = name.split("/", 1)
        ch_map.setdefault(ch, set()).add(proc)
    return {ch: sorted(list(procs)) for ch, procs in ch_map.items()}

def read_th1(f, ch, proc):
    h = f[f"{ch}/{proc}"]

    # in-range bins (for plotting / shapes)
    vals = h.values(flow=False).astype(float)
    edges = h.axis().edges().astype(float)

    vars_ = h.variances(flow=False)
    if vars_ is None:
        errs = None
    else:
        errs = np.sqrt(np.maximum(vars_.astype(float), 0.0))

    # flow bins (for yield bookkeeping)
    vals_flow = h.values(flow=True).astype(float)
    uf = vals_flow[0]
    of = vals_flow[-1]

    if h.variances(flow=True) is not None:
        vars_flow = h.variances(flow=True).astype(float)
        vuf = vars_flow[0]
        vof = vars_flow[-1]
    else:
        vuf = vof = None

    return edges, vals, errs, uf, of, vuf, vof

def infer_shape(nbins_flat, axes_nbins):
    """
    axes_nbins: list like [n_ll, n_mtt] (or [n_ll] for 1D).
    Return tuple shape if consistent, else None.
    """
    prod = 1
    for n in axes_nbins:
        prod *= n
    if nbins_flat == prod:
        return tuple(axes_nbins)
    return None

def unflatten(vals_1d, shape):
    return np.asarray(vals_1d).reshape(shape, order="C")

def project_onto_axis(vals_nd, errs_nd, axis_to_keep):
    """
    Sum over all other axes -> 1D.
    """
    axes = list(range(vals_nd.ndim))
    sum_axes = tuple(a for a in axes if a != axis_to_keep)

    v = np.sum(vals_nd, axis=sum_axes)
    if errs_nd is None:
        e = None
    else:
        # variances add under sum
        e = np.sqrt(np.sum(errs_nd**2, axis=sum_axes))
    return v, e

def yields_from_hist(vals, errs, edges, uf, of, vuf=None, vof=None):
    """
    vals, errs are already EVENTS PER BIN (Combine convention).
    Do NOT multiply by bin width.
    """

    # in-range contribution (counts)
    y = np.sum(vals)
    var = np.sum(errs**2) if errs is not None else y

    # add under/overflow (already counts)
    y += uf + of
    if vuf is not None:
        var += vuf + vof

    return float(y), float(np.sqrt(var)), False

def get_reference_yield(hist_eft):
    """
    Return the SM reference yield INCLUDING under/overflow.
    quadratic_term index 0 corresponds to cSM^2.
    """
    arr = np.asarray(ak.to_numpy(hist_eft.values(flow=True)))

    coef_axis = None
    for i, ax in enumerate(hist_eft.axes):
        if ax.name == "quadratic_term":
            coef_axis = i
            break
    if coef_axis is None:
        raise RuntimeError("No quadratic_term axis found")

    # SM^2 term
    sm_coeff = np.take(arr, indices=0, axis=coef_axis)

    return float(np.sum(sm_coeff))

def make_stack_ratio_plot(outpath, title, xedges, data_vals, data_errs,
                          mc_hists, mc_order, xlabel):
    """
    mc_hists: dict proc -> (vals, errs)
    """
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    fig, (ax, rax) = plt.subplots(
        2, 1, figsize=(8, 8),
        gridspec_kw={"height_ratios": (3, 1), "hspace": 0},
        sharex=True
    )
    hep.cms.label("Work In Progress", data=True, ax=ax)

    # Prepare stacked arrays (values only)
    stack_vals = []
    stack_labels = []
    stack_colors = []

    for p in mc_order:
        if p not in mc_hists:
            continue
        v, _e = mc_hists[p]
        stack_vals.append(v)
        stack_labels.append(p)
        stack_colors.append(PROCESS_COLORS.get(p, "#999999"))

    # Bin centers for mplhep
    centers = 0.5 * (xedges[:-1] + xedges[1:])

    # Draw stack
    hep.histplot(
        stack_vals, xedges,
        stack=True, histtype="fill",
        label=stack_labels,
        color=stack_colors,
        edgecolor="black",
        linewidth=1.0,
        ax=ax
    )

    # MC total + uncertainty band (if available)
    mc_tot = np.zeros_like(data_vals, dtype=float)
    mc_var = np.zeros_like(data_vals, dtype=float)

    for p, (v, e) in mc_hists.items():
        mc_tot += v
        if e is None:
            mc_var += np.maximum(v, 0.0)
        else:
            mc_var += e**2

    mc_err = np.sqrt(np.maximum(mc_var, 0.0))
    # band
    ax.fill_between(
        centers,
        mc_tot - mc_err,
        mc_tot + mc_err,
        step="mid",
        alpha=0.25,
        label="Uncertainty"
    )

    # Data
    if data_errs is None:
        data_errs = np.sqrt(np.maximum(data_vals, 0.0))
    ax.errorbar(
        centers, data_vals, yerr=data_errs,
        fmt="o", color="black", label="Data", capsize=2
    )

    ax.set_ylabel("Events / bin")
    ax.set_title(title)
    ax.legend(ncol=2, fontsize=10)

    # Ratio
    denom = mc_tot.copy()
    denom_safe = np.where(denom == 0, 1.0, denom)

    ratio = data_vals / denom_safe
    # ratio uncertainty (approx): sigma(data)/MC
    ratio_err = data_errs / denom_safe

    rax.axhline(1.0, color="black", linewidth=1)
    rax.errorbar(
        centers, ratio, yerr=ratio_err,
        fmt="o", color="black", capsize=2
    )
    rax.set_ylabel("Data/Pred.")
    rax.set_xlabel(xlabel)
    rax.set_ylim(0.8, 1.2)

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print(f"[OK] wrote {outpath}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("templates_root", help="templates.root produced for Combine")
    ap.add_argument("--yaml", required=True, help="your binning yaml (variables: ...)")
    ap.add_argument("--outdir", default="closure_plots_physical")
    ap.add_argument("--channels", default="", help="comma-separated channels (default: all)")
    ap.add_argument("--stack_order", default="",
                    help="comma-separated custom MC stack order (default: internal)")
    ap.add_argument("--dump_json", default="yields_from_templates_root.json")
    args = ap.parse_args()

    with open(args.yaml) as f:
        y = yaml.safe_load(f)

    # preserve insertion order from YAML (python>=3.7)
    var_names = list(y["variables"].keys())
    var_edges = {k: np.asarray(y["variables"][k][0], dtype=float) for k in var_names}
    var_nbins = {k: (len(var_edges[k]) - 1) for k in var_names}

    f = uproot.open(args.templates_root)
    ch_map = list_channels_processes(f)

    selected = set(ch_map.keys())
    if args.channels.strip():
        selected = set([c.strip() for c in args.channels.split(",") if c.strip()])

    mc_order = DEFAULT_STACK_ORDER
    if args.stack_order.strip():
        mc_order = [p.strip() for p in args.stack_order.split(",") if p.strip()]

    yields = {}
    any_missing_errs = False

    # We support:
    # - 1D: templates has nbins == N(var)
    # - 2D: templates has nbins == N(var0)*N(var1) and we project
    for ch in sorted(selected):
        if ch not in ch_map:
            continue

        yields[ch] = {}
        procs = ch_map[ch]

        # read all TH1
        th1 = {}
        for proc in procs:
            edges, vals, errs, uf, of, vuf, vof = read_th1(f, ch, proc)
            th1[proc] = (edges, vals, errs, uf, of, vuf, vof)

        # Choose which variable(s) exist in this template by matching bin count
        # Prefer 2D if possible.
        example_proc = procs[0]
        _edges, _vals, _errs, _uf, _of, _vuf, _vof = th1[example_proc]
        nb_flat = _vals.size

        # Try 2D using first two vars from YAML (common case)
        shape2 = None
        if len(var_names) >= 2:
            v0, v1 = var_names[0], var_names[1]
            shape2 = infer_shape(nb_flat, [var_nbins[v0], var_nbins[v1]])

        # Try 1D using first var
        shape1 = infer_shape(nb_flat, [var_nbins[var_names[0]]])

        # ---- Make plots per variable ----
        # For 2D template: make 1D projections for both axes.
        # For 1D template: plot that variable.
        plot_tasks = []

        if shape2 is not None:
            v0, v1 = var_names[0], var_names[1]
            plot_tasks.append(("proj", v0, 0, shape2, var_edges[v0]))
            plot_tasks.append(("proj", v1, 1, shape2, var_edges[v1]))
        elif shape1 is not None:
            v0 = var_names[0]
            plot_tasks.append(("1d", v0, 0, shape1, var_edges[v0]))
        else:
            # Hard fail with an informative message (this is usually “coeff axis got flattened”)
            msg = (
                f"[FATAL] Cannot interpret flattened bin count = {nb_flat} using YAML.\n"
                f"  YAML vars = {var_names}\n"
                f"  nbins(var0)={var_nbins[var_names[0]]}"
            )
            if len(var_names) >= 2:
                msg += f", nbins(var1)={var_nbins[var_names[1]]}, product={var_nbins[var_names[0]]*var_nbins[var_names[1]]}"
            msg += (
                "\nThis usually means your templates.root was built by flattening a HistEFT *including* the coeff axis "
                "(e.g. 6*153=918 bins), or you flattened extra category axes.\n"
                "Fix: write templates.root from hist_eft.as_hist(values=wc_start) (evaluated), not from hist_eft.values()."
            )
            raise RuntimeError(msg)

        # yields: per proc from *total* TH1 (matches Combine normalization)
        mc_total_y = 0.0
        mc_total_var = 0.0
        
        for proc, (edges, v, err, uf, of, vuf, vof) in th1.items():
            yv, ys, used_poisson = yields_from_hist(
                v, err, edges, uf, of, vuf, vof
            )
            any_missing_errs = any_missing_errs or used_poisson
            yields[ch][proc] = {"Nominal Value": yv, "Std Dev": ys}
        
            if proc != "data_obs":
                mc_total_y += yv
                mc_total_var += ys**2


        yields[ch]["MC_total"] = {"Nominal Value": float(mc_total_y), "Std Dev": float(np.sqrt(mc_total_var))}

        # produce plots
        for _mode, vname, axis_keep, shape, xedges in plot_tasks:
            # build projected hists
            mc_hists = {}
            data_vals = None
            data_errs = None

            for proc in procs:
                _edges, flat_vals, flat_errs, _uf, _of, _vuf, _vof = th1[proc]

                vals_nd = unflatten(flat_vals, shape)
                errs_nd = None if flat_errs is None else unflatten(flat_errs, shape)

                vals_1d, errs_1d = project_onto_axis(vals_nd, errs_nd, axis_keep)

                if proc == "data_obs":
                    data_vals, data_errs = vals_1d, errs_1d
                else:
                    mc_hists[proc] = (vals_1d, errs_1d)

            if data_vals is None:
                # pseudo-data fallback: use MC_total
                data_vals = np.zeros(len(xedges) - 1, dtype=float)
                data_errs = np.zeros_like(data_vals)
                for p, (v, e) in mc_hists.items():
                    data_vals += v
                    if e is None:
                        data_errs = np.sqrt(np.maximum(data_vals, 0.0))
                    else:
                        data_errs = np.sqrt(np.maximum(data_errs**2 + e**2, 0.0))

            outpath = os.path.join(args.outdir, ch, f"{ch}_{vname}.png")
            make_stack_ratio_plot(
                outpath=outpath,
                title=f"",
                xedges=xedges,
                data_vals=data_vals,
                data_errs=data_errs,
                mc_hists=mc_hists,
                mc_order=mc_order,
                xlabel=vname
            )

    with open(args.dump_json, "w") as fp:
        json.dump(yields, fp, indent=4)
    print(f"[OK] wrote {args.dump_json}")

    if any_missing_errs:
        print(
            "\n[WARN] Some TH1s in templates.root have no stored Sumw2/variances.\n"
            "       Then 'Std Dev' becomes sqrt(sumw) fallback, which will NOT match weighted coffea yields.\n"
            "       Fix: when writing templates.root, also write TH1 bin errors (SetBinError / Sumw2).\n"
        )

if __name__ == "__main__":
    main()
