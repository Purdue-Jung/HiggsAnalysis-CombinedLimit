#!/usr/bin/env python3

import json
import numpy as np
import uproot
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
templates_root = "/depot/cms/top/he614/cmseft/statistics/EFT_Combine/2_vars_7_9-0/templates.root"
scaling_json   = "/depot/cms/top/he614/cmseft/statistics/EFT_Combine/2_vars_7_9-0/scaling.json"

channel = "combined"
process = "ttbar_smeft"

# WC to test
wc_name = "ctGRe"

# expansion points relative to START
deltas = [-2, -1, 0, +1, +2]

# your EFT START point (must match YAML)
wc_start = {
    "ctGRe": -0.5,
    "ctGIm": 0.0,
    "cQj18": 1.5,
    "cQj38": 1.5,
    "cQj11": 1.5,
    "cQj31": 1.5,
    "ctu8": 0.0,
    "ctd8": 0.0,
    "ctj8": 0.0,
    "cQu8": 0.0,
    "cQd8": 0.0,
    "ctu1": 0.0,
    "ctd1": 0.0,
    "ctj1": 0.0,
    "cQu1": 0.0,
    "cQd1": 0.0,
}

# ------------------------------------------------------------
def eval_scaling_row(coeffs, wc_names, wc_values):
    """
    Evaluate quadratic EFT polynomial for one bin.
    """
    p = np.array([1.0] + [wc_values[w] for w in wc_names])
    n = len(p)

    val = 0.0
    k = 0
    for i in range(n):
        for j in range(i, n):
            val += coeffs[k] * p[i] * p[j]
            k += 1
    return val

# ------------------------------------------------------------
# Load inputs
# ------------------------------------------------------------
with uproot.open(templates_root) as f:
    h = f[f"{channel}/{process}"]
    template_vals = h.values(flow=False)
    edges = h.axis().edges()

with open(scaling_json) as f:
    scaling = json.load(f)

entry = None
for row in scaling:
    if row["channel"] == channel and row["process"] == process:
        entry = row
        break

if entry is None:
    raise RuntimeError("Scaling entry not found")

wc_names = [p.split("[")[0] for p in entry["parameters"] if p != "cSM[1]"]
scaling_rows = np.asarray(entry["scaling"])

# ------------------------------------------------------------
# Build shifted histograms
# ------------------------------------------------------------
plt.figure(figsize=(8, 6))

for d in deltas:
    wc_vals = dict(wc_start)
    wc_vals[wc_name] = wc_start[wc_name] + d

    scale = np.array([
        eval_scaling_row(coeffs, wc_names, wc_vals)
        for coeffs in scaling_rows
    ])

    y = template_vals * scale

    label = f"{wc_name} = {wc_vals[wc_name]:+.1f}"
    plt.step(edges[:-1], y, where="post", label=label)

# ------------------------------------------------------------
# Plot cosmetics
# ------------------------------------------------------------
plt.xlabel("Bin")
plt.ylabel("Yield")
plt.title(f"EFT expansion around START ({wc_name})")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()
