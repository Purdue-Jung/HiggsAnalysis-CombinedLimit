#!/usr/bin/env python3

import json
import yaml
import numpy as np
import uproot
import matplotlib.pyplot as plt
import os

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
yaml_file      = "hist_input_validation.yml"
templates_root = "templates.root"
scaling_json   = "scaling.json"

outdir = "eft_expansion"
os.makedirs(outdir, exist_ok=True)

channel = "combined"
process = "ttbar_smeft"

# choose WC to scan
wc_to_scan = "cQu8"

# relative steps around START
deltas = [-2, -1, 0, +1, +2]

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def eval_scaling_row(coeffs, wc_names, wc_vals):
    """Evaluate quadratic EFT polynomial for one bin."""
    p = np.array([1.0] + [wc_vals[w] for w in wc_names])
    n = len(p)

    val = 0.0
    k = 0
    for i in range(n):
        for j in range(i, n):
            val += coeffs[k] * p[i] * p[j]
            k += 1
    return val


# ------------------------------------------------------------
# Load YAML (single source of truth)
# ------------------------------------------------------------
with open(yaml_file) as f:
    ycfg = yaml.safe_load(f)

wc_names = ycfg["eft"]["wc_names"]
wc_start = ycfg["eft"]["wc_start"]

assert wc_to_scan in wc_names, f"{wc_to_scan} not in wc_names"

print("[INFO] Loaded EFT START point:")
for w in wc_names:
    print(f"  {w:8s} = {wc_start[w]}")

# ------------------------------------------------------------
# Load template histogram
# ------------------------------------------------------------
with uproot.open(templates_root) as f:
    h = f[f"{channel}/{process}"]
    template_vals = h.values(flow=False)
    edges = h.axis().edges()

# ------------------------------------------------------------
# Load scaling.json
# ------------------------------------------------------------
with open(scaling_json) as f:
    scaling_all = json.load(f)

entry = None
for row in scaling_all:
    if row["channel"] == channel and row["process"] == process:
        entry = row
        break

if entry is None:
    raise RuntimeError("Scaling entry not found")

scaling_rows = np.asarray(entry["scaling"])

# ------------------------------------------------------------
# EFT expansion plot
# ------------------------------------------------------------
plt.figure(figsize=(8, 6))

y_start = None

for d in deltas:
    wc_vals = dict(wc_start)
    wc_vals[wc_to_scan] = wc_start[wc_to_scan] + d

    scale = np.array([
        eval_scaling_row(coeffs, wc_names, wc_vals)
        for coeffs in scaling_rows
    ])

    y = template_vals * scale

    if d == 0:
        y_start = y.copy()

    label = f"{wc_to_scan} = {wc_vals[wc_to_scan]:+.1f}"
    plt.step(edges[:-1], y, where="post", label=label)

plt.xlabel("bin")
plt.ylabel("yield")
plt.title(f"EFT expansion around START ({wc_to_scan})")
plt.legend()
plt.grid(True)
plt.tight_layout()

out_main = os.path.join(
    outdir, f"expansion_{channel}_{process}_{wc_to_scan}.png"
)
plt.savefig(out_main, dpi=150)
print(f"[INFO] Saved plot → {out_main}")
plt.close()


# ------------------------------------------------------------
# Ratio-to-START plot (MOST IMPORTANT CHECK)
# ------------------------------------------------------------
plt.figure(figsize=(8, 4))

for d in deltas:
    wc_vals = dict(wc_start)
    wc_vals[wc_to_scan] = wc_start[wc_to_scan] + d

    scale = np.array([
        eval_scaling_row(coeffs, wc_names, wc_vals)
        for coeffs in scaling_rows
    ])

    y = template_vals * scale
    ratio = y / y_start

    plt.step(edges[:-1], ratio, where="post", label=f"Δ = {d:+d}")

plt.axhline(1.0, color="black", linestyle="--")
plt.xlabel("bin")
plt.ylabel("ratio to START")
plt.title(f"EFT ratio to START ({wc_to_scan})")
plt.legend()
plt.grid(True)
plt.tight_layout()

out_ratio = os.path.join(
    outdir, f"ratio_{channel}_{process}_{wc_to_scan}.png"
)
plt.savefig(out_ratio, dpi=150)
print(f"[INFO] Saved ratio plot → {out_ratio}")
plt.close()
