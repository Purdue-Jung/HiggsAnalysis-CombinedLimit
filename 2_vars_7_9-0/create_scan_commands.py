#!/usr/bin/env python3
import itertools
import argparse

WC_LIST = [
    "ctGRe","ctGIm","cQj18","cQj38","cQj11","cQj31",
    "ctu8","ctd8","ctj8","cQu8","cQd8","ctu1",
    "ctd1","ctj1","cQu1","cQd1"
]

GRID_POINTS_1D = 10000
GRID_POINTS_2D_X = 100
GRID_POINTS_2D_Y = 100
GRID_POINTS_2D_TOTAL = GRID_POINTS_2D_X * GRID_POINTS_2D_Y


def main():
    ap = argparse.ArgumentParser(
        description="Generate Combine MultiDimFit scan commands"
    )
    ap.add_argument(
        "--mode",
        required=True,
        help="Tag used to select workspace_<mode>.root (e.g. asimov, data)"
    )
    ap.add_argument(
        "--range",
        default="20",
        help="Absolute EFT scan range (default: 20 → [-20,20])"
    )

    args = ap.parse_args()
    scan_range = float(args.range)

    # ------------------------------------------------------------
    # Workspace selection purely from tag
    # ------------------------------------------------------------
    tag = args.mode
    workspace = f"workspace_{tag}.root"

    # ------------------------------------------------------------
    # Mode-dependent extra flags
    # ------------------------------------------------------------
    extra_flags = ""
    if tag == "asimov":
        sm_snapshot = "cSM=1," + ",".join(f"{wc}=0" for wc in WC_LIST)
        extra_flags = (
            f"--setParameters {sm_snapshot} "
            "-t -1 "
        )

    COMMON_FLAGS = (
        "--X-rtd MINIMIZER_no_analytic "
        "--cminDefaultMinimizerStrategy 1 "
        "--cminFallbackAlgo Minuit2,0:0.1 "
        "--robustFit 1 "
    )

    print("#!/bin/bash\n")

    print("# Load CMSSW + Combine")
    print("export SCRAM_ARCH=el9_amd64_gcc12")
    print("cd /eos/home-l/lingqian/cmseft/statistics/CMSSW_14_1_0_pre4/src")
    print("eval `scramv1 runtime -sh`")
    print("cd -\n")

    print("###########################################")
    print(f"# SCANS (mode = {tag})")
    print("###########################################\n")

    # ============================================================
    # ASIMOV BEST-FIT CHECKS (algo none)
    # ============================================================
    if tag == "asimov":
        print("# Asimov best-fit checks (algo none)\n")
        for poi in WC_LIST:
            freeze = [x for x in WC_LIST if x != poi]
            freeze_str = "cSM," + ",".join(freeze)

            cmd = (
                f"combine {workspace} "
                f"-M MultiDimFit --algo none "
                f"--redefineSignalPOIs {poi} "
                f"--freezeParameters {freeze_str} "
                f"{extra_flags}"
                f"-n BF_Asimov.{poi}"
            )
            print(cmd)

        print("\n")

    # ============================================================
    # 1D GRID SCANS
    # ============================================================
    print("# 1D scans\n")
    for poi in WC_LIST:
        freeze = [x for x in WC_LIST if x != poi]
        freeze_str = "cSM," + ",".join(freeze)

        cmd = (
            f"combineTool.py {workspace} "
            f"-M MultiDimFit --algo grid --points {GRID_POINTS_1D} "
            f"--redefineSignalPOIs {poi} "
            f"--freezeParameters {freeze_str} "
            f"--setParameterRanges {poi}=-{scan_range},{scan_range} "
            f"{COMMON_FLAGS} "
            f"{extra_flags}"
            f"-n Scan1D_{tag}.{poi}"
        )
        print(cmd)

    # ============================================================
    # 2D GRID SCANS (CORRECT FOR InterferenceModel)
    # ============================================================
    print("\n# 2D scans\n")

    for p0, p1 in itertools.combinations(WC_LIST, 2):
        freeze = [x for x in WC_LIST if x not in (p0, p1)]
        freeze_str = "cSM," + ",".join(freeze)

        cmd = (
            f"combineTool.py {workspace} "
            f"-M MultiDimFit --algo grid "
            f"--points {GRID_POINTS_2D_TOTAL} "
            f"--redefineSignalPOIs {p0},{p1} "
            f"--freezeParameters {freeze_str} "
            f"--setParameterRanges "
            f"{p0}=-{scan_range},{scan_range}:"
            f"{p1}=-{scan_range},{scan_range} "
            f"{COMMON_FLAGS} "
            f"{extra_flags}"
            f"-n Scan2D_{tag}.{p0}.{p1}"
        )
        print(cmd)


if __name__ == "__main__":
    main()
