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


def comma_join(xs):
    return ",".join(xs)


def main():
    ap = argparse.ArgumentParser(
        description="Generate text2workspace, scan, and impact commands for EFT InterferenceModel fits"
    )
    ap.add_argument(
        "--mode",
        required=True,
        help="Tag used to build workspace_<mode>.root and output names (e.g. asimov, data)"
    )
    ap.add_argument(
        "--datacard",
        default="eft_card_combined.txt",
        help="Input datacard used to build the workspace"
    )
    ap.add_argument(
        "--workspace",
        default=None,
        help="Optional explicit workspace name. Default: workspace_<mode>.root"
    )
    ap.add_argument(
        "--range",
        default="20",
        help="Absolute EFT scan range (default: 20 -> [-20,20])"
    )
    ap.add_argument(
        "--physics-model",
        default="HiggsAnalysis.CombinedLimit.InterferenceModels:interferenceModel",
        help=(
            "Physics model passed to text2workspace.py with -P. "
            "Adjust if your local InterferenceModel entry point has a different name."
        )
    )
    ap.add_argument(
        "--t2w-extra",
        default="",
        help="Extra raw arguments appended to text2workspace.py"
    )
    ap.add_argument(
        "--impact-pois",
        default="all",
        help="Comma-separated list of POIs for impact plots, or 'all'"
    )
    ap.add_argument(
        "--cmssw-dir",
        default="/eos/home-l/lingqian/cmseft/statistics/CMSSW_14_1_0_pre4/src",
        help="CMSSW area used to set up Combine runtime"
    )

    args = ap.parse_args()
    scan_range = float(args.range)

    tag = args.mode
    workspace = args.workspace if args.workspace is not None else f"workspace_{tag}.root"

    if args.impact_pois.strip().lower() == "all":
        impact_pois = WC_LIST[:]
    else:
        impact_pois = [x.strip() for x in args.impact_pois.split(",") if x.strip()]

    sm_snapshot = "cSM=1," + ",".join(f"{wc}=0" for wc in WC_LIST)

    extra_flags = ""
    if tag == "asimov":
        extra_flags = f"--setParameters {sm_snapshot} -t -1 "

    COMMON_FLAGS = (
        "--X-rtd MINIMIZER_no_analytic "
        "--cminDefaultMinimizerStrategy 1 "
        "--cminFallbackAlgo Minuit2,0:0.1 "
        "--robustFit 1 "
    )

    print("#!/bin/bash\n")
    #print("set -e\n")

    print("source ~/.profile_el8")
    print("cd /depot/cms/top/he614/cmseft/statistics/CMSSW_14_1_0_pre4/src/")
    print("cmsenv")
    print("cd -\n")

    print("###########################################")
    print(f"# TEXT2WORKSPACE (mode = {tag})")
    print("###########################################\n")
    
    default_t2w_opts = "--PO verbose --PO scalingData=scaling.json --X-allow-no-background"
    t2w_extra = f"{default_t2w_opts} {args.t2w_extra}".strip()
    
    t2w_cmd = (
        f"PYTHONPATH=$PWD:$PYTHONPATH "
        f"text2workspace.py {args.datacard} "
        f"-o {workspace} "
        f"-P {args.physics_model} "
        f"{t2w_extra}".rstrip()
    )
    print(t2w_cmd)
    print()

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
            freeze_str = "cSM," + comma_join(freeze)

            cmd = (
                f"combine {workspace} "
                f"-M MultiDimFit --algo none "
                f"--redefineSignalPOIs {poi} "
                f"--floatOtherPOIs 0 "
                f"--freezeParameters {freeze_str} "
                f"{extra_flags}"
                f"-n BF_Asimov.{poi}"
            )
            print(cmd)
        print()

    # ============================================================
    # 1D GRID SCANS
    # ============================================================
    print("# 1D scans\n")
    for poi in WC_LIST:
        freeze = [x for x in WC_LIST if x != poi]
        freeze_str = "cSM," + comma_join(freeze)

        cmd = (
            f"combineTool.py {workspace} "
            f"-M MultiDimFit --algo grid --points {GRID_POINTS_1D} "
            f"--redefineSignalPOIs {poi} "
            f"--floatOtherPOIs 0 "
            f"--freezeParameters {freeze_str} "
            f"--setParameterRanges {poi}=-{scan_range},{scan_range} "
            f"{COMMON_FLAGS}"
            f"{extra_flags}"
            f"-n Scan1D_{tag}.{poi}"
        )
        print(cmd)

    # ============================================================
    # 2D GRID SCANS
    # ============================================================
    print("\n# 2D scans\n")
    for p0, p1 in itertools.combinations(WC_LIST, 2):
        freeze = [x for x in WC_LIST if x not in (p0, p1)]
        freeze_str = "cSM," + comma_join(freeze)

        cmd = (
            f"combineTool.py {workspace} "
            f"-M MultiDimFit --algo grid "
            f"--points {GRID_POINTS_2D_TOTAL} "
            f"--redefineSignalPOIs {p0},{p1} "
            f"--floatOtherPOIs 0 "
            f"--freezeParameters {freeze_str} "
            f"--setParameterRanges "
            f"{p0}=-{scan_range},{scan_range}:"
            f"{p1}=-{scan_range},{scan_range} "
            f"{COMMON_FLAGS}"
            f"{extra_flags}"
            f"-n Scan2D_{tag}.{p0}.{p1}"
        )
        print(cmd)

    # ============================================================
    # IMPACTS
    # ============================================================
    print("\n###########################################")
    print(f"# IMPACTS (mode = {tag})")
    print("###########################################\n")

    print("# Run these blocks POI-by-POI to get nuisance impacts\n")
    for poi in impact_pois:
        freeze = [x for x in WC_LIST if x != poi]
        freeze_str = "cSM," + comma_join(freeze)
        impact_json = f"impacts_{tag}.{poi}.json"
        impact_base = f"impacts_{tag}.{poi}"

        print(f"# Impacts for {poi}")

        initial_fit = (
            f"combineTool.py -M Impacts -d {workspace} "
            f"--redefineSignalPOIs {poi} "
            f"--floatOtherPOIs 0 "
            f"--freezeParameters {freeze_str} "
            f"--setParameterRanges {poi}=-{scan_range},{scan_range} "
            f"{COMMON_FLAGS}"
            f"{extra_flags}"
            f"--doInitialFit"
        )
        print(initial_fit)

        fits = (
            f"combineTool.py -M Impacts -d {workspace} "
            f"--redefineSignalPOIs {poi} "
            f"--floatOtherPOIs 0 "
            f"--freezeParameters {freeze_str} "
            f"--setParameterRanges {poi}=-{scan_range},{scan_range} "
            f"{COMMON_FLAGS}"
            f"{extra_flags}"
            f"--doFits"
        )
        print(fits)

        collect = (
            f"combineTool.py -M Impacts -d {workspace} "
            f"--redefineSignalPOIs {poi} "
            f"--floatOtherPOIs 0 "
            f"--freezeParameters {freeze_str} "
            f"-o {impact_json}"
        )
        print(collect)

        plot = f"plotImpacts.py -i {impact_json} -o {impact_base}"
        print(plot)
        print()


if __name__ == "__main__":
    main()