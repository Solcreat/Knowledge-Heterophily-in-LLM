#!/usr/bin/env python3
"""
Calculate the Dirichlet energy (real and permutated) and count all the rifts (razloms).

python scripts/run_heterophily.py --help

python scripts/run_heterophily.py --codex-dir data/raw/codex --trex-dir data/raw/trex --pharmkg-dir data/raw/pharmkg -rp "my_project/LLama8BCodex-S GPR 91.23" "my_project/LLama8BCodex-S SAGE 74.11" -rp "my_project/LLama8BTrex GPR 95.48" "my_project/LLama8BTrex SAGE 93.55" -rp "my_project/LLama8BPharmKG GPR 15.62" "my_project/LLama8BPharmKG SAGE 7.94" -rp "my_project/Mistral7BCodex-S GPR 99.18" "my_project/Mistral7BCodex-S SAGE 84.38" -rp "my_project/Mistral7BTrex GPR 95.69" "my_project/Mistral7BTrex SAGE 96.05" -rp "my_project/Mistral7BLPharmKG GPR 99.25" "my_project/Mistral7BPharmKG SAGE 94.63"
"""
import argparse
import os
import sys
import os
import sys
# we have to allow to run this script from any directory. To do this:
# It's the very path our script is running, __file__ has the full directory to the script and dirname cuts the filename
script_dir = os.path.dirname(os.path.abspath(__file__))
# this is the parent directory, os.path.join(..., "..") does the step back.
project_root = os.path.abspath(os.path.join(script_dir, ".."))

if project_root not in sys.path:
    sys.path.append(project_root)
# allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from my_project.heterophily import compute_heterophily_metrics


def main():
    p = argparse.ArgumentParser(
        description="Calculate the Dirichlet energy (real and permutated) and count all the rifts (razloms)."
    )
    p.add_argument(
        "--codex-dir", required=True,
        help="The relative path to the folder which contains codex.csv dataset. Please use forward slashes '/'."
    )
    p.add_argument(
        "--trex-dir", required=True,
        help="The relative path to the folder which contains trex.csv dataset. Please use forward slashes '/'.",
    )
    p.add_argument(
        "--pharmkg-dir", required=True,
        help="The relative path to the folder which contains pharmkg.csv dataset. Please use forward slashes '/'.",
    )
    p.add_argument(
        "-rp",
        action="append",
        nargs=2,
        metavar=("KEY", "VALUE"),
        required=True,
        help="You have to pass 12 Relative Paths (rp) in 6 pairs of the folders which contain for every model and dataset "
             "/finetune/anchor_results.csv and /finetune/anchor_entity_scores.csv. Read the example in Readme.md. "
             "Please use forward slashes '/'."
    )
    args = p.parse_args()

    if not os.path.isdir(args.codex_dir):
        raise FileNotFoundError(f"Input dir not found: {args.codex_dir}")
    if not os.path.isdir(args.trex_dir):
        raise FileNotFoundError(f"Input dir not found: {args.trex_dir}")
    if not os.path.isdir(args.pharmkg_dir):
        raise FileNotFoundError(f"Input dir not found: {args.pharmkg_dir}")
    if (len(args.rp) != 6):
        print("You have to pass exactly 6 pairs of the folders which contain for every model and dataset /finetune/anchor_results.csv and /finetune/anchor_entity_scores.csv. Read the example in Readme.md")
        print(f"Now you have passed {len(args.rp)} pairs, which is less or more than 6.")
        exit()

    compute_heterophily_metrics(
        codex_path=args.codex_dir,
        trex_path=args.trex_dir,
        pharmkg_path=args.pharmkg_dir,
        all_paths=args.rp
    )


if __name__ == "__main__":
    main()
