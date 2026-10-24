#!/usr/bin/env python3
import os
import sys
# we have to allow to run this script from any directory. To do this:
# It's the very path our script is running, __file__ has the full directory to the script and dirname cuts the filename
script_dir = os.path.dirname(os.path.abspath(__file__))
# this is the parent directory, os.path.join(..., "..") does the step back.
project_root = os.path.abspath(os.path.join(script_dir, ".."))

if project_root not in sys.path:
    sys.path.append(project_root)
import argparse
from my_project.gnn import run_gnn

if __name__=="__main__":
    p = argparse.ArgumentParser(
        description="Step 4: Train GPR-GNN & predict entity kg_values"
    )
    p.add_argument('--input-dir', required=True,
                   help="Processed dataset folder (with metadata.json, entity_kg_values.csv, etc.)")
    args = p.parse_args()

    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(f"No such folder: {args.input_dir}")
    run_gnn(args.input_dir)
