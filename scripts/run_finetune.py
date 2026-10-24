#!/usr/bin/env python3
"""
Step: Ignorance-guided knowledge injection + fine-tuning  (Part 2)

Examples
--------
# Default (open-source LLM, budget from config):
python scripts/run_finetune.py \
    --input-dir  data/processed/trex \
    --output-dir data/processed/trex/finetune

Pipeline
--------
  1. Select anchor triplets (20% of budget, degree-stratified, capped per entity)
  2. Query anchors against base LLM  → ground-truth knowledge scores
  3. Train GNN on anchor entity scores → predict all entity scores
  4. Ignorance-guided selection of remaining 80% budget triplets
  5. Fine-tune LLM on anchor + selected (combined)
  6. Evaluate on held-out 2% test set  → evaluation_summary.json
"""

import argparse
import os
import sys
# we have to allow to run this script from any directory. To do this:
# It's the very path our script is running, __file__ has the full directory to the script and dirname cuts the filename
script_dir = os.path.dirname(os.path.abspath(__file__))
# this is the parent directory, os.path.join(..., "..") does the step back.
project_root = os.path.abspath(os.path.join(script_dir, ".."))

if project_root not in sys.path:
    sys.path.append(project_root)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from my_project.finetune import run_finetune


def main():
    p = argparse.ArgumentParser(
        description="Part 2: Ignorance-guided triplet selection + LLM fine-tuning."
    )
    p.add_argument(
        "--input-dir", required=True,
        help="Processed dataset folder (metadata.json, triplet_prompts.csv, raw KG)",
    )
    p.add_argument(
        "--output-dir", required=True,
        help="Where to write checkpoints, model, evaluation results",
    )
    p.add_argument(
        "--test-fraction", type=float, default=0.02,
        help="Fraction of all triplets reserved as held-out test set (default: 0.02)",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )

    args = p.parse_args()

    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(f"Input dir not found: {args.input_dir}")

    run_finetune(
        input_dir=args.input_dir,    # e.g. data/processed/codex
        output_dir=args.output_dir,  # e.g. data/processed/codex/finetune
        test_fraction=args.test_fraction,  # 0.02 by default
        random_seed=args.seed,  # 42
    )


if __name__ == "__main__":
    main()
