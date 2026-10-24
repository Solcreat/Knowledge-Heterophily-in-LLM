#!/usr/bin/env python3
import argparse
import os
import sys
# we have to allow to run this script from any directory. To do this:
# It's the very path our script is running, __file__ has the full directory to the script and dirname cuts the filename
script_dir = os.path.dirname(os.path.abspath(__file__))
# this is the parent directory, os.path.join(..., "..") takes a step back.
project_root = os.path.abspath(os.path.join(script_dir, ".."))

from collections import Counter

if project_root not in sys.path:
    sys.path.append(project_root)

from my_project.io import (
    ensure_dir, discover_csvs,
    load_csv, save_csv, write_metadata
)
from my_project.preprocess import uniform_headers
from my_project.selection import compute_entities, select_triplets
from my_project.prompting import load_templates, build_prompts

def main(input_dir, output_base):
    kg_files = discover_csvs(input_dir) # just returns the list of paths like \data\raw\codex\codex.csv or pharmkg.csv at the end, but not the templates.csv
    for kg_path in kg_files:
        # os.path.splitext(...) returns a tuple of two items: file_path and its extension like ('\data\raw\codex\codex', '.csv')
        name = os.path.splitext(os.path.basename(kg_path))[0]  # os.path.basename returns the string of the file name like 'codex.csv'
        # so, the var name will be just codex, trex, pharmkg etc.
        out_dir = os.path.join(output_base, name)  # this will return the directory like \data\processed\codex
        ensure_dir(out_dir)  # it's just os.makedirs(out_dir, exist_ok=True) => \data\processed\codex or \...\trex

        kg = load_csv(kg_path)  # just a pd.read_csv(kg_path). Returns a DataFrame. kg_path is \data\raw\codex\codex.csv
        kg = uniform_headers(kg)      # pass kg as a DataFrame. this will standardize the name of columns and return DF.
        entities = compute_entities(kg)  # pass kg as a DataFrame. returns a DF like:
#                                       entity  out_degree  in_degree  total_degree stratum
# 0                                     Russian           0        140           140      Q4
# 1                               Holly Johnson          15          0            15      Q1
# ...
# 2032                               Paul Simon          24          2            26      Q3
# 2033                               Soulja Boy          17          0            17      Q2
        sel_ids, sel_entities = select_triplets(kg, entities)  # By default, sel_ids will be a list that has 800 random triplets' IDs and sel_entities is a list, too, with some entities, usually 30-40.


        import pandas as pd
        df_ent = pd.DataFrame({'entity_label': sel_entities})  # just a DataFrame of one column and all the entities, that happened to be in Triplets.
        save_csv(df_ent, os.path.join(out_dir, 'selected_entities.csv'))  # it leaves us with data/processed/selected_entities.csv

        tpl_path = os.path.join(input_dir, f"{name}_templates.csv")  # it's, for example, 'data/raw/codex_templates.csv':
        # rel	rel_label	           prompt_template
        # P36   capital                The capital of {subject} is {object}.
        # P361  part of                {subject} is part of {object}.
        # P159  headquarters location  The headquarters of {subject} is or was {object}.
        # ...
        templates = load_templates(tpl_path)  # returns an elegant dictionary:
        # {'capital': 'The capital of {subject} is {object}.',
        #  'part of': '{subject} is part of {object}.',
        #  'headquarters location': 'The headquarters of {subject} is or was {object}.',
        #  'country': ...,
        #  ...}


        # df_prompts = build_prompts(sel_ids, kg, templates)  # we pass a list with IDs, our standardized DF and an elegant Dict.
        # print(kg[~kg["rel_label"].isin(templates.keys())])
        valid_ids = kg[kg["rel_label"].isin(templates.keys())]["triplet_id"].tolist()
        df_prompts = build_prompts(valid_ids, kg, templates)
        # def save_csv(df, path): is df.to_csv(path, index=False)
        save_csv(df_prompts, os.path.join(out_dir, 'triplet_prompts.csv')) # save the builded prompts in /data/processed/{name}/triplet_prompts.csv

        write_metadata(out_dir, {  # save the json file (the full function is in io.py)
            'input_kg'  : kg_path,
            'template'  : tpl_path,
            'entities'  : 'selected_entities.csv',
            'prompts'   : 'triplet_prompts.csv'
        })

        print(f"✔ Finished dataset '{name}'. Outputs in {out_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run entity selection + prompt generation on a raw KG dataset."
    )
    parser.add_argument('--input-dir', required=True,
                        help="Folder with <dataset>.csv and <dataset>_templates.csv")
    parser.add_argument('--output-base', default='./data/processed',
                        help="Where to create per-dataset output folders")
    args = parser.parse_args()
    main(args.input_dir, args.output_base)
