import pandas as pd


def load_templates(template_path):
    # expects two columns: rel_label, prompt_template
    return pd.read_csv(template_path).set_index('rel_label')['prompt_template'].to_dict()


def build_prompts(selected_ids, df_kg, templates):
    sub = df_kg.set_index('triplet_id')['sub_label']  # DF of only subjects: Leonhard Euler, Carl Djerassi, Colombia, ..
    obj = df_kg.set_index('triplet_id')['obj_label']  # DF of only objects:   German, cancer, International Finance ...
    rel = df_kg.set_index('triplet_id')['rel_label']  # DF of only relations: languages spoken, written, or signed ...

    rows = []
    for tid in selected_ids:
        t = templates.get(rel[tid])  # by knowing the triplet id (tid), we can get a relation for it in rel (rel[tid]) and then get a preestablished template from the elegant dictionary.
        if not t:  # if it is a relation that we haven't stated in {name - codex/trex/pharmkg}_templates.csv
            raise KeyError(f"No template for relation {rel[tid]}")
        # for example, t can look like '{subject} is part of {object}.'.
        prompt = t.replace('{subject}', sub[tid]).replace('{object}', obj[tid]) # making it neat like 'Hong Kong is part of People's Republic of China'

        rows.append({'triplet_id': tid, 'triplet_prompt': prompt})  # creating a DataFrame of prompts...
    return pd.DataFrame(rows)
