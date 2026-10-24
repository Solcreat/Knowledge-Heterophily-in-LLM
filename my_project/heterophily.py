import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter
from scipy import sparse

def compute_heterophily(df_triplets, df_scores, raw_kg_path):
    """
    df_triplets: DataFrame from anchor_results.csv
    df_scores: DataFrame from anchor_entity_scores.csv
    raw_kg_path: the path to dataset codex.csv/trex.csv/pharmkg.csv (with sub_label and obj_label)
    """
    sub_col = 'sub_label'
    obj_col = 'obj_label'
    ent_col = 'entity'
    score_col = 'kg_value'

    kg_df = pd.read_csv(raw_kg_path)
    # convert triplet_id to str to prevent .merge() from fail
    df_triplets['triplet_id'] = df_triplets['triplet_id'].astype(str)
    kg_df['triplet_id'] = kg_df['triplet_id'].astype(str)

    df_triplets_merged = df_triplets.merge(kg_df[['triplet_id', 'sub_label', 'obj_label']],
                                           on='triplet_id',
                                           how='inner')

    # map the entities with their scores.
    score_dict = dict(zip(df_scores[ent_col], df_scores[score_col]))

    G = nx.from_pandas_edgelist(df_triplets_merged, source=sub_col, target=obj_col)
    nodes = list(G.nodes())

    k = np.array([score_dict.get(node, 0.0) for node in nodes])

    # create the adjacency matrix.
    N = len(nodes)
    A = nx.adjacency_matrix(G, nodelist=nodes).astype(float)
    degrees = np.array(A.sum(axis=1)).flatten()
    degrees[degrees == 0] = 1  # the np.sqrt(1) = 1 in D_inv_sqrt below.

    # Normalized Laplacian L_sym = I - D^-1/2 A D^-1/2
    D_inv_sqrt = sparse.diags(1.0 / np.sqrt(degrees))
    L_sym = sparse.eye(N) - D_inv_sqrt @ A @ D_inv_sqrt

    # The Dirichlet energy. Real and permutated one.
    dirichlet_real = float((k.T @ L_sym @ k))

    k_perm = np.random.permutation(k)
    dirichlet_perm = float((k_perm.T @ L_sym @ k_perm))
    diffs = []
    max_diff = 0 # for calculating the maximum difference on edges
    min_diff = 2
    for _, row in df_triplets_merged.iterrows():
        s_score = score_dict.get(row['sub_label'], 0.0)
        o_score = score_dict.get(row['obj_label'], 0.0)
        current_diff = abs(s_score - o_score) # this is the difference of knowledge on edges in one triplet
        diffs.append(current_diff)
        if(current_diff > max_diff):
            max_diff = current_diff
            saved_prompt = row['triplet_prompt']
            s_score_saved = s_score
            o_score_saved = o_score
            subject_saved = row['sub_label']
            object_saved = row['obj_label']
        if(current_diff < min_diff):
            min_diff = current_diff
            saved_prompt_min = row['triplet_prompt']
            o_score_min_saved = o_score
            s_score_min_saved = s_score
            subject_min_saved = row['sub_label']
            object_min_saved = row['obj_label']

    diffs = np.array(diffs)

    print("=" * 50)
    print(f"Total of analyzed anchor edges: {len(diffs)}")
    print(f"Fraction of edges with knowledge gap >= 0.4: {(diffs >= 0.4).mean() * 100:.2f}%")
    print(f"The exact number of edges with knowledge gap >= 0.4: {Counter((diffs >= 0.4))[True]}")
    print(f"Fraction of edges with knowledge gap >= 0.5: {(diffs >= 0.5).mean() * 100:.2f}%")
    print(f"The exact number of edges with knowledge gap >= 0.5: {Counter((diffs >= 0.5))[True]}")
    print(f"The average edge's knowledge gap: {diffs.mean():.4f}")
    print(f"The maximum difference is: {max_diff}. On the prompt: '{saved_prompt}'.")
    print(f"Where the knowledge of object '{object_saved}' is {o_score_saved}, and the knowdlege of subject '{subject_saved}' is {s_score_saved}.")
    print(f"The minimum difference is: {min_diff}. On the prompt: '{saved_prompt_min}'.")
    print(f"Where the knowledge of object '{object_min_saved}' is {o_score_min_saved}, and the knowdlege of subject '{subject_min_saved}' is {s_score_min_saved}.")
    print("=" * 50)

    return {
        "Number_Of_Entities_In_Anchor_SubGraph": N,
        "Dirichlet_Real": dirichlet_real,
        "Dirichlet_Permuted (Null Model)": dirichlet_perm,
        "Dirichlet_Ratio (Real/Null)": dirichlet_real / dirichlet_perm
    }

def print_heterophily_metrics(dataSetPath, pairOfpathsOfResults, nameOfDataset, nameOfModel):
    endingAnchorRes = "/finetune/anchor_results.csv"
    endingAnchorScores = "/finetune/anchor_entity_scores.csv"
    i = 0
    for path in pairOfpathsOfResults:
        allAnchorTriplets = pd.read_csv(path + endingAnchorRes)
        anchorEntityScores = pd.read_csv(path + endingAnchorScores)
        print(f"For {nameOfDataset} using {nameOfModel} running " + str(i % 2))
        for x, y in compute_heterophily(allAnchorTriplets, anchorEntityScores,
                                        dataSetPath).items():
            print(f"for {x} we have {float(y)}")
        print("-" * 50)
        i += 1


def compute_heterophily_metrics(codex_path, trex_path, pharmkg_path, all_paths):
    RAW_CODEX_PATH = codex_path + "/codex.csv"
    RAW_TREX_PATH = trex_path + "/trex.csv"
    RAW_PHARMKG_PATH = pharmkg_path + "/pharmkg.csv"
    rawPaths = [RAW_CODEX_PATH, RAW_TREX_PATH, RAW_PHARMKG_PATH]
    nameOfDataSets = ["CoDEx", "T-REx", "PharmKG"]
    namesOfTheModels = ["LLama", "Mistral"]
    print("=" * 50)
    print("Anchor analysis for (!) LLama8B (!):")
    curr_index = 0
    for i in range(2):  # for LLama and Mistral
        for ds, dsName in zip(rawPaths, nameOfDataSets):  # for every dataset and its name for the current model.
            print_heterophily_metrics(ds, all_paths[curr_index], dsName, namesOfTheModels[i])
            curr_index += 1
        print("="*50)
        print("Anchor analysis for (!) Mistral7B (!):")