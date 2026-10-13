import numpy as np
import pandas as pd
import networkx as nx
import sys
import os
from scipy import sparse

def compute_heterophily_metrics(df_triplets, df_scores, raw_kg_path):
    """
    df_triplets: DataFrame from anchor_results.csv
    df_scores: DataFrame from anchor_entity_scores.csv
    raw_kg_path: the path to codex.csv (with sub_label and obj_label)
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
    for _, row in df_triplets_merged.iterrows():
        s_score = score_dict.get(row['sub_label'], 0.0)
        o_score = score_dict.get(row['obj_label'], 0.0)
        diffs.append(abs(s_score - o_score))

    diffs = np.array(diffs)

    print("=" * 50)
    print(f"Total of analyzed anchor edges: {len(diffs)}")
    print(f"Fraction of edges with knowledge gap >= 0.4: {(diffs >= 0.4).mean() * 100:.2f}%")
    print(f"Fraction of edges with knowledge gap >= 0.5: {(diffs >= 0.5).mean() * 100:.2f}%")
    print(f"The average edge's knowledge gap: {diffs.mean():.4f}")
    print("=" * 50)

    return {
        "Number_Of_Entities_In_Anchor_SubGraph": N,
        "Dirichlet_Real": dirichlet_real,
        "Dirichlet_Permuted (Null Model)": dirichlet_perm,
        "Dirichlet_Ratio (Real/Null)": dirichlet_real / dirichlet_perm
    }


if __name__ == "__main__":
    RAW_CODEX_PATH = "C:/Users/Solange/Desktop/Knowledge Heterophily in Large Language Models/kgc/data/raw/codex/codex.csv"
    RAW_TREX_PATH = "C:/Users/Solange/Desktop/Knowledge Heterophily in Large Language Models/kgc/data/raw/trex/trex.csv"
    RAW_PHARMKG_PATH ="C:/Users/Solange/Desktop/Knowledge Heterophily in Large Language Models/kgc/data/raw/pharmkg/pharmkg.csv"
    allAnchorTripletsCodexLLamaGPR = pd.read_csv("C:/Users/Solange/Desktop/Knowledge Heterophily in Large Language Models/kgc/my_project/LLama8BCodex-S GPR 91.23/finetune/anchor_results.csv")
    anchorEntityScoresCodexLLamaGPR = pd.read_csv("C:/Users/Solange/Desktop/Knowledge Heterophily in Large Language Models/kgc/my_project/LLama8BCodex-S GPR 91.23/finetune/anchor_entity_scores.csv")
    allAnchorTripletsCodexLLamaSAGE = pd.read_csv("C:/Users/Solange/Desktop/Knowledge Heterophily in Large Language Models/kgc/my_project/LLama8BCodex-S SAGE 74.11/finetune/anchor_results.csv")
    anchorEntityScoresCodexLLamaSAGE = pd.read_csv("C:/Users/Solange/Desktop/Knowledge Heterophily in Large Language Models/kgc/my_project/LLama8BCodex-S SAGE 74.11/finetune/anchor_entity_scores.csv")
    allAnchorTripletsTrexLLamaGPR = pd.read_csv("C:/Users/Solange/Desktop/Knowledge Heterophily in Large Language Models/kgc/my_project/LLama8BTrex GPR 95.48/finetune/anchor_results.csv")
    anchorEntityScoresTrexLLamaGPR = pd.read_csv("C:/Users/Solange/Desktop/Knowledge Heterophily in Large Language Models/kgc/my_project/LLama8BTrex GPR 95.48/finetune/anchor_entity_scores.csv")
    allAnchorTripletsTrexLLamaSAGE = pd.read_csv("C:/Users/Solange/Desktop/Knowledge Heterophily in Large Language Models/kgc/my_project/LLama8BTrex SAGE 93.55/finetune/anchor_results.csv")
    anchorEntityScoresTrexLLamaSAGE = pd.read_csv("C:/Users/Solange/Desktop/Knowledge Heterophily in Large Language Models/kgc/my_project/LLama8BTrex SAGE 93.55/finetune/anchor_entity_scores.csv")
    allAnchorTripletsPharmKGLLamaGPR = pd.read_csv("C:/Users/Solange/Desktop/Knowledge Heterophily in Large Language Models/kgc/my_project/LLama8BPharmKG GPR 15.62/finetune/anchor_results.csv")
    anchorEntityScoresPharmKGLLamaGPR = pd.read_csv("C:/Users/Solange/Desktop/Knowledge Heterophily in Large Language Models/kgc/my_project/LLama8BPharmKG GPR 15.62/finetune/anchor_entity_scores.csv")
    allAnchorTripletsPharmKGLLamaSAGE = pd.read_csv("C:/Users/Solange/Desktop/Knowledge Heterophily in Large Language Models/kgc/my_project/LLama8BPharmKG SAGE 7.94/finetune/anchor_results.csv")
    anchorEntityScoresPharmKGLLamaSAGE = pd.read_csv("C:/Users/Solange/Desktop/Knowledge Heterophily in Large Language Models/kgc/my_project/LLama8BPharmKG SAGE 7.94/finetune/anchor_entity_scores.csv")
    print("="*50)
    print("Anchor analysis for (!) LLama8B (!):")
    print("For Codex using LLama 8B and GPR-GNN:")
    for x, y in compute_heterophily_metrics(allAnchorTripletsCodexLLamaGPR, anchorEntityScoresCodexLLamaGPR, RAW_CODEX_PATH).items():
        print(f"for {x} we have {float(y)}")
    print("-" * 50)
    print("For Codex using LLama 8B and Graph-SAGE:")
    for x, y in compute_heterophily_metrics(allAnchorTripletsCodexLLamaSAGE, anchorEntityScoresCodexLLamaSAGE, RAW_CODEX_PATH).items():
        print(f"for {x} we have {float(y)}")
    print("-" * 50)
    print("For Trex using LLama 8B and GPR-GNN:")
    for x, y in compute_heterophily_metrics(allAnchorTripletsTrexLLamaGPR, anchorEntityScoresTrexLLamaGPR,
                                            RAW_TREX_PATH).items():
        print(f"for {x} we have {float(y)}")
    print("-" * 50)
    print("For Trex using LLama 8B and Graph-SAGE:")
    for x, y in compute_heterophily_metrics(allAnchorTripletsTrexLLamaSAGE, anchorEntityScoresTrexLLamaSAGE,
                                            RAW_TREX_PATH).items():
        print(f"for {x} we have {float(y)}")
    print("-" * 50)
    print("For PharmKG using LLama 8B and GPR-GNN:")
    for x, y in compute_heterophily_metrics(allAnchorTripletsPharmKGLLamaGPR, anchorEntityScoresPharmKGLLamaGPR,
                                            RAW_PHARMKG_PATH).items():
        print(f"for {x} we have {float(y)}")
    print("-" * 50)
    print("For PharmKG using LLama 8B and Graph-SAGE:")
    for x, y in compute_heterophily_metrics(allAnchorTripletsPharmKGLLamaSAGE, anchorEntityScoresPharmKGLLamaSAGE,
                                            RAW_PHARMKG_PATH).items():
        print(f"for {x} we have {float(y)}")

    allAnchorTripletsCodexMistralGPR = pd.read_csv(
        "C:/Users/Solange/Desktop/Knowledge Heterophily in Large Language Models/kgc/my_project/Mistral7BCodex-S GPR 99.18/finetune/anchor_results.csv")
    anchorEntityScoresCodexMistralGPR = pd.read_csv(
        "C:/Users/Solange/Desktop/Knowledge Heterophily in Large Language Models/kgc/my_project/Mistral7BCodex-S GPR 99.18/finetune/anchor_entity_scores.csv")
    allAnchorTripletsCodexMistralSAGE = pd.read_csv(
        "C:/Users/Solange/Desktop/Knowledge Heterophily in Large Language Models/kgc/my_project/Mistral7BCodex-S SAGE 84.38/finetune/anchor_results.csv")
    anchorEntityScoresCodexMistralSAGE = pd.read_csv(
        "C:/Users/Solange/Desktop/Knowledge Heterophily in Large Language Models/kgc/my_project/Mistral7BCodex-S SAGE 84.38/finetune/anchor_entity_scores.csv")
    allAnchorTripletsTrexMistralGPR = pd.read_csv(
        "C:/Users/Solange/Desktop/Knowledge Heterophily in Large Language Models/kgc/my_project/Mistral7BTrex GPR 95.69/finetune/anchor_results.csv")
    anchorEntityScoresTrexMistralGPR = pd.read_csv(
        "C:/Users/Solange/Desktop/Knowledge Heterophily in Large Language Models/kgc/my_project/Mistral7BTrex GPR 95.69/finetune/anchor_entity_scores.csv")
    allAnchorTripletsTrexMistralSAGE = pd.read_csv(
        "C:/Users/Solange/Desktop/Knowledge Heterophily in Large Language Models/kgc/my_project/Mistral7BTrex SAGE 96.05/finetune/anchor_results.csv")
    anchorEntityScoresTrexMistralSAGE = pd.read_csv(
        "C:/Users/Solange/Desktop/Knowledge Heterophily in Large Language Models/kgc/my_project/Mistral7BTrex SAGE 96.05/finetune/anchor_entity_scores.csv")
    allAnchorTripletsPharmKGMistralGPR = pd.read_csv(
        "C:/Users/Solange/Desktop/Knowledge Heterophily in Large Language Models/kgc/my_project/Mistral7BPharmKG GPR 66.55/finetune/anchor_results.csv")
    anchorEntityScoresPharmKGMistralGPR = pd.read_csv(
        "C:/Users/Solange/Desktop/Knowledge Heterophily in Large Language Models/kgc/my_project/Mistral7BPharmKG GPR 66.55/finetune/anchor_entity_scores.csv")
    allAnchorTripletsPharmKGMistralSAGE = pd.read_csv(
        "C:/Users/Solange/Desktop/Knowledge Heterophily in Large Language Models/kgc/my_project/Mistral7BPharmKG SAGE 94.63/finetune/anchor_results.csv")
    anchorEntityScoresPharmKGMistralSAGE = pd.read_csv(
        "C:/Users/Solange/Desktop/Knowledge Heterophily in Large Language Models/kgc/my_project/Mistral7BPharmKG SAGE 94.63/finetune/anchor_entity_scores.csv")
    print("="*50)
    print("Anchor analysis for (!) Mistral7B (!):")
    print("For Codex using Mistral 7B and GPR-GNN:")
    for x, y in compute_heterophily_metrics(allAnchorTripletsCodexMistralGPR, anchorEntityScoresCodexMistralGPR, RAW_CODEX_PATH).items():
        print(f"for {x} we have {float(y)}")
    print("-" * 50)
    print("For Codex using Mistral 7B and Graph-SAGE:")
    for x, y in compute_heterophily_metrics(allAnchorTripletsCodexMistralSAGE, anchorEntityScoresCodexMistralSAGE, RAW_CODEX_PATH).items():
        print(f"for {x} we have {float(y)}")
    print("-" * 50)
    print("For Trex using Mistral 7B and GPR-GNN:")
    for x, y in compute_heterophily_metrics(allAnchorTripletsTrexMistralGPR, anchorEntityScoresTrexMistralGPR,
                                            RAW_TREX_PATH).items():
        print(f"for {x} we have {float(y)}")
    print("-" * 50)
    print("For Trex using Mistral 7B and Graph-SAGE:")
    for x, y in compute_heterophily_metrics(allAnchorTripletsTrexMistralSAGE, anchorEntityScoresTrexMistralSAGE,
                                            RAW_TREX_PATH).items():
        print(f"for {x} we have {float(y)}")
    print("-" * 50)
    print("For PharmKG using Mistral 7B and GPR-GNN:")
    for x, y in compute_heterophily_metrics(allAnchorTripletsPharmKGMistralGPR, anchorEntityScoresPharmKGMistralGPR,
                                            RAW_PHARMKG_PATH).items():
        print(f"for {x} we have {float(y)}")
    print("-" * 50)
    print("For PharmKG using Mistral 7B and Graph-SAGE:")
    for x, y in compute_heterophily_metrics(allAnchorTripletsPharmKGMistralSAGE, anchorEntityScoresPharmKGMistralSAGE,
                                            RAW_PHARMKG_PATH).items():
        print(f"for {x} we have {float(y)}")
