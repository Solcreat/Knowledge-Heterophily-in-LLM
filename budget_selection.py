"""
budget_selection.py
-------------------
Two functions:

  select_anchor_triplets(...)
      Entity-centric, degree-stratified sampling for the 20% anchor set.
      Improvement over original: round-robin across degree bins so the GNN
      trains on a diverse neighborhood signal (not just high-degree hubs).

  select_ignorance_guided(...)
      Implements the pseudocode greedy budget allocation.
      Improvement over pseudocode: the CASE-B fallback candidate pool still
      excludes explicitly-excluded triplets (anchors), so they can never be
      accidentally re-queued.
"""

from __future__ import annotations

import math
import random

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Anchor (20 %) selection
# ─────────────────────────────────────────────────────────────────────────────

def select_anchor_triplets(
    kg_df: pd.DataFrame,   # e.g. codex.csv
    total_budget: int,  # by default, 4000
    anchor_fraction: float = 0.20,
    max_per_entity: int = 40,
    num_bins: int = 4,
    random_seed: int = 42,
) -> tuple[list[str], list[str]]:
    """
    Degree-stratified anchor selection.

    Returns
    -------
    anchor_triplet_ids   : list[str]
    anchor_entity_labels : list[str]
    """
    rng = random.Random(random_seed)  # isolated, independent instance of a pseudo-random number generator
    anchor_budget = int(total_budget * anchor_fraction)  # by default, anchor_budget will be 800

    def _get_ids(ent: str) -> list[str]:  # returns all the triplets' IDs that has this entity without dupes
        s = kg_df.loc[kg_df["sub_label"] == ent, "triplet_id"].tolist()
        o = kg_df.loc[kg_df["obj_label"] == ent, "triplet_id"].tolist()
        return list(set(s + o))

    all_entities = list(
        set(kg_df["sub_label"].tolist()) | set(kg_df["obj_label"].tolist())  # operator | is like .union()
    )
    # print(len(all_entities))  # 2034 entities
    # print(all_entities)
    # print("=========================++++++++++++++++++++++++")
    out_deg = kg_df["sub_label"].value_counts()
    in_deg  = kg_df["obj_label"].value_counts()
    ent_df  = pd.DataFrame({"entity": all_entities})
    ent_df["degree"] = (
        ent_df["entity"].map(out_deg).fillna(0)
        + ent_df["entity"].map(in_deg).fillna(0)
    ).astype(int)
    # now ent_df has total occurrences of every entity.
    try:
        ent_df["stratum"] = pd.qcut(
            ent_df["degree"], q=num_bins,
            labels=list(range(num_bins)), duplicates="drop",
        ).astype(int)
    except Exception:
        ent_df["stratum"] = 0
    # now ent_df almost has Q1...Q4 (quantiles), but they_are labeled just in order 1, 2, 3, 4.
    strata    = sorted(ent_df["stratum"].unique()) # 1 2 3 4
    bins_list = [  # splits the entities into 4 tiers: from the rarest to the most common.
        ent_df[ent_df["stratum"] == s]["entity"]
        .sample(frac=1, random_state=random_seed).tolist()
        for s in strata
    ]
    # print(len(bins_list[0])+len(bins_list[1])+len(bins_list[2])+len(bins_list[3]))  # in all tiers, there are 2034 ent
    # round-robin across strata
    i = 0
    ordered_entities: list[str] = []
    while any(bins_list):  # while there are still entities in the list of lists
        for b in bins_list:
            if b:
                ordered_entities.append(b.pop(0)) # append the current entity from the tier e.g. 0. Then do the same to tier 1, etc. after this, we'll have one less entities in every tier
                # so, ordered_entities will have all 2034 entities in that strange order...

    selected_ids: set[str] = set()
    selected_entities: list[str] = []

    for ent in ordered_entities:  # now, we will go through every entity in that strange order
        if len(selected_ids) >= anchor_budget:  # if our set is more than 800 (by default)
            break
        ids = _get_ids(ent)  # now ids has all the triplets' IDS that have it.
        if not ids: # if there aren't any triplets...
            continue
        if len(ids) > max_per_entity:  # if there are more triplets' IDs, than 40 (by default) for this entity...
            ids = rng.sample(ids, max_per_entity)  # we shuffle it and take only 40 triplets' IDs.
        new_ids = [i for i in ids if i not in selected_ids]  # leave only new triplets' IDs to exclude duplicate IDs consisting two entities we have.
        remain  = anchor_budget - len(selected_ids)  # it will get fewer and fewer through iterations
        take    = new_ids[:remain]  # we don't take more, than we need.
        if take:
            selected_ids.update(take)  # now we have new triplets' IDs from this entity
            selected_entities.append(ent)  # push back the entity!

    print(
        f"[anchor] budget={anchor_budget}  "
        f"selected={len(selected_ids)}  "
        f"entities={len(selected_entities)}"
    )
    return list(selected_ids), selected_entities  # returns 800 triplets' IDs and all the entities


# ─────────────────────────────────────────────────────────────────────────────
# Ignorance-guided selection (pseudocode algorithm)
# ─────────────────────────────────────────────────────────────────────────────

def select_ignorance_guided(
    entity_scores_df: pd.DataFrame,
    triplet_graph_df: pd.DataFrame,
    triplet_prompt_df: pd.DataFrame,
    excluded_triplet_ids: set[str],
    total_budget: int,
    min_triplets_per_partial_slice: int,
    fraction_triplets_for_partial: float,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Greedy budget allocation toward most-ignorant entities.

    entity_scores_df columns:
        entity_id, predicted_knowledge_score,
        [optional] observed_knowledge_score

    triplet_graph_df columns:
        triplet_id, subject_entity_id, object_entity_id

    triplet_prompt_df columns:
        triplet_id, triplet_prompt

    Returns (selected_triplets_df, selected_entities_df)
    """
    rng = random.Random(random_seed)
    excluded: set[str] = set(excluded_triplet_ids)

    # ── ignorance per entity ──────────────────────────────────────────────────
    obs_col = "observed_knowledge_score"
    has_obs = obs_col in entity_scores_df.columns

    ignorance_map: dict[str, float] = {}
    for _, row in entity_scores_df.iterrows():
        eid = str(row["entity_id"])
        if has_obs and pd.notna(row.get(obs_col)):
            k = float(row[obs_col])
        else:
            k = float(row["predicted_knowledge_score"])
        ignorance_map[eid] = max(0.0, min(1.0, 1.0 - k))

    entity_list = list(ignorance_map.keys())

    # ── entity → triplets ─────────────────────────────────────────────────────
    triplets_by_entity: dict[str, set[str]] = {e: set() for e in entity_list}
    entity_set = set(entity_list)

    for _, row in triplet_graph_df.iterrows():
        tid  = str(row["triplet_id"])
        subj = str(row["subject_entity_id"])
        obj  = str(row["object_entity_id"])
        if subj in entity_set:
            triplets_by_entity[subj].add(tid)
        if obj in entity_set:
            triplets_by_entity[obj].add(tid)

    entity_degree: dict[str, int] = {e: len(triplets_by_entity[e]) for e in entity_list}

    # ── rank: high ignorance → low degree → random ───────────────────────────
    tie = {e: rng.random() for e in entity_list}
    ranked = sorted(
        entity_list,
        key=lambda e: (-ignorance_map[e], entity_degree[e], tie[e]),
    )

    # ── greedy loop ───────────────────────────────────────────────────────────
    remaining = total_budget
    seen: set[str] = set()
    records: list[tuple[str, object, set[str]]] = []

    for ent in ranked:
        if remaining <= 0:
            break
        deg = entity_degree[ent]
        if deg == 0:
            continue

        all_t      = triplets_by_entity[ent]
        new_t      = all_t - seen - excluded
        cost_full  = len(new_t)

        # CASE A – full selection fits
        if 0 < cost_full <= remaining:
            records.append((ent, "all", set(new_t)))
            seen     |= all_t
            remaining -= cost_full
            continue

        # CASE B – partial slice
        if cost_full > 0:
            target    = math.floor(fraction_triplets_for_partial * deg)
            ideal     = max(min_triplets_per_partial_slice, target)
            capped    = min(ideal, deg)
            sz        = min(capped, remaining)

            if sz < min_triplets_per_partial_slice:
                continue

            pool = list(new_t)
            if len(pool) < sz:
                # improvement: fallback still excludes anchor/excluded triplets
                pool = list(all_t - excluded)
            if len(pool) < sz:
                continue

            picked = set(rng.sample(pool, sz))
            records.append((ent, sz, picked))
            remaining -= sz
            continue

        # CASE C – nothing new, skip

    # ── flatten ───────────────────────────────────────────────────────────────
    selected_ids: set[str] = set()
    for _, _, tids in records:
        selected_ids |= tids

    # ── join with prompts ─────────────────────────────────────────────────────
    sel_df = (
        triplet_prompt_df[
            triplet_prompt_df["triplet_id"].astype(str).isin(selected_ids)
        ]
        .drop_duplicates(subset=["triplet_id"])
        .reset_index(drop=True)
    )

    sel_ent_df = pd.DataFrame(
        {"entity_id": [e for e, _, tids in records if tids]}
    )

    print(
        f"[ignorance_selection] budget={total_budget}  "
        f"selected_triplets={len(selected_ids)}  "
        f"selected_entities={len(sel_ent_df)}  "
        f"remaining={remaining}"
    )
    return sel_df, sel_ent_df
