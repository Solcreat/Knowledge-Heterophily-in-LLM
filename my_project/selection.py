import pandas as pd
import random
from my_project.config import T_QUERY, Q4_CAP, MAX_FULL_TRY, NUM_BINS

def compute_entities(df_uris):
    out_deg = df_uris['sub_label'].value_counts()  # counts the frequency of unique values in sub_label column. returns a Series
    in_deg = df_uris['obj_label'].value_counts()  # counts the frequency of unique values in obj_label column.
    # BECAUSE OF THE SET APPEARS THE RANDOM ORDER IN DATAFRAME
    all_ents = set(df_uris['sub_label']).union(df_uris['obj_label'])  # this will return a set of all subjects and objects like {'Leonhard Euler', 'German', ..., 'theater director'}
    # print(all_ents) # HERE COMES THE RANDOM: FROM THE SET
    ent_df = pd.DataFrame({'entity': list(all_ents)}) # creates a DataFrame only consisting of one column
    # print(ent_df) this DataFrame is always DIFFERENT, no matter that further Authors use the same random_states!
    ent_df['out_degree'] = ent_df['entity'].map(out_deg).fillna(0).astype(int) # .map(out_deg) will replace entities names that were in sub_label column before with their number of occurences in DataFrame (if they weren't in sub_label (subject) then they'll be zero, due to .fillna(0))
    ent_df['in_degree']  = ent_df['entity'].map(in_deg).fillna(0).astype(int)  # same for the entities that were in obj_label
    ent_df['total_degree'] = ent_df['in_degree'] + ent_df['out_degree']  # total amount of current object and subject

    # stratify into Q1..Qn
    # Cut the entities Q1 - the rarest entities and Q4 (quantile) - the most common entities. Look below for an example.
    cut = pd.qcut(ent_df['total_degree'], q=NUM_BINS, labels=[f"Q{i}" for i in range(1,NUM_BINS+1)], duplicates='drop')
    # NUM_BINS = 4 in default settings
    ent_df['stratum'] = cut
    # print(ent_df)
    return ent_df
"""
for example for Codex:
                                       entity  out_degree  in_degree  total_degree stratum
0                                     Russian           0        140           140      Q4
1                               Holly Johnson          15          0            15      Q1
2     International Hydrographic Organization           0         86            86      Q4
3                              peace activist           0         17            17      Q2
4                                    Interpol           0        190           190      Q4
...                                       ...         ...        ...           ...     ...
2029                           David Walliams          15          0            15      Q1
2030                         Miranda Cosgrove          15          0            15      Q1
2031                           Bertolt Brecht          20          4            24      Q3
2032                               Paul Simon          24          2            26      Q3
2033                               Soulja Boy          17          0            17      Q2

[2034 rows x 5 columns]

"""


def select_triplets(df_uris, entities):  # df_uris = KG (standardized DF), entities = entities.
    def get_ids(e):
        # print(e)
        s = df_uris.loc[df_uris['sub_label']==e, 'triplet_id']  # returns Series of all the triplet IDs where current entity was in sub_label
        o = df_uris.loc[df_uris['obj_label']==e, 'triplet_id']  # returns triplet IDs where this entity was in obj_label
        return set(s).union(o)  # returns the RANDOM SET of all Triplet ID's for this current entity like:
    # Alphonse Daudet
    # {21218, 18084, 25223, 11145, 19403, 25740, 12494, 2320, 1203, 11891, 14484, 35128, 18399, 21535, 3615}

    pool = entities.sample(frac=1, random_state=43)['entity'] # basically, .sample used to randomly shuffle an entire DataFrame while making the process reproducible
    # but the problem is - BECAUSE OF THE SET, IT WILL BE ALWAYS DIFFERENT, NO MATTER OF THE SAME RANDOM_STATE!
    # setting frac to 1 (or 100%) means it returns every single row from the original DataFrame. So we have a pool of entities.
    # print(pool)
    #                    entity
    # 0              Irene Cara
    # 1            Tila Tequila
    # 2        contemporary R&B
    # ...
    # 2033        archaeologist
    selected = set()  # this is the set with all the triplets' IDs
    E_final = []  # all the entities that are in those triplets with those IDs
    skipped_full = 0

    for e in pool:
        # T_QUERY = int(TOTAL_TRIPLET_BUDGET * ANCHOR_FRACTION) (by default it's 4000 * 0.2 = 800) So, T_QUERY = 800
        if len(selected) >= T_QUERY:
            break
        ids = list(get_ids(e))  # it will give us a random list of ALL the triplets' IDs having the entity from the pool
        if entities.loc[entities['entity']==e, 'stratum'].iloc[0] == 'Q4' and len(ids) > Q4_CAP:  # Q4_CAP = 40
            ids = random.sample(ids, Q4_CAP)
            # if an entity is very common (Q4 quantile) and there is more than 40 triplets' IDs of it, then we sample and leave only 40 of triplets containing it.

        new_ids = [i for i in ids if i not in selected]  # leave only new triplets' IDs to exclude duplicates IDs consisting two entities we have.
        remain = T_QUERY - len(selected)
        # if a new entity have less or equal to amount of IDs we need (max 800 and it gets fewer), then
        if len(new_ids) <= remain:
            selected.update(new_ids)  # add the new triplets' IDs.
            E_final.append(e)  # add the entity to the list
            skipped_full = 0
        else:
            skipped_full += 1
            # if the new entities were in the number of triplets that more, than we need, and it happened MAX_FULL_TRY, then
            if skipped_full > MAX_FULL_TRY and remain > 0:  # MAX_FULL_TRY = 2
                selected.update(new_ids[:remain])  # just add as many triplets as we need at the end of the list
                E_final.append(e)  # push back the entity!
                break  # enough is enough.

    return list(selected), E_final  # returns the Triplets IDs (800 by default) and list of entities that there were.
