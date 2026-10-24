from __future__ import annotations
import os
import json
import torch
import re
import pickle as pkl
from typing import Literal

import pandas as pd
from tqdm import tqdm

from my_project.config import (
    MODEL_DIR, MAX_SEQ_LENGTH, LOAD_IN_4BIT, TEMPERATURE, MAX_NEW_TOKENS,
    CHECKPOINT_STEP, KNOWLEDGE_CSV,
)
from my_project.knowledge    import compute_knowledge_scores
from my_project.gnn          import run_gnn


# ─────────────────────────────────────────────────────────────────────────────
# LLM query helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_tf(text: str, cid    : str):
    p = text.strip().lower()
    if re.fullmatch(r"true[.!?]*",  p): return 1
    if re.fullmatch(r"false[.!?]*", p): return 0
    return (-1, f"Bad response for {cid}: '{text}'")


def _build_opensource_querier(model_dir: str):
    """Returns a query_fn(prompt, triplet_id) -> int|tuple."""
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_dir,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=LOAD_IN_4BIT,
    )
    FastLanguageModel.for_inference(model)
    tokenizer = get_chat_template(tokenizer, chat_template="mistral") # chat_template="llama-3"

    SYSTEM = (
        'Evaluate the following statement based on your knowledge and '
        'respond only with "true" or "false". This is for research purposes only.'
    )

    def query_fn(prompt: str, cid: str):
        try:
            msgs = [
                {"role": "system",  "content": SYSTEM},
                {"role": "user",    "content": prompt},
            ]
            inputs = tokenizer.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True,
                return_tensors="pt"
            ).to("cuda")
            attention_mask = torch.ones_like(inputs)
            outs = model.generate(
                input_ids=inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                attention_mask=attention_mask,
                temperature=None if TEMPERATURE == 0 else TEMPERATURE,
                do_sample=TEMPERATURE > 0,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
            answer = tokenizer.decode(
                outs[0, inputs.shape[1]:], skip_special_tokens=True
            )
            return _parse_tf(answer, cid)
        except Exception as e:
            return (-1, f"Exception for {cid}: {repr(e)}")

    return query_fn

def _run_llm_on_prompts(
    df_prompts: pd.DataFrame,
    query_fn,
    checkpoint_dir: str,
    checkpoint_prefix: str = "llm",
) -> pd.DataFrame:
    """
    Queries query_fn on every row of df_prompts (columns: triplet_id, triplet_prompt).
    Supports resume from checkpoint.
    Returns df_prompts with an added 'tf_value' column.
    """
    save_path  = os.path.join(checkpoint_dir, f"{checkpoint_prefix}_results.pkl")
    error_path = os.path.join(checkpoint_dir, f"{checkpoint_prefix}_errors.pkl")

    df = df_prompts.copy()
    df["tf_value"] = pd.NA

    results, errors = [], []
    if os.path.exists(save_path) and os.path.exists(error_path):
        with open(save_path, "rb") as f: results = pkl.load(f)
        with open(error_path, "rb") as f: errors  = pkl.load(f)
        done = {cid for cid, _ in results}
        for i, row in df.iterrows():
            if row["triplet_id"] in done:
                val = next(v for c, v in results if c == row["triplet_id"])
                df.at[i, "tf_value"] = val

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="LLM query"):
        if not pd.isna(row["tf_value"]):
            continue
        cid = row["triplet_id"]
        res = query_fn(row["triplet_prompt"], cid)
        if isinstance(res, tuple) and res[0] == -1:
            results.append((cid, -1))
            errors.append({"id": cid, "error": res[1]})
            df.at[idx, "tf_value"] = -1
        else:
            results.append((cid, res))
            df.at[idx, "tf_value"] = res

        if (idx + 1) % CHECKPOINT_STEP == 0 or (idx + 1) == len(df):
            with open(save_path,  "wb") as f: pkl.dump(results, f)
            with open(error_path, "wb") as f: pkl.dump(errors,  f)

    df.to_csv(os.path.join(checkpoint_dir, f"{checkpoint_prefix}_results.csv"), index=False)
    return df