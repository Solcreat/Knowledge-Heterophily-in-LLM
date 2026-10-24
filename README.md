# Knowledge Heterophily in Large Language Models

This repository contains all the code, experimental results, training logs, associated files and even some .safetensors files for the models that were fine-tuned using modified GPR-GNN! The paper explaining and proving all of this is not published yet.

### Abstract
---
# IN SHORT what we did
### Mended the parts that fell short after running.
### Introduced Normalized Dirichlet Energy and rifts (sharp changes of knowledge) statistics directly measuring knowledge disparities (heterophily) on the graph.
### Replaced the baseline GraphSAGE aggregator with a Generalized PageRank GNN (GPR-GNN) to capture high-frequency knowledge boundaries.
### Resolved evaluation set degeneracy issues.
---


# Overview

The pipeline has two  parts:

---
## Repository Structure
```
data/
│    ├── raw/        # Original datasets and templates
│
my_project/
│   ├── config.py               # All hyperparameters and paths
│   ├── budget_selection.py     # Anchor selection + ignorance-guided selection
|   ├── heterophily.py          # Computing homophily/heterophily metrics when LLM was already queried. 
│   ├── homophily.py            # The LLM query helpers
│   ├── finetune.py             # Querying LLM, Training GPR-GNN, Fine-tuning the LLM on ignorance-selected triplets and computing the Generalization Gain.
│   ├── gnn.py                  # GPR-GNN training (with all hyperparameters) + prediction.
│   ├── knowledge.py            # Computing the knowledge scores for entities. Creating entity_kg_values.csv
│   ├── selection.py            # Helpers for creating {dataset}_templates.csv
│   ├── prompting.py            # Even more helpers for creating {dataset}_templates.csv
│   └── io.py / preprocess.py   # CSV I/O and column unification 
│
scripts/
│   ├── run_dataset.py          # building prompts
│   ├── run_finetune.py         # For running the process of querying LLM, Training GPR-GNN, Fine-tuning the LLM on ignorance-selected triplets and computing the Generalization Gain.
│   ├── run_heterophily.py      # Calculate the Dirichlet energy (real and permutated) and count all the rifts (sharp changes of knowledge).
│   ├── run_gnn.py              # train GNN or predict by it only
│   └── run_knowledge.py        # compute entity knowledge scores only
│

```

---
# Installation and running all the project in a notebook

### Cell 1 (dependencies):
```
!pip install transformers unsloth sentence-transformers torch-geometric networkx
!pip install "sentence-transformers==2.7.0"
```

### Cell 2 (models LLama-8B-4Bit and Mistral7B):
```
import os
from huggingface_hub import snapshot_download

TOKEN = "{your_Hugging_Face_Token}"
print("Downloading the configs...")
snapshot_download(
    repo_id="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    local_dir="./llama_local",
    token=TOKEN,
    ignore_patterns=["*.safetensors"]
)
print("Done! ✅ LLama's configs are here!")

!apt-get install -y aria2 > /dev/null

!aria2c --header="Authorization: Bearer {your_Hugging_Face_Token}" \
        -c -x 16 -s 16 -k 1M \
        https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit/resolve/main/model.safetensors \
        -d ./llama_local -o model.safetensors

print("The LLama's tensors have downloaded ✅")

print("Downloading the configs...")
snapshot_download(
    repo_id="unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    local_dir="./Mistral_local",
    token=TOKEN,
    ignore_patterns=["*.safetensors"]
)
print("Done! ✅ Mistral's Configs are here!")

!aria2c --header="Authorization: Bearer {your_Hugging_Face_Token}" \
        -c -x 16 -s 16 -k 1M \
        https://huggingface.co/unsloth/mistral-7b-instruct-v0.3-bnb-4bit/resolve/main/model.safetensors \
        -d ./Mistral_local -o model.safetensors

print("The Mistral's tensors have downloaded ✅")
```

### Cell 3 (to prevent any discrepancies):
```
!pip uninstall -y torchcodec
```

### Cell 4 (for further appropriate running):
```
import sys
print(sys.executable) # you have to copy the path it will give you
```

### Cell 5 (all the pipeline for all Datasets for the model in `my_project/config.py` (REMEMBER TO CHANGE THE `chat_template` IN `my_project/homophily.py` AND `my_project/finetune.py`):
```
%%bash
PY={you pass here exactly what printed the cell 4}
DATASETS="codex pharmkg trex"

for DS in $DATASETS; do
    echo "====================================================="
    echo "🚀 STARTING LOCAL PIPELINE FOR DATASET: $DS"
    echo "====================================================="

    $PY scripts/run_dataset.py --input-dir data/raw/$DS --output-base data/processed > log_dataset_$DS.txt 2>&1
   

    $PY scripts/run_finetune.py --input-dir data/processed/$DS --output-dir data/processed/$DS/finetune > log_finetune_$DS.txt 2>&1

    mkdir -p /content/Paper_Results/$DS

    rm -rf data/processed/$DS/finetune/checkpoint-*
    rm -rf data/processed/$DS/finetune/finetuned_model/checkpoint-*

    cp -r data/processed/$DS/finetune /content/Paper_Results/$DS/


    echo "✅ DATASET $DS SUCCESSFULLY COMPLETED!"
done
```
### Cell 6 (to archive your gained results without large files like models' weights):
```
!zip -r /content/scientific_results.zip /content/Paper_Results -x "*.safetensors" -x "*.bin" -x "*.pt"
```
To save your work, you need to download the `/content/scientific_results.zip` manually from the notebook data storage.
### Cell 7 (if you want to see the results immediately):
```
!tail -n 30 log_finetune_codex.txt
!tail -n 30 log_finetune_trex.txt
!tail -n 30 log_finetune_pharmkg.txt
```

### Cell 8 (if you want to upload the models' weights and associated files to your Hugging Face account):
```
from huggingface_hub import HfApi

api = HfApi()
api.create_repo(
    repo_id="{Your HF Name}/codex-adapters",
    repo_type="model",
    private=True,
    exist_ok=True
)
api.create_repo(
    repo_id="{Your HF Name}/trex-adapters",
    repo_type="model",
    private=True,
    exist_ok=True
)
api.create_repo(
    repo_id="{Your HF Name}/pharmkg-adapters",
    repo_type="model",
    private=True,
    exist_ok=True
)

api.upload_folder(
    folder_path="/content/Paper_Results/codex/finetune/finetuned_model",
    repo_id="{Your HF Name}/codex-adapters",
    repo_type="model"
)
api.upload_folder(
    folder_path="/content/Paper_Results/pharmkg/finetune/finetuned_model",
    repo_id="{Your HF Name}/trex-adapters",
    repo_type="model"
)
api.upload_folder(
    folder_path="/content/Paper_Results/trex/finetune/finetuned_model",
    repo_id="{Your HF Name}/pharmkg-adapters",
    repo_type="model"
)
```
After pushing to Hugging Face, you can easily download all the files through your local terminal:
```
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='{Your HF Name}/{dataset name}-adapters', local_dir=r'{the directory where you want to download it}')"
```
---
# What happens before, during and after the fine-tuning stage

1. **Anchor selection** — Degree-stratified sampling based on 4 quantiles for prompting LLMs.
2. **LLM querying** — Anchor triplets are queried by LLMs for true/false (know or not) labels.
3. **GNN training** — GPR-GNN trained on anchor entity scores; predicts scores for all entities to choose the least knowledgeable entities.
4. **Ignorance-guided selection** — Selects the least knowledgeable entities with the most connections.
5. **Fine-tuning** — `UnslothTrainer` runs on the combined anchor + selected triplet set
6. **Evaluation** — Fine-tuned model is queried on a held-out 2% test set; results saved to `evaluation_summary.json`

---
## Configuration
---



# The datasets

We evaluate both homophily and heterophily and also fine-tune the models on three datasets:

### T-Rex: https://hadyelsahar.github.io/t-rex/, General (Wikipedia) 46 891 entities and 193 781 triplets.

### CoDEx-S: https://github.com/tsafavi/codex, General (Wikidata/Wikipedia) 2 034 entities and 36 543 triplets.

### PharmKG8K: https://github.com/MindRank-Biotech/PharmKG, Genes, diseases, Pharmaceuticals 6 877 entities and 99 572 triplets.

Raw files go in `data/raw/{dataset_name}`. Each dataset folder needs a `{dataset_name}_templates.csv` for building prompts.

# Acknowledgements
> This repository is being build upon and continues to extend the foundational codebase from the paper:
"Knowledge Homophily in Large Language Models" (Utkarsh Sahu, Zhisheng Qi et al., WSDM '26) [[Their Paper]](https://arxiv.org/abs/2509.23773), [[GitHub Repository]](https://github.com/utkarshxsahu/kgc).
> We sincerely thank the authors and co-authors for open-sourcing their base implementation. Many parts of their project were built perfectly; the ones we considered not so ideal, we meticulously analyzed, fixed up and refined!
