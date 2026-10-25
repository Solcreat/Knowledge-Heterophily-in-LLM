# Knowledge Heterophily in Large Language Models

### This repository contains all the code, experimental results, training logs, associated files and even some .safetensors files for the models that were fine-tuned using modified GPR-GNN! The paper explaining and proving all of this is being peer-reviewed.

# Abstract (briefly)
> **Research related to the structure of LLM's knowledge is still in its early stages and previous studies have argued that it is strictly homophilic. We tested this hypothesis. We adapted the concept of normalized Dirichlet energy and introduced a new concept of “rift” to demonstrate pronounced local heterophily. We found that the percentage of edges with a knowledge difference of more than 0.4 units among 800 interconnected entities averages 26.97%. To more accurately predict the knowledge of unlabeled entities, we adapted a GPR Graph Neural Network, which enabled us to achieve a 23.98% relative increase in the knowledge of “Mistral-7B” and “Llama3-8B.”**

---
# IN SHORT what we did
### - Mended the parts of previous homophily-related work that fell short after running.
### - Introduced Normalized Dirichlet Energy and "rifts" (sharp changes of knowledge) statistics directly measuring knowledge disparities (heterophily) on the graph.
### - Replaced the baseline GraphSAGE aggregator that does not consider heterophily with a Generalized PageRank Graph Neural Network (GPR-GNN) to capture high-frequency knowledge boundaries.
### - Resolved inference and testing sets degeneracy issues.
### - Gained metrics that beat standard GraphSAGE by a 23.98% relative increase in the knowledge!
---


# Overview

The pipeline has three parts. The first one is pretty small, the second one is the heart of the project, and the third is for obtaining the homophily/heterophily metrics.
How they go:
1) Preparing all the datasets and building the large table of all possible prompts (it will create `data/processed` directory).
2) Selecting anchor prompts. Inferencing LLM's on those anchor prompts whether it knows a fact or not. Train GPR-GNN to predict the knowledge of unlabeled by LLM entities. Select the entities which a model knows least and which have the most number of connection. Fine-tune LLM on them + anchor prompts. Evaluate the knowledge on current Dataset now! (You can change the GNN to compare, for example GraphSAGE).
3) Evaluate the normalized Dirichlet energy and metrics directly connected to rifts (sharp changes of knowledge)!
   
---
## Repository Structure
```
data/
│  └── raw/                    # Original datasets and templates.
│
my_project/
│  ├── config.py               # All hyperparameters and paths.
│  ├── budget_selection.py     # Anchor selection + ignorance-guided selection.
|  ├── heterophily.py          # Computing homophily/heterophily metrics when LLM was already queried. 
│  ├── homophily.py            # The LLM query helpers.
│  ├── finetune.py             # Querying LLM, Training GPR-GNN, Fine-tuning the LLM on ignorance-selected triplets and computing the knowledge score.
│  ├── gnn.py                  # GPR-GNN training (with all hyperparameters) + prediction.
│  ├── knowledge.py            # Computing the knowledge scores for entities. Creating entity_kg_values.csv.
│  ├── selection.py            # Helpers for creating {dataset}_templates.csv.
│  ├── prompting.py            # Even more helpers for creating {dataset}_templates.csv.
│  └── io.py / preprocess.py   # CSV I/O and column unification.
│
scripts/
   ├── run_dataset.py          # building prompts.
   ├── run_finetune.py         # For running the process of querying LLM, Training GPR-GNN, Fine-tuning the LLM on ignorance-selected triplets and computing the knowledge score.
   ├── run_heterophily.py      # Calculate the Dirichlet energy (real and permutated) and count all the connected to rifts metrics.
   ├── run_gnn.py              # train GNN or predict by it only.
   └── run_knowledge.py        # compute entity knowledge scores only.
```
---
# Installation and running all the project in a notebook

### Cell 1 (dependencies):
```
!pip install transformers unsloth sentence-transformers torch-geometric networkx
!pip install "sentence-transformers==2.7.0"
```

### Cell 2 (models Llama-3.1-8B-Instruct-bnb-4bit and mistral-7b-instruct-v0.3-bnb-4bit):
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

### Cell 5 (all the pipeline for all Datasets for the model in `my_project/config.py` (REMEMBER TO CHANGE THE `chat_template` IN `my_project/homophily.py` AND `my_project/finetune.py`)):
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
### Cell 8:
This script was used for the paper and expects that you have tested each model (for example Mistral and LLama) at least twice to evaluate the normalized Dirichlet energy and metrics directly connected to rifts on different anchor prompt inference. But if you didn't, you can just pass in pairs same directories like `-rp resLlamaCodex resLlamaCodex -rp resLlamaTrex resLlamaTrex ...`
```
%%bash
$PY scripts/run_heterophily.py --help (to see what you need to pass more clearly)
$PY scripts/run_heterophily.py --codex-dir data/raw/codex --trex-dir data/raw/trex --pharmkg-dir data/raw/pharmkg -rp "LLamaCodex1" "LLamaCodex2" -rp "LLamaTrex1" "LLamaTrex2" -rp "LLamaPharmkg1" "LLamaPharmkg2" -rp "MistralCodex1" "MistralCodex2" -rp "MistralTrex1" "MistralTrex2" -rp "MistralPharmkg1" "MistralPharmkg2"  
```
A raw example like we used it:
```
python scripts/run_heterophily.py --codex-dir data/raw/codex --trex-dir data/raw/trex --pharmkg-dir data/raw/pharmkg -rp "my_project/LLama8BCodex-S GPR 91.23" "my_project/LLama8BCodex-S SAGE 74.11" -rp "my_project/LLama8BTrex GPR 95.48" "my_project/LLama8BTrex SAGE 93.55" -rp "my_project/LLama8BPharmKG GPR 15.62" "my_project/LLama8BPharmKG SAGE 7.94" -rp "my_project/Mistral7BCodex-S GPR 99.18" "my_project/Mistral7BCodex-S SAGE 84.38" -rp "my_project/Mistral7BTrex GPR 95.69" "my_project/Mistral7BTrex SAGE 96.05" -rp "my_project/Mistral7BLPharmKG GPR 99.25" "my_project/Mistral7BPharmKG SAGE 94.63"
```
### Cell 9 (if you want to upload the models' weights and associated files to your Hugging Face account):
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

### Building Anchor prompts - Degree-stratified sampling based on 4 quantiles for prompting LLMs.
### LLM being queried by Anchor Prompts - Anchor prompts are queried for true/false (know or not) labels.
### GNN training - GPR-GNN trained on anchor entity scores; predicts scores for all unlabeled by the model entities.
### Ignorance-guided selection - Selects the least knowledgeable entities with the most connections.
### Fine-tuning - `UnslothTrainer` runs on the combined anchor + selected triplet set
### Evaluation - Once the model is fine-tuned, it is being queried on a test set from the graph + anchor prompts. From here we gain knowledge score!
---
# Configuration

You can change precisely how GNN is training and how LLMs are being fine-tuned. For GNN you will need `my_project/gnn.py`. There:
```
EMBEDDING_MODEL      = "all-MiniLM-L6-v2" - the texting embedding model. It process the text semantics through 384-dimensional vectors for GNN.
HIDDEN_DIM           = 64                 - this is the number of vector components after going through first linear layer in forward message passing.
LR                   = 0.05 # 0.01        - the starting learning rate for GPR-GNN
EPOCHS               = 10000 # 100        - for how long will your or GPR-GNN will train.
TEST_SIZE            = 0.30               - for testing GNN
VAL_FRACTION_OF_TEST = 0.50               - for validating GNN
SPLIT_RANDOM_STATE   = 42                 - a random state for splitting the train and test.
BEST_VAL_LOSS = float('inf')              - for saving the best (the lowest) loss for GNN.
BEST_MODEL_STATE = None                   - for saving the best weights of GNN.
PATIENCE = 100                            - how many epochs we give the model to improve its score
BAD_EPOCHS = 0                            - counter for not-improving-epochs
MIN_LR = 1e-7                             - the minimum learning rate before stopping
LR_DECAY_FACTOR = 0.25                    - how many times (the coefficient) we decrease LR
```
For fine-tuning the LLM in `my_project/config.py`:
```
MODEL_DIR        = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit" # "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit" # "unsloth/Llama-3.2-3B-Instruct-bnb-4bit" - the model used for querying 
MAX_NEW_TOKENS   = 1 - how many tokens are produced by LLM for an answer (we recommend to leave it alone).
FINETUNE_MODEL_DIR     = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"  #  "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit" # "unsloth/Llama-3.2-3B-Instruct-bnb-4bit" - the model for fine-tuning (of course it should be equal to MODEL_DIR).
FINETUNE_LORA_R        = 16 - the rank of the matrix for LORA fine-tuning
FINETUNE_LORA_ALPHA    = 16 - a scaling factor that controls the magnitude and strength of the low-rank adapter updates
FINETUNE_TARGET_MODULES = [ - what layers of LLM are being fine-tuned - the lower - the cheaper both computing and results. 
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
FINETUNE_EPOCHS        = 3 - the number of full fine-tuning cycles (3 is the standard)
FINETUNE_BATCH_SIZE    = 4 - the number of training examples processed by the model for once
FINETUNE_GRAD_ACCUM    = 4 - the number of times we process FINETUNE_BATCH_SIZE of batches before updating the weights.
FINETUNE_LR            = 2e-4 - the learning rate
```
And also you can change the whole GPR GNN with your own GNN! Just comment all `class GPRPropagation(MessagePassing)`, `class GPRGNN(torch.nn.Module)` and all the training loop. After this, change
```
model = GPRGNN(data.num_node_features, HIDDEN_DIM, K=10, alpha=0.1).to(DEVICE)
```
by your own GNN class matching `MessagePassing`, construct the `training loop` to match the logic of the project, start the `cell 5` from `Installation` and see the **knowledge score!**

---



# The datasets

We evaluate both **homophily** and **heterophily** and also fine-tune the models on three datasets:

### T-Rex: https://hadyelsahar.github.io/t-rex/, General (Wikipedia) 46 891 entities and 193 781 triplets.

### CoDEx-S: https://github.com/tsafavi/codex, General (Wikidata/Wikipedia) 2 034 entities and 36 543 triplets.

### PharmKG8K: https://github.com/MindRank-Biotech/PharmKG, Genes, diseases, Pharmaceuticals 6 877 entities and 99 572 triplets.

Raw files go in `data/raw/{dataset_name}`. Each dataset folder needs a `{dataset_name}_templates.csv` for building prompts.

# Acknowledgements
> This repository is being build upon and continues to extend the foundational codebase from the paper:
"Knowledge Homophily in Large Language Models" (Utkarsh Sahu, Zhisheng Qi et al., WSDM '26) [[Their Paper]](https://arxiv.org/abs/2509.23773), [[GitHub Repository]](https://github.com/utkarshxsahu/kgc).
> We sincerely thank the authors and co-authors for open-sourcing their base implementation. Many parts of their project were built perfectly; the ones we considered not so ideal, we meticulously analyzed, fixed up and refined!
