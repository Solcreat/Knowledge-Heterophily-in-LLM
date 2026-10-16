# src/my_project/gnn.py

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, SAGEConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

# ── HYPERPARAMETERS ─────────────────────────────────────────────────────────────
EMBEDDING_MODEL      = "all-MiniLM-L6-v2"
HIDDEN_DIM           = 64
LR                   = 0.05 # 0.01
EPOCHS               = 10000 # 100
TEST_SIZE            = 0.30     # fraction for (val+test)
VAL_FRACTION_OF_TEST = 0.50     # fraction of that to use as validation
SPLIT_RANDOM_STATE   = 42
DEVICE               = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


BEST_VAL_LOSS = float('inf')
BEST_MODEL_STATE = None
PATIENCE = 100  # how many epochs we give the model to improve its score
BAD_EPOCHS = 0  # counter for not-improving-epochs
MIN_LR = 1e-7  # the minimum learning rate before stopping
LR_DECAY_FACTOR = 0.25  # how many times we decrease LR

# Filenames (inside input_dir)
KG_VALUES_CSV        = "entity_kg_values.csv"
OUTPUT_PRED_CSV      = "entity_kg_predictions.csv"

# class GraphSAGE(torch.nn.Module):
#     def __init__(self, in_ch, hidden_ch):
#         super().__init__()
#         self.conv1 = SAGEConv(in_ch, hidden_ch)
#         self.conv2 = SAGEConv(hidden_ch, hidden_ch)
#         self.lin   = torch.nn.Linear(hidden_ch, 1)
#
#     def forward(self, x, edge_index):
#         x = F.relu(self.conv1(x, edge_index))
#         x = F.relu(self.conv2(x, edge_index))
#         return self.lin(x)

# our own layer of adaptible PageRank-passing
class GPRPropagation(MessagePassing):
    def __init__(self, K, alpha):
        super().__init__(aggr='add')
        self.K = K
        self.apha = alpha

        # initialize Gamma weights, according to classic Personalized Pagerank (PPR)
        init_gammas = alpha * ((1-alpha)**np.arange(K+1))
        init_gammas[-1] = (1-alpha)**K
        # make these weights learnable by NNetwork
        self.gamma = torch.nn.Parameter(torch.tensor(init_gammas, dtype=torch.float))
    def forward(self, x, edge_index):
        # Get a normalized simmetry adjacent matrix (like in GCN):
        edge_index, edge_weight = gcn_norm(
            edge_index, num_nodes=x.size(0), add_self_loops=True, dtype=x.dtype
        )
        # Conduct K diffusion steps and save temporary states H^k
        h_states = [x]
        h = x
        for k in range(self.K):
            # Message passing (multiply by normalized adjacent matrix)
            h = self.propagate(edge_index, x=h, edge_weight=edge_weight)
            h_states.append(h)
        # sum all the states to the learnable weights Gamma
        out = torch.zeros_like(x)
        for k in range(self.K + 1):
            out += self.gamma[k] * h_states[k]
        return out
    def message(self, x_j, edge_weight):
        # multiply neighbours' features by adjacent matrix's normalization coefficients
        return edge_weight.view(-1, 1) * x_j
# # the final GPR-GNN model:
class GPRGNN(torch.nn.Module):
    def __init__(self, in_ch, hidden_ch, K=10, alpha=0.1):
        super().__init__()
        # Feature Transformation (MLP)
        self.lin1 = torch.nn.Linear(in_ch, hidden_ch)
        self.lin2 = torch.nn.Linear(hidden_ch, 1)
        self.dropout = torch.nn.Dropout(p=0.2)

        # our configurated message passing layer
        self.prop = GPRPropagation(K=K, alpha=alpha)

    def forward(self, x, edge_index):
        x = self.dropout(x)
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        # to let process 64-dim semantic vectors
        x = self.prop(x, edge_index)
        out = self.lin2(x)
        return out

        
def run_gnn(input_dir):
    global BEST_VAL_LOSS, BEST_MODEL_STATE, BAD_EPOCHS
    # 1. load metadata.json
    with open(os.path.join(input_dir, "metadata.json")) as fp:
        meta = json.load(fp)

    # 2. load raw KG and knowledge‐value CSV
    kg_df      = pd.read_csv(meta["input_kg"]) 
    values_df  = pd.read_csv(os.path.join(input_dir, KG_VALUES_CSV))
    # 3. prepare triplet edges
    # rename for clarity
    df_edges = kg_df.rename(columns={'sub_label':'head','obj_label':'tail'})
    entities = pd.unique(df_edges[['head','tail']].values.ravel())
    ent2idx  = {e:i for i,e in enumerate(entities)}
    df_edges['head_idx'] = df_edges['head'].map(ent2idx)
    df_edges['tail_idx'] = df_edges['tail'].map(ent2idx)

    # 4. node features: embeddings of entity strings
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings  = embed_model.encode(list(entities), convert_to_tensor=True)

    # 5. build edge_index tensor
    edge_index = torch.tensor(
        [df_edges['head_idx'].tolist(), df_edges['tail_idx'].tolist()],
        dtype=torch.long
    ).to(DEVICE)

    # 6. prepare labels & masks
    # map kg_values to indices
    values_df = values_df.rename(columns={'entity':'entity_label'})
    values_df['idx'] = values_df['entity_label'].map(ent2idx)
    values_df = values_df.dropna(subset=['idx'])
    values_df['idx'] = values_df['idx'].astype(int)

    # train/val/test splits
    train_idx, val_test_idx = train_test_split(
        values_df['idx'], test_size=TEST_SIZE, random_state=SPLIT_RANDOM_STATE
    )
    val_idx, test_idx = train_test_split(
        val_test_idx, test_size=VAL_FRACTION_OF_TEST, random_state=SPLIT_RANDOM_STATE
    )

    num_nodes = len(entities)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask   = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask  = torch.zeros(num_nodes, dtype=torch.bool)
    labels     = torch.zeros(num_nodes, dtype=torch.float)

    for _, row in values_df.iterrows():
        labels[row['idx']] = row['kg_value']
    # train_mask[train_idx] = True
    # val_mask[val_idx] = True
        # test_mask[test_idx] = True
    for x in train_idx:
        train_mask[x] = True
    for y in val_idx:
        val_mask[y] = True
    for z in test_idx:
        test_mask[z] = True

    # 7. build PyG Data object
    data = Data(
        x         = embeddings.to(DEVICE),
        edge_index= edge_index,
        y         = labels.unsqueeze(1).to(DEVICE)
    )
    data.train_mask = train_mask
    data.val_mask   = val_mask
    data.test_mask  = test_mask

    # 8. model, optimizer
    # model = GraphSAGE(embeddings.shape[1], HIDDEN_DIM).to(DEVICE)
    model = GPRGNN(data.num_node_features, HIDDEN_DIM, K=10, alpha=0.1).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)

    # 9. training loop

    # cloning and detaching features from the inference mode letting the model to save them
    data.x = data.x.clone().detach()
    import copy
    for epoch in range(1, EPOCHS+1):
        model.train()
        opt.zero_grad()
        out = model(data.x, data.edge_index).squeeze()
        loss  = F.mse_loss(out[data.train_mask], data.y[data.train_mask].squeeze())
        loss.backward()
        opt.step()

        # if epoch % 10 == 0 or epoch==EPOCHS:
        #     model.eval()
        #     with torch.no_grad():
        #         val_out   = model(data.x, data.edge_index).squeeze()
        #         val_loss  = F.mse_loss(
        #             val_out[data.val_mask],
        #             data.y[data.val_mask].squeeze()
        #         )
        #     print(f"Epoch {epoch:03d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}")

        model.eval()
        with torch.no_grad():
            val_out = model(data.x, data.edge_index).squeeze()
            val_loss = F.mse_loss(val_out[data.val_mask], data.y[data.val_mask].squeeze())

        # new best score!
        if val_loss < BEST_VAL_LOSS:
            BEST_VAL_LOSS = val_loss
            BEST_MODEL_STATE = copy.deepcopy(model.state_dict())
            BAD_EPOCHS = 0  # now we give the model another 20 epochs to improve!
        else:
            BAD_EPOCHS += 1

        if epoch % 10 == 0:
            current_lr = opt.param_groups[0]['lr']
            print(f"Epoch {epoch:05d} | LR: {current_lr:.8f} | Train Loss: {loss.item():.7f} | Val Loss: {val_loss:.7f}")
        # if model has not been improving its score for too long
        if BAD_EPOCHS >= PATIENCE:
            current_lr = opt.param_groups[0]['lr']
            # the condition for full stop
            if current_lr < MIN_LR:
                print(f"🛑 Learning rate reached minimum. Early stopping at epoch {epoch}!")
                break

            print(f"📉 Val Loss hasn't improved for {PATIENCE} epochs. Rolling back weights and reducing LR!")
            # weight rollback and decreasing LR:
            for param_group in opt.param_groups:
                param_group['lr'] *= LR_DECAY_FACTOR
            model.load_state_dict(BEST_MODEL_STATE)
            BAD_EPOCHS = 0
    gammas = model.prop.gamma.detach().cpu().numpy()
    if BEST_MODEL_STATE is not None:
        model.load_state_dict(BEST_MODEL_STATE)
        print(f"🏆 Final GPR-GNN Model loaded with Best Val Loss: {BEST_VAL_LOSS:.7f}")
    print("__________________________________")
    print("LEARNED GPR-GNN GAMMA WEIGHTS:")
    for k, g in enumerate(gammas):
        print(f"  Step {k}: {g:.4f}")
    print("__________________________________")
    # 10. predict for all nodes
    model.eval()
    with torch.no_grad():
        preds = model(data.x, data.edge_index).squeeze().cpu().numpy()

    # 11. build result DataFrame
    result = pd.DataFrame({
        'entity_label': entities,
        'predicted_kg_value': preds,
    })
    # merge in true values & assign split
    result = result.merge(
        values_df[['entity_label','kg_value','idx']],
        on='entity_label',
        how='left'
    )
    def which_split(i):
        if i in train_idx.values: return 'train'
        if i in val_idx.values:   return 'valid'
        if i in test_idx.values:  return 'test'
        return 'none'
    result['split'] = result['idx'].apply(
        lambda x: which_split(x) if pd.notna(x) else 'none'
    )
    result = result[['entity_label','predicted_kg_value','kg_value','split']]

    # 12. save
    out_path = os.path.join(input_dir, OUTPUT_PRED_CSV)
    result.to_csv(out_path, index=False)
    print(f"Saved predictions → {out_path}")
