"""
P2-2: GNN 嵌入随数据规模演化
在 CEPDB 不同样本量下训练 GNN 并提取图嵌入
计算嵌入与 Morgan fingerprint 的 CKA 相似度
绘制样本量 vs CKA 相似度曲线
"""
import os, sys, json, random, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

os.environ['RDKIT_SILENCE'] = '1'
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import AllChem

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool, global_max_pool, global_add_pool

FP_DIM = 512
SEED = 42
RESULTS_FILE = 'external_results/embedding_evolution_results.json'
os.makedirs('external_results', exist_ok=True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def smiles_to_graph(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        atom_features = []
        for atom in mol.GetAtoms():
            atomic_num = atom.GetAtomicNum(); degree = atom.GetDegree()
            formal_charge = atom.GetFormalCharge(); is_aromatic = int(atom.GetIsAromatic())
            is_in_ring = int(atom.IsInRing())
            try:
                hybridization = int(atom.GetHybridization()); num_h = atom.GetTotalNumHs()
                valence = atom.GetTotalValence(); r3 = int(atom.IsInRingSize(3))
                r4 = int(atom.IsInRingSize(4)); r5 = int(atom.IsInRingSize(5))
                r6 = int(atom.IsInRingSize(6))
            except: hybridization = num_h = valence = r3 = r4 = r5 = r6 = 0
            common_atoms = [1, 6, 7, 8, 9, 15, 16, 17, 35]
            feat = [atomic_num/100.0, degree/6.0, formal_charge/8.0, num_h/4.0, valence/8.0,
                    is_aromatic, is_in_ring, r3, r4, r5, r6] \
                + [int(atomic_num == a) for a in common_atoms] \
                + [int(degree == d) for d in range(5)] \
                + [int(hybridization == h) for h in range(1, 6)]
            atom_features.append(feat)
        edge_indices = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_indices += [[i, j], [j, i]]
        if not edge_indices: return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=FP_DIM)
        fp_tensor = torch.tensor(np.array(fp, dtype=np.float32)).unsqueeze(0)
        x = torch.tensor(atom_features, dtype=torch.float)
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        return Data(x=x, edge_index=edge_index, fp=fp_tensor)
    except:
        return None

class HighPCERegressorV3(nn.Module):
    def __init__(self, in_channels=30, hidden=128, fp_dim=FP_DIM, dropout=0.3):
        super().__init__()
        self.gcn1 = GCNConv(in_channels, hidden)
        self.gcn2 = GCNConv(hidden, hidden)
        self.gat1 = GATConv(in_channels, hidden//4, heads=4, dropout=dropout)
        self.gat2 = GATConv(hidden, hidden, heads=1, dropout=dropout)
        self.sage1 = SAGEConv(in_channels, hidden)
        self.sage2 = SAGEConv(hidden, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.bn3 = nn.BatchNorm1d(hidden)
        self.fp_encoder = nn.Sequential(
            nn.Linear(fp_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(),
        )
        fused_dim = hidden * 9 + 128
        self.regressor = nn.Sequential(
            nn.Linear(fused_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x_gcn = F.relu(self.bn1(self.gcn1(x, edge_index)))
        x_gcn = F.relu(self.gcn2(x_gcn, edge_index))
        x_gat = F.relu(self.gat1(x, edge_index))
        x_gat = F.relu(self.bn2(self.gat2(x_gat, edge_index)))
        x_sage = F.relu(self.sage1(x, edge_index))
        x_sage = F.relu(self.bn3(self.sage2(x_sage, edge_index)))
        def pool3(h):
            return torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch), global_add_pool(h, batch)], dim=1)
        g = torch.cat([pool3(x_gcn), pool3(x_gat), pool3(x_sage)], dim=1)
        fp_feat = self.fp_encoder(data.fp)
        g = torch.cat([g, fp_feat], dim=1)
        return self.regressor(g).squeeze(1), g  # Return both prediction and embedding

    def extract_embedding(self, data):
        """Extract graph embedding without prediction head."""
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x_gcn = F.relu(self.bn1(self.gcn1(x, edge_index)))
        x_gcn = F.relu(self.gcn2(x_gcn, edge_index))
        x_gat = F.relu(self.gat1(x, edge_index))
        x_gat = F.relu(self.bn2(self.gat2(x_gat, edge_index)))
        x_sage = F.relu(self.sage1(x, edge_index))
        x_sage = F.relu(self.bn3(self.sage2(x_sage, edge_index)))
        def pool3(h):
            return torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch), global_add_pool(h, batch)], dim=1)
        g = torch.cat([pool3(x_gcn), pool3(x_gat), pool3(x_sage)], dim=1)
        fp_feat = self.fp_encoder(data.fp)
        return torch.cat([g, fp_feat], dim=1)


def compute_cka(X, Y):
    """Linear CKA similarity."""
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    XX = X @ X.T
    YY = Y @ Y.T
    hsic = (XX * YY).sum()
    sqrt_hsic_x = (XX * XX).sum() ** 0.5
    sqrt_hsic_y = (YY * YY).sum() ** 0.5
    if sqrt_hsic_x * sqrt_hsic_y == 0:
        return 0.0
    return (hsic / (sqrt_hsic_x * sqrt_hsic_y)).item()


print("="*60)
print("P2-2: GNN Embedding Evolution with Data Size")
print("="*60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Load CEPDB data
df = pd.read_csv('/tmp/CEPDB_25000.csv')
df = df[df['pce'] > 0.01].copy().reset_index(drop=True)
df_high = df[df['pce'] > 3.0].copy().reset_index(drop=True)
print(f"CEPDB high-PCE: {len(df_high)}")

# Build graphs (cap at 2000 for consistency)
graphs = []
for i, (_, row) in enumerate(df_high.iterrows()):
    if i >= 2000:
        break
    g = smiles_to_graph(row['SMILES_str'])
    if g is not None:
        g.y = torch.tensor([float(row['pce'])], dtype=torch.float)
        graphs.append(g)
print(f"Graphs: {len(graphs)}")

# Fixed 3-fold split
N_FOLDS = 3
split_size = len(graphs) // N_FOLDS
indices = np.random.permutation(len(graphs))
fold_splits = []
for f in range(N_FOLDS):
    test_idx = indices[f*split_size:(f+1)*split_size] if f < N_FOLDS-1 else indices[f*split_size:]
    train_idx = np.concatenate([indices[:f*split_size], indices[(f+1)*split_size:]])
    fold_splits.append((train_idx, test_idx))

N_VALUES = [250, 500, 1000, 2000]
in_dim = graphs[0].x.shape[1]

results = {}

for n_train in N_VALUES:
    print(f"\n--- n={n_train} ---")
    fold_cka = []
    fold_r2 = []

    for fold, (train_idx, test_idx) in enumerate(fold_splits):
        actual_n = min(n_train, len(train_idx))
        fold_train = [graphs[i] for i in train_idx[:actual_n]]
        fold_test = [graphs[i] for i in test_idx]
        val_size = max(1, len(fold_train) // 5)
        fold_val = fold_train[-val_size:]
        fold_train_sub = fold_train[:-val_size]

        train_loader = GeoDataLoader(fold_train_sub, batch_size=64, shuffle=True)
        val_loader = GeoDataLoader(fold_val, batch_size=64)
        test_loader = GeoDataLoader(fold_test, batch_size=64)

        # Train model
        set_seed(SEED)
        model = HighPCERegressorV3(in_channels=in_dim).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.HuberLoss(delta=1.0)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

        best_val_mae = float('inf')
        patience_counter = 0
        for epoch in range(1, 101):
            model.train()
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                pred, _ = model(batch)
                loss = criterion(pred, batch.y.view(-1))
                loss.backward()
                optimizer.step()
            model.eval()
            preds, targets = [], []
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    pred, _ = model(batch)
                    preds.extend(pred.cpu().numpy())
                    targets.extend(batch.y.view(-1).cpu().numpy())
            val_mae = mean_absolute_error(targets, preds)
            scheduler.step(val_mae)
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= 15:
                break

        # Test R²
        model.eval()
        emb_list = []
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                pred, embed = model(batch)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(batch.y.view(-1).cpu().numpy())
                emb_list.append(embed.cpu())
        r2 = r2_score(all_targets, all_preds)
        fold_r2.append(r2)

        # Extract embeddings + compute fingerprints for CKA
        gnn_emb = torch.cat(emb_list, dim=0).numpy()

        fps_list = []
        for i in test_idx:
            smi = df_high.iloc[i]['SMILES_str']
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=FP_DIM)
                fps_list.append(np.array(fp, dtype=np.float32))
        fps_arr = np.stack(fps_list)

        cka = compute_cka(torch.tensor(gnn_emb), torch.tensor(fps_arr))
        fold_cka.append(cka)
        print(f"  fold={fold}: R²={r2:.4f}, CKA={cka:.4f}")

    results[str(n_train)] = {
        'r2_mean': float(np.mean(fold_r2)),
        'r2_std': float(np.std(fold_r2)),
        'cka_mean': float(np.mean(fold_cka)),
        'cka_std': float(np.std(fold_cka)),
    }
    print(f"  >>> n={n_train}: R²={np.mean(fold_r2):.4f}±{np.std(fold_r2):.4f}, CKA={np.mean(fold_cka):.4f}±{np.std(fold_cka):.4f}")

# Save
with open(RESULTS_FILE, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {RESULTS_FILE}")

# Print trend
print(f"\n{'='*50}")
print(f"CKA Similarity Trend: GNN Embedding vs Morgan Fingerprint")
print(f"{'='*50}")
for n in N_VALUES:
    r = results[str(n)]
    print(f"  n={n:4d}: CKA={r['cka_mean']:.4f}±{r['cka_std']:.4f}")
print("Done!")
