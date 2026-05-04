"""
P1-3: CEPDB GNN 多随机种子验证
在相变点附近的关键样本量 (n=1,500, 2,000, 3,000) 跑 4 个随机种子
每个种子内做 3-fold CV，共 12 个 runs / n
"""
import os, sys, json, random
import numpy as np
import pandas as pd
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
            atomic_num = atom.GetAtomicNum()
            degree = atom.GetDegree()
            formal_charge = atom.GetFormalCharge()
            is_aromatic = int(atom.GetIsAromatic())
            is_in_ring = int(atom.IsInRing())
            try:
                hybridization = int(atom.GetHybridization())
                num_h = atom.GetTotalNumHs()
                valence = atom.GetTotalValence()
                r3 = int(atom.IsInRingSize(3))
                r4 = int(atom.IsInRingSize(4))
                r5 = int(atom.IsInRingSize(5))
                r6 = int(atom.IsInRingSize(6))
            except:
                hybridization = num_h = valence = r3 = r4 = r5 = r6 = 0
            common_atoms = [1, 6, 7, 8, 9, 15, 16, 17, 35]
            feat = [
                atomic_num / 100.0, degree / 6.0, formal_charge / 8.0,
                num_h / 4.0, valence / 8.0, is_aromatic, is_in_ring,
                r3, r4, r5, r6,
            ] + [int(atomic_num == a) for a in common_atoms] \
              + [int(degree == d) for d in range(5)] \
              + [int(hybridization == h) for h in range(1, 6)]
            atom_features.append(feat)
        edge_indices = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_indices += [[i, j], [j, i]]
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
        return self.regressor(g).squeeze(1)

# ========== MAIN ==========
print("="*50)
print("P1-3: CEPDB GNN Multi-Seed Verification")
print("="*50)

df = pd.read_csv('/tmp/CEPDB_25000.csv')
df = df[df['pce'] > 0.01].copy().reset_index(drop=True)
df_high = df[df['pce'] > 3.0].copy().reset_index(drop=True)
print(f"High-PCE samples: {len(df_high)}")

graphs = []
for i, (_, row) in enumerate(df_high.iterrows()):
    if i >= 2000:
        break
    g = smiles_to_graph(row['SMILES_str'])
    if g is not None:
        g.y = torch.tensor([float(row['pce'])], dtype=torch.float)
        graphs.append(g)
print(f"Graphs: {len(graphs)}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

N_VALUES = [1500, 2000, 3000]
SEEDS = [42, 123, 456, 789]
in_dim = graphs[0].x.shape[1]

results = {}

for n_train in N_VALUES:
    print(f"\n{'='*40}")
    print(f"n = {n_train}")
    print(f"{'='*40}")
    per_seed = {}

    for seed in SEEDS:
        set_seed(seed)
        fold_r2 = []

        for fold in range(3):
            split_idx = len(graphs) // 5
            test_graphs = graphs[fold*split_idx:(fold+1)*split_idx]
            train_graphs_all = graphs[:fold*split_idx] + graphs[(fold+1)*split_idx:]
            train_graphs = train_graphs_all[:n_train]
            val_size = max(1, len(train_graphs) // 5)
            val_graphs = train_graphs[-val_size:]
            train_graphs_sub = train_graphs[:-val_size]

            train_loader = GeoDataLoader(train_graphs_sub, batch_size=64, shuffle=True)
            val_loader = GeoDataLoader(val_graphs, batch_size=64)
            test_loader = GeoDataLoader(test_graphs, batch_size=64)

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
                    pred = model(batch)
                    loss = criterion(pred, batch.y.view(-1))
                    loss.backward()
                    optimizer.step()

                model.eval()
                preds, targets = [], []
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(device)
                        pred = model(batch)
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

            model.eval()
            preds, targets = [], []
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    pred = model(batch)
                    preds.extend(pred.cpu().numpy())
                    targets.extend(batch.y.view(-1).cpu().numpy())
            r2 = r2_score(targets, preds)
            fold_r2.append(r2)
            print(f"  seed={seed:3d} fold={fold}  R²={r2:.4f}")

        per_seed[str(seed)] = {
            'r2_mean': float(np.mean(fold_r2)),
            'r2_std': float(np.std(fold_r2)),
            'r2_per_fold': [float(v) for v in fold_r2],
        }
        print(f"  >>> seed={seed:3d}: R²={np.mean(fold_r2):.4f}±{np.std(fold_r2):.4f}")

    # Aggregate across all seeds
    all_r2 = []
    for s in SEEDS:
        all_r2.extend(per_seed[str(s)]['r2_per_fold'])
    results[str(n_train)] = {
        'r2_mean': float(np.mean(all_r2)),
        'r2_std': float(np.std(all_r2)),
        'n_seeds': len(SEEDS),
        'n_folds_per_seed': 3,
        'per_seed': per_seed,
    }
    print(f"\n  *** n={n_train}: overall R²={np.mean(all_r2):.4f}±{np.std(all_r2):.4f} (across {len(all_r2)} runs) ***")

# Save
os.makedirs('external_results', exist_ok=True)
with open('external_results/cepdb_gnn_multiseed.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to external_results/cepdb_gnn_multiseed.json")
