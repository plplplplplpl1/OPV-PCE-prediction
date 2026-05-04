"""
P1-4: MoleculeNet 跨数据集验证
在 ESOL / FreeSolv / Lipophilicity 上比较 XGBoost vs GNN 随样本量的表现
检验 XGBoost-better → GNN-better 相变是否跨数据集复现
"""
import os, sys, json, random
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

os.environ['RDKIT_SILENCE'] = '1'
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import AllChem

from xgboost import XGBRegressor

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.datasets import MoleculeNet

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
print("="*60)
print("P1-4: MoleculeNet Cross-Dataset Validation")
print("="*60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

datasets_config = {
    'ESOL': {'name': 'esol', 'max_samples': 1128},
    'FreeSolv': {'name': 'freesolv', 'max_samples': 642},
    'Lipophilicity': {'name': 'lipo', 'max_samples': 4200},
}
N_TRAIN = [100, 250, 500, 1000, 2000]
SEED = 42

all_results = {}

for ds_display, ds_cfg in datasets_config.items():
    ds_name = ds_cfg['name']
    max_s = ds_cfg['max_samples']
    print(f"{'='*50}")
    print(f"Dataset: {ds_display} ({max_s} samples)")
    print(f"{'='*50}")

    # Load data
    molnet = MoleculeNet(root='/tmp/molnet', name=ds_name)
    smiles_list = [d['smiles'] for d in molnet]
    y_all = np.array([d.y.item() for d in molnet])

    # Build graphs & fingerprints
    graphs = []
    fps = []
    valid_indices = []
    for i, smi in enumerate(smiles_list):
        g = smiles_to_graph(smi)
        if g is not None:
            g.y = torch.tensor([y_all[i]], dtype=torch.float)
            graphs.append(g)
            fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), radius=2, nBits=FP_DIM)
            fps.append(np.array(fp, dtype=np.float32))
            valid_indices.append(i)
    y_valid = y_all[valid_indices]
    n_total = len(graphs)
    print(f"  Valid molecules: {n_total}/{max_s}")

    # Determine n values
    n_values = [n for n in N_TRAIN if n < n_total] + [n_total]
    n_values = sorted(set(n_values))
    print(f"  Training sizes: {n_values}")

    # 3-fold split indices
    set_seed(SEED)
    indices = np.random.permutation(n_total)
    fold_splits = []
    split_size = n_total // 3
    for f in range(3):
        test_idx = indices[f*split_size:(f+1)*split_size] if f < 2 else indices[2*split_size:]
        train_idx = np.concatenate([indices[:f*split_size], indices[(f+1)*split_size:]]) if f < 2 else indices[:2*split_size]
        fold_splits.append((train_idx, test_idx))

    ds_results = {}

    for n_train in n_values:
        print(f"\n  --- n={n_train} ---")
        xgb_r2_list = []
        gnn_r2_list = []
        n_folds_used = 0

        for fold, (train_idx, test_idx) in enumerate(fold_splits):
            n_train_actual = min(n_train, len(train_idx))
            fold_train_idx = train_idx[:n_train_actual]

            # Datasets
            xgb_train_fps = [fps[i] for i in fold_train_idx]
            xgb_test_fps = [fps[i] for i in test_idx]
            y_train = y_valid[fold_train_idx]
            y_test = y_valid[test_idx]
            gnn_train = [graphs[i] for i in fold_train_idx]
            gnn_test = [graphs[i] for i in test_idx]

            # Skip if too little training data
            fold_valid = True

            # --- XGBoost ---
            try:
                xgb = XGBRegressor(n_estimators=500, max_depth=8, learning_rate=0.1,
                                   subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                                   n_jobs=-1)
                xgb.fit(np.array(xgb_train_fps), y_train)
                xgb_pred = xgb.predict(np.array(xgb_test_fps))
                xgb_r2 = r2_score(y_test, xgb_pred)
                xgb_r2_list.append(xgb_r2)
            except Exception as e:
                print(f"    XGBoost fold {fold} failed: {e}")
                fold_valid = False

            # --- GNN ---
            val_size = max(1, len(gnn_train) // 5)
            val_graphs = gnn_train[-val_size:]
            train_graphs_sub = gnn_train[:-val_size]

            if len(train_graphs_sub) < 10:  # too little data
                fold_valid = False
                continue

            train_loader = GeoDataLoader(train_graphs_sub, batch_size=64, shuffle=True)
            val_loader = GeoDataLoader(val_graphs, batch_size=64)
            test_loader = GeoDataLoader(gnn_test, batch_size=64)

            in_dim = graphs[0].x.shape[1]
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

            # Test GNN
            model.eval()
            preds, targets = [], []
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    pred = model(batch)
                    preds.extend(pred.cpu().numpy())
                    targets.extend(batch.y.view(-1).cpu().numpy())
            gnn_r2 = r2_score(targets, preds)
            gnn_r2_list.append(gnn_r2)

            n_folds_used += 1

        if xgb_r2_list:
            xgb_mean, xgb_std = np.mean(xgb_r2_list), np.std(xgb_r2_list)
        else:
            xgb_mean, xgb_std = None, None
        if gnn_r2_list:
            gnn_mean, gnn_std = np.mean(gnn_r2_list), np.std(gnn_r2_list)
        else:
            gnn_mean, gnn_std = None, None

        ds_results[str(n_train)] = {
            'xgb_r2_mean': xgb_mean,
            'xgb_r2_std': xgb_std,
            'gnn_r2_mean': gnn_mean,
            'gnn_r2_std': gnn_std,
            'n_folds': n_folds_used,
        }
        print(f"    XGBoost: R²={xgb_mean:.4f}±{xgb_std:.4f}" if xgb_mean else "    XGBoost: failed")
        print(f"    GNN:    R²={gnn_mean:.4f}±{gnn_std:.4f}" if gnn_mean else "    GNN:    failed")

    all_results[ds_display] = {
        'n_total': n_total,
        'n_values': n_values,
        'results': ds_results,
    }

# Save
os.makedirs('external_results', exist_ok=True)
with open('external_results/molenet_cross_validation.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\nResults saved to external_results/molenet_cross_validation.json")
