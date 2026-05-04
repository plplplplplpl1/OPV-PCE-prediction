"""
CEPDB Large-Scale Extension: verify phase transition at 2000-5000 samples
Extends the original experiment to training sizes 2000, 3000, 5000, 10000
"""
import os, sys, json, random
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
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

SEED = 42
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

# ========== Main ==========
print("="*60)
print("CEPDB Large-Scale Extension")
print("="*60)

# Load dataset
df = pd.read_csv('/tmp/CEPDB_25000.csv')
df = df[df['pce'] > 0.01].copy().reset_index(drop=True)

# Use high-PCE subset
df_high = df[df['pce'] > 3.0].copy().reset_index(drop=True)
print(f"Total CEPDB molecules: {len(df)}")
print(f"High-PCE (PCE>3%): {len(df_high)}")

# Convert to graphs (use all high-PCE molecules)
graphs, smiles_list = [], []
failed = 0
for i, (_, row) in enumerate(df_high.iterrows()):
    g = smiles_to_graph(row['SMILES_str'])
    if g is not None:
        g.y = torch.tensor([float(row['pce'])], dtype=torch.float)
        graphs.append(g)
        smiles_list.append(row['SMILES_str'])
    else:
        failed += 1

print(f"Graph conversion: {len(graphs)} success, {failed} failed")
print(f"Using all {len(graphs)} high-PCE molecules")
pce_vals = [g.y.item() for g in graphs]

# Training sizes for phase transition verification
# Original: 100, 250, 500, 1000, 1500
# Extended: 2000, 3000, 5000, 10000
train_sizes = [2000, 3000, 5000, 10000]
n_folds = 3

xgb_results = {}
gnn_results = {}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

# Use KFold on all graphs
kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)

for n_train in train_sizes:
    print(f"\n{'='*50}")
    print(f"Training size: n_train={n_train}")
    print(f"{'='*50}")

    # ---- XGBoost ----
    print(f"\n--- XGBoost n={n_train} ---")
    xgb_r2_folds = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(graphs)):
        # Subset training to n_train
        train_idx_sub = train_idx[:n_train]

        train_fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles_list[i]), 2, 4096) for i in train_idx_sub]
        test_fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles_list[i]), 2, 4096) for i in test_idx]
        X_train = np.array(train_fps, dtype=np.float32)
        X_test = np.array(test_fps, dtype=np.float32)
        y_train = np.array([pce_vals[i] for i in train_idx_sub], dtype=np.float32)
        y_test = np.array([pce_vals[i] for i in test_idx], dtype=np.float32)

        model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                             random_state=SEED+fold, verbosity=0, n_jobs=8)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        r2 = r2_score(y_test, pred)
        xgb_r2_folds.append(r2)
        print(f"  Fold {fold}: R²={r2:.4f}")

    xgb_mean, xgb_std = float(np.mean(xgb_r2_folds)), float(np.std(xgb_r2_folds))
    xgb_results[str(n_train)] = {'r2_mean': xgb_mean, 'r2_std': xgb_std}
    print(f"  XGBoost n={n_train}: R²={xgb_mean:.4f}±{xgb_std:.4f}")

    # ---- GNN ----
    print(f"\n--- GNN n={n_train} ---")
    gnn_r2_folds = []
    for fold in range(n_folds):
        set_seed(SEED + fold)

        split_idx = len(graphs) // n_folds
        test_graphs = graphs[fold*split_idx:(fold+1)*split_idx]
        train_graphs_all = graphs[:fold*split_idx] + graphs[(fold+1)*split_idx:]
        train_graphs = train_graphs_all[:n_train]
        val_size = max(1, len(train_graphs) // 5)
        val_graphs = train_graphs[-val_size:]
        train_graphs_sub = train_graphs[:-val_size]

        batch_size = 128 if n_train >= 5000 else 64
        train_loader = GeoDataLoader(train_graphs_sub, batch_size=batch_size, shuffle=True)
        val_loader = GeoDataLoader(val_graphs, batch_size=batch_size)
        test_loader = GeoDataLoader(test_graphs, batch_size=batch_size)

        in_dim = graphs[0].x.shape[1]
        model = HighPCERegressorV3(in_channels=in_dim).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.HuberLoss(delta=1.0)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

        best_val_mae = float('inf')
        patience_counter = 0
        max_epochs = 80 if n_train >= 5000 else 100

        for epoch in range(1, max_epochs + 1):
            model.train()
            total_loss = 0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                pred = model(batch)
                loss = criterion(pred, batch.y.view(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch.num_graphs

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
                torch.save(model.state_dict(), f'external_results/cepdb_large_n{n_train}_fold{fold}.pth')
            else:
                patience_counter += 1

            if epoch % 20 == 0:
                val_r2 = r2_score(targets, preds)
                print(f"  n={n_train} fold={fold} epoch={epoch}/{max_epochs} val_MAE={val_mae:.4f} val_R²={val_r2:.4f}")

            if patience_counter >= 15:
                break

        # Test
        model.load_state_dict(torch.load(
            f'external_results/cepdb_large_n{n_train}_fold{fold}.pth', weights_only=True))
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                pred = model(batch)
                preds.extend(pred.cpu().numpy())
                targets.extend(batch.y.view(-1).cpu().numpy())
        r2 = r2_score(targets, preds)
        gnn_r2_folds.append(r2)
        print(f"  >>> n={n_train} fold={fold} TEST R²={r2:.4f}")
        sys.stdout.flush()
        torch.cuda.empty_cache()

    if gnn_r2_folds:
        gnn_mean, gnn_std = float(np.mean(gnn_r2_folds)), float(np.std(gnn_r2_folds))
        gnn_results[str(n_train)] = {'r2_mean': gnn_mean, 'r2_std': gnn_std}
        print(f"  GNN n={n_train}: R²={gnn_mean:.4f}±{gnn_std:.4f}")

    # Save incrementally after each training size
    os.makedirs('external_results', exist_ok=True)
    with open('external_results/cepdb_large_xgb.json', 'w') as f:
        json.dump(xgb_results, f, indent=2)
    with open('external_results/cepdb_large_gnn.json', 'w') as f:
        json.dump(gnn_results, f, indent=2)
    sys.stdout.flush()

# Summary
print("\n" + "="*60)
print("CEPDB Large-Scale Extension - Summary")
print("="*60)
all_sizes = train_sizes
for n in all_sizes:
    xgb = xgb_results.get(str(n), {})
    gnn = gnn_results.get(str(n), {})
    xgb_str = f"{xgb.get('r2_mean', 0):.4f}±{xgb.get('r2_std', 0):.4f}" if xgb else "N/A"
    gnn_str = f"{gnn.get('r2_mean', 0):.4f}±{gnn.get('r2_std', 0):.4f}" if gnn else "N/A"
    delta = float(xgb.get('r2_mean', 0)) - float(gnn.get('r2_mean', 0)) if xgb and gnn else 0
    print(f"  n={n:5d}: XGBoost R²={xgb_str} | GNN R²={gnn_str} | Δ={delta:.4f}")

print("\nDone!")
