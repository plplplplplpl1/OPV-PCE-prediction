"""
HOPV15 外部验证
在哈佛 OPV 数据集上独立训练并比较 XGBoost vs GNN
"""
import os, sys, json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['RDKIT_SILENCE'] = '1'
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import AllChem

# XGBoost
from xgboost import XGBRegressor

# Torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool, global_max_pool, global_add_pool

SEED = 42
FP_DIM = 512
BATCH_SIZE = 16

def set_seed(seed):
    import random
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

# ========== 加载 HOPV15 ==========
print("="*50)
print("HOPV15 External Validation")
print("="*50)

df = pd.read_csv('/tmp/HOPV_15_revised_2_processed_homo_5fold.csv')
# Filter to high-PCE only (>3%) to match our paper's task
df_high = df[df['pce'] > 3.0].copy().reset_index(drop=True)
print(f"High-PCE samples (PCE>3%): {len(df_high)}")

# Convert SMILES to graphs
graphs, pce_vals, failed = [], [], []
for _, row in df_high.iterrows():
    g = smiles_to_graph(row['smiles'])
    if g is not None:
        g.y = torch.tensor([float(row['pce'])], dtype=torch.float)
        graphs.append(g)
        pce_vals.append(float(row['pce']))
    else:
        failed.append(row['smiles'])

print(f"Graph conversion: {len(graphs)} success, {len(failed)} failed")

# ========== XGBoost Baseline (5-fold CV) ==========
print("\n--- XGBoost 5-fold CV ---")
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
xgb_r2, xgb_mae, xgb_rmse = [], [], []
fold = 0
for train_idx, test_idx in kf.split(graphs):
    fold += 1
    train_smiles = [df_high.iloc[i]['smiles'] for i in train_idx]
    train_pce = [df_high.iloc[i]['pce'] for i in train_idx]
    test_smiles = [df_high.iloc[i]['smiles'] for i in test_idx]
    test_pce = [df_high.iloc[i]['pce'] for i in test_idx]

    train_fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 4096) for s in train_smiles]
    test_fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 4096) for s in test_smiles]
    X_train = np.array(train_fps, dtype=np.float32)
    X_test = np.array(test_fps, dtype=np.float32)
    y_train = np.array(train_pce, dtype=np.float32)
    y_test = np.array(test_pce, dtype=np.float32)

    model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                         random_state=SEED, verbosity=0)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    xgb_r2.append(r2_score(y_test, pred))
    xgb_mae.append(mean_absolute_error(y_test, pred))
    xgb_rmse.append(float(np.sqrt(np.mean((y_test - pred)**2))))
    print(f"  Fold {fold}: R²={xgb_r2[-1]:.4f}, MAE={xgb_mae[-1]:.4f}, RMSE={xgb_rmse[-1]:.4f}")

print(f"\nXGBoost HOPV15: R²={np.mean(xgb_r2):.4f}±{np.std(xgb_r2):.4f}")

# Save XGBoost results
xgb_results = {
    'dataset': 'HOPV15',
    'model': 'XGBoost',
    'n_samples': len(graphs),
    'r2_mean': float(np.mean(xgb_r2)),
    'r2_std': float(np.std(xgb_r2)),
    'mae_mean': float(np.mean(xgb_mae)),
    'mae_std': float(np.std(xgb_mae)),
    'rmse_mean': float(np.mean(xgb_rmse)),
    'rmse_std': float(np.std(xgb_rmse)),
    'per_fold': [{'r2': r, 'mae': m, 'rmse': rms} for r, m, rms in zip(xgb_r2, xgb_mae, xgb_rmse)]
}
with open('external_results/hopv15_xgb.json', 'w') as f:
    json.dump(xgb_results, f, indent=2)
print("XGBoost results saved to external_results/hopv15_xgb.json")

# ========== GNN 5-fold CV ==========
print("\n--- GNN 5-fold CV ---")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

gnn_r2, gnn_mae, gnn_rmse = [], [], []
fold = 0
for train_idx, test_idx in kf.split(graphs):
    fold += 1
    set_seed(SEED + fold)

    train_graphs = [graphs[i] for i in train_idx]
    test_graphs = [graphs[i] for i in test_idx]
    train_loader = GeoDataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = GeoDataLoader(test_graphs, batch_size=BATCH_SIZE)

    in_dim = graphs[0].x.shape[1]
    model = HighPCERegressorV3(in_channels=in_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.HuberLoss(delta=1.0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    best_val_mae = float('inf')
    patience_counter = 0
    # Use 20% of training as validation
    val_size = max(1, len(train_graphs) // 5)
    val_graphs = train_graphs[-val_size:]
    train_graphs_sub = train_graphs[:-val_size]
    val_loader = GeoDataLoader(val_graphs, batch_size=BATCH_SIZE)
    train_loader_sub = GeoDataLoader(train_graphs_sub, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(1, 101):
        model.train()
        total_loss = 0
        for batch in train_loader_sub:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = criterion(pred, batch.y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        train_loss = total_loss / len(train_loader_sub.dataset)

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
            torch.save(model.state_dict(), f'external_results/hopv15_gnn_fold{fold}.pth')
        else:
            patience_counter += 1

        if epoch % 20 == 0 or epoch == 1:
            val_r2 = r2_score(targets, preds)
            val_rmse = float(np.sqrt(np.mean((np.array(targets)-np.array(preds))**2)))
            print(f"  Fold {fold} Epoch {epoch:3d} | train_loss={train_loss:.4f} | val_MAE={val_mae:.4f} | val_R²={val_r2:.4f} | patience={patience_counter}")

        if patience_counter >= 15:
            break

    # Test
    model.load_state_dict(torch.load(f'external_results/hopv15_gnn_fold{fold}.pth', weights_only=True))
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch)
            preds.extend(pred.cpu().numpy())
            targets.extend(batch.y.view(-1).cpu().numpy())
    r2 = r2_score(targets, preds)
    mae = mean_absolute_error(targets, preds)
    rmse = float(np.sqrt(np.mean((np.array(targets)-np.array(preds))**2)))
    gnn_r2.append(r2)
    gnn_mae.append(mae)
    gnn_rmse.append(rmse)
    print(f"  >>> Fold {fold} TEST: R²={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")

print(f"\nGNN HOPV15: R²={np.mean(gnn_r2):.4f}±{np.std(gnn_r2):.4f}")

gnn_results = {
    'dataset': 'HOPV15',
    'model': 'HighPCERegressorV3',
    'n_samples': len(graphs),
    'r2_mean': float(np.mean(gnn_r2)),
    'r2_std': float(np.std(gnn_r2)),
    'mae_mean': float(np.mean(gnn_mae)),
    'mae_std': float(np.std(gnn_mae)),
    'rmse_mean': float(np.mean(gnn_rmse)),
    'rmse_std': float(np.std(gnn_rmse)),
    'per_fold': [{'r2': r, 'mae': m, 'rmse': rms} for r, m, rms in zip(gnn_r2, gnn_mae, gnn_rmse)]
}
with open('external_results/hopv15_gnn.json', 'w') as f:
    json.dump(gnn_results, f, indent=2)
print("GNN results saved to external_results/hopv15_gnn.json")

# Comparison
print("\n" + "="*50)
print("HOPV15 External Validation Results")
print("="*50)
print(f"XGBoost: R²={np.mean(xgb_r2):.4f}±{np.std(xgb_r2):.4f}, MAE={np.mean(xgb_mae):.4f}±{np.std(xgb_mae):.4f}")
print(f"GNN:     R²={np.mean(gnn_r2):.4f}±{np.std(gnn_r2):.4f}, MAE={np.mean(gnn_mae):.4f}±{np.std(gnn_mae):.4f}")
delta = np.mean(xgb_r2) - np.mean(gnn_r2)
print(f"ΔR²: {delta:.4f} (XGBoost {'wins' if delta > 0 else 'loses'})")
