"""
P2-1: CEPDB Pretrain → OPV Finetune
在 CEPDB 全量数据上预训练 GNN，然后在 OPV 实验数据上微调
比较：预训练+微调 GNN vs XGBoost baseline vs 从头训练 GNN
"""
import os, sys, json, random, time, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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

FP_DIM = 512
SEED = 9999
PCE_THRESHOLD = 3.0
RESULTS_FILE = 'external_results/pretrain_finetune_results.json'
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
        return self.regressor(g).squeeze(1)


def train_model(model, train_loader, val_loader, device, epochs=100, patience=15):
    """Train GNN model, return best validation MAE."""
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.HuberLoss(delta=1.0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    best_val_mae = float('inf')
    patience_counter = 0
    for epoch in range(1, epochs + 1):
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
            # Save best checkpoint
            torch.save(model.state_dict(), 'external_results/pretrain_best.pth')
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break
    return best_val_mae


def evaluate(model, data_loader, device):
    """Evaluate model on a data loader, return R² and MAE."""
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            pred = model(batch)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(batch.y.view(-1).cpu().numpy())
    return r2_score(all_targets, all_preds), mean_absolute_error(all_targets, all_preds)


# ========== Step 0: Build OPV graphs (for finetuning & evaluation) ==========
print("="*60)
print("P2-1: CEPDB Pretrain → OPV Finetune")
print("="*60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Load OPV data
DATA_CSV = 'data/data_merged.csv' if os.path.exists('data/data_merged.csv') else 'data/data.csv'
print(f"\n[1/5] Loading OPV data from {DATA_CSV}...")
df = pd.read_csv(DATA_CSV, encoding='latin-1')
pce_col = df.columns[2]
smiles_col = df.columns[-1]
df[pce_col] = pd.to_numeric(df[pce_col], errors='coerce')
df[smiles_col] = df[smiles_col].astype(str).str.strip()
df = df.dropna(subset=[pce_col, smiles_col])
df = df[df[smiles_col] != 'nan'].reset_index(drop=True)
df_high = df[df[pce_col] > PCE_THRESHOLD].copy().reset_index(drop=True)

opv_graphs, opv_smiles = [], []
for _, row in df_high.iterrows():
    g = smiles_to_graph(row[smiles_col])
    if g is not None:
        g.y = torch.tensor([float(row[pce_col])], dtype=torch.float)
        opv_graphs.append(g)
        opv_smiles.append(row[smiles_col])
print(f"  OPV high-PCE: {len(opv_graphs)} graphs")

# OPV split (same as main paper)
set_seed(SEED)
indices = list(range(len(opv_graphs)))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=SEED)
train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=SEED)

opv_train = [opv_graphs[i] for i in train_idx]
opv_val = [opv_graphs[i] for i in val_idx]
opv_test = [opv_graphs[i] for i in test_idx]
print(f"  OPV split: train={len(opv_train)}, val={len(opv_val)}, test={len(opv_test)}")

in_dim = opv_graphs[0].x.shape[1]

# ========== Step 1: Load CEPDB data for pretraining ==========
print(f"\n[2/5] Loading CEPDB data for pretraining...")
cepdbs_df = pd.read_csv('/tmp/CEPDB_25000.csv')
cepdbs_df = cepdbs_df[cepdbs_df['pce'] > 0.01].copy().reset_index(drop=True)
cepdbs_high = cepdbs_df[cepdbs_df['pce'] > PCE_THRESHOLD].copy().reset_index(drop=True)
print(f"  CEPDB high-PCE: {len(cepdbs_high)} rows (cap at 15000 for pretraining)")

# Build CEPDB graphs (cap at 15000 for practical pretraining)
cepdbs_graphs = []
n_cepdbs = min(15000, len(cepdbs_high))
for i in range(n_cepdbs):
    g = smiles_to_graph(cepdbs_high.iloc[i]['SMILES_str'])
    if g is not None:
        g.y = torch.tensor([float(cepdbs_high.iloc[i]['pce'])], dtype=torch.float)
        cepdbs_graphs.append(g)
print(f"  CEPDB graphs: {len(cepdbs_graphs)}")

# Split CEPDB into train/val
cepdbs_indices = list(range(len(cepdbs_graphs)))
cepdbs_train_idx, cepdbs_val_idx = train_test_split(cepdbs_indices, test_size=0.1, random_state=SEED)
cepdbs_train = [cepdbs_graphs[i] for i in cepdbs_train_idx]
cepdbs_val = [cepdbs_graphs[i] for i in cepdbs_val_idx]
print(f"  CEPDB split: train={len(cepdbs_train)}, val={len(cepdbs_val)}")

# ========== Step 2: Pretrain on CEPDB ==========
print(f"\n[3/5] Pretraining GNN on CEPDB ({len(cepdbs_train)} samples)...")
set_seed(SEED)
pretrain_model = HighPCERegressorV3(in_channels=in_dim).to(device)
pt_train_loader = GeoDataLoader(cepdbs_train, batch_size=64, shuffle=True)
pt_val_loader = GeoDataLoader(cepdbs_val, batch_size=64)

t0 = time.time()
pretrain_val_mae = train_model(pretrain_model, pt_train_loader, pt_val_loader, device, epochs=80, patience=12)
t1 = time.time()
print(f"  Pretrain done: {t1-t0:.0f}s, best val MAE={pretrain_val_mae:.4f}")

# Evaluate pretrained on CEPDB test set
pt_test_loader = GeoDataLoader(cepdbs_val, batch_size=64)
pt_r2, pt_mae = evaluate(pretrain_model, pt_test_loader, device)
print(f"  Pretrained on CEPDB: R²={pt_r2:.4f}, MAE={pt_mae:.4f}")

# ========== Step 3: Finetune on OPV ==========
print(f"\n[4/5] Finetuning pretrained GNN on OPV...")
finetune_model = HighPCERegressorV3(in_channels=in_dim).to(device)
# Load pretrained weights
finetune_model.load_state_dict(torch.load('external_results/pretrain_best.pth'))

# Use lower LR for finetuning
ft_train_loader = GeoDataLoader(opv_train, batch_size=32, shuffle=True)
ft_val_loader = GeoDataLoader(opv_val, batch_size=32)
ft_test_loader = GeoDataLoader(opv_test, batch_size=32)

optimizer = optim.AdamW(finetune_model.parameters(), lr=0.0003, weight_decay=1e-4)
criterion = nn.HuberLoss(delta=1.0)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5)

best_val_mae = float('inf')
patience_counter = 0
for epoch in range(1, 101):
    finetune_model.train()
    for batch in ft_train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = finetune_model(batch)
        loss = criterion(pred, batch.y.view(-1))
        loss.backward()
        optimizer.step()

    finetune_model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in ft_val_loader:
            batch = batch.to(device)
            pred = finetune_model(batch)
            preds.extend(pred.cpu().numpy())
            targets.extend(batch.y.view(-1).cpu().numpy())
    val_mae = mean_absolute_error(targets, preds)
    scheduler.step(val_mae)

    if val_mae < best_val_mae:
        best_val_mae = val_mae
        patience_counter = 0
        torch.save(finetune_model.state_dict(), 'external_results/finetune_best.pth')
    else:
        patience_counter += 1
    if patience_counter >= 15:
        break

finetune_model.load_state_dict(torch.load('external_results/finetune_best.pth', weights_only=True))
ft_r2, ft_mae = evaluate(finetune_model, ft_test_loader, device)
print(f"  Finetuned on OPV: R²={ft_r2:.4f}, MAE={ft_mae:.4f}")

# ========== Step 3b: Baseline: Train from scratch on OPV ==========
print(f"\n  Baseline: Training GNN from scratch on OPV...")
set_seed(SEED)
scratch_model = HighPCERegressorV3(in_channels=in_dim).to(device)
train_model(scratch_model, ft_train_loader, ft_val_loader, device)
scratch_r2, scratch_mae = evaluate(scratch_model, ft_test_loader, device)
print(f"  From scratch: R²={scratch_r2:.4f}, MAE={scratch_mae:.4f}")

# ========== Step 4: XGBoost baseline ==========
print(f"\n[5/5] XGBoost baseline on OPV...")
set_seed(SEED)
# Hash-based XGBoost features
opv_train_fps = []
for i in train_idx:
    fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(opv_smiles[i]), 2, 4096)
    opv_train_fps.append(np.array(fp, dtype=np.float32))
opv_test_fps = []
for i in test_idx:
    fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(opv_smiles[i]), 2, 4096)
    opv_test_fps.append(np.array(fp, dtype=np.float32))

X_train = np.array(opv_train_fps)
X_test = np.array(opv_test_fps)
y_train = np.array([opv_graphs[i].y.item() for i in train_idx])
y_test = np.array([opv_graphs[i].y.item() for i in test_idx])

xgb = XGBRegressor(n_estimators=2000, learning_rate=0.0117, max_depth=6,
                   min_child_weight=5, subsample=0.595, colsample_bytree=0.626,
                   reg_alpha=0.1, reg_lambda=1.0,
                   random_state=SEED, verbosity=0, n_jobs=-1)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
xgb_r2 = r2_score(y_test, xgb_pred)
xgb_mae = mean_absolute_error(y_test, xgb_pred)
print(f"  XGBoost: R²={xgb_r2:.4f}, MAE={xgb_mae:.4f}")

# ========== Results ==========
print(f"\n{'='*50}")
print(f"P2-1 Results Summary")
print(f"{'='*50}")
print(f"  XGBoost baseline:       R²={xgb_r2:.4f}, MAE={xgb_mae:.4f}")
print(f"  GNN from scratch:       R²={scratch_r2:.4f}, MAE={scratch_mae:.4f}")
print(f"  GNN pretrain+finetune:  R²={ft_r2:.4f}, MAE={ft_mae:.4f}")
print(f"  Pretrain improvement:   ΔR²={ft_r2 - scratch_r2:.4f}")

results = {
    'xgb_baseline': {'r2': float(xgb_r2), 'mae': float(xgb_mae)},
    'gnn_from_scratch': {'r2': float(scratch_r2), 'mae': float(scratch_mae)},
    'gnn_pretrain_finetune': {'r2': float(ft_r2), 'mae': float(ft_mae)},
    'pretrain_improvement_r2': float(ft_r2 - scratch_r2),
    'pretrain_data': f'CEPDB high-PCE n={len(cepdbs_graphs)}',
    'pretrain_cepd_r2': float(pt_r2),
    'pretrain_cepd_mae': float(pt_mae),
}
with open(RESULTS_FILE, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {RESULTS_FILE}")
print("Done!")
