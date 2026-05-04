"""
GNN Capacity Scan: 固定样本量 n=1,000，扫描模型容量（隐藏维度）
==============================================================
核心假设：在固定小样本下，低容量GNN（小hidden）优于高容量GNN（大hidden），
因为高容量模型的"估计误差"淹没了"表示优势"。
这与论文3.1节的VC维理论框架直接对应。

协议：
- 固定训练样本 n=1,000（OPV高PCE子集）
- 扫描 hidden = [16, 32, 64, 128, 256, 512]
- 每个配置 4 种随机种子
- XGBoost基线（Optuna优化参数）作为参照
- 预测：R² 随 hidden 增大先升后降（或单调降），最优容量出现在小hidden处
"""
import os, sys, json, random, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

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

import xgboost as xgb

# ─── Config ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'data.csv')
RESULTS_FILE = os.path.join(BASE_DIR, 'external_results', 'gnn_capacity_scan.json')
PCE_THRESHOLD = 3.0
FP_DIM = 512
N_TRAIN = 1000      # 固定训练样本量
SEEDS = [42, 123, 456, 9999]
HIDDEN_DIMS = [16, 32, 64, 128, 256, 512]
XGB_PARAMS = {
    'n_estimators': 2000, 'learning_rate': 0.0117, 'max_depth': 6,
    'min_child_weight': 5, 'subsample': 0.595, 'colsample_bytree': 0.626,
    'reg_alpha': 0.01, 'reg_lambda': 0.01, 'gamma': 0.1,
    'random_state': 42, 'verbosity': 0, 'n_jobs': -1
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
print(f'Fixed training size: {N_TRAIN}')
print(f'Hidden dims: {HIDDEN_DIMS}')
print(f'Seeds: {SEEDS}')

# ─── 1. Load data ───────────────────────────────────────────────────────────
print('\nLoading data...')
df = pd.read_csv(DATA_PATH, encoding='latin-1')
df.columns = df.columns.str.strip()
pce_col = [c for c in df.columns if 'pce' in c.lower()][0]
smiles_col = [c for c in df.columns if 'smiles' in c.lower()][0]
df[pce_col] = pd.to_numeric(df[pce_col], errors='coerce')
df = df.dropna(subset=[pce_col, smiles_col]).reset_index(drop=True)
df_high = df[df[pce_col] > PCE_THRESHOLD].copy().reset_index(drop=True)
print(f'High-PCE samples: {len(df_high)}')

# ─── 2. Compute graphs and fingerprints ─────────────────────────────────────
def smiles_to_graph(smiles, fp_dim=FP_DIM):
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
                r3 = int(atom.IsInRingSize(3)); r4 = int(atom.IsInRingSize(4))
                r5 = int(atom.IsInRingSize(5)); r6 = int(atom.IsInRingSize(6))
            except:
                hybridization = num_h = valence = r3 = r4 = r5 = r6 = 0
            common_atoms = [1, 6, 7, 8, 9, 15, 16, 17, 35]
            feat = [atomic_num/100., degree/6., formal_charge/8., num_h/4., valence/8.,
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
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=fp_dim)
        fp_t = torch.tensor(np.array(fp, dtype=np.float32)).unsqueeze(0)
        x = torch.tensor(atom_features, dtype=torch.float)
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        return Data(x=x, edge_index=edge_index, fp=fp_t)
    except:
        return None

graphs = []
y_list = []
fps_2048 = []  # for XGBoost baseline
failed = 0
for _, row in df_high.iterrows():
    smi = row[smiles_col]
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        failed += 1
        continue
    g = smiles_to_graph(smi)
    if g is None:
        failed += 1
        continue
    g.y = torch.tensor([float(row[pce_col])], dtype=torch.float)
    graphs.append(g)
    y_list.append(float(row[pce_col]))
    fp2048 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    fps_2048.append(np.array(fp2048, dtype=np.float32))

y_all = np.array(y_list)
n_total = len(graphs)
print(f'Valid: {n_total}, failed: {failed}')

in_dim = graphs[0].x.shape[1]

# ─── 3. Model ────────────────────────────────────────────────────────────────
class CapacityGNN(nn.Module):
    """HighPCERegressorV3 with parameterized hidden dim."""
    def __init__(self, in_channels, hidden, fp_dim=FP_DIM, dropout=0.3):
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

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x_gcn = F.relu(self.bn1(self.gcn1(x, edge_index)))
        x_gcn = F.relu(self.gcn2(x_gcn, edge_index))
        x_gat = F.relu(self.gat1(x, edge_index))
        x_gat = F.relu(self.bn2(self.gat2(x_gat, edge_index)))
        x_sage = F.relu(self.sage1(x, edge_index))
        x_sage = F.relu(self.bn3(self.sage2(x_sage, edge_index)))
        def pool3(h):
            return torch.cat([global_mean_pool(h, batch),
                              global_max_pool(h, batch),
                              global_add_pool(h, batch)], dim=1)
        g = torch.cat([pool3(x_gcn), pool3(x_gat), pool3(x_sage)], dim=1)
        fp_feat = self.fp_encoder(data.fp)
        g = torch.cat([g, fp_feat], dim=1)
        return self.regressor(g).squeeze(1)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_gnn(train_graphs, val_graphs, hidden_dim, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = CapacityGNN(in_channels=in_dim, hidden=hidden_dim).to(device)
    n_params = count_params(model)
    opt = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.HuberLoss(delta=1.0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=8, factor=0.5)

    train_loader = GeoDataLoader(train_graphs, batch_size=32, shuffle=True)
    val_loader = GeoDataLoader(val_graphs, batch_size=32)

    best_val_mae = float('inf')
    best_sd = None
    patience = 0

    for epoch in range(1, 151):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            loss = criterion(model(batch), batch.y.view(-1))
            loss.backward()
            opt.step()

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
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 20:
                break

    model.load_state_dict(best_sd)
    return model, n_params

# ─── 4. Main experiment ──────────────────────────────────────────────────────
# 固定测试集（seed 9999划分），各模型共享相同train/val/test
all_indices = np.arange(n_total)
tr_idx, te_idx = train_test_split(all_indices, test_size=0.2, random_state=9999)
tr_idx, va_idx = train_test_split(tr_idx, test_size=0.1, random_state=9999)

# 限制训练集为 N_TRAIN
if len(tr_idx) > N_TRAIN:
    tr_idx = tr_idx[:N_TRAIN]

test_graphs = [graphs[i] for i in te_idx]
y_test = y_all[te_idx]
print(f'Train: {len(tr_idx)}, Val: {len(va_idx)}, Test: {len(te_idx)}')

# ─── XGBoost baseline ────────────────────────────────────────────────────────
print('\n── XGBoost baseline ──')
X_tr_fp = np.array([fps_2048[i] for i in tr_idx])
X_va_fp = np.array([fps_2048[i] for i in va_idx])
X_te_fp = np.array([fps_2048[i] for i in te_idx])
y_tr_arr = y_all[tr_idx]
y_va_arr = y_all[va_idx]

xgb_model = xgb.XGBRegressor(**XGB_PARAMS)
xgb_model.fit(np.vstack([X_tr_fp, X_va_fp]),
              np.concatenate([y_tr_arr, y_va_arr]),
              eval_set=[(X_te_fp, y_test)], verbose=False)
xgb_pred = xgb_model.predict(X_te_fp)
xgb_r2 = r2_score(y_test, xgb_pred)
xgb_mae = mean_absolute_error(y_test, xgb_pred)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
print(f'  XGBoost: R²={xgb_r2:.4f}, MAE={xgb_mae:.4f}, RMSE={xgb_rmse:.4f}')

# ─── GNN capacity scan ───────────────────────────────────────────────────────
results = {}
for hidden in HIDDEN_DIMS:
    print(f'\n── hidden={hidden} ──')
    seed_results = []
    for seed in SEEDS:
        # 每个种子重新划分train/val（固定test集不变）
        tr_idx_s, va_idx_s = train_test_split(
            tr_idx, test_size=0.1, random_state=int(seed))
        tr_graphs = [graphs[i] for i in tr_idx_s]
        va_graphs = [graphs[i] for i in va_idx_s]

        t0 = time.time()
        model, n_params = train_gnn(tr_graphs, va_graphs, hidden, seed)
        train_time = time.time() - t0

        # Evaluate
        model.eval()
        te_loader = GeoDataLoader(test_graphs, batch_size=32)
        preds = []
        with torch.no_grad():
            for batch in te_loader:
                batch = batch.to(device)
                preds.append(model(batch).cpu())
        y_pred = torch.cat(preds).numpy()

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        seed_results.append({
            'seed': seed, 'r2': float(r2), 'mae': float(mae), 'rmse': float(rmse),
            'n_params': n_params, 'train_time_s': round(train_time, 1)
        })
        print(f'  seed={seed}: R²={r2:.4f}, MAE={mae:.4f}, params={n_params:,}')

    r2_vals = [r['r2'] for r in seed_results]
    mae_vals = [r['mae'] for r in seed_results]
    results[str(hidden)] = {
        'r2_mean': float(np.mean(r2_vals)),
        'r2_std': float(np.std(r2_vals)),
        'mae_mean': float(np.mean(mae_vals)),
        'mae_std': float(np.std(mae_vals)),
        'n_params': seed_results[0]['n_params'],
        'seed_results': seed_results,
    }

# ─── 5. Summarize ────────────────────────────────────────────────────────────
print(f'\n{"="*65}')
print('GNN Capacity Scan Results (n=1,000 fixed)')
print(f'{"="*65}')
print(f'{"Hidden":<8} {"Params":<12} {"R² mean":<12} {"R² std":<12} {"MAE mean":<12}')
print(f'{"-"*8} {"-"*12} {"-"*12} {"-"*12} {"-"*12} {"-"*12}')
for h in HIDDEN_DIMS:
    r = results[str(h)]
    print(f'{h:<8} {r["n_params"]:<12,} {r["r2_mean"]:<12.4f} {r["r2_std"]:<12.4f} {r["mae_mean"]:<12.4f}')
print(f'{"-"*65}')
print(f'{"XGBoost":<8} {"—":<12} {xgb_r2:<12.4f} {"—":<12} {xgb_mae:<12.4f}')

# ─── 6. Save ─────────────────────────────────────────────────────────────────
output = {
    'experiment': 'GNN capacity scan at fixed n=1,000',
    'n_train': N_TRAIN,
    'hidden_dims': HIDDEN_DIMS,
    'seeds': SEEDS,
    'xgb_baseline': {'r2': xgb_r2, 'mae': xgb_mae, 'rmse': xgb_rmse},
    'gnn_results': {str(h): {
        'r2_mean': results[str(h)]['r2_mean'],
        'r2_std': results[str(h)]['r2_std'],
        'mae_mean': results[str(h)]['mae_mean'],
        'mae_std': results[str(h)]['mae_std'],
        'n_params': results[str(h)]['n_params'],
    } for h in HIDDEN_DIMS},
    'detailed': results,
}
os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
with open(RESULTS_FILE, 'w') as f:
    json.dump(output, f, indent=2)
print(f'\nResults saved to {RESULTS_FILE}')
print('Done.')
