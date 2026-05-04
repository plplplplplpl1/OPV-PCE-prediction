"""
QM9 Full GNN (upgraded from SimpleGCN to 2-branch GCN+GAT)
============================================================
将QM9的GNN升级到与主实验HighPCERegressorV3等价的架构
（2分支：GCN+GAT，隐藏128维，融合512-bit指纹）
确保QM9的GNN不比主实验弱，消除审稿人的架构不对称质疑。
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import global_mean_pool as gmp
from rdkit import Chem
from rdkit.Chem import AllChem
import xgboost as xgb
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import json, os, time, warnings, random

os.environ['RDKIT_SILENCE'] = '1'
warnings.filterwarnings('ignore')

DATA_URL = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv'
TARGET = 'gap'
FP_DIM = 512
RESULTS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'external_results', 'qm9_full_gnn.json')
TEST_SIZE = 10000
VAL_FRAC = 0.1
BATCH_SIZE = 64
LR = 0.001
N_VALUES = [100, 500, 1000, 5000, 20000, 50000]
SEEDS = [42, 123, 456]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}', flush=True)

# ─── 1. Load data ────────────────────────────────────────────────────────────
print('Loading QM9 data...', flush=True)
df = pd.read_csv(DATA_URL)
df = df[df['smiles'].notna() & df[TARGET].notna()].reset_index(drop=True)
y_all = df[TARGET].values.astype(np.float32)
print(f'Total: {len(df)}, {TARGET}={y_all.mean():.4f}±{y_all.std():.4f}', flush=True)

# ─── 2. Fingerprints ─────────────────────────────────────────────────────────
print('Computing fingerprints...', flush=True)
t0 = time.time()
fp_list = []
for smi in df['smiles']:
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=FP_DIM)
        arr = np.zeros(FP_DIM, dtype=np.float32)
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    else:
        arr = np.zeros(FP_DIM, dtype=np.float32)
    fp_list.append(arr)
X_fp = np.stack(fp_list)
print(f'Fingerprints: {X_fp.shape}, {time.time()-t0:.1f}s', flush=True)

# XGBoost baseline fingerprints (2048-bit)
fp2048_list = []
for smi in df['smiles']:
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        arr = np.zeros(2048, dtype=np.float32)
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    else:
        arr = np.zeros(2048, dtype=np.float32)
    fp2048_list.append(arr)
X_fp2048 = np.stack(fp2048_list)

# ─── 3. Graph construction (main experiment features: 30-dim) ────────────────
ATOM_DIM = 30

FULL_ATOM_FEATS = {
    'atomic_num': [1, 6, 7, 8, 9, 15, 16, 17, 35],
    'degree': list(range(5)),
    'hybridization': list(range(1, 6)),
}

def make_graph_full(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except:
        return None

    xs = []
    for a in mol.GetAtoms():
        atomic_num = a.GetAtomicNum()
        degree = a.GetDegree()
        formal_charge = a.GetFormalCharge()
        is_aromatic = int(a.GetIsAromatic())
        is_in_ring = int(a.IsInRing())
        try:
            hybridization = int(a.GetHybridization())
            num_h = a.GetTotalNumHs()
            valence = a.GetTotalValence()
            r3 = int(a.IsInRingSize(3))
            r4 = int(a.IsInRingSize(4))
            r5 = int(a.IsInRingSize(5))
            r6 = int(a.IsInRingSize(6))
        except:
            hybridization = num_h = valence = r3 = r4 = r5 = r6 = 0

        feat = [
            atomic_num / 100.0, degree / 6.0, formal_charge / 8.0,
            num_h / 4.0, valence / 8.0, is_aromatic, is_in_ring,
            r3, r4, r5, r6,
        ] + [int(atomic_num == a) for a in FULL_ATOM_FEATS['atomic_num']] \
          + [int(degree == d) for d in FULL_ATOM_FEATS['degree']] \
          + [int(hybridization == h) for h in FULL_ATOM_FEATS['hybridization']]
        xs.append(feat)

    edges = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        edges.extend([(i, j), (j, i)])
    if not edges:
        return None

    return Data(
        x=torch.tensor(xs, dtype=torch.float),
        edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous()
    )

# ─── 4. Select test set + training pool ───────────────────────────────────────
rng42 = np.random.RandomState(42)
all_idx = np.arange(len(df))
test_idx = set(rng42.choice(len(df), TEST_SIZE, replace=False))
test_arr = np.array([i for i in range(len(df)) if i in test_idx])
train_pool_arr = np.array([i for i in range(len(df)) if i not in test_idx])
y_test = y_all[test_arr]
print(f'Test: {len(test_arr)}, Train pool: {len(train_pool_arr)}', flush=True)

# Build graphs for test set only
print('Building test set graphs...', flush=True)
t0 = time.time()
test_graphs = []
test_valid_idx = []
for idx, i in enumerate(test_arr):
    g = make_graph_full(df.iloc[i]['smiles'])
    if g is not None:
        g.y = torch.tensor([y_all[i]], dtype=torch.float)
        g.fp = torch.tensor(X_fp[i], dtype=torch.float).unsqueeze(0)
        test_graphs.append(g)
        test_valid_idx.append(idx)
# Filter test arrays to match valid graphs
test_arr = test_arr[test_valid_idx]
y_test = y_all[test_arr]
X_test_fp = X_fp[test_arr]
X_test_fp2048 = X_fp2048[test_arr]
print(f'Test graphs: {len(test_graphs)}, {time.time()-t0:.1f}s', flush=True)

# ─── 5. Model: QM9GNN (2-branch GCN+GAT + fingerprint fusion) ────────────────
class QM9GNN(nn.Module):
    """2-branch GNN (GCN+GAT) with fingerprint fusion, matching main experiment style."""
    def __init__(self, in_channels=ATOM_DIM, hidden=128, fp_dim=FP_DIM, dropout=0.3):
        super().__init__()
        self.gcn1 = GCNConv(in_channels, hidden)
        self.gcn2 = GCNConv(hidden, hidden)
        self.gat1 = GATConv(in_channels, hidden//4, heads=4, dropout=dropout)
        self.gat2 = GATConv(hidden, hidden, heads=1, dropout=dropout)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.bn3 = nn.BatchNorm1d(hidden)
        self.fp_encoder = nn.Sequential(
            nn.Linear(fp_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(),
        )
        fused_dim = hidden * 6 + 128  # GCN(3 pools) + GAT(3 pools) + fp
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
        def pool3(h):
            return torch.cat([gmp(h, batch), global_max_pool(h, batch), global_add_pool(h, batch)], dim=1)
        g = torch.cat([pool3(x_gcn), pool3(x_gat)], dim=1)
        fp_feat = self.fp_encoder(data.fp)
        g = torch.cat([g, fp_feat], dim=1)
        return self.regressor(g).squeeze(1)

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

# ─── 6. Training ──────────────────────────────────────────────────────────────
def train_gnn(train_data, val_data, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = QM9GNN(in_channels=ATOM_DIM).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.HuberLoss(delta=1.0)

    tr_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

    best_mae = float('inf')
    best_sd = None
    patience = 0

    for epoch in range(1, 151):
        model.train()
        for b in tr_loader:
            b = b.to(device)
            opt.zero_grad()
            criterion(model(b), b.y.view(-1)).backward()
            opt.step()

        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for b in va_loader:
                b = b.to(device)
                preds.extend(model(b).cpu().numpy())
                targets.extend(b.y.view(-1).cpu().numpy())
        mae = mean_absolute_error(targets, preds)

        if mae < best_mae:
            best_mae = mae
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 20:
                break

    model.load_state_dict(best_sd)
    return model

@torch.no_grad()
def eval_gnn(model, loader):
    model.eval()
    preds = []
    for b in loader:
        preds.append(model(b.to(device)))
    return torch.cat(preds).cpu().numpy()

# ─── 7. Run ───────────────────────────────────────────────────────────────────
results = {}

# XGBoost full data baseline (matching original QM9 hyperparams)
xgb_default = xgb.XGBRegressor(
    n_estimators=1000, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    reg_alpha=0.01, reg_lambda=0.01,
    random_state=42, verbosity=0, n_jobs=-1)
xgb_default.fit(X_fp2048[train_pool_arr], y_all[train_pool_arr],
                eval_set=[(X_fp2048[test_arr], y_test)], verbose=False)
xgb_full_pred = xgb_default.predict(X_test_fp2048)
xgb_full_r2 = r2_score(y_test, xgb_full_pred)
print(f'\nXGBoost full: R²={xgb_full_r2:.4f}', flush=True)

# Test loader (fixed)
test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE)

for n in N_VALUES:
    print(f'\n=== n={n} ===', flush=True)
    n_results = []

    for trial, seed in enumerate(SEEDS):
        rng = np.random.RandomState(seed + n)
        chosen = rng.choice(train_pool_arr, n, replace=False)
        tr, va = train_test_split(chosen, test_size=VAL_FRAC, random_state=int(seed + n))

        # ── XGBoost (matching original QM9 hyperparams) ──
        xgb_m = xgb.XGBRegressor(
            n_estimators=1000, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            reg_alpha=0.01, reg_lambda=0.01,
            early_stopping_rounds=50, random_state=seed, verbosity=0, n_jobs=-1)
        xgb_m.fit(X_fp2048[tr], y_all[tr], eval_set=[(X_fp2048[va], y_all[va])], verbose=False)
        xp = xgb_m.predict(X_test_fp2048)
        xgb_r2 = r2_score(y_test, xp)

        # ── GNN ──
        tr_graphs = []
        for i in tr:
            g = make_graph_full(df.iloc[i]['smiles'])
            if g is not None:
                g.y = torch.tensor([y_all[i]], dtype=torch.float)
                g.fp = torch.tensor(X_fp[i], dtype=torch.float).unsqueeze(0)
                tr_graphs.append(g)

        va_graphs = []
        for i in va:
            g = make_graph_full(df.iloc[i]['smiles'])
            if g is not None:
                g.y = torch.tensor([y_all[i]], dtype=torch.float)
                g.fp = torch.tensor(X_fp[i], dtype=torch.float).unsqueeze(0)
                va_graphs.append(g)

        if len(tr_graphs) < 10 or len(va_graphs) < 5:
            print(f'  Trial {trial+1}: too few graphs, skipping', flush=True)
            continue

        t0 = time.time()
        gm = train_gnn(tr_graphs, va_graphs, seed)
        gp = eval_gnn(gm, test_loader)
        gnn_r2 = r2_score(y_test, gp)
        gnn_mae = mean_absolute_error(y_test, gp)
        gnn_rmse = np.sqrt(mean_squared_error(y_test, gp))
        train_time = time.time() - t0

        n_results.append({
            'seed': seed, 'xgb_r2': float(xgb_r2), 'gnn_r2': float(gnn_r2),
            'gnn_mae': float(gnn_mae), 'gnn_rmse': float(gnn_rmse),
            'train_time_s': round(train_time, 1),
        })
        print(f'  Trial {trial+1}: XGB={xgb_r2:.4f} GNN={gnn_r2:.4f} ({train_time:.0f}s)', flush=True)

    xgb_r2s = [r['xgb_r2'] for r in n_results]
    gnn_r2s = [r['gnn_r2'] for r in n_results]
    print(f'  → XGB={np.mean(xgb_r2s):.4f}±{np.std(xgb_r2s):.4f}  GNN={np.mean(gnn_r2s):.4f}±{np.std(gnn_r2s):.4f}', flush=True)

    results[str(n)] = {
        'xgb_r2_mean': float(np.mean(xgb_r2s)), 'xgb_r2_std': float(np.std(xgb_r2s)),
        'gnn_r2_mean': float(np.mean(gnn_r2s)), 'gnn_r2_std': float(np.std(gnn_r2s)),
        'delta_r2_mean': float(np.mean(gnn_r2s) - np.mean(xgb_r2s)),
        'n_trials': len(n_results),
        'xgb_full_r2': float(xgb_full_r2),
    }

    # Save incremental
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)

# ─── 8. Summary ──────────────────────────────────────────────────────────────
print(f'\n{"="*55}', flush=True)
print('QM9 Full GNN Results', flush=True)
print(f'{"="*55}', flush=True)
print(f'{"n":<8} {"XGB R²":<15} {"GNN R²":<15} {"ΔR²":<15}', flush=True)
print('-'*55, flush=True)
for n in N_VALUES:
    r = results[str(n)]
    print(f'{n:<8} {r["xgb_r2_mean"]:<15.4f} {r["gnn_r2_mean"]:<15.4f} {r["delta_r2_mean"]:<15.4f}', flush=True)
print(f'{"Full":<8} {xgb_full_r2:<15.4f} {"—":<15}', flush=True)

print(f'\nComparison with SimpleGCN (manuscript Table 10):', flush=True)
print(f'  Original: n=100: XGB=0.6564 GNN=0.5773 | n=500: XGB=0.7442 GNN=0.8153', flush=True)
print(f'  Upgraded: n=100: XGB={results["100"]["xgb_r2_mean"]:.4f} GNN={results["100"]["gnn_r2_mean"]:.4f}', flush=True)
print('Done.', flush=True)
