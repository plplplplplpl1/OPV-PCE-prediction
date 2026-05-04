"""
NREL OPV Database: Prospective Literature Validation
=====================================================
Validates the phase transition on an independent OPV-specific computational
database (NREL OPV ~95k molecules, HOMO-LUMO gap target).

This is a strictly independent dataset: NREL DFT-computed properties for
OPV-relevant monomers and small molecules, never used in our original study.

Adds a 6th independent dataset to the paper's phase transition analysis,
specifically targeting the OPV domain (complementing QM9 which is general
quantum chemistry and CEPDB which is general organic photovoltaics).
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool, global_add_pool
from rdkit import Chem
from rdkit.Chem import AllChem
import xgboost as xgb
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import json, os, time, warnings, random

os.environ['RDKIT_SILENCE'] = '1'
warnings.filterwarnings('ignore')

# ─── Config ─────────────────────────────────────────────────────────────────
TARGET = 'gap'
FP_DIM = 512
N_VALUES = [100, 500, 1000, 5000, 20000, 50000]
SEEDS = [42, 123, 456]
TEST_SIZE = 20000
VAL_FRAC = 0.1
BATCH_SIZE = 64
LR = 0.001
RESULTS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'external_results', 'nrel_opv_validation.json')
DATA_CACHE = '/tmp/opv_db.csv.gz'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}', flush=True)

# ─── 1. Load ────────────────────────────────────────────────────────────────
print('Loading NREL OPV database...', flush=True)
df = pd.read_csv(DATA_CACHE)
df = df[df['smile'].notna() & df[TARGET].notna()].drop_duplicates(subset='smile').reset_index(drop=True)
y_all = df[TARGET].values.astype(np.float32)
print(f'Unique molecules: {len(df)}, gap={y_all.mean():.4f}±{y_all.std():.4f}', flush=True)

# ─── 2. Fingerprints ─────────────────────────────────────────────────────────
print('Computing fingerprints...', flush=True)
t0 = time.time()
fp_list, fp2048_list = [], []
for smi in df['smile']:
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=FP_DIM)
        arr = np.zeros(FP_DIM, dtype=np.float32)
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
        fp_list.append(arr)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        arr2 = np.zeros(2048, dtype=np.float32)
        AllChem.DataStructs.ConvertToNumpyArray(fp2, arr2)
        fp2048_list.append(arr2)
    else:
        fp_list.append(np.zeros(FP_DIM, dtype=np.float32))
        fp2048_list.append(np.zeros(2048, dtype=np.float32))
X_fp = np.stack(fp_list)
X_fp2048 = np.stack(fp2048_list)
print(f'Fingerprints: {X_fp.shape}, {time.time()-t0:.1f}s', flush=True)

# ─── 3. Graphs ────────────────────────────────────────────────────────────────
ATOM_DIM = 30
FULL_ATOM_FEATS = {
    'atomic_num': [1, 6, 7, 8, 9, 15, 16, 17, 35],
    'degree': list(range(5)),
    'hybridization': list(range(1, 6)),
}

def make_graph(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except:
        return None
    xs = []
    for a in mol.GetAtoms():
        n = a.GetAtomicNum(); d = a.GetDegree(); fc = a.GetFormalCharge()
        ar = int(a.GetIsAromatic()); ri = int(a.IsInRing())
        try:
            hyb = int(a.GetHybridization()); nh = a.GetTotalNumHs(); v = a.GetTotalValence()
            r3 = int(a.IsInRingSize(3)); r4 = int(a.IsInRingSize(4))
            r5 = int(a.IsInRingSize(5)); r6 = int(a.IsInRingSize(6))
        except:
            hyb = nh = v = r3 = r4 = r5 = r6 = 0
        feat = [n/100., d/6., fc/8., nh/4., v/8., ar, ri, r3, r4, r5, r6] \
            + [int(n == a) for a in FULL_ATOM_FEATS['atomic_num']] \
            + [int(d == dd) for dd in FULL_ATOM_FEATS['degree']] \
            + [int(hyb == h) for h in FULL_ATOM_FEATS['hybridization']]
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

# ─── 4. Split ──────────────────────────────────────────────────────────────────
rng42 = np.random.RandomState(42)
all_idx = np.arange(len(df))
test_idx = set(rng42.choice(len(df), TEST_SIZE, replace=False))
test_arr = np.array([i for i in range(len(df)) if i in test_idx])
train_pool_arr = np.array([i for i in range(len(df)) if i not in test_idx])
y_test = y_all[test_arr]
print(f'Test: {len(test_arr)}, Train pool: {len(train_pool_arr)}', flush=True)

# Build test set graphs
print('Building test graphs...', flush=True)
t0 = time.time()
test_graphs = []
test_valid_idx = []
for idx, i in enumerate(test_arr):
    g = make_graph(df.iloc[i]['smile'])
    if g is not None:
        g.y = torch.tensor([y_all[i]], dtype=torch.float)
        g.fp = torch.tensor(X_fp[i], dtype=torch.float).unsqueeze(0)
        test_graphs.append(g)
        test_valid_idx.append(idx)
test_arr = test_arr[test_valid_idx]
y_test = y_all[test_arr]
X_test_fp = X_fp[test_arr]
X_test_fp2048 = X_fp2048[test_arr]
print(f'Test graphs: {len(test_graphs)}, {time.time()-t0:.1f}s', flush=True)

# ─── 5. Model ──────────────────────────────────────────────────────────────────
class OPVGNN(nn.Module):
    """2-branch GCN+GAT + fingerprint fusion (same as QM9GNN)."""
    def __init__(self, in_channels=ATOM_DIM, hidden=128, fp_dim=FP_DIM, dropout=0.3):
        super().__init__()
        self.gcn1 = GCNConv(in_channels, hidden)
        self.gcn2 = GCNConv(hidden, hidden)
        self.gat1 = GATConv(in_channels, hidden//4, heads=4, dropout=dropout)
        self.gat2 = GATConv(hidden, hidden, heads=1, dropout=dropout)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.fp_encoder = nn.Sequential(
            nn.Linear(fp_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(),
        )
        fused_dim = hidden * 6 + 128
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
            return torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch), global_add_pool(h, batch)], dim=1)
        g = torch.cat([pool3(x_gcn), pool3(x_gat)], dim=1)
        fp_feat = self.fp_encoder(data.fp)
        g = torch.cat([g, fp_feat], dim=1)
        return self.regressor(g).squeeze(1)

def train_gnn(train_data, val_data, seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = OPVGNN(in_channels=ATOM_DIM).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.HuberLoss(delta=1.0)
    tr_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
    best_mae = float('inf'); best_sd = None; patience = 0
    for epoch in range(1, 151):
        model.train()
        for b in tr_loader:
            b = b.to(device); opt.zero_grad()
            criterion(model(b), b.y.view(-1)).backward(); opt.step()
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
    model.eval(); preds = []
    for b in loader:
        preds.append(model(b.to(device)).cpu())
    return torch.cat(preds).numpy()

# ─── 6. Run ────────────────────────────────────────────────────────────────────
print('\n' + '='*55, flush=True)
print('NREL OPV Validation: XGBoost vs GNN', flush=True)
print('='*55, flush=True)

results = {}

# XGBoost full data baseline
xgb_full = xgb.XGBRegressor(
    n_estimators=1000, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    reg_alpha=0.01, reg_lambda=0.01,
    random_state=42, verbosity=0, n_jobs=-1)
xgb_full.fit(X_fp2048[train_pool_arr], y_all[train_pool_arr],
             eval_set=[(X_fp2048[test_arr], y_test)], verbose=False)
xgb_full_pred = xgb_full.predict(X_test_fp2048)
xgb_full_r2 = r2_score(y_test, xgb_full_pred)
print(f'XGBoost full (n={len(train_pool_arr)}): R²={xgb_full_r2:.4f}', flush=True)

test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE)

for n in N_VALUES:
    print(f'\n=== n={n} ===', flush=True)
    n_results = []

    for trial, seed in enumerate(SEEDS):
        rng = np.random.RandomState(seed + n)
        chosen = rng.choice(train_pool_arr, n, replace=False)
        tr, va = train_test_split(chosen, test_size=VAL_FRAC, random_state=int(seed + n))

        # XGBoost
        xgb_m = xgb.XGBRegressor(
            n_estimators=1000, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            reg_alpha=0.01, reg_lambda=0.01,
            early_stopping_rounds=50, random_state=seed, verbosity=0, n_jobs=-1)
        xgb_m.fit(X_fp2048[tr], y_all[tr], eval_set=[(X_fp2048[va], y_all[va])], verbose=False)
        xgb_r2 = r2_score(y_test, xgb_m.predict(X_test_fp2048))

        # GNN
        tr_graphs = []
        for i in tr:
            g = make_graph(df.iloc[i]['smile'])
            if g is not None:
                g.y = torch.tensor([y_all[i]], dtype=torch.float)
                g.fp = torch.tensor(X_fp[i], dtype=torch.float).unsqueeze(0)
                tr_graphs.append(g)
        va_graphs = []
        for i in va:
            g = make_graph(df.iloc[i]['smile'])
            if g is not None:
                g.y = torch.tensor([y_all[i]], dtype=torch.float)
                g.fp = torch.tensor(X_fp[i], dtype=torch.float).unsqueeze(0)
                va_graphs.append(g)

        if len(tr_graphs) < 10 or len(va_graphs) < 5:
            print(f'  Trial {trial+1}: too few graphs, skip', flush=True)
            continue

        t0 = time.time()
        gm = train_gnn(tr_graphs, va_graphs, seed)
        gp = eval_gnn(gm, test_loader)
        gnn_r2 = r2_score(y_test, gp)
        train_time = time.time() - t0

        n_results.append({
            'seed': seed, 'xgb_r2': float(xgb_r2), 'gnn_r2': float(gnn_r2),
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

    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)

# ─── 7. Summary ──────────────────────────────────────────────────────────────
print(f'\n{"="*55}', flush=True)
print('NREL OPV Validation Results', flush=True)
print(f'{"="*55}', flush=True)
print(f'{"n":<8} {"XGB R²":<15} {"GNN R²":<15} {"ΔR²":<15}', flush=True)
print('-'*55, flush=True)
for n in N_VALUES:
    r = results[str(n)]
    print(f'{n:<8} {r["xgb_r2_mean"]:<15.4f} {r["gnn_r2_mean"]:<15.4f} {r["delta_r2_mean"]:<15.4f}', flush=True)
print(f'{"Full":<8} {xgb_full_r2:<15.4f} {"—":<15}', flush=True)
print(f'\nComparison with QM9:')
print(f'  QM9:   n=100: XGB≈0.64 GNN≈0.60 | n=500: XGB≈0.75 GNN≈0.81')
print(f'  NREL:  n=100: XGB={results["100"]["xgb_r2_mean"]:.4f} GNN={results["100"]["gnn_r2_mean"]:.4f}', flush=True)
print('Done.', flush=True)
