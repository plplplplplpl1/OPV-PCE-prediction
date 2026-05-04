"""
Virtual Screening Simulation: XGBoost vs GNN
=============================================
模拟虚拟筛选场景：给定有限的训练数据，用各模型对候选库打分，
看top-k中有多少真正的高效材料。

直接回答：XGBoost的样本效率优势能否转化为更好的筛选效果？
"""
import os, sys, json, random, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

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
RESULTS_FILE = os.path.join(BASE_DIR, 'external_results', 'screening_simulation.json')
PCE_THRESHOLD = 3.0
FP_DIM = 512
SCREENING_TOP_K = [5, 10, 20, 50, 100]
TRAIN_SIZES = [68, 137, 344, 689, 1033, 1378]
SEEDS = [42, 123, 456, 9999]
XGB_PARAMS = {
    'n_estimators': 2000, 'learning_rate': 0.0117, 'max_depth': 6,
    'min_child_weight': 5, 'subsample': 0.595, 'colsample_bytree': 0.626,
    'reg_alpha': 0.01, 'reg_lambda': 0.01, 'gamma': 0.1,
    'random_state': 42, 'verbosity': 0, 'n_jobs': -1
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# ─── 1. Load ────────────────────────────────────────────────────────────────
print('Loading data...')
df = pd.read_csv(DATA_PATH, encoding='latin-1')
df.columns = df.columns.str.strip()
pce_col = [c for c in df.columns if 'pce' in c.lower()][0]
smiles_col = [c for c in df.columns if 'smiles' in c.lower()][0]
df[pce_col] = pd.to_numeric(df[pce_col], errors='coerce')
df = df.dropna(subset=[pce_col, smiles_col]).reset_index(drop=True)
df_high = df[df[pce_col] > PCE_THRESHOLD].copy().reset_index(drop=True)
print(f'Total: {len(df)}, High-PCE: {len(df_high)}')

# ─── 2. Compute fingerprints & graphs ──────────────────────────────────────
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
y_vals = []
fps_xgb_list = []
smiles_valid = []
for _, row in df_high.iterrows():
    smi = row[smiles_col]
    mol = Chem.MolFromSmiles(smi)
    if mol is None: continue
    g = smiles_to_graph(smi)
    if g is None: continue
    g.y = torch.tensor([float(row[pce_col])], dtype=torch.float)
    graphs.append(g)
    y_vals.append(float(row[pce_col]))
    fp2048 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    fps_xgb_list.append(np.array(fp2048, dtype=np.float32))
    smiles_valid.append(smi)

y_vals = np.array(y_vals)
fps_xgb = np.array(fps_xgb_list)
n_total = len(graphs)
in_dim = graphs[0].x.shape[1]
print(f'Valid: {n_total}')

# ─── 3. Model ────────────────────────────────────────────────────────────────
class ScreeningGNN(nn.Module):
    def __init__(self, in_channels, hidden=128, fp_dim=FP_DIM, dropout=0.3):
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

def train_gnn(train_g, val_g, seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = ScreeningGNN(in_channels=in_dim).to(device)
    opt = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.HuberLoss(delta=1.0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=8, factor=0.5)
    tr_loader = GeoDataLoader(train_g, batch_size=32, shuffle=True)
    va_loader = GeoDataLoader(val_g, batch_size=32)
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
        scheduler.step(mae)
        if mae < best_mae: best_mae = mae; best_sd = {k: v.clone() for k, v in model.state_dict().items()}; patience = 0
        else: patience += 1
        if patience >= 20: break
    model.load_state_dict(best_sd)
    return model

# ─── 4. Screening simulation ────────────────────────────────────────────────
def hit_rate(y_true, y_pred, top_k, threshold=5.0):
    """How many of top-k predicted are actually above threshold?"""
    top_idx = np.argsort(y_pred)[-top_k:]
    hits = np.sum(y_true[top_idx] > threshold)
    return hits / top_k

results = {}

for n_train in TRAIN_SIZES:
    print(f'\n{"="*50}')
    print(f'Training size: {n_train}')
    print(f'{"="*50}')

    xgb_hit_rates = {k: [] for k in SCREENING_TOP_K}
    gnn_hit_rates = {k: [] for k in SCREENING_TOP_K}
    xgb_r2_list = []
    gnn_r2_list = []

    for seed in SEEDS:
        # Split train/test (fixed test = candidate library)
        all_idx = np.arange(n_total)
        tr_i, te_i = train_test_split(all_idx, test_size=0.3, random_state=seed)

        # Subset training size
        if len(tr_i) > n_train:
            rng = np.random.RandomState(seed)
            tr_i = np.sort(rng.choice(tr_i, n_train, replace=False))

        # Validation (20% of training)
        tr_i2, va_i = train_test_split(tr_i, test_size=0.2, random_state=seed)
        tr_i = tr_i2

        # ── XGBoost ──
        xgb_m = xgb.XGBRegressor(**XGB_PARAMS)
        xgb_m.fit(fps_xgb[tr_i], y_vals[tr_i],
                  eval_set=[(fps_xgb[va_i], y_vals[va_i])], verbose=False)
        xgb_pred = xgb_m.predict(fps_xgb[te_i])
        xgb_r2_list.append(r2_score(y_vals[te_i], xgb_pred))

        for k in SCREENING_TOP_K:
            hr = hit_rate(y_vals[te_i], xgb_pred, k)
            xgb_hit_rates[k].append(hr)

        # ── GNN ──
        tr_g = [graphs[i] for i in tr_i]
        va_g = [graphs[i] for i in va_i]
        te_g = [graphs[i] for i in te_i]

        if len(tr_g) >= 10 and len(va_g) >= 5:
            try:
                gnn_m = train_gnn(tr_g, va_g, seed)
                gnn_m.eval()
                te_loader = GeoDataLoader(te_g, batch_size=32)
                preds = []
                with torch.no_grad():
                    for b in te_loader:
                        b = b.to(device)
                        preds.append(gnn_m(b).cpu())
                gnn_pred = torch.cat(preds).numpy()
                gnn_r2_list.append(r2_score(y_vals[te_i], gnn_pred))

                for k in SCREENING_TOP_K:
                    hr = hit_rate(y_vals[te_i], gnn_pred, k)
                    gnn_hit_rates[k].append(hr)
            except Exception as e:
                print(f'  GNN seed={seed} failed: {e}')
        else:
            print(f'  GNN seed={seed}: too few training samples')

    # Aggregate
    n_seeds = len(SEEDS)
    summary = {'xgb': {}, 'gnn': {}}

    for k in SCREENING_TOP_K:
        summary['xgb'][f'top{k}'] = {
            'mean': float(np.mean(xgb_hit_rates[k])),
            'std': float(np.std(xgb_hit_rates[k])),
        }
        if gnn_hit_rates[k]:
            summary['gnn'][f'top{k}'] = {
                'mean': float(np.mean(gnn_hit_rates[k])),
                'std': float(np.std(gnn_hit_rates[k])),
            }

    summary['xgb']['r2'] = {'mean': float(np.mean(xgb_r2_list)), 'std': float(np.std(xgb_r2_list))}
    if gnn_r2_list:
        summary['gnn']['r2'] = {'mean': float(np.mean(gnn_r2_list)), 'std': float(np.std(gnn_r2_list))}

    results[str(n_train)] = summary

    # Print
    print(f'\n  XGBoost R²: {np.mean(xgb_r2_list):.4f}±{np.std(xgb_r2_list):.4f}')
    if gnn_r2_list:
        print(f'  GNN R²:     {np.mean(gnn_r2_list):.4f}±{np.std(gnn_r2_list):.4f}')
    print(f'\n  Screening hit rates (target >5% PCE):')
    print(f'  {"Top-k":<8} {"XGBoost":<20} {"GNN":<20}')
    for k in SCREENING_TOP_K:
        xgb_hr = summary['xgb'][f'top{k}']
        gnn_hr = summary['gnn'].get(f'top{k}', {'mean': float('nan'), 'std': float('nan')})
        print(f'  {k:<8} {xgb_hr["mean"]:.3f}±{xgb_hr["std"]:.3f}     {gnn_hr["mean"]:.3f}±{gnn_hr["std"]:.3f}')

# ─── 5. Save ────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
with open(RESULTS_FILE, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nResults saved to {RESULTS_FILE}')
print('Done.')
