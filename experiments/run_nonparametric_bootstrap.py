"""
非参数 Bootstrap 统计检验
===============================
对 OPV 高PCE回归任务进行非参数Bootstrap验证。
方法：在测试集上有放回重采样（残差重采样），
计算 XGBoost vs GNN 的 ΔR²、ΔMAE、ΔRMSE 的Bootstrap置信区间。

与参数化Bootstrap的区别：直接重采样预测残差而非假设正态分布。
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool, global_max_pool, global_add_pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from collections import OrderedDict
import os, sys, json, warnings

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'data.csv')
MODEL_PATH = os.path.join(BASE_DIR, '保存的模型', 'best_high_pce_regressor_v3_seed9999.pth')
RESULTS_FILE = os.path.join(BASE_DIR, 'external_results', 'nonparametric_bootstrap.json')
print(f'Base dir: {BASE_DIR}')
print(f'Data: {DATA_PATH}')
print(f'Model: {MODEL_PATH}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# ─── Constants ────────────────────────────────────────────────────────────
FP_DIM_XGB = 4096
FP_DIM_GNN = 512
PCE_THRESHOLD = 3.0
TEST_SIZE = 0.2
RANDOM_SEED = 9999
N_BOOTSTRAP = 2000

# ─── 1. Load Data ─────────────────────────────────────────────────────────
print('Loading data...')
df = pd.read_csv(DATA_PATH, encoding='latin-1')
df.columns = df.columns.str.strip()
pce_col = [c for c in df.columns if 'pce' in c.lower()][0]
smiles_col = [c for c in df.columns if 'smiles' in c.lower()][0]
df[pce_col] = pd.to_numeric(df[pce_col], errors='coerce')
df = df.dropna(subset=[pce_col, smiles_col]).reset_index(drop=True)
df_high = df[df[pce_col] > PCE_THRESHOLD].copy().reset_index(drop=True)
print(f'High-PCE samples: {len(df_high)}')

# ─── 2. Compute features for XGBoost ─────────────────────────────────────
CORE_DESC = OrderedDict([
    ('MolWt', Descriptors.MolWt), ('MolLogP', Descriptors.MolLogP),
    ('TPSA', Descriptors.TPSA), ('AromaticRings', Descriptors.NumAromaticRings),
    ('AliphaticRings', Descriptors.NumAliphaticRings),
    ('HDonors', Descriptors.NumHDonors), ('HAcceptors', Descriptors.NumHAcceptors),
    ('RotBonds', Descriptors.NumRotatableBonds), ('RingCount', Descriptors.RingCount),
    ('HeavyAtomCount', Descriptors.HeavyAtomCount),
])

print('Computing XGBoost features...')
X_fp_list, y_list, smiles_valid = [], [], []
for _, row in df_high.iterrows():
    mol = Chem.MolFromSmiles(row[smiles_col])
    if mol is None:
        continue
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=FP_DIM_XGB)
        arr = np.zeros(FP_DIM_XGB, dtype=np.float32)
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
        desc = np.array([fn(mol) for fn in CORE_DESC.values()], dtype=np.float32)
        X_fp_list.append(np.concatenate([arr, desc]))
        y_list.append(float(row[pce_col]))
        smiles_valid.append(row[smiles_col])
    except Exception:
        continue

X_all = np.array(X_fp_list)
y_all = np.array(y_list)
print(f'X shape: {X_all.shape}, y range: [{y_all.min():.2f}, {y_all.max():.2f}]')

# ─── 3. Compute graphs for GNN ───────────────────────────────────────────
print('Computing molecular graphs...')

def smiles_to_graph(smiles, fp_dim=FP_DIM_GNN):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
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
            except Exception:
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
        if not edge_indices:
            return None

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=fp_dim)
        fp_tensor = torch.tensor(np.array(fp, dtype=np.float32)).unsqueeze(0)

        x = torch.tensor(np.array(atom_features, dtype=np.float32))
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        return Data(x=x, edge_index=edge_index, fp=fp_tensor)
    except Exception:
        return None

graphs = []
valid_indices = []
for i, smi in enumerate(smiles_valid):
    g = smiles_to_graph(smi)
    if g is not None:
        graphs.append(g)
        valid_indices.append(i)

valid_indices = np.array(valid_indices)
X_all = X_all[valid_indices]
y_all = y_all[valid_indices]
n_total = len(graphs)
print(f'Valid graphs: {n_total}')

# ─── 4. Train/Test Split ─────────────────────────────────────────────────
tr_idx, te_idx = train_test_split(
    np.arange(n_total), test_size=TEST_SIZE, random_state=RANDOM_SEED)
print(f'Train: {len(tr_idx)}, Test: {len(te_idx)}')

X_tr, X_te = X_all[tr_idx], X_all[te_idx]
y_tr, y_te = y_all[tr_idx], y_all[te_idx]

graph_te = [graphs[i] for i in te_idx]
for i, g in enumerate(graphs):
    g.y = torch.tensor([y_all[i]], dtype=torch.float)

# ─── 5. Model: HighPCERegressorV3 ────────────────────────────────────────
class HighPCERegressorV3(nn.Module):
    def __init__(self, in_channels=30, hidden=128, fp_dim=FP_DIM_GNN, dropout=0.3):
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
        fp = data.fp
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
        fp_feat = self.fp_encoder(fp)
        g = torch.cat([g, fp_feat], dim=1)
        return self.regressor(g).squeeze(1)

# ─── 6. Load GNN Checkpoint ──────────────────────────────────────────────
print('Loading GNN checkpoint...')
gnn_model = HighPCERegressorV3().to(device)
state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
gnn_model.load_state_dict(state_dict)
gnn_model.eval()
print('  GNN loaded successfully.')

# ─── 7. Train XGBoost ────────────────────────────────────────────────────
# Use Optuna-optimized hyperparameters from the paper
print('Training XGBoost...')
xgb_model = xgb.XGBRegressor(
    n_estimators=2000, learning_rate=0.0117, max_depth=6,
    min_child_weight=5, subsample=0.595, colsample_bytree=0.626,
    reg_alpha=0.01, reg_lambda=0.01, gamma=0.1,
    early_stopping_rounds=50, random_state=RANDOM_SEED, verbosity=0, n_jobs=-1)
xgb_model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
y_pred_xgb = xgb_model.predict(X_te)
r2_xgb = r2_score(y_te, y_pred_xgb)
mae_xgb = mean_absolute_error(y_te, y_pred_xgb)
print(f'  XGBoost: R²={r2_xgb:.4f}, MAE={mae_xgb:.4f}')

# ─── 8. Get GNN Predictions ──────────────────────────────────────────────
@torch.no_grad()
def predict_gnn(model, graph_list, batch_size=32):
    loader = GeoDataLoader(graph_list, batch_size=batch_size)
    preds = []
    for batch in loader:
        preds.append(model(batch.to(device)).cpu())
    return torch.cat(preds).numpy()

y_pred_gnn = predict_gnn(gnn_model, graph_te)
r2_gnn = r2_score(y_te, y_pred_gnn)
mae_gnn = mean_absolute_error(y_te, y_pred_gnn)
rmse_gnn = np.sqrt(mean_squared_error(y_te, y_pred_gnn))
print(f'  GNN:      R²={r2_gnn:.4f}, MAE={mae_gnn:.4f}')

n_test = len(y_te)

# ─── 9. Non-parametric Bootstrap ─────────────────────────────────────────
print(f'\nRunning {N_BOOTSTRAP} non-parametric bootstrap iterations...')
rng = np.random.RandomState(42)

delta_r2_list, delta_mae_list, delta_rmse_list = [], [], []
xgb_r2_list, gnn_r2_list = [], []

for i in range(N_BOOTSTRAP):
    # Resample test indices with replacement
    idx = rng.randint(0, n_test, n_test)

    y_boot = y_te[idx]
    y_xgb_boot = y_pred_xgb[idx]
    y_gnn_boot = y_pred_gnn[idx]

    r2_x = r2_score(y_boot, y_xgb_boot)
    r2_g = r2_score(y_boot, y_gnn_boot)
    mae_x = mean_absolute_error(y_boot, y_xgb_boot)
    mae_g = mean_absolute_error(y_boot, y_gnn_boot)
    rmse_x = np.sqrt(mean_squared_error(y_boot, y_xgb_boot))
    rmse_g = np.sqrt(mean_squared_error(y_boot, y_gnn_boot))

    delta_r2_list.append(r2_x - r2_g)
    delta_mae_list.append(mae_g - mae_x)  # positive = GNN has higher MAE = XGB better
    delta_rmse_list.append(rmse_g - rmse_x)
    xgb_r2_list.append(r2_x)
    gnn_r2_list.append(r2_g)

    if (i + 1) % 500 == 0:
        print(f'  {i+1}/{N_BOOTSTRAP}')

delta_r2 = np.array(delta_r2_list)
delta_mae = np.array(delta_mae_list)
delta_rmse = np.array(delta_rmse_list)

def summarize(deltas, label):
    mean_d = float(np.mean(deltas))
    std_d = float(np.std(deltas))
    ci_low = float(np.percentile(deltas, 2.5))
    ci_high = float(np.percentile(deltas, 97.5))
    if label.startswith('R'):
        p_val = float((deltas <= 0).mean())
    else:
        p_val = float((deltas <= 0).mean())
    sig = 'YES' if p_val < 0.05 else 'NO'
    print(f'  Δ{label}: mean={mean_d:.4f}, 95% CI=[{ci_low:.4f}, {ci_high:.4f}], p={p_val:.4f} [{sig}]')
    return {'mean': mean_d, 'std': std_d, 'ci_95': [ci_low, ci_high], 'p_value': p_val}

print('\n── Non-parametric Bootstrap Results ──')
r2_stats = summarize(delta_r2, 'R²')
mae_stats = summarize(delta_mae, 'MAE')
rmse_stats = summarize(delta_rmse, 'RMSE')

# ─── 10. Save ────────────────────────────────────────────────────────────
results = {
    'method': 'non-parametric bootstrap (residual resampling)',
    'n_bootstrap': N_BOOTSTRAP,
    'n_test': int(n_test),
    'split_seed': RANDOM_SEED,
    'xgb_r2': float(r2_xgb),
    'xgb_mae': float(mae_xgb),
    'xgb_rmse': float(np.sqrt(mean_squared_error(y_te, y_pred_xgb))),
    'gnn_r2': float(r2_gnn),
    'gnn_mae': float(mae_gnn),
    'gnn_rmse': float(rmse_gnn),
    'delta_r2': r2_stats,
    'delta_mae': mae_stats,
    'delta_rmse': rmse_stats,
}
with open(RESULTS_FILE, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nResults saved to {RESULTS_FILE}')

print('\n── Comparison with Parametric Bootstrap (manuscript) ──')
print(f'  Manuscript reports: ΔR² 95% CI=[-0.0246, 0.1234], p=0.084')
print(f'  Non-parametric:     ΔR² 95% CI=[{r2_stats["ci_95"][0]:.4f}, {r2_stats["ci_95"][1]:.4f}], p={r2_stats["p_value"]:.4f}')
print(f'  Manuscript reports: ΔMAE p=0.036')
print(f'  Non-parametric:     ΔMAE p={mae_stats["p_value"]:.4f}')
print(f'  Manuscript reports: ΔRMSE p=0.012')
print(f'  Non-parametric:     ΔRMSE p={rmse_stats["p_value"]:.4f}')
print('Done.')
