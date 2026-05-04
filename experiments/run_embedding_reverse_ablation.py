"""
GNN Embedding Reverse Ablation Experiment
===========================================
Purpose: Test whether GNN embeddings capture information BEYOND what
Morgan fingerprints + RDKit descriptors provide.

Design:
- "Hard mode": XGBoost with ONLY 12 RDKit descriptors (no Morgan fingerprints)
  trained on SMALL subsets (n=68 to 1378)
- Compare: XGBoost(12_desc) vs XGBoost(12_desc + GNN_384_embeddings)
- If GNN embeddings significantly improve XGBoost in hard mode, it means GNN
  learned structure info that simple descriptors miss.

Additional comparison: XGBoost(512_fp) vs XGBoost(512_fp + GNN_emb)
to test if GNN embeddings are redundant with fingerprints specifically.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import xgboost as xgb
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
import os, sys, json, warnings, random

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'data.csv')
MODEL_PATH = os.path.join(BASE_DIR, '保存的模型', 'best_high_pce_regressor_v3_seed9999.pth')
RESULTS_FILE = os.path.join(BASE_DIR, 'external_results', 'embedding_reverse_ablation.json')

PCE_THRESHOLD = 3.0
FP_DIM = 512
N_VALUES = [68, 137, 344, 689, 1033, 1378]
SEEDS = [42, 123, 333, 9999]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# ─── 1. Model Definition (same as HighPCERegressorV3) ──────────────────────
class GNNEmbeddingExtractor(nn.Module):
    """HighPCERegressorV3 with embedding extraction."""
    def __init__(self, in_channels=30, hidden=128, fp_dim=FP_DIM, dropout=0.3):
        super().__init__()
        self.gcn1 = GCNConv(in_channels, hidden)
        self.gcn2 = GCNConv(hidden, hidden)
        self.gat1 = GATConv(in_channels, hidden // 4, heads=4, dropout=dropout)
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
        self.hidden = hidden

    def forward(self, data, return_embedding=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        fp = data.fp
        x_gcn = F.relu(self.bn1(self.gcn1(x, edge_index)))
        x_gcn = F.relu(self.gcn2(x_gcn, edge_index))
        x_gat = F.relu(self.gat1(x, edge_index))
        x_gat = F.relu(self.bn2(self.gat2(x_gat, edge_index)))
        x_sage = F.relu(self.sage1(x, edge_index))
        x_sage = F.relu(self.bn3(self.sage2(x_sage, edge_index)))

        def pool3(h):
            return torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch), global_add_pool(h, batch)], dim=1)

        g = torch.cat([pool3(x_gcn), pool3(x_gat), pool3(x_sage)], dim=1)
        if return_embedding:
            return g  # 384-dim graph embedding (no fp encoder)
        fp_feat = self.fp_encoder(fp)
        g = torch.cat([g, fp_feat], dim=1)
        return self.regressor(g).squeeze(1)

# ─── 2. Data Loading ────────────────────────────────────────────────────────
print('Loading data...')
df = pd.read_csv(DATA_PATH, encoding='latin-1')
df.columns = df.columns.str.strip()
pce_col = [c for c in df.columns if 'pce' in c.lower()][0]
smiles_col = [c for c in df.columns if 'smiles' in c.lower()][0]

df[pce_col] = pd.to_numeric(df[pce_col], errors='coerce')
df = df[df[pce_col] > PCE_THRESHOLD].dropna(subset=[pce_col, smiles_col]).reset_index(drop=True)
y_all = df[pce_col].values.astype(np.float32)
N = len(df)
print(f'High-PCE samples: {N}')

# ─── 3. Compute Features ────────────────────────────────────────────────────
# Morgan fingerprints
print('Computing Morgan fingerprints...')
fp_list = []
for smi in df[smiles_col]:
    mol = Chem.MolFromSmiles(smi)
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=FP_DIM)
        arr = np.zeros(FP_DIM, dtype=np.float32)
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    else:
        arr = np.zeros(FP_DIM, dtype=np.float32)
    fp_list.append(arr)
X_fp = np.stack(fp_list)
print(f'  Fingerprints: {X_fp.shape}')

# RDKit 12 core descriptors
DESC_FUNCS = [
    ('MolWt', Descriptors.MolWt), ('MolLogP', Descriptors.MolLogP),
    ('TPSA', Descriptors.TPSA), ('NumHDonors', Descriptors.NumHDonors),
    ('NumHAcceptors', Descriptors.NumHAcceptors), ('NumRotatableBonds', Descriptors.NumRotatableBonds),
    ('RingCount', Descriptors.RingCount), ('NumAromaticRings', Descriptors.NumAromaticRings),
    ('FractionCSP3', Descriptors.FractionCSP3), ('HeavyAtomCount', Descriptors.HeavyAtomCount),
    ('NHOHCount', Descriptors.NHOHCount), ('NOCount', Descriptors.NOCount),
]
print('Computing RDKit descriptors...')
desc_list = []
for smi in df[smiles_col]:
    mol = Chem.MolFromSmiles(smi)
    if mol:
        desc_list.append([f(mol) for _, f in DESC_FUNCS])
    else:
        desc_list.append([0.0] * len(DESC_FUNCS))
X_desc = np.array(desc_list, dtype=np.float32)
print(f'  Descriptors: {X_desc.shape}')

# ─── 4. Build Graphs & Extract GNN Embeddings ──────────────────────────────
FULL_ATOM_FEATS = {
    'atomic_num': [1, 6, 7, 8, 9, 15, 16, 17, 35],
    'degree': list(range(5)),
    'hybridization': list(range(1, 6)),
}

def make_graph(smi, fp_vec):
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
        edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
        fp=torch.tensor(fp_vec, dtype=torch.float).unsqueeze(0),
    )

# Build graphs
print('Building graphs...')
all_graphs = []
valid_idx = []
for i in range(N):
    g = make_graph(df[smiles_col].iloc[i], X_fp[i])
    if g is not None:
        g.y = torch.tensor([y_all[i]], dtype=torch.float)
        all_graphs.append(g)
        valid_idx.append(i)
print(f'  Valid graphs: {len(all_graphs)}')

# Filter to molecules with valid graphs only
valid_idx = np.array(valid_idx)
y_all = y_all[valid_idx]
X_desc = X_desc[valid_idx]
X_fp = X_fp[valid_idx]
gnn_emb = None  # will be set after extraction

# Extract GNN embeddings (384-dim, pre-fp_encoder)
print(f'Loading GNN model from {MODEL_PATH}...')
in_dim = all_graphs[0].x.shape[1]
model = GNNEmbeddingExtractor(in_channels=in_dim).to(device)
state = torch.load(MODEL_PATH, map_location=device, weights_only=False)
# Handle possible key mismatches
model_state = model.state_dict()
filtered_state = {k: v for k, v in state.items() if k in model_state and v.shape == model_state[k].shape}
model_state.update(filtered_state)
model.load_state_dict(model_state)
model.eval()

print('Extracting 384-dim graph embeddings...')
loader = GeoDataLoader(all_graphs, batch_size=64)
all_embeddings = []
with torch.no_grad():
    for batch in loader:
        batch = batch.to(device)
        emb = model(batch, return_embedding=True)
        all_embeddings.append(emb.cpu().numpy())
gnn_emb = np.concatenate(all_embeddings, axis=0)
print(f'  GNN embeddings: {gnn_emb.shape}')

# ─── 5. Train/Test Split ────────────────────────────────────────────────────
# Use seed=9999 split consistent with the pre-trained GNN
rng = np.random.RandomState(9999)
indices = np.arange(len(all_graphs))
test_idx = rng.choice(len(all_graphs), int(len(all_graphs) * 0.2), replace=False)
train_pool = np.array([i for i in range(len(all_graphs)) if i not in test_idx])

y_test = y_all[test_idx]
X_desc_test = X_desc[test_idx]
X_fp_test = X_fp[test_idx]
gnn_emb_test = gnn_emb[test_idx]
print(f'Train pool: {len(train_pool)}, Test: {len(test_idx)}')

# ─── 6. Run Reverse Ablation ────────────────────────────────────────────────
results = {}
for n in N_VALUES:
    print(f'\n=== n={n} ===')
    n_results = []

    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        chosen = rng.choice(train_pool, min(n, len(train_pool)), replace=False)
        tr, va = train_test_split(chosen, test_size=0.1, random_state=seed)

        # XGBoost with ONLY 12 descriptors (hard mode)
        xgb_hard = xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8,
            random_state=seed, verbosity=0, n_jobs=-1)
        xgb_hard.fit(X_desc[tr], y_all[tr], eval_set=[(X_desc[va], y_all[va])], verbose=False)
        pred_hard = xgb_hard.predict(X_desc_test)
        r2_hard = r2_score(y_test, pred_hard)

        # XGBoost with 12 descriptors + GNN embeddings
        X_aug = np.concatenate([X_desc, gnn_emb], axis=1)
        xgb_aug = xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8,
            random_state=seed, verbosity=0, n_jobs=-1)
        xgb_aug.fit(X_aug[tr], y_all[tr], eval_set=[(X_aug[va], y_all[va])], verbose=False)
        pred_aug = xgb_aug.predict(X_aug[test_idx])
        r2_aug = r2_score(y_test, pred_aug)

        # XGBoost with 512-bit Morgan (reference)
        xgb_fp = xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8,
            random_state=seed, verbosity=0, n_jobs=-1)
        xgb_fp.fit(X_fp[tr], y_all[tr], eval_set=[(X_fp[va], y_all[va])], verbose=False)
        pred_fp = xgb_fp.predict(X_fp_test)
        r2_fp = r2_score(y_test, pred_fp)

        # XGBoost with 512-bit Morgan + GNN embeddings
        X_fp_aug = np.concatenate([X_fp, gnn_emb], axis=1)
        xgb_fp_aug = xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8,
            random_state=seed, verbosity=0, n_jobs=-1)
        xgb_fp_aug.fit(X_fp_aug[tr], y_all[tr], eval_set=[(X_fp_aug[va], y_all[va])], verbose=False)
        pred_fp_aug = xgb_fp_aug.predict(X_fp_aug[test_idx])
        r2_fp_aug = r2_score(y_test, pred_fp_aug)

        n_results.append({
            'seed': seed,
            'r2_desc_only': float(r2_hard),
            'r2_desc_plus_gnn': float(r2_aug),
            'delta_desc': float(r2_aug - r2_hard),
            'r2_fp_only': float(r2_fp),
            'r2_fp_plus_gnn': float(r2_fp_aug),
            'delta_fp': float(r2_fp_aug - r2_fp),
        })
        print(f'  seed={seed:4d} | desc={r2_hard:.4f} desc+gnn={r2_aug:.4f} Δ={r2_aug-r2_hard:+.4f} | fp={r2_fp:.4f} fp+gnn={r2_fp_aug:.4f} Δ={r2_fp_aug-r2_fp:+.4f}')

    # Aggregate
    desc_means = [r['r2_desc_only'] for r in n_results]
    desc_gnn_means = [r['r2_desc_plus_gnn'] for r in n_results]
    fp_means = [r['r2_fp_only'] for r in n_results]
    fp_gnn_means = [r['r2_fp_plus_gnn'] for r in n_results]

    results[str(n)] = {
        'desc_only': {'mean': float(np.mean(desc_means)), 'std': float(np.std(desc_means))},
        'desc_plus_gnn': {'mean': float(np.mean(desc_gnn_means)), 'std': float(np.std(desc_gnn_means))},
        'delta_desc': float(np.mean([r['delta_desc'] for r in n_results])),
        'fp_only': {'mean': float(np.mean(fp_means)), 'std': float(np.std(fp_means))},
        'fp_plus_gnn': {'mean': float(np.mean(fp_gnn_means)), 'std': float(np.std(fp_gnn_means))},
        'delta_fp': float(np.mean([r['delta_fp'] for r in n_results])),
        'n_trials': len(n_results),
    }

    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)

# ─── 7. Full-data reference ─────────────────────────────────────────────────
print(f'\n{"="*55}')
print('Reverse Ablation Summary')
print(f'{"="*55}')
print(f'{"n":<6} {"Desc R²":<12} {"Desc+GNN R²":<14} {"ΔDesc":<10} {"FP R²":<12} {"FP+GNN R²":<14} {"ΔFP":<10}')
print('-'*72)
for n in N_VALUES:
    r = results[str(n)]
    print(f'{n:<6} {r["desc_only"]["mean"]:<8.4f}±{r["desc_only"]["std"]:<.4f}  '
          f'{r["desc_plus_gnn"]["mean"]:<8.4f}±{r["desc_plus_gnn"]["std"]:<.4f}  '
          f'{r["delta_desc"]:<+10.4f}  '
          f'{r["fp_only"]["mean"]:<8.4f}±{r["fp_only"]["std"]:<.4f}  '
          f'{r["fp_plus_gnn"]["mean"]:<8.4f}±{r["fp_plus_gnn"]["std"]:<.4f}  '
          f'{r["delta_fp"]:<+10.4f}')

print(f'\nInterpretation:')
print(f'  ΔDesc = R²(desc+GNN) - R²(desc only): how much GNN embeddings add beyond simple descriptors')
print(f'  ΔFP   = R²(FP+GNN) - R²(FP only): how much GNN embeddings add beyond Morgan fingerprints')
print(f'  If ΔDesc > 0.02 and ΔFP ≈ 0: GNN learns structure info (captured well by graph) but redundant with FP')
print(f'  If both ≈ 0: GNN embedding is largely noise / already captured by both feature types')
print(f'  If both > 0.02: GNN truly learns complementary information not captured by either fingerprint or descriptors')

print('\nDone.')
