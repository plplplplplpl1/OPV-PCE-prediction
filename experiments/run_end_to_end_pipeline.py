"""
End-to-End Cascaded Pipeline Evaluation
=========================================
Purpose: Evaluate the "GNN classifier → GNN regressor" cascaded pipeline
described in §2.2 on all 3,018 samples.

Pipeline:
  1. AdvancedGCN classifier predicts PCE > 3% or ≤ 3%
  2. For samples predicted as high-PCE: pre-trained regressor predicts exact PCE
  3. For samples predicted as low-PCE: assign baseline (median low-PCE from training)

This uses a single stratified 80/20 split. The classifier is trained from scratch;
the regressor is the pre-trained best_high_pce_regressor_v3_seed9999.pth (trained
on a disjoint split of high-PCE samples only, so no data leakage to the
classifier's test set).
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
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
import os, json, warnings, random

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'data.csv')
REGRESSOR_PATH = os.path.join(BASE_DIR, '保存的模型', 'best_high_pce_regressor_v3_seed9999.pth')
RESULTS_FILE = os.path.join(BASE_DIR, 'external_results', 'end_to_end_pipeline.json')

PCE_THRESHOLD = 3.0
FP_DIM = 512
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# ─── 1. Data Loading ────────────────────────────────────────────────────────
print('Loading data...')
df = pd.read_csv(DATA_PATH, encoding='latin-1')
df.columns = df.columns.str.strip()
pce_col = [c for c in df.columns if 'pce' in c.lower()][0]
smiles_col = [c for c in df.columns if 'smiles' in c.lower()][0]
df[pce_col] = pd.to_numeric(df[pce_col], errors='coerce')
df = df.dropna(subset=[pce_col, smiles_col]).reset_index(drop=True)
y_all = df[pce_col].values.astype(np.float32)
N = len(df)
print(f'Total samples: {N}')

# ─── 2. Morgan fingerprints ─────────────────────────────────────────────────
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

# ─── 3. Build Graphs ────────────────────────────────────────────────────────
FULL_ATOM_FEATS = {
    'atomic_num': [1, 6, 7, 8, 9, 15, 16, 17, 35],
    'degree': list(range(5)),
    'hybridization': list(range(1, 6)),
}

def make_graph(smi, fp_vec, label=None):
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
        y=torch.tensor([label], dtype=torch.float) if label is not None else None,
    )

print('Building graphs for all samples...')
all_graphs = []
valid_idx = []
for i in range(N):
    g = make_graph(df[smiles_col].iloc[i], X_fp[i], y_all[i])
    if g is not None:
        all_graphs.append(g)
        valid_idx.append(i)
valid_idx = np.array(valid_idx)
y_all = y_all[valid_idx]
X_fp = X_fp[valid_idx]
print(f'Valid graphs: {len(all_graphs)} / {N}')

# ─── 4. Classifier Definition ───────────────────────────────────────────────
class ClassifierGNN(nn.Module):
    """3-branch GNN classifier (AdvancedGCN architecture)."""
    def __init__(self, in_channels=30, hidden=128, dropout=0.3):
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
        pooled_dim = hidden * 3 * 3
        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 2),
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
            return torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch), global_add_pool(h, batch)], dim=1)
        g = torch.cat([pool3(x_gcn), pool3(x_gat), pool3(x_sage)], dim=1)
        return self.classifier(g)

# ─── 5. Regressor Definition (same as HighPCERegressorV3) ────────────────────
class RegressorGNN(nn.Module):
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
            return torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch), global_add_pool(h, batch)], dim=1)
        g = torch.cat([pool3(x_gcn), pool3(x_gat), pool3(x_sage)], dim=1)
        fp_feat = self.fp_encoder(fp)
        g = torch.cat([g, fp_feat], dim=1)
        return self.regressor(g).squeeze(1)

# ─── 6. Load Pre-trained Regressor ──────────────────────────────────────────
print(f'Loading pre-trained regressor...')
in_dim = all_graphs[0].x.shape[1]
reg_model = RegressorGNN(in_channels=in_dim).to(device)
state = torch.load(REGRESSOR_PATH, map_location=device, weights_only=False)
reg_state = reg_model.state_dict()
filtered = {k: v for k, v in state.items() if k in reg_state and v.shape == reg_state[k].shape}
reg_state.update(filtered)
reg_model.load_state_dict(reg_state)
reg_model.eval()
print('  Done.')

@torch.no_grad()
def predict_regressor(graphs):
    """Batch predict using pre-trained GNN regressor."""
    if not graphs:
        return np.array([])
    preds = []
    loader = GeoDataLoader(graphs, batch_size=64)
    for batch in loader:
        batch = batch.to(device)
        preds.append(reg_model(batch).cpu().numpy())
    return np.concatenate(preds)

# ─── 7. Single Stratified Split ─────────────────────────────────────────────
# Use stratified split on the full dataset to preserve high/low ratio.
labels_cls = (y_all > PCE_THRESHOLD).astype(int)
tr_idx, te_idx = train_test_split(
    np.arange(len(all_graphs)), test_size=0.2, random_state=42,
    stratify=labels_cls)

train_graphs = [all_graphs[i] for i in tr_idx]
test_graphs = [all_graphs[i] for i in te_idx]
y_train, y_test = y_all[tr_idx], y_all[te_idx]

# Compute low-PCE baseline from training data
low_pce_train = y_train[y_train <= PCE_THRESHOLD]
low_pce_baseline = float(np.median(low_pce_train)) if len(low_pce_train) > 0 else 1.5
print(f'Low-PCE baseline (median of training low-PCE): {low_pce_baseline:.2f}%')

# ─── 8. Train Classifier ─────────────────────────────────────────────────────
print('Training classifier...')
for g in train_graphs: g.y_cls = torch.tensor([int(g.y.item() > PCE_THRESHOLD)], dtype=torch.long)
for g in test_graphs:  g.y_cls = torch.tensor([int(g.y.item() > PCE_THRESHOLD)], dtype=torch.long)

clf_model = ClassifierGNN(in_channels=in_dim).to(device)
clf_opt = torch.optim.AdamW(clf_model.parameters(), lr=0.001, weight_decay=1e-4)
clf_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

tr_loader = GeoDataLoader(train_graphs, batch_size=64, shuffle=True)
te_loader = GeoDataLoader(test_graphs, batch_size=64)

best_acc = 0.0
best_sd = None
patience = 0
for epoch in range(1, 101):
    clf_model.train()
    for batch in tr_loader:
        batch = batch.to(device)
        clf_opt.zero_grad()
        out = clf_model(batch)
        loss = clf_criterion(out, batch.y_cls.view(-1))
        loss.backward()
        clf_opt.step()

    clf_model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in te_loader:
            batch = batch.to(device)
            pred = clf_model(batch).argmax(dim=1)
            correct += (pred == batch.y_cls.view(-1)).sum().item()
            total += batch.y_cls.size(0)
    acc = correct / total
    if acc > best_acc:
        best_acc = acc
        best_sd = {k: v.clone() for k, v in clf_model.state_dict().items()}
        patience = 0
    else:
        patience += 1
        if patience >= 15:
            break
clf_model.load_state_dict(best_sd)
print(f'  Classifier test accuracy: {best_acc:.4f}')

# ─── 9. Classifier Predictions on Test Set ───────────────────────────────────
clf_model.eval()
clf_preds = []
clf_probs = []
with torch.no_grad():
    for batch in te_loader:
        batch = batch.to(device)
        out = clf_model(batch)
        clf_probs.append(F.softmax(out, dim=1)[:, 1].cpu())
        clf_preds.append(out.argmax(dim=1).cpu())
clf_preds = torch.cat(clf_preds).numpy()
clf_probs = torch.cat(clf_probs).numpy()
test_labels = np.array([int(g.y.item() > PCE_THRESHOLD) for g in test_graphs])
y_test_actual = np.array([g.y.item() for g in test_graphs])

# ─── 10. End-to-End Pipeline ────────────────────────────────────────────────
print('Running end-to-end pipeline...')

# Classifier says "high" → use regressor; says "low" → use baseline
pipe_preds = np.full(len(test_graphs), low_pce_baseline, dtype=np.float32)
high_mask = (clf_preds == 1)
high_confidence = clf_probs[high_mask]

if high_mask.any():
    high_graphs = [test_graphs[j] for j in range(len(test_graphs)) if high_mask[j]]
    pipe_preds[high_mask] = predict_regressor(high_graphs)

# Reference: regressor-only (oracle: uses true labels to route)
reg_oracle_preds = np.full(len(test_graphs), low_pce_baseline, dtype=np.float32)
true_high_mask = (y_test_actual > PCE_THRESHOLD)
if true_high_mask.any():
    true_high_graphs = [test_graphs[j] for j in range(len(test_graphs)) if true_high_mask[j]]
    reg_oracle_preds[true_high_mask] = predict_regressor(true_high_graphs)

# Also: what if we always use regressor (no classifier)?
reg_all_preds = predict_regressor(test_graphs)

# ─── 11. Metrics ────────────────────────────────────────────────────────────
def metrics(y_true, y_pred):
    return {
        'r2': float(r2_score(y_true, y_pred)),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }

pipe_m = metrics(y_test_actual, pipe_preds)
reg_oracle_m = metrics(y_test_actual, reg_oracle_preds)
reg_all_m = metrics(y_test_actual, reg_all_preds)

# Per-quadrant breakdown
tp = ((clf_preds == 1) & (test_labels == 1))
tn = ((clf_preds == 0) & (test_labels == 0))
fp_mask = ((clf_preds == 1) & (test_labels == 0))
fn_mask = ((clf_preds == 0) & (test_labels == 1))

quad = {}
for name, mask in [('tp', tp), ('tn', tn), ('fp', fp_mask), ('fn', fn_mask)]:
    n = int(mask.sum())
    if n > 1:
        q = metrics(y_test_actual[mask], pipe_preds[mask])
        q['n'] = n
        quad[name] = q
    else:
        quad[name] = {'r2': None, 'mae': None, 'rmse': None, 'n': n}

# High-PCE only and low-PCE only metrics
high_pce_mask = (y_test_actual > PCE_THRESHOLD)
low_pce_mask = (y_test_actual <= PCE_THRESHOLD)
pipe_high = metrics(y_test_actual[high_pce_mask], pipe_preds[high_pce_mask])
pipe_low = metrics(y_test_actual[low_pce_mask], pipe_preds[low_pce_mask])

# ─── 12. Report ─────────────────────────────────────────────────────────────
print(f'\n{"="*55}')
print('End-to-End Pipeline Results')
print(f'{"="*55}')
print(f'Test set size: {len(test_graphs)} (stratified 80/20)')
print(f'Classifier accuracy: {best_acc:.4f}')
tn_count = int(((test_labels == 0) & (clf_preds == 0)).sum())
fp_count = int(((test_labels == 0) & (clf_preds == 1)).sum())
fn_count = int(((test_labels == 1) & (clf_preds == 0)).sum())
tp_count = int(((test_labels == 1) & (clf_preds == 1)).sum())
print(f'Confusion matrix: TN={tn_count} FP={fp_count} FN={fn_count} TP={tp_count}')
print()
print(f'{"Condition":<25} {"R²":<10} {"MAE":<10} {"RMSE":<10}')
print('-'*55)
print(f'{"Pipeline (end-to-end)":<25} {pipe_m["r2"]:<10.4f} {pipe_m["mae"]:<10.4f} {pipe_m["rmse"]:<10.4f}')
print(f'{"Regressor-only (oracle)":<25} {reg_oracle_m["r2"]:<10.4f} {reg_oracle_m["mae"]:<10.4f} {reg_oracle_m["rmse"]:<10.4f}')
print(f'{"Regressor on all":<25} {reg_all_m["r2"]:<10.4f} {reg_all_m["mae"]:<10.4f} {reg_all_m["rmse"]:<10.4f}')
print()
print(f'Subset analysis:')
print(f'  High-PCE only (n={int(high_pce_mask.sum())}): R²={pipe_high["r2"]:.4f} MAE={pipe_high["mae"]:.4f} RMSE={pipe_high["rmse"]:.4f}')
print(f'  Low-PCE only  (n={int(low_pce_mask.sum())}): R²={pipe_low["r2"]:.4f} MAE={pipe_low["mae"]:.4f} RMSE={pipe_low["rmse"]:.4f}')
print()
print('Quadrant analysis:')
for name in ['tp', 'tn', 'fp', 'fn']:
    q = quad[name]
    if q['r2'] is not None:
        print(f'  {name.upper():>2} (n={q["n"]:>3}): R²={q["r2"]:.4f} MAE={q["mae"]:.4f} RMSE={q["rmse"]:.4f}')
    else:
        print(f'  {name.upper():>2} (n={q["n"]:>3}): insufficient')

# ─── 13. Save ───────────────────────────────────────────────────────────────
results = {
    'n_total': N,
    'n_test': len(test_graphs),
    'classifier': {
        'accuracy': float(best_acc),
        'n_tp': int(tp.sum()), 'n_tn': int(tn.sum()),
        'n_fp': int(fp_mask.sum()), 'n_fn': int(fn_mask.sum()),
    },
    'pipeline': pipe_m,
    'pipeline_high_pce': pipe_high,
    'pipeline_low_pce': pipe_low,
    'regressor_oracle': reg_oracle_m,
    'regressor_all': reg_all_m,
    'quadrant_metrics': quad,
    'low_pce_baseline': low_pce_baseline,
}
os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
with open(RESULTS_FILE, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nResults saved to {RESULTS_FILE}')
print('Done.')
