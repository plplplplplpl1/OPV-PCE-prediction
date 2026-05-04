"""
Paper Figures and Statistical Analysis
Generates:
1. Figure: OPV learning curves with power law fits (Figure 3)
2. Figure: Model comparison bar chart (Figure 1 supplement)
3. Statistical significance tests (paired t-test, Wilcoxon)
"""
import os, sys, json, random, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import ttest_rel, wilcoxon, pearsonr

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

# ===== Configuration =====
SEED = 9999
FP_DIM = 512
FIG_DIR = '论文写作指导/论文草稿/figures'
RESULTS_DIR = 'external_results'
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Best hyperparams from Optuna search (trial 13, R²=0.6385)
BEST_LR = 0.0006107648790956326
BEST_HIDDEN = 256
BEST_DROPOUT = 0.1735093669041254
BEST_WEIGHT_DECAY = 0.00015717090515338398
BATCH_SIZE = 16

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
            except:
                hybridization = num_h = valence = 0
            r3 = int(atom.IsInRingSize(3)) if atom.IsInRing() else 0
            r4 = int(atom.IsInRingSize(4)) if atom.IsInRing() else 0
            r5 = int(atom.IsInRingSize(5)) if atom.IsInRing() else 0
            r6 = int(atom.IsInRingSize(6)) if atom.IsInRing() else 0
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

# ===== Load Data =====
DATA_CSV = 'data/data_merged.csv' if os.path.exists('data/data_merged.csv') else 'data/data.csv'
PCE_THRESHOLD = 3.0

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

print("Loading data ...")
df = pd.read_csv(DATA_CSV, encoding='latin-1')
pce_col = df.columns[2]
smiles_col = df.columns[-1]
df[pce_col] = pd.to_numeric(df[pce_col], errors='coerce')
df[smiles_col] = df[smiles_col].astype(str).str.strip()
df = df.dropna(subset=[pce_col, smiles_col])
df = df[df[smiles_col] != 'nan'].reset_index(drop=True)
df_high = df[df[pce_col] > PCE_THRESHOLD].copy().reset_index(drop=True)

graphs, smiles_list = [], []
for _, row in df_high.iterrows():
    g = smiles_to_graph(row[smiles_col])
    if g is not None:
        g.y = torch.tensor([float(row[pce_col])], dtype=torch.float)
        graphs.append(g)
        smiles_list.append(row[smiles_col])
print(f"Loaded {len(graphs)} high-PCE graphs from {len(df_high)} rows")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ========================================================
# Part 1: Statistical Significance Test (seed=9999)
# ========================================================
print("\n" + "="*60)
print("PART 1: Statistical Significance Test")
print("="*60)

set_seed(SEED)
indices = list(range(len(graphs)))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=SEED)
train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=SEED)

train_graphs = [graphs[i] for i in train_idx]
val_graphs = [graphs[i] for i in val_idx]
test_graphs = [graphs[i] for i in test_idx]

test_smiles = [smiles_list[i] for i in test_idx]
test_pce = [graphs[i].y.item() for i in test_idx]

# ---- XGBoost ----
print("\nTraining XGBoost...")
train_fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles_list[i]), 2, 4096) for i in train_idx]
test_fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles_list[i]), 2, 4096) for i in test_idx]
X_train = np.array(train_fps, dtype=np.float32)
X_test = np.array(test_fps, dtype=np.float32)
y_train = np.array([graphs[i].y.item() for i in train_idx], dtype=np.float32)
y_test = np.array([graphs[i].y.item() for i in test_idx], dtype=np.float32)

xgb_model = XGBRegressor(n_estimators=2000, learning_rate=0.0117, max_depth=6,
                          min_child_weight=5, subsample=0.595, colsample_bytree=0.626,
                          reg_alpha=0.1, reg_lambda=1.0,
                          random_state=SEED, verbosity=0, n_jobs=8)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_r2 = r2_score(y_test, xgb_pred)
xgb_mae = mean_absolute_error(y_test, xgb_pred)
print(f"  XGBoost R²={xgb_r2:.4f}, MAE={xgb_mae:.4f}")

# ---- GNN ----
print("\nTraining GNN (Optuna-best config)...")
set_seed(SEED)
train_loader = GeoDataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
val_loader = GeoDataLoader(val_graphs, batch_size=BATCH_SIZE)
test_loader = GeoDataLoader(test_graphs, batch_size=BATCH_SIZE)

in_dim = graphs[0].x.shape[1]
model = HighPCERegressorV3(in_channels=in_dim, hidden=BEST_HIDDEN, dropout=BEST_DROPOUT).to(device)
optimizer = optim.AdamW(model.parameters(), lr=BEST_LR, weight_decay=BEST_WEIGHT_DECAY)
criterion = nn.HuberLoss(delta=1.0)

best_val_mae = float('inf')
patience_counter = 0
EPOCHS = 100
PATIENCE = 15

for epoch in range(1, EPOCHS + 1):
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

    if val_mae < best_val_mae:
        best_val_mae = val_mae
        patience_counter = 0
    else:
        patience_counter += 1
    if patience_counter >= PATIENCE:
        break

# Test
model.eval()
gnn_pred = []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        pred = model(batch)
        gnn_pred.extend(pred.cpu().numpy())
gnn_pred = np.array(gnn_pred)
gnn_r2 = r2_score(y_test, gnn_pred)
gnn_mae = mean_absolute_error(y_test, gnn_pred)
print(f"  GNN R²={gnn_r2:.4f}, MAE={gnn_mae:.4f}")

# Paired tests
print("\n--- Statistical Tests ---")
print(f"  Test set size: {len(y_test)}")

# Paired t-test on absolute errors
xgb_errors = np.abs(y_test - xgb_pred)
gnn_errors = np.abs(y_test - gnn_pred)

t_stat, t_pval = ttest_rel(xgb_errors, gnn_errors, alternative='less')
print(f"  Paired t-test (XGBoost error < GNN error): t={t_stat:.4f}, p={t_pval:.6f}")
print(f"    Mean abs error: XGB={xgb_errors.mean():.4f}, GNN={gnn_errors.mean():.4f}")

# Wilcoxon signed-rank test
try:
    w_stat, w_pval = wilcoxon(xgb_errors, gnn_errors, alternative='less')
    print(f"  Wilcoxon signed-rank (XGB error < GNN error): W={w_stat:.1f}, p={w_pval:.6f}")
except ValueError as e:
    print(f"  Wilcoxon test skipped: {e}")
    w_stat, w_pval = None, None

# Correlation of predictions
corr, corr_pval = pearsonr(xgb_pred, gnn_pred)
print(f"  Prediction correlation: r={corr:.4f}, p={corr_pval:.6f}")

# MSE ratio (how much larger is GNN MSE than XGB MSE?)
xgb_mse = np.mean((y_test - xgb_pred)**2)
gnn_mse = np.mean((y_test - gnn_pred)**2)
print(f"  MSE ratio (GNN/XGBoost): {gnn_mse/xgb_mse:.3f}x")

# Save results
stats_results = {
    'seed': SEED,
    'test_n': len(y_test),
    'xgb_r2': float(xgb_r2),
    'gnn_r2': float(gnn_r2),
    'xgb_mae': float(xgb_mae),
    'gnn_mae': float(gnn_mae),
    'xgb_mse': float(xgb_mse),
    'gnn_mse': float(gnn_mse),
    'mse_ratio': float(gnn_mse / xgb_mse),
    'paired_t': {'statistic': float(t_stat), 'p_value': float(t_pval)},
    'wilcoxon': {'statistic': float(w_stat), 'p_value': float(w_pval)} if w_stat is not None else None,
    'prediction_correlation': {'r': float(corr), 'p_value': float(corr_pval)},
    'xgb_mae_mean_std': [float(xgb_errors.mean()), float(xgb_errors.std())],
    'gnn_mae_mean_std': [float(gnn_errors.mean()), float(gnn_errors.std())],
}
with open(f'{RESULTS_DIR}/statistical_tests.json', 'w') as f:
    json.dump(stats_results, f, indent=2)
print(f"\nResults saved to {RESULTS_DIR}/statistical_tests.json")

# ========================================================
# Part 2: OPV Learning Curve Figure (Figure 3)
# ========================================================
print("\n" + "="*60)
print("PART 2: OPV Learning Curve Figure")
print("="*60)

# Learning curve data from the paper (Section 2.5)
n_values = np.array([68, 137, 344, 689, 1033, 1378], dtype=float)
xgb_lc = np.array([0.511, 0.558, 0.677, 0.695, 0.725, 0.730])
gnn_lc = np.array([0.145, 0.457, 0.433, 0.542, 0.503, 0.598])

# Power law fit
def power_law(n, a, b, c):
    return a - b * np.power(n, -c)

popt_xgb, _ = curve_fit(power_law, n_values, xgb_lc, p0=[0.8, 5, 0.5], maxfev=10000)
popt_gnn, _ = curve_fit(power_law, n_values, gnn_lc, p0=[0.8, 5, 0.5], maxfev=10000)

a_xgb, b_xgb, c_xgb = popt_xgb
a_gnn, b_gnn, c_gnn = popt_gnn

print(f"XGBoost: R²(n) = {a_xgb:.4f} - {b_xgb:.4f} · n^(-{c_xgb:.4f})")
print(f"GNN:     R²(n) = {a_gnn:.4f} - {b_gnn:.4f} · n^(-{c_gnn:.4f})")

# Extrapolation
n_smooth = np.logspace(np.log10(50), np.log10(10000), 200)
xgb_smooth = power_law(n_smooth, *popt_xgb)
gnn_smooth = power_law(n_smooth, *popt_gnn)

# Figure
fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(n_values, xgb_lc, 'o-', color='#E74C3C', markersize=8, linewidth=2, label='XGBoost (observed)')
ax.plot(n_values, gnn_lc, 's-', color='#3498DB', markersize=8, linewidth=2, label='GNN (observed)')

ax.plot(n_smooth, xgb_smooth, '--', color='#E74C3C', alpha=0.4, linewidth=1,
        label=f'XGBoost fit (R²asymp={a_xgb:.3f})')
ax.plot(n_smooth, gnn_smooth, '--', color='#3498DB', alpha=0.4, linewidth=1,
        label=f'GNN fit (R²asymp={a_gnn:.3f})')

# Annotation: XGBoost catches GNN at ~250 samples
ax.axhline(y=gnn_lc[-1], color='#3498DB', linestyle=':', alpha=0.4)
target_r2 = gnn_lc[-1]
interp_n = np.interp(target_r2, xgb_lc, n_values)
ax.annotate(f'XGBoost matches GNN\nfull-data at n≈{interp_n:.0f}',
            xy=(interp_n, target_r2), xytext=(interp_n*2.5, target_r2-0.08),
            fontsize=10, color='#2C3E50', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=1.5))

ax.set_xscale('log')
ax.set_xlabel('Training Samples (n)', fontsize=12)
ax.set_ylabel('R²', fontsize=12)
ax.set_title('OPV Learning Curves: XGBoost vs GNN', fontsize=13, fontweight='bold')
ax.set_ylim(0, 0.95)
ax.set_xlim(50, 10000)
ax.legend(fontsize=10, loc='lower right')
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=10)

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/opv_learning_curves.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{FIG_DIR}/opv_learning_curves.pdf', bbox_inches='tight')
plt.close()
print(f"Figure saved to {FIG_DIR}/opv_learning_curves.png/pdf")

# ========================================================
# Part 3: Model Comparison Bar Chart
# ========================================================
print("\n" + "="*60)
print("PART 3: Model Comparison Bar Chart")
print("="*60)

models = ['XGBoost\n(Optuna best)', 'XGBoost\n(4-seed avg)', 'GNN\n(4-seed avg)', 'GraphGPS\n(4-seed avg)']
r2_means = [0.7360, 0.686, 0.635, 0.616]
r2_stds =  [0,       0.026, 0.039, 0.051]
colors =   ['#C0392B', '#E74C3C', '#3498DB', '#2ECC71']

fig, ax = plt.subplots(figsize=(8, 5.5))
bars = ax.bar(range(len(models)), r2_means, yerr=r2_stds, color=colors,
              capsize=5, edgecolor='white', linewidth=1.2, width=0.5, error_kw={'linewidth': 1.5})

# Value labels on bars
for i, (bar, mean, std) in enumerate(zip(bars, r2_means, r2_stds)):
    if std > 0:
        ax.text(i, mean + std + 0.012, f'{mean:.3f}±{std:.3f}',
                ha='center', fontsize=10, fontweight='bold', color=colors[i])
    else:
        ax.text(i, mean + 0.012, f'{mean:.4f}',
                ha='center', fontsize=10, fontweight='bold', color=colors[i])

# XGBoost best reference line
ax.axhline(y=0.736, color='#C0392B', linestyle='--', alpha=0.5, linewidth=1)
ax.text(3.5, 0.738, 'XGBoost best: 0.736', fontsize=9, color='#C0392B', fontweight='bold')

# GNN multi-seed mean reference
ax.axhline(y=0.635, color='#3498DB', linestyle=':', alpha=0.4, linewidth=1)

ax.set_ylabel('R²', fontsize=12)
ax.set_title('Model Comparison on High-PCE Regression', fontsize=13, fontweight='bold')
ax.set_xticks(range(len(models)))
ax.set_xticklabels(models, fontsize=10)
ax.set_ylim(0.5, 0.85)
ax.grid(True, alpha=0.3, axis='y')
ax.tick_params(labelsize=10)

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/model_comparison.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{FIG_DIR}/model_comparison.pdf', bbox_inches='tight')
plt.close()
print(f"Figure saved to {FIG_DIR}/model_comparison.png/pdf")

# ========================================================
# Part 4: Per-sample prediction scatter plot
# ========================================================
print("\n" + "="*60)
print("PART 4: Prediction Scatter Plot")
print("="*60)

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

# XGBoost predictions
ax = axes[0]
ax.scatter(y_test, xgb_pred, alpha=0.5, color='#E74C3C', edgecolors='white', linewidth=0.5)
min_val = min(y_test.min(), xgb_pred.min())
max_val = max(y_test.max(), xgb_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=1)
ax.set_xlabel('True PCE (%)', fontsize=11)
ax.set_ylabel('Predicted PCE (%)', fontsize=11)
ax.set_title(f'XGBoost (R²={xgb_r2:.4f})', fontsize=12, fontweight='bold', color='#E74C3C')
ax.grid(True, alpha=0.3)

# GNN predictions
ax = axes[1]
ax.scatter(y_test, gnn_pred, alpha=0.5, color='#3498DB', edgecolors='white', linewidth=0.5)
ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=1)
ax.set_xlabel('True PCE (%)', fontsize=11)
ax.set_ylabel('Predicted PCE (%)', fontsize=11)
ax.set_title(f'GNN (R²={gnn_r2:.4f})', fontsize=12, fontweight='bold', color='#3498DB')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/prediction_scatter.png', dpi=200, bbox_inches='tight')
plt.savefig(f'{FIG_DIR}/prediction_scatter.pdf', bbox_inches='tight')
plt.close()
print(f"Figure saved to {FIG_DIR}/prediction_scatter.png/pdf")

# ========================================================
# Summary
# ========================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"XGBoost R²={xgb_r2:.4f}, MAE={xgb_mae:.4f}")
print(f"GNN R²={gnn_r2:.4f}, MAE={gnn_mae:.4f}")
print(f"ΔR² = {xgb_r2 - gnn_r2:.4f}")
print(f"Paired t-test: p={t_pval:.6f}")
if w_pval is not None:
    print(f"Wilcoxon: p={w_pval:.6f}")
print(f"\nFigures generated:")
print(f"  - {FIG_DIR}/opv_learning_curves.png/pdf")
print(f"  - {FIG_DIR}/model_comparison.png/pdf")
print(f"  - {FIG_DIR}/prediction_scatter.png/pdf")
print("Done!")
