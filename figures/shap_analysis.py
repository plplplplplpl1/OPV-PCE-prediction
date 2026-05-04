#!/usr/bin/env python3
"""
SHAP analysis for best XGBoost model.
Generates: (1) SHAP summary bar + beeswarm, (2) SHAP dependence plots,
(3) SHAP force plot for representative molecules.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
import os, warnings
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
warnings.filterwarnings('ignore')

BASE_DIR = "/root/第四版r2=0.72/最小版本"
DATA_PATH = os.path.join(BASE_DIR, "data/data.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans', 'Arial'],
    'font.size': 9, 'axes.linewidth': 0.6,
    'figure.dpi': 300, 'savefig.dpi': 600,
})

# ── Load data ──
df = pd.read_csv(DATA_PATH, encoding='latin-1')
df.columns = df.columns.str.strip()
for c in df.columns:
    if 'pce' in c.lower(): df = df.rename(columns={c: 'PCE'})
    if 'smiles' in c.lower(): df = df.rename(columns={c: 'SMILES'})
df_high = df[df['PCE'] > 3].copy()
smiles_list = df_high['SMILES'].tolist()

# ── Feature engineering (matching manuscript: 4096-bit Morgan) ──
def featurize(smi, nbits=4096, radius=2):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    return np.array(fp, dtype=np.float32)

X_list, y_list = [], []
for _, r in df_high.iterrows():
    f = featurize(r['SMILES'])
    if f is not None:
        X_list.append(f)
        y_list.append(r['PCE'])
X = np.array(X_list)
y = np.array(y_list)
print(f"X shape: {X.shape}, y range: [{y.min():.2f}, {y.max():.2f}]")

# ── Train XGBoost (Optuna-best config from manuscript) ──
model = xgb.XGBRegressor(
    n_estimators=2000,
    learning_rate=0.0117,
    max_depth=6,
    min_child_weight=5,
    subsample=0.595,
    colsample_bytree=0.626,
    reg_alpha=0.1,
    reg_lambda=1.0,
    gamma=0.0,
    random_state=9999,
    verbosity=0,
    tree_method='hist',
)
model.fit(X, y, verbose=False)
print("XGBoost trained.")

# ── SHAP analysis ──
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# ── Fig A: SHAP summary bar (top 15) ──
fig, ax = plt.subplots(figsize=(5.5, 4.5))
shap.summary_plot(shap_values, X, plot_type="bar", max_display=15, show=False,
                  color_bar=False, axis_color='#333333', title='')
ax.set_xlabel('mean |SHAP value|', fontsize=9)
ax.set_title('Top 15 Morgan Fingerprint Bits by SHAP Importance', fontweight='bold', fontsize=10)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'figS2a_shap_bar.png'), dpi=600, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("Saved figS2a_shap_bar.png")

# ── Fig B: SHAP beeswarm (top 15, direction-aware) ──
fig, ax = plt.subplots(figsize=(5.5, 4.5))
shap.summary_plot(shap_values, X, plot_type="dot", max_display=15, show=False,
                  color_bar=True, axis_color='#333333', title='')
ax.set_xlabel('SHAP value (impact on PCE)', fontsize=9)
ax.set_title('Directional Impact of Top Fingerprint Bits', fontweight='bold', fontsize=10)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'figS2b_shap_beeswarm.png'), dpi=600, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("Saved figS2b_shap_beeswarm.png")

# ── Fig C: SHAP force plot for best/worst predicted molecules ──
# Split train/test to get predictions
from sklearn.model_selection import train_test_split
idx = np.arange(len(y))
train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=9999)
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
model2 = xgb.XGBRegressor(
    n_estimators=2000, learning_rate=0.0117, max_depth=6,
    min_child_weight=5, subsample=0.595, colsample_bytree=0.626,
    reg_alpha=0.1, reg_lambda=1.0, gamma=0.0,
    random_state=9999, verbosity=0, tree_method='hist',
)
model2.fit(X_train, y_train, verbose=False)
explainer2 = shap.TreeExplainer(model2)
shap_test = explainer2.shap_values(X_test)
preds = model2.predict(X_test)
errors = np.abs(preds - y_test)

# Best-predicted molecule (smallest error)
best_idx = np.argmin(errors)
worst_idx = np.argmax(errors)

for label, idx_sel in [('best', best_idx), ('worst', worst_idx)]:
    fig = shap.force_plot(explainer2.expected_value, shap_test[idx_sel], X_test[idx_sel],
                          matplotlib=True, show=False, figsize=(8, 2))
    fig.savefig(os.path.join(OUTPUT_DIR, f'figS2c_shap_force_{label}.png'),
                dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved figS2c_shap_force_{label}.png (true={y_test[idx_sel]:.2f}, pred={preds[idx_sel]:.2f})")

# ── Print top-15 with direction info ──
feat_imp = np.abs(shap_values).mean(axis=0)
top15 = np.argsort(feat_imp)[-15:][::-1]
print("\nTop 15 SHAP features and their directional impact:")
print(f"{'Rank':<5} {'Bit':<6} {'|SHAP|':<10} {'SHAP+':<10} {'SHAP-':<10} {'Net direction':<15}")
print("-" * 56)
for rank, bit in enumerate(top15, 1):
    pos = shap_values[shap_values[:, bit] > 0, bit].mean() if (shap_values[:, bit] > 0).sum() > 0 else 0
    neg = shap_values[shap_values[:, bit] < 0, bit].mean() if (shap_values[:, bit] < 0).sum() > 0 else 0
    direction = "bit=1 → PCE↑" if abs(pos) > abs(neg) else "bit=1 → PCE↓"
    print(f"{rank:<5} {bit:<6} {feat_imp[bit]:<10.4f} {pos:<10.4f} {neg:<10.4f} {direction}")

print("\nDone. Generated figures in figures/")
