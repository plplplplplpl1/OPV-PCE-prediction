#!/usr/bin/env python3
"""
Generate publication-quality figures for OPV PCE prediction paper.
Nature-journal style: minimalist, professional color palette, clean typography.
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb

warnings.filterwarnings('ignore')

BASE_DIR    = "/root/第四版r2=0.72/最小版本"
DATA_PATH   = os.path.join(BASE_DIR, "data/data.csv")
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Nature-style theme ────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7.5,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Professional color palette (Nature Communications style)
C_HIGH = '#C0392B'
C_LOW  = '#2980B9'
C_MAIN = '#2C3E50'
C_ACCENT = ['#2980B9', '#E74C3C', '#27AE60', '#F39C12', '#8E44AD', '#1ABC9C']
C_GRID = '#ECF0F1'

# ====================================================================
# Data loading
# ====================================================================
def load_data():
    df = pd.read_csv(DATA_PATH, encoding='latin-1')
    df.columns = df.columns.str.strip()
    rename = {}
    for c in df.columns:
        low = c.lower().replace(' ', '_')
        if 'pce' in low: rename[c] = 'PCE'
        elif 'smiles' in low: rename[c] = 'SMILES'
    df = df.rename(columns=rename)
    return df

def morgan_fp(smiles, nbits=2048, radius=2):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    return np.array(fp, dtype=np.float32)

CORE_DESC = OrderedDict([
    ('MolWt', Descriptors.MolWt),
    ('MolLogP', Descriptors.MolLogP),
    ('TPSA', Descriptors.TPSA),
    ('AromaticRings', rdMolDescriptors.CalcNumAromaticRings),
    ('AliphaticRings', rdMolDescriptors.CalcNumAliphaticRings),
    ('HDonors', Descriptors.NumHDonors),
    ('HAcceptors', Descriptors.NumHAcceptors),
    ('RotBonds', Descriptors.NumRotatableBonds),
    ('RingCount', Descriptors.RingCount),
    ('HeavyAtomCount', Descriptors.HeavyAtomCount),
])

def compute_desc(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    return [fn(mol) for fn in CORE_DESC.values()]

# ====================================================================
# FIGURE 2: PCE Distribution
# ====================================================================
def fig2_pce_distribution(df):
    print("[Fig 2] PCE distribution histogram ...")
    pce = df['PCE'].values
    n_high = (pce > 3).sum()
    n_low  = (pce <= 3).sum()

    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    bins = np.linspace(0, 20, 41)
    ax.hist(pce[pce <= 3], bins=bins, color=C_LOW, alpha=0.75, label=f'Low PCE (≤3%, n={n_low})')
    ax.hist(pce[pce > 3], bins=bins, color=C_HIGH, alpha=0.75, label=f'High PCE (>3%, n={n_high})')
    ax.axvline(3.0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('PCE (%)'); ax.set_ylabel('Count')
    ax.set_title('PCE Distribution of OPV Dataset')
    ax.legend(frameon=False, fontsize=7)
    ax.set_xlim(0, 20)
    # Stats
    stats = f'Total: {len(pce)}\nMean: {pce.mean():.2f}%\nMedian: {np.median(pce):.2f}%\nMax: {pce.max():.2f}%'
    ax.text(0.97, 0.94, stats, transform=ax.transAxes, fontsize=7, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F8F9FA', edgecolor='#DDD', linewidth=0.5))
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig2_pce_distribution.png')
    fig.savefig(path); plt.close(fig); print(f"  -> {path}")

# ====================================================================
# FIGURE 1: Scatter + Model Comparison
# ====================================================================
def fig1_scatter_comparison(df):
    print("[Fig 1] Scatter + model comparison ...")
    df_high = df[df['PCE'] > 3].copy()
    X, y = [], []
    for _, r in df_high.iterrows():
        fp = morgan_fp(r['SMILES'], 2048)
        if fp is not None: X.append(fp); y.append(r['PCE'])
    X, y = np.array(X), np.array(y)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(n_estimators=2000, learning_rate=0.03, max_depth=6,
                             min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                             reg_lambda=5, gamma=0.1, random_state=42, verbosity=0,
                             tree_method='hist')
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    y_pred = model.predict(X_te)
    r2 = r2_score(y_te, y_pred)
    mae = mean_absolute_error(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    print(f"  XGB R²={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")

    fig = plt.figure(figsize=(7.2, 3.2))
    gs = GridSpec(1, 2, figure=fig, wspace=0.4)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_te, y_pred, s=12, alpha=0.55, c=C_HIGH, edgecolors='white', linewidth=0.3, zorder=3)
    lims = [min(y_te.min(), y_pred.min()), max(y_te.max(), y_pred.max())]
    ax1.plot(lims, lims, 'k-', alpha=0.5, linewidth=0.8, zorder=1)
    ax1.set_xlim(lims); ax1.set_ylim(lims)
    ax1.set_xlabel('True PCE (%)'); ax1.set_ylabel('Predicted PCE (%)')
    ax1.set_title('XGBoost Predictions (High PCE)')
    ax1.set_aspect('equal')
    ax1.text(0.05, 0.94, f'R² = {r2:.4f}\nMAE = {mae:.2f}%', transform=ax1.transAxes,
             fontsize=8, va='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.8))

    ax2 = fig.add_subplot(gs[0, 1])
    models_lbl = ['XGBoost\n(Optuna)', 'XGBoost\n(Default)', 'CatBoost', 'GNN\n(RegV3)', 'GNN\n(3-Ensemble)']
    r2_vals = [r2, 0.7247, 0.7180, 0.6432, 0.6466]
    colors = [C_ACCENT[1], C_ACCENT[3], C_ACCENT[5], C_ACCENT[0], '#5DADE2']
    bars = ax2.bar(models_lbl, r2_vals, color=colors, edgecolor='white', linewidth=0.5, width=0.55)
    for bar, val in zip(bars, r2_vals):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.008, f'{val:.4f}',
                 ha='center', va='bottom', fontsize=7, fontweight='bold')
    ax2.set_ylabel('R²'); ax2.set_title('Model Comparison')
    ax2.set_ylim(0, 0.85)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig1_scatter_comparison.png')
    fig.savefig(path); plt.close(fig); print(f"  -> {path}")
    return r2, mae, rmse

# ====================================================================
# FIGURE 3: Feature Importance
# ====================================================================
def fig3_feature_importance(df):
    print("[Fig 3] Feature importance ...")
    df_high = df[df['PCE'] > 3].copy()
    X, y = [], []
    for _, r in df_high.iterrows():
        fp = morgan_fp(r['SMILES'], 4096)
        if fp is not None: X.append(fp); y.append(r['PCE'])
    X, y = np.array(X), np.array(y)
    model = xgb.XGBRegressor(n_estimators=2000, learning_rate=0.03, max_depth=6,
                             min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                             reg_lambda=5, gamma=0.1, random_state=42, verbosity=0,
                             tree_method='hist')
    model.fit(X, y, verbose=False)

    imp = model.feature_importances_
    top_idx = np.argsort(imp)[-20:]
    top_vals = imp[top_idx]

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.barh(range(len(top_idx)), top_vals, color=C_ACCENT[0], edgecolor='none', height=0.7)
    ax.set_yticks(range(len(top_idx)))
    ax.set_yticklabels([f'Bit {i}' for i in top_idx], fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance')
    ax.set_title('Top 20 Morgan Fingerprint Bits (XGBoost)')
    ax.set_xlim(0, max(top_vals) * 1.15)
    for i, v in enumerate(top_vals):
        ax.text(v + 0.0005, i, f'{v:.4f}', va='center', fontsize=6.5)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig3_feature_importance.png')
    fig.savefig(path); plt.close(fig); print(f"  -> {path}")

# ====================================================================
# FIGURE 4: Learning Curve
# ====================================================================
def fig4_learning_curve(df):
    print("[Fig 4] Learning curve ...")
    df_high = df[df['PCE'] > 3].copy()
    rng = np.random.RandomState(42)
    X, y = [], []
    for _, r in df_high.iterrows():
        fp = morgan_fp(r['SMILES'], 2048)
        if fp is not None: X.append(fp); y.append(r['PCE'])
    X, y = np.array(X), np.array(y)
    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]

    n_test = int(len(y) * 0.2)
    X_te, y_te = X[:n_test], y[:n_test]
    X_tr_full, y_tr_full = X[n_test:], y[n_test:]

    fractions = [0.01, 0.02, 0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 1.0]
    results = []
    for frac in fractions:
        n = max(10, int(len(y_tr_full) * frac))
        X_sub = X_tr_full[:n]
        y_sub = y_tr_full[:n]
        model = xgb.XGBRegressor(n_estimators=2000, learning_rate=0.03, max_depth=6,
                                 min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                                 reg_lambda=5, gamma=0.1, random_state=42, verbosity=0,
                                 tree_method='hist')
        model.fit(X_sub, y_sub, verbose=False)
        r2 = r2_score(y_te, model.predict(X_te))
        results.append((n, r2))
        print(f"    n={n:4d}  R²={r2:.4f}")

    results = np.array(results)
    n_vals, r2_vals = results[:, 0], results[:, 2]

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.plot(n_vals, r2_vals, 'o-', color=C_ACCENT[1], linewidth=1.5, markersize=5)
    gnn_r2 = 0.6432
    ax.axhline(gnn_r2, color=C_ACCENT[0], linestyle='--', linewidth=1, label=f'GNN baseline (R²={gnn_r2})')

    cross = np.where(r2_vals >= gnn_r2)[0]
    if len(cross) > 0:
        cn = n_vals[cross[0]]
        cr = r2_vals[cross[0]]
        ax.axvline(cn, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
        ax.annotate(f'~{int(cn)} samples', xy=(cn, cr), xytext=(cn*1.3, cr-0.07),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.8),
                    fontsize=7.5, color='gray')

    ax.set_xlabel('Training Samples'); ax.set_ylabel('R²')
    ax.set_title('Learning Curve')
    ax.legend(frameon=False, fontsize=7)
    ax.set_xlim(0, n_vals.max()*1.1)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig4_learning_curve.png')
    fig.savefig(path); plt.close(fig); print(f"  -> {path}")

# ====================================================================
# FIGURE 5: t-SNE Chemical Space
# ====================================================================
def fig5_tsne(df):
    print("[Fig 5] t-SNE ...")
    N = min(2000, len(df))
    df_s = df.sample(N, random_state=42) if len(df) > N else df

    fps = []
    valid_idx = []
    for i, r in df_s.iterrows():
        fp = morgan_fp(r['SMILES'], 2048)
        if fp is not None: fps.append(fp); valid_idx.append(i)
    fps = np.array(fps)
    print(f"  Computing t-SNE on {len(fps)} molecules ...")
    xy = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000, verbose=0).fit_transform(fps)
    print(f"  Done.")

    pce = df_s.loc[valid_idx, 'PCE'].values
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2))

    sc = ax1.scatter(xy[:,0], xy[:,1], c=pce, cmap='viridis', s=4, alpha=0.7, edgecolors='none')
    plt.colorbar(sc, ax=ax1, label='PCE (%)', shrink=0.8)
    ax1.set_title('Colored by PCE'); ax1.set_xlabel('t-SNE 1'); ax1.set_ylabel('t-SNE 2')
    ax1.set_xticks([]); ax1.set_yticks([])

    colors = np.where(pce > 3, C_HIGH, C_LOW)
    ax2.scatter(xy[:,0], xy[:,1], c=colors, s=4, alpha=0.7, edgecolors='none')
    ax2.legend(handles=[Patch(color=C_HIGH, label='High PCE'), Patch(color=C_LOW, label='Low PCE')],
               frameon=False, fontsize=7)
    ax2.set_title('High vs Low PCE'); ax2.set_xlabel('t-SNE 1'); ax2.set_ylabel('t-SNE 2')
    ax2.set_xticks([]); ax2.set_yticks([])

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig5_tsne_chemical_space.png')
    fig.savefig(path); plt.close(fig); print(f"  -> {path}")

# ====================================================================
# FIGURE 6: Residuals
# ====================================================================
def fig6_residuals(df):
    print("[Fig 6] Residuals ...")
    df_high = df[df['PCE'] > 3].copy()
    X, y = [], []
    for _, r in df_high.iterrows():
        fp = morgan_fp(r['SMILES'], 2048)
        if fp is not None: X.append(fp); y.append(r['PCE'])
    X, y = np.array(X), np.array(y)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(n_estimators=2000, learning_rate=0.03, max_depth=6,
                             min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                             reg_lambda=5, gamma=0.1, random_state=42, verbosity=0,
                             tree_method='hist')
    model.fit(X_tr, y_tr, verbose=False)
    y_pred = model.predict(X_te)
    res = y_te - y_pred

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2))

    ax1.hist(res, bins=25, color='#5D6D7E', edgecolor='white', alpha=0.8)
    ax1.axvline(0, color='black', linestyle='--', linewidth=0.8)
    ax1.set_xlabel('Residual (%)'); ax1.set_ylabel('Count')
    ax1.set_title('Residual Distribution')
    ax1.text(0.95, 0.94, f'Mean: {res.mean():.3f}%\nStd:  {res.std():.3f}%',
             transform=ax1.transAxes, fontsize=8, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='#F8F9FA', edgecolor='#DDD', linewidth=0.5))

    ax2.scatter(y_te, res, s=10, alpha=0.5, c=C_HIGH, edgecolors='white', linewidth=0.3)
    ax2.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax2.set_xlabel('True PCE (%)'); ax2.set_ylabel('Residual (%)')
    ax2.set_title('Residuals vs True PCE')

    # Bin means
    bins = [3, 5, 8, 12, 20]
    for i in range(len(bins)-1):
        m = (y_te >= bins[i]) & (y_te < bins[i+1])
        if m.sum() > 0:
            mr = res[m].mean()
            mid = (bins[i]+bins[i+1])/2
            ax2.plot(mid, mr, 's', color='black', markersize=6, zorder=5)
            ax2.annotate(f'{mr:+.2f}%', xy=(mid, mr), xytext=(mid+0.6, mr+0.4),
                        fontsize=6.5, color='black')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig6_residuals.png')
    fig.savefig(path); plt.close(fig); print(f"  -> {path}")

# ====================================================================
# FIGURE 7: Descriptor Pair Plot
# ====================================================================
def fig7_descriptor_pairs(df):
    print("[Fig 7] Descriptor pairs ...")
    N = min(2000, len(df))
    df_s = df.sample(N, random_state=42) if len(df) > N else df

    desc_data = {name: [] for name in CORE_DESC}
    pce_vals = []
    for _, r in df_s.iterrows():
        d = compute_desc(r['SMILES'])
        if d is None: continue
        for k, v in zip(CORE_DESC.keys(), d): desc_data[k].append(v)
        pce_vals.append(r['PCE'])
    desc_df = pd.DataFrame(desc_data)
    pce = np.array(pce_vals)
    high = pce > 3

    sel = ['MolWt', 'MolLogP', 'TPSA', 'RingCount']
    n = len(sel)
    fig, axes = plt.subplots(n, n, figsize=(6.5, 6.5))
    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.08, top=0.93, hspace=0.12, wspace=0.12)

    for i, ni in enumerate(sel):
        for j, nj in enumerate(sel):
            ax = axes[i, j]
            if i == j:
                ax.hist(desc_df[ni][high], bins=20, alpha=0.6, color=C_HIGH)
                ax.hist(desc_df[ni][~high], bins=20, alpha=0.5, color=C_LOW)
                ax.set_title(ni, fontsize=7, fontweight='bold')
            else:
                ax.scatter(desc_df[nj][high], desc_df[ni][high], s=3, alpha=0.35, c=C_HIGH, edgecolors='none')
                ax.scatter(desc_df[nj][~high], desc_df[ni][~high], s=3, alpha=0.35, c=C_LOW, edgecolors='none')
            ax.tick_params(labelsize=6)
            if j > 0: ax.set_yticklabels([])
            if i < n-1: ax.set_xticklabels([])
            if j == 0: ax.set_ylabel(ni, fontsize=7)
            if i == n-1: ax.set_xlabel(nj, fontsize=7)

    fig.legend(handles=[Patch(color=C_HIGH, label='High PCE (>3%)'),
                        Patch(color=C_LOW, label='Low PCE (≤3%)')],
               loc='lower center', ncol=2, fontsize=8, frameon=False)
    fig.suptitle('Core Molecular Descriptor Distributions', fontweight='bold', fontsize=10, y=0.97)
    path = os.path.join(OUTPUT_DIR, 'fig7_descriptor_pairs.png')
    fig.savefig(path); plt.close(fig); print(f"  -> {path}")

# ====================================================================
# Main
# ====================================================================
if __name__ == '__main__':
    print("="*50)
    print("  Generating publication-quality figures")
    print("="*50)
    df = load_data()
    fig2_pce_distribution(df)
    r2, mae, rmse = fig1_scatter_comparison(df)
    fig3_feature_importance(df)
    fig4_learning_curve(df)
    fig6_residuals(df)
    fig7_descriptor_pairs(df)
    fig5_tsne(df)
    print("\n" + "="*50)
    print(f"  All figures in: {OUTPUT_DIR}")
    print("="*50)
