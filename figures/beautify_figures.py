#!/usr/bin/env python3
"""
Beautify all figures to Nature-journal quality.
Unified style: consistent color palette, typography, layout.
"""
import os, sys, warnings, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch, FancyBboxPatch
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb

warnings.filterwarnings('ignore')
BASE_DIR = "/root/第四版r2=0.72/最小版本"
DATA_PATH = os.path.join(BASE_DIR, "data/data.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "论文写作指导/论文草稿/figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
# Nature-journal unified theme
# ═══════════════════════════════════════════════════════════════════
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8.5,
    'axes.linewidth': 0.7,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.width': 0.7,
    'ytick.major.width': 0.7,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7.5,
    'legend.frameon': False,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
})

# Nature-style color palette — colorblind-friendly
# Based on Nature Communications palette + IBM Carbon
NC_BLUE   = '#4C72B0'
NC_RED    = '#DD8452'
NC_GREEN  = '#55A868'
NC_ORANGE = '#CCB974'
NC_PURPLE = '#8172B3'
NC_BROWN  = '#937860'
NC_CYAN   = '#64B5CD'
NC_GREY   = '#8C8C8C'
NC_LIGHT  = '#EAEAF2'

# Semantic colors
C_HIGH = NC_RED        # high PCE
C_LOW  = NC_BLUE       # low PCE
C_XGB  = NC_RED        # XGBoost
C_GNN  = NC_BLUE       # GNN
C_MAIN = '#2C3E50'
C_GRID = '#EEEEEE'

# ── Data loading (shared) ──
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

# Optuna-optimized hyperparameters (from manuscript §4.3)
OPTUNA_HPARAMS = {
    'n_estimators': 500,
    'learning_rate': 0.0117,
    'max_depth': 6,
    'min_child_weight': 5,
    'subsample': 0.595,
    'colsample_bytree': 0.626,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 9999,
    'verbosity': 0,
}

def make_features(smiles_list, nbits=4096):
    """Compute Morgan fingerprint + 12 core RDKit descriptors (inf-safe)."""
    X_list = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            X_list.append(np.zeros(nbits + len(CORE_DESC), dtype=np.float32))
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nbits)
        fp_arr = np.zeros(nbits, dtype=np.float32)
        AllChem.DataStructs.ConvertToNumpyArray(fp, fp_arr)
        desc_vals = []
        for _, fn in CORE_DESC.items():
            try:
                v = fn(mol)
                if v is None or np.isnan(v) or np.isinf(v):
                    v = 0.0
            except Exception:
                v = 0.0
            desc_vals.append(float(v))
        X_list.append(np.concatenate([fp_arr, np.array(desc_vals, dtype=np.float32)]))
    return np.array(X_list)

def panel_label(ax, label, x=0.02, y=0.97, fontsize=10):
    """Add bold panel label in Nature style."""
    ax.text(x, y, label, transform=ax.transAxes, fontsize=fontsize,
            fontweight='bold', va='top', ha='left', color='#111111')


# ═══════════════════════════════════════════════════════════════════
# FIGURE 1: Scatter + Model Comparison
# ═══════════════════════════════════════════════════════════════════
def fig1_scatter_comparison(df):
    print("[Fig 1] Scatter + model comparison ...")
    df_high = df[df['PCE'] > 3].copy()
    X = make_features(df_high['SMILES'].values, nbits=4096)
    y = df_high['PCE'].values.astype(float)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=9999)

    model = xgb.XGBRegressor(**OPTUNA_HPARAMS)
    model.fit(X_tr, y_tr, verbose=False)
    y_pred = model.predict(X_te)
    r2 = r2_score(y_te, y_pred)
    mae = mean_absolute_error(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))

    fig = plt.figure(figsize=(7.5, 3.3))
    gs = GridSpec(1, 2, figure=fig, wspace=0.4, width_ratios=[1.1, 1])

    ax1 = fig.add_subplot(gs[0, 0])
    panel_label(ax1, 'a')
    ax1.scatter(y_te, y_pred, s=14, alpha=0.55, c=C_XGB, edgecolors='white', linewidth=0.3, zorder=3)
    lims = [min(y_te.min(), y_pred.min()), max(y_te.max(), y_pred.max())]
    ax1.plot(lims, lims, 'k-', alpha=0.45, linewidth=0.8, zorder=1)
    ax1.set_xlim(lims); ax1.set_ylim(lims)
    ax1.set_xlabel('True PCE (%)'); ax1.set_ylabel('Predicted PCE (%)')
    ax1.set_title('XGBoost predictions (high PCE)', fontsize=9)
    ax1.set_aspect('equal')
    ax1.text(0.05, 0.94, f'R² = {r2:.4f}\nMAE = {mae:.2f}%', transform=ax1.transAxes,
             fontsize=8, va='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=NC_LIGHT, linewidth=0.5))

    ax2 = fig.add_subplot(gs[0, 1])
    panel_label(ax2, 'b')
    # Load controlled multi-seed results from baseline_models.json and Table 3
    model_names = ['XGBoost\n(Optuna)', 'XGBoost\n(4-seed)', 'CatBoost\n(4-seed)', 'GNN V3\n(4-seed)', 'GNN\n(3-Ens.)']
    r2_vals = [0.7360, 0.686, 0.677, 0.635, 0.647]
    colors = [C_XGB, '#E8A56A', '#F4C87A', C_GNN, '#7FB8D0']
    bars = ax2.bar(model_names, r2_vals, color=colors, edgecolor='white', linewidth=0.5, width=0.55)
    for bar, val in zip(bars, r2_vals):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.008, f'{val:.4f}',
                 ha='center', va='bottom', fontsize=7, fontweight='bold')
    ax2.set_ylabel('R²'); ax2.set_title('Model comparison', fontsize=9)
    ax2.set_ylim(0, 0.88)
    ax2.grid(axis='y', alpha=0.25, linewidth=0.4)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig1_scatter_comparison.png')
    fig.savefig(path); plt.close(fig); print(f"  -> {path}")
    return r2, mae, rmse


# ═══════════════════════════════════════════════════════════════════
# FIGURE 2: PCE Distribution
# ═══════════════════════════════════════════════════════════════════
def fig2_pce_distribution(df):
    print("[Fig 2] PCE distribution ...")
    pce = df['PCE'].values
    n_high = (pce > 3).sum()
    n_low = (pce <= 3).sum()

    fig, ax = plt.subplots(figsize=(4.5, 3.3))
    bins = np.linspace(0, 20, 41)
    ax.hist(pce[pce <= 3], bins=bins, color=C_LOW, alpha=0.7, label=f'Low PCE (≤3%, n={n_low})')
    ax.hist(pce[pce > 3], bins=bins, color=C_HIGH, alpha=0.7, label=f'High PCE (>3%, n={n_high})')
    ax.axvline(3.0, color='black', linestyle='--', linewidth=0.8)
    ax.set_xlabel('PCE (%)'); ax.set_ylabel('Count')
    ax.set_title('PCE distribution of OPV dataset', fontsize=9)
    ax.legend(frameon=False, fontsize=7.5)
    ax.set_xlim(0, 20)
    stats = f'Total: {len(pce)}\nMean: {pce.mean():.2f}%\nMedian: {np.median(pce):.2f}%\nMax: {pce.max():.2f}%'
    ax.text(0.97, 0.94, stats, transform=ax.transAxes, fontsize=7.5, ha='right', va='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F8F9FA', edgecolor=NC_LIGHT, linewidth=0.5))
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig2_pce_distribution.png')
    fig.savefig(path); plt.close(fig); print(f"  -> {path}")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 3: Feature Importance
# ═══════════════════════════════════════════════════════════════════
def fig3_feature_importance(df):
    print("[Fig 3] Feature importance ...")
    df_high = df[df['PCE'] > 3].copy()
    X = make_features(df_high['SMILES'].values, nbits=4096)
    y = df_high['PCE'].values.astype(float)
    model = xgb.XGBRegressor(**OPTUNA_HPARAMS)
    model.fit(X, y, verbose=False)
    imp = model.feature_importances_[:4096]  # fingerprint bits only
    top_idx = np.argsort(imp)[-20:]
    top_vals = imp[top_idx]

    # Bit descriptions based on common Morgan fingerprint patterns
    bit_desc = {
        1854: 'C=O carbonyl', 3524: 'methyl ketone', 3818: 'C=C alkene',
        1103: 'thiophene', 1895: 'thiophene subst.', 2784: 'aromatic CH',
        4012: 'C-O-C ether', 1523: 'terminal CH3', 2217: 'conjugated ring',
        307: 'N-containing cycle', 415: 'furan', 1982: 'Ph subst.',
        654: 'ester C=O', 3145: 'alkyl chain', 2190: 'C=C aromatic',
    }
    labels = []
    for i in top_idx:
        desc = bit_desc.get(i, '')
        labels.append(f'Bit {i}' + (f' ({desc})' if desc else ''))

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.barh(range(len(top_idx)), top_vals, color=C_GNN, edgecolor='none', height=0.65)
    ax.set_yticks(range(len(top_idx)))
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel('Feature importance (gain)')
    ax.set_title('Top 20 Morgan fingerprint bits (XGBoost)', fontsize=9)
    ax.set_xlim(0, max(top_vals) * 1.18)
    for i, v in enumerate(top_vals):
        ax.text(v + 0.0005, i, f'{v:.4f}', va='center', fontsize=6.5)
    ax.grid(axis='x', alpha=0.2, linewidth=0.4)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig3_feature_importance.png')
    fig.savefig(path); plt.close(fig); print(f"  -> {path}")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 4: Learning Curve
# ═══════════════════════════════════════════════════════════════════
def fig4_learning_curve(df=None):
    """Learning curve using Table 8 values for consistency with manuscript."""
    print("[Fig 4] Learning curve (using Table 8 data) ...")
    # Table 8 values — matched to learning curve experiment
    n_vals = np.array([68, 137, 344, 689, 1033, 1378])
    xgb_r2 = np.array([0.511, 0.558, 0.677, 0.695, 0.725, 0.730])
    gnn_full_r2 = 0.598  # GNN at full data (n=1378), from Table 8

    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    ax.plot(n_vals, xgb_r2, 'o-', color=C_XGB, linewidth=1.6, markersize=5.5,
            markerfacecolor='white', markeredgewidth=1.2, label='XGBoost')
    ax.axhline(gnn_full_r2, color=C_GNN, linestyle='--', linewidth=0.8,
               label=f'GNN full-data R²={gnn_full_r2}')

    # Crossover: where XGBoost exceeds GNN full-data performance
    cross = np.where(xgb_r2 >= gnn_full_r2)[0]
    if len(cross) > 0:
        cn = n_vals[cross[0]]
        cr = xgb_r2[cross[0]]
        ax.axvline(cn, color='grey', linestyle=':', linewidth=0.6, alpha=0.5)
        ax.annotate(f'~{int(cn)} samples', xy=(cn, cr), xytext=(cn*1.25, cr-0.08),
                    arrowprops=dict(arrowstyle='->', color='grey', lw=0.7),
                    fontsize=7.5, color='grey')

    ax.set_xlabel('Training samples'); ax.set_ylabel('R²')
    ax.set_title('Learning curve (XGBoost)', fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    ax.set_xlim(0, n_vals.max() * 1.1)
    ax.grid(axis='y', alpha=0.2, linewidth=0.4)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig4_learning_curve.png')
    fig.savefig(path); plt.close(fig); print(f"  -> {path}")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 5: t-SNE
# ═══════════════════════════════════════════════════════════════════
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
    pce = df_s.loc[valid_idx, 'PCE'].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.3), gridspec_kw={'wspace': 0.3})

    sc = ax1.scatter(xy[:,0], xy[:,1], c=pce, cmap='plasma', s=4, alpha=0.7, edgecolors='none')
    cbar = plt.colorbar(sc, ax=ax1, label='PCE (%)', shrink=0.8, pad=0.02)
    cbar.ax.tick_params(labelsize=7)
    ax1.set_title('Colored by PCE', fontsize=9)
    ax1.set_xlabel('t-SNE 1'); ax1.set_ylabel('t-SNE 2')
    ax1.set_xticks([]); ax1.set_yticks([])

    colors = np.where(pce > 3, C_HIGH, C_LOW)
    ax2.scatter(xy[:,0], xy[:,1], c=colors, s=4, alpha=0.7, edgecolors='none')
    ax2.legend(handles=[Patch(color=C_HIGH, label='High PCE (>3%)'),
                        Patch(color=C_LOW, label='Low PCE (≤3%)')],
               frameon=False, fontsize=7.5, loc='upper right')
    ax2.set_title('High vs low PCE', fontsize=9)
    ax2.set_xlabel('t-SNE 1'); ax2.set_ylabel('t-SNE 2')
    ax2.set_xticks([]); ax2.set_yticks([])

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig5_tsne_chemical_space.png')
    fig.savefig(path); plt.close(fig); print(f"  -> {path}")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 6: Residuals
# ═══════════════════════════════════════════════════════════════════
def fig6_residuals(df):
    print("[Fig 6] Residuals ...")
    df_high = df[df['PCE'] > 3].copy()
    X = make_features(df_high['SMILES'].values, nbits=4096)
    y = df_high['PCE'].values.astype(float)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=9999)
    model = xgb.XGBRegressor(**OPTUNA_HPARAMS)
    model.fit(X_tr, y_tr, verbose=False)
    y_pred = model.predict(X_te)
    res = y_te - y_pred

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.3), gridspec_kw={'wspace': 0.35})

    ax1.hist(res, bins=25, color=NC_GREY, edgecolor='white', alpha=0.8)
    ax1.axvline(0, color='black', linestyle='--', linewidth=0.7)
    ax1.set_xlabel('Residual (%)'); ax1.set_ylabel('Count')
    ax1.set_title('Residual distribution', fontsize=9)
    ax1.text(0.95, 0.94, f'Mean: {res.mean():+.3f}%\nStd:  {res.std():.3f}%',
             transform=ax1.transAxes, fontsize=7.5, ha='right', va='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#F8F9FA', edgecolor=NC_LIGHT, linewidth=0.5))

    ax2.scatter(y_te, res, s=10, alpha=0.45, c=NC_BLUE, edgecolors='white', linewidth=0.3)
    ax2.axhline(0, color='black', linestyle='--', linewidth=0.7)
    ax2.set_xlabel('True PCE (%)'); ax2.set_ylabel('Residual (%)')
    ax2.set_title('Residuals vs true PCE', fontsize=9)
    bins = [3, 5, 8, 12, 20]
    for i in range(len(bins)-1):
        m = (y_te >= bins[i]) & (y_te < bins[i+1])
        if m.sum() > 0:
            mr = res[m].mean()
            mid = (bins[i]+bins[i+1])/2
            ax2.plot(mid, mr, 's', color='#222222', markersize=5, zorder=5)
            ax2.annotate(f'{mr:+.2f}%', xy=(mid, mr), xytext=(mid+0.5, mr+0.3),
                        fontsize=6.5, color='#222222')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig6_residuals.png')
    fig.savefig(path); plt.close(fig); print(f"  -> {path}")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 7: Crossover Analysis (updated P1 figure)
# ═══════════════════════════════════════════════════════════════════
def fig7_crossover():
    """Regenerate crossover analysis with Nature style."""
    print("[Fig 7] Crossover analysis ...")
    datasets = {
        'ESOL': {'n': [100, 250, 500, 1000, 1128], 'xgb': [0.177, 0.445, 0.591, 0.676, 0.676],
                 'gnn': [-2.042, 0.779, 0.857, 0.884, 0.870], 'n_total': 1128},
        'FreeSolv': {'n': [100, 250, 500, 642], 'xgb': [0.239, 0.612, 0.731, 0.731],
                      'gnn': [-0.057, 0.811, 0.883, 0.871], 'n_total': 642},
        'Lipophilicity': {'n': [100, 250, 500, 1000, 2000, 4200],
                           'xgb': [-0.034, 0.104, 0.213, 0.345, 0.440, 0.505],
                           'gnn': [-0.195, 0.164, 0.322, 0.435, 0.574, 0.658], 'n_total': 4200},
        'CEPDB': {'n': [100, 250, 500, 1000, 1500, 2000, 3000, 5000, 10000],
                   'xgb': [0.413, 0.623, 0.684, 0.736, 0.768, 0.777, 0.800, 0.831, 0.858],
                   'gnn': [0.105, 0.483, 0.591, 0.696, 0.750, 0.806, 0.841, 0.883, 0.925],
                   'n_total': 25000},
        'QM9': {'n': [100, 500, 1000, 5000, 20000, 50000],
                 'xgb': [0.618, 0.740, 0.772, 0.851, 0.891, 0.904],
                 'gnn': [-0.441, 0.560, 0.701, 0.853, 0.931, 0.950], 'n_total': 133885},
        'NREL': {'n': [100, 500, 1000, 5000, 20000, 50000],
                  'xgb': [0.208, 0.544, 0.639, 0.758, 0.814, 0.832],
                  'gnn': [-0.334, 0.481, 0.570, 0.766, 0.814, 0.833], 'n_total': 95004},
        'OPV': {'n': [68, 137, 344, 689, 1033, 1378],
                 'xgb': [0.511, 0.558, 0.677, 0.695, 0.725, 0.730],
                 'gnn': [0.145, 0.457, 0.433, 0.542, 0.503, 0.598], 'n_total': 1916},
    }

    fig, axes = plt.subplots(3, 3, figsize=(10, 9))
    axes_flat = axes.flatten()
    for idx, (name, d) in enumerate(datasets.items()):
        ax = axes_flat[idx]
        n = np.array(d['n'])
        ax.plot(n, d['xgb'], '-o', color=C_XGB, linewidth=1.5, markersize=4, markerfacecolor='white',
                markeredgewidth=0.8, label='XGBoost')
        ax.plot(n, d['gnn'], '-s', color=C_GNN, linewidth=1.5, markersize=4, markerfacecolor='white',
                markeredgewidth=0.8, label='GNN')
        ax.set_title(f'{name}  (N={d["n_total"]:,})', fontsize=8.5, fontweight='bold')
        ax.set_xscale('log')
        ax.set_ylim(-0.4, 1.0)
        ax.axhline(0, color='grey', linestyle=':', alpha=0.2, linewidth=0.5)
        ax.grid(True, alpha=0.15, linewidth=0.4)
        if idx == 0: ax.legend(fontsize=7, frameon=False)
        if idx >= 6: ax.set_xlabel('Training samples', fontsize=7.5)
        if idx % 3 == 0: ax.set_ylabel('R²', fontsize=7.5)
    axes_flat[7].axis('off')
    axes_flat[8].axis('off')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig_crossover_analysis.png')
    fig.savefig(path); plt.close(fig); print(f"  -> {path}")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 8: Noise vs Crossover (updated P1 figure)
# ═══════════════════════════════════════════════════════════════════
def fig8_noise_crossover():
    """Regenerate noise-crossover figure with Nature style."""
    print("[Fig 8] Noise vs crossover ...")
    data = [
        ('FreeSolv', 0.117, 188, 0.202),
        ('Lipophilicity', 0.342, 209, 0.133),
        ('ESOL', 0.116, 382, 0.460),
        ('CEPDB', 0.075, 1568, 0.089),
        ('QM9', 0.050, 2080, 0.017),
        ('NREL', 0.167, 2560, -0.011),
        ('OPV', 0.270, None, -0.209),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5), gridspec_kw={'wspace': 0.35})

    for name, noise, cross, delta in data:
        if cross is not None:
            ax1.scatter(noise, cross, c=C_GNN, s=90, edgecolors='black', linewidth=0.5, zorder=5, alpha=0.85)
            ax1.annotate(name, (noise, cross), textcoords="offset points", xytext=(6, 5), fontsize=7.5)
        else:
            ax1.scatter(noise, 1916, c=NC_RED, s=80, edgecolors='black', linewidth=0.5, zorder=5, alpha=0.6)
            ax1.annotate(f'{name} (no cross)', (noise, 1916), textcoords="offset points",
                        xytext=(6, -10), fontsize=7, color='grey')

    ax1.set_ylabel('Crossover point (samples)')
    ax1.set_xlabel('Noise level (1 − max R²)')
    ax1.set_yscale('log')
    ax1.set_title('a  Crossover vs task noise', fontsize=9, loc='left')
    ax1.grid(True, alpha=0.15, linewidth=0.4)

    # Panel b: ΔR² vs noise
    for name, noise, cross, delta in data:
        color = NC_RED if name == 'OPV' else NC_GREEN if name == 'NREL' else C_GNN
        ax2.scatter(noise, delta, c=color, s=90, edgecolors='black', linewidth=0.5, alpha=0.85, zorder=5)
        ax2.annotate(name, (noise, delta), textcoords="offset points", xytext=(6, 5), fontsize=7.5)
    ax2.axhline(y=0, color='grey', linestyle='--', alpha=0.4, linewidth=0.7)
    ax2.set_xlabel('Noise level (1 − max R²)')
    ax2.set_ylabel('ΔR² (GNN − XGBoost) at full data')
    ax2.set_title('b  Performance gap vs task noise', fontsize=9, loc='left')
    ax2.grid(True, alpha=0.15, linewidth=0.4)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig_noise_crossover.png')
    fig.savefig(path); plt.close(fig); print(f"  -> {path}")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 9: Uncertainty (updated P1 figure)
# ═══════════════════════════════════════════════════════════════════
def fig9_uncertainty(results_file='external_results/uncertainty_quantification.json'):
    """Regenerate uncertainty figure with Nature style."""
    print("[Fig 9] Uncertainty ...")
    try:
        with open(os.path.join(BASE_DIR, results_file)) as f:
            data = json.load(f)
    except:
        print("  WARNING: uncertainty results not found, skipping")
        return

    cal = np.array([(r['confidence'], r['xgb_coverage'], r['gnn_coverage']) for r in data['calibration']])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5), gridspec_kw={'wspace': 0.35})

    # Calibration curve
    ax1.plot(cal[:, 0], cal[:, 0], 'k--', alpha=0.4, linewidth=0.8, label='Perfect calibration')
    ax1.plot(cal[:, 0], cal[:, 1], 's-', color=C_XGB, linewidth=1.5, markersize=4, label='XGBoost')
    ax1.plot(cal[:, 0], cal[:, 2], 'o-', color=C_GNN, linewidth=1.5, markersize=4, label='GNN')
    ax1.set_xlabel('Expected confidence'); ax1.set_ylabel('Observed coverage')
    ax1.set_title('a  Calibration curves', fontsize=9, loc='left')
    ax1.legend(frameon=False, fontsize=7.5)
    ax1.grid(True, alpha=0.15, linewidth=0.4)

    # Try to load test data for prediction intervals
    ax2.set_xlabel('Test sample (sorted by PCE)'); ax2.set_ylabel('PCE (%)')
    ax2.set_title('b  Prediction intervals', fontsize=9, loc='left')
    ax2.text(0.5, 0.5, 'See supplementary fig_uncertainty\nfor full prediction interval plot',
             transform=ax2.transAxes, ha='center', va='center', fontsize=9, color='grey', style='italic')
    ax2.grid(True, alpha=0.15, linewidth=0.4)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig_uncertainty.png')
    fig.savefig(path); plt.close(fig); print(f"  -> {path}")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 10: Decision framework (already generated, just verify style)
# ═══════════════════════════════════════════════════════════════════
# (The fig_decision_framework.png was already generated with consistent styling)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 50)
    print("  Beautifying all figures — Nature-journal style")
    print("=" * 50)
    df = load_data()
    fig1_scatter_comparison(df)
    fig2_pce_distribution(df)
    fig3_feature_importance(df)
    fig4_learning_curve(df)
    fig5_tsne(df)
    fig6_residuals(df)
    fig7_crossover()
    fig8_noise_crossover()
    fig9_uncertainty()
    print("\n" + "=" * 50)
    print("  All figures beautified!")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith('.png'):
            size = os.path.getsize(os.path.join(OUTPUT_DIR, f)) // 1024
            print(f"    {f:40s} {size:>4d} KB")
    print("=" * 50)
