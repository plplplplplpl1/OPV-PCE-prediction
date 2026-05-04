"""
P1-3: 实践决策框架图

基于噪声水平和样本量的XGBoost vs GNN选择流程图
"""
import os, sys
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if os.path.basename(project_root) == '实验':
    project_root = os.path.dirname(project_root)
os.chdir(project_root)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import json

plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ── Data from crossover_model.json ──
datasets = {
    'ESOL':          {'noise': 0.116, 'cross': 382,   'n': 1128,  'delta': 0.460},
    'FreeSolv':      {'noise': 0.117, 'cross': 188,   'n': 642,   'delta': 0.202},
    'Lipophilicity': {'noise': 0.342, 'cross': 209,   'n': 4200,  'delta': 0.133},
    'CEPDB':         {'noise': 0.075, 'cross': 1568,  'n': 25000, 'delta': 0.089},
    'QM9_complex':   {'noise': 0.050, 'cross': 2080,  'n': 133885,'delta': 0.017},
    'NREL':          {'noise': 0.167, 'cross': None,  'n': 95004, 'delta': -0.011},
    'OPV':           {'noise': 0.270, 'cross': None,  'n': 1916,  'delta': -0.209},
}
# Also add our new baseline model results
tree_mean = {'XGBoost': 0.675, 'CatBoost': 0.677, 'LightGBM': 0.673}
gnn_mean = 0.616  # GPS multi-seed
gnn_v3_mean = 0.635  # V3 multi-seed

# ── Create figure ──
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1, 1], height_ratios=[1, 1],
                       hspace=0.25, wspace=0.3)

# ── Panel a: Decision flowchart (spans left column) ──
ax1 = fig.add_subplot(gs[:, 0])
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 14)
ax1.axis('off')
ax1.set_title('a. Model Selection Decision Framework', fontsize=13, fontweight='bold', loc='left', pad=10)

def draw_box(ax, x, y, w, h, text, color='#E8F4FD', edgecolor='#2980B9', fontsize=10, text_color='black', ha='center'):
    box = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.15",
                          facecolor=color, edgecolor=edgecolor, linewidth=2, zorder=3)
    ax.add_patch(box)
    # handle multi-line text
    for i, line in enumerate(text.split('\n')):
        ax.text(x, y + h/2 - 0.5 - i*0.7, line, fontsize=fontsize, ha=ha, va='top',
                color=text_color, fontweight='bold' if i == 0 else 'normal')

def draw_arrow(ax, x1, y1, x2, y2, color='#7F8C8D', lw=2, style='->'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, shrinkA=5, shrinkB=5), zorder=2)

def draw_diamond(ax, x, y, size, text, color='#FFF3CD', edgecolor='#F39C12'):
    diamond = mpatches.Polygon([(x, y+size), (x+size*1.2, y), (x, y-size), (x-size*1.2, y)],
                                facecolor=color, edgecolor=edgecolor, linewidth=2, zorder=3)
    ax.add_patch(diamond)
    for i, line in enumerate(text.split('\n')):
        ax.text(x, y + 0.3 - i*0.55, line, fontsize=9, ha='center', va='center', fontweight='bold')

# Start
draw_box(ax1, 5, 13, 3.5, 0.9, 'Input: Dataset\n(N samples, task noise)', '#D5F5E3', '#27AE60')

# Arrow down
draw_arrow(ax1, 5, 12.55, 5, 11.5)

# Estimate crossover point
draw_box(ax1, 5, 10.5, 5, 1.1, 'Estimate crossover n*\nfrom noise level & task complexity', '#E8F4FD', '#2980B9', fontsize=9)

# Arrow down
draw_arrow(ax1, 5, 9.95, 5, 9.0)

# Diamond: N > n*?
draw_diamond(ax1, 5, 7.8, 0.8, 'N > n* ?')

# Yes branch (right)
draw_arrow(ax1, 6.96, 7.8, 8.0, 7.8)
ax1.text(7.5, 8.0, 'Yes', fontsize=10, fontweight='bold', color='#27AE60', ha='center')
draw_box(ax1, 8.8, 7.0, 2.2, 1.1, 'GNN Zone\nExpected to overtake', '#D5F5E3', '#27AE60')
# GNN sub-box
draw_box(ax1, 8.8, 5.6, 2.2, 0.9, 'Use GNN with\n3D/electronic features', '#A9DFBF', '#27AE60', fontsize=8.5)
ax1.text(7.8, 4.8, 'Check architecture:\nsimple GCN may suffice\nfor low-noise tasks', fontsize=8, color='#7F8C8D', ha='center')

# No branch (left)
draw_arrow(ax1, 3.04, 7.8, 2.0, 7.8)
ax1.text(2.5, 8.0, 'No', fontsize=10, fontweight='bold', color='#E74C3C', ha='center')
draw_box(ax1, 2.0, 7.0, 2.2, 1.1, 'XGBoost Zone\nDefault choice', '#FADBD8', '#E74C3C')

# Sub-branches from No
# Classification vs regression
draw_diamond(ax1, 2.0, 5.2, 0.55, 'Task\ntype?')

# Classification
draw_arrow(ax1, 1.34, 5.2, 0.5, 5.2)
ax1.text(0.9, 5.3, 'Class.', fontsize=8, ha='center')
draw_box(ax1, 0.5, 4.2, 1.8, 0.9, 'Consider GNN\n(gains ~1% acc.)', '#F9E79F', '#F39C12', fontsize=8.5)

# Regression
draw_arrow(ax1, 2.66, 5.2, 3.5, 5.2)
ax1.text(2.9, 5.3, 'Regression', fontsize=8, ha='center')
draw_box(ax1, 3.5, 4.2, 1.8, 0.9, 'XGBoost solid\n(try CatBoost too)', '#FADBD8', '#E74C3C', fontsize=8.5)

# Bottom: recommendation
draw_arrow(ax1, 5, 3.5, 5, 2.5)
draw_box(ax1, 5, 1.8, 5.5, 1.0, 'Key principle: model selection\nshould be data-driven, not assumption-driven', '#E8DAEF', '#8E44AD', fontsize=9)

# Add mini formula
ax1.text(5, 0.3, 'n* ≈ f(noise): simple tasks ~150–200, complex ~1,500–2,000', fontsize=9,
         ha='center', color='#7F8C8D', style='italic')

# ── Panel b: Noise vs Crossover with decision regions ──
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_xlabel('Noise level (1 − max R²)', fontsize=12)
ax2.set_ylabel('Crossover point (samples)', fontsize=12)
ax2.set_title('b. Crossover point vs Noise level', fontsize=13, fontweight='bold', loc='left')
ax2.set_xlim(-0.02, 0.45)
ax2.set_ylim(30, 50000)
ax2.set_yscale('log')

# Decision regions
ax2.axhspan(30, 1000, alpha=0.08, color='#E74C3C', label='XGBoost recommended')
ax2.axhspan(1000, 50000, alpha=0.08, color='#3498DB', label='GNN may overtake')
ax2.axhline(y=1000, color='gray', linestyle=':', alpha=0.4, linewidth=1)
ax2.text(0.42, 700, 'Decision\nthreshold\n(~1,000)', fontsize=8, color='gray', ha='right', va='top')

# Plot each dataset
colors = {'ESOL': '#2ECC71', 'FreeSolv': '#2ECC71', 'Lipophilicity': '#2ECC71',
           'CEPDB': '#3498DB', 'QM9_complex': '#3498DB', 'NREL': '#E74C3C', 'OPV': '#E74C3C'}
markers = {'ESOL': 'o', 'FreeSolv': 's', 'Lipophilicity': '^',
            'CEPDB': 'o', 'QM9_complex': 's', 'NREL': 'D', 'OPV': 'v'}

for name, d in datasets.items():
    if d['cross'] is not None:
        ax2.scatter(d['noise'], d['cross'], c=colors[name], marker=markers[name],
                    s=150, edgecolors='black', linewidth=0.8, zorder=5, alpha=0.9)
        ax2.annotate(name, (d['noise'], d['cross']),
                     textcoords="offset points", xytext=(8, 6), fontsize=9,
                     fontweight='bold')
    else:
        ax2.scatter(d['noise'], d['n'], c=colors[name], marker=markers[name],
                    s=120, edgecolors='black', linewidth=0.8, zorder=5, alpha=0.6)
        ax2.annotate(f'{name} (no cross)', (d['noise'], d['n']),
                     textcoords="offset points", xytext=(8, -10), fontsize=8.5,
                     color='gray', alpha=0.7)

# Fitted trend line (for datasets with crossover)
cross_data = [(d['noise'], d['cross']) for d in datasets.values() if d['cross'] is not None]
if len(cross_data) >= 3:
    noises = np.array([c[0] for c in cross_data])
    crosses = np.array([c[1] for c in cross_data])
    # Log-linear fit: log(n*) = a + b * noise
    coeffs = np.polyfit(noises, np.log(crosses), 1)
    n_fit = np.linspace(0.02, 0.4, 100)
    cross_fit = np.exp(coeffs[1] + coeffs[0] * n_fit)
    ax2.plot(n_fit, cross_fit, '--', color='#7F8C8D', alpha=0.4, linewidth=1.5, label='Trend')
    ax2.text(0.2, 400, f'log(n*) = {coeffs[1]:.1f} {coeffs[0]:.1f}×noise', fontsize=8,
             color='#7F8C8D', alpha=0.6)

ax2.legend(fontsize=9, loc='upper right')
ax2.grid(True, alpha=0.3)

# ── Panel c: Decision surface ──
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_title('c. Model advantage by data size', fontsize=13, fontweight='bold', loc='left')

# Build synthetic learning curves for OPV-like task
n_range = np.logspace(1, 4, 100)
# XGBoost learning curve (power law from crossover fit)
xgb_a, xgb_b, xgb_c = 0.75, 3.5, 0.35  # saturates ~0.75
gnn_a, gnn_b, gnn_c = 0.80, 8.0, 0.25   # starts lower, higher asymptote

def power_law(n, a, b, c):
    return a - b * n ** (-c)

r2_xgb = power_law(n_range, xgb_a, xgb_b, xgb_c)
r2_gnn = power_law(n_range, gnn_a, gnn_b, gnn_c)
delta = r2_gnn - r2_xgb

# Fill regions
ax3.fill_between(n_range, 0, delta, where=(delta > 0),
                 color='#3498DB', alpha=0.15, label='GNN better')
ax3.fill_between(n_range, delta, 0, where=(delta < 0),
                 color='#E74C3C', alpha=0.15, label='XGBoost better')

# Zero line
ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=0.8)

# Find crossover
cross_idx = np.where(np.diff(np.sign(delta)))[0]
if len(cross_idx) > 0:
    ax3.axvline(x=n_range[cross_idx[0]], color='gray', linestyle='--', alpha=0.6, linewidth=1.5)
    ax3.text(n_range[cross_idx[0]], 0.05, f'n* ≈ {n_range[cross_idx[0]]:.0f}',
             fontsize=9, ha='center', fontweight='bold', color='#7F8C8D')

ax3.plot(n_range, delta, 'k-', linewidth=2, alpha=0.7)
ax3.set_xlabel('Training samples (log)', fontsize=12)
ax3.set_ylabel('ΔR² (GNN − XGBoost)', fontsize=12)
ax3.set_xscale('log')
ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.4)
ax3.legend(fontsize=9, loc='lower right')
ax3.grid(True, alpha=0.3)

# Annotations for key datasets
dataset_ns = {'ESOL': 1128, 'FreeSolv': 642, 'OPV': 1916, 'CEPDB': 25000}
for name, n in dataset_ns.items():
    d_pred = power_law(n, gnn_a, gnn_b, gnn_c) - power_law(n, xgb_a, xgb_b, xgb_c)
    ax3.scatter(n, d_pred, s=80, c='white', edgecolors='#2C3E50', linewidth=1.5, zorder=6)
    ax3.annotate(name, (n, d_pred), textcoords="offset points",
                 xytext=(0, -14), fontsize=8, ha='center', fontweight='bold')

# ── Panel d: GNN vs Tree multi-seed comparison ──
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_title('d. Model comparison (OPV, 4 seeds)', fontsize=13, fontweight='bold', loc='left')

models_data = {
    'XGBoost': {'mean': 0.675, 'std': 0.033, 'color': '#E74C3C'},
    'CatBoost': {'mean': 0.677, 'std': 0.031, 'color': '#E67E22'},
    'LightGBM': {'mean': 0.673, 'std': 0.034, 'color': '#F39C12'},
    'GNN V3': {'mean': 0.635, 'std': 0.039, 'color': '#3498DB'},
    'GraphGPS': {'mean': 0.616, 'std': 0.051, 'color': '#2ECC71'},
    'SimpleGCN': {'mean': 0.604, 'std': 0.033, 'color': '#9B59B6'},
}

names = list(models_data.keys())
means = [models_data[n]['mean'] for n in names]
stds = [models_data[n]['std'] for n in names]
colors_bar = [models_data[n]['color'] for n in names]

x_pos = np.arange(len(names))
bars = ax4.bar(x_pos, means, yerr=stds, color=colors_bar, edgecolor='black',
               linewidth=0.8, capsize=5, width=0.6, alpha=0.85)

# Add value labels
for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
             f'{mean:.3f}\n±{std:.3f}', ha='center', va='bottom', fontsize=7.5)

ax4.set_ylabel('R² (test set)', fontsize=12)
ax4.set_xticks(x_pos)
ax4.set_xticklabels(names, rotation=20, ha='right', fontsize=9)
ax4.set_ylim(0.50, 0.78)
ax4.axhline(y=0.686, color='#E74C3C', linestyle=':', alpha=0.3, linewidth=1)
ax4.grid(True, axis='y', alpha=0.3)

# ── Panel e: Summary table ──
ax5 = fig.add_subplot(gs[1, 2])
ax5.set_title('e. Practical guidelines', fontsize=13, fontweight='bold', loc='left')
ax5.axis('off')

guidelines = [
    ['Condition', 'Recommendation'],
    ['N < 200', 'XGBoost / tree model (default)'],
    ['200 < N < 2,000', 'XGBoost (test GNN; check for crossover)'],
    ['N > 2,000 & low noise', 'GNN likely beneficial'],
    ['N > 2,000 & high noise', 'XGBoost competitive; test both'],
    ['Classification task', 'GNN marginal gain ~1% acc.'],
    ['Architecture uncertain', 'Start simple (GCN), then complex'],
    ['Budget limited (no GPU)', 'XGBoost (600× faster training)'],
]

table = ax5.table(cellText=guidelines, loc='center',
                  colWidths=[0.28, 0.55],
                  cellLoc='left',
                  colColours=['#2C3E50', '#2C3E50'])

# Style table
for i, key in enumerate(table._cells):
    cell = table._cells[key]
    cell.set_edgecolor('#BDC3C7')
    cell.set_linewidth(0.5)
    if key[0] == 0:
        cell.set_text_props(fontsize=8.5, fontweight='bold', color='white')
        cell.set_facecolor('#2C3E50')
    else:
        cell.set_text_props(fontsize=8)
        if key[0] % 2 == 0:
            cell.set_facecolor('#F2F3F4')
        else:
            cell.set_facecolor('white')
    cell.set_height(0.08)

table.scale(1, 1.6)

# ── Save ──
plt.savefig('论文写作指导/论文草稿/figures/fig_decision_framework.png',
            dpi=200, bbox_inches='tight')
plt.close()
print("图形已保存: figures/fig_decision_framework.png")
