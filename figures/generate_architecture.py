#!/usr/bin/env python3
"""
Professional model architecture diagram — Nature-journal quality.

Three-branch GNN (GCN + GAT + GraphSAGE) with fingerprint fusion
and hierarchical classification + regression pipeline.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Nature-style theme ────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8,
    'axes.linewidth': 0.6,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
})

# Professional muted palette
C = {
    'bg':        '#F9FAFB',
    'text':      '#1A1A2E',
    'text_light':'#6B7280',
    'line':      '#9CA3AF',
    'input':     '#E8F0FE',
    'input_edge':'#5B9BD5',
    'gcn':       '#FCE4EC',
    'gcn_edge':  '#E91E63',
    'gat':       '#E8F5E9',
    'gat_edge':  '#4CAF50',
    'sage':      '#FFF3E0',
    'sage_edge': '#FF9800',
    'pool':      '#EDE7F6',
    'pool_edge': '#673AB7',
    'fp':        '#E3F2FD',
    'fp_edge':   '#2196F3',
    'fusion':    '#FBE9E7',
    'fusion_edge':'#FF5722',
    'reg':       '#E8F5E9',
    'reg_edge':  '#2E7D32',
    'cls':       '#FCE4EC',
    'cls_edge':  '#C62828',
    'output':    '#FFFDE7',
    'output_edge':'#F9A825',
    'arrow':     '#78909C',
    'sep':       '#E0E0E0',
}


def box(ax, x, y, w, h, color, edgecolor, text, fontsize=8, alpha=0.92, text2=None, text_color='#333333'):
    """Draw a rounded rectangle with centered text."""
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=4", facecolor=color,
                          edgecolor=edgecolor, linewidth=1.5, alpha=alpha)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color=text_color)
    if text2:
        ax.text(x + w/2, y + h*0.25, text2, ha='center', va='center',
                fontsize=fontsize-1.5, color=C['text_light'])


def arrow(ax, x1, y1, x2, y2, style='arc3,rad=0', lw=1.2):
    """Draw a clean arrow."""
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                 arrowstyle='-|>', color=C['arrow'],
                                 linewidth=lw, connectionstyle=style,
                                 mutation_scale=12))


def label(ax, x, y, text, fontsize=9, weight='bold'):
    """Add a section label."""
    ax.text(x, y, text, fontsize=fontsize, fontweight=weight,
            color=C['text'], ha='left', va='center')


def main():
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    # ══════════════════════════════════════════════════════════════
    # SECTION A: Classification Pipeline (y ≈ 6.2–7.8)
    # ══════════════════════════════════════════════════════════════
    sep_y = 5.95
    ax.axhline(sep_y, xmin=0.05, xmax=0.95, color=C['sep'], linewidth=1, linestyle='-')
    label(ax, 0.3, 7.5, 'A  GNN Classifier (Stage 1)', fontsize=10)

    box(ax, 0.5, 6.7, 1.8, 0.7, C['input'], C['input_edge'], 'Molecule', text2='Graph + Features')
    arrow(ax, 2.3, 7.05, 3.2, 7.05)

    box(ax, 3.2, 6.7, 1.8, 0.7, C['cls'], C['cls_edge'], 'AdvancedGCN', text2='3-Branch GNN')
    arrow(ax, 5.0, 7.15, 5.8, 7.15)

    box(ax, 5.8, 6.85, 1.2, 0.4, '#E8F5E9', '#2E7D32', 'High PCE\n(>3%)', fontsize=7)
    arrow(ax, 5.0, 6.95, 5.8, 6.95)

    box(ax, 5.8, 6.25, 1.2, 0.4, '#E3F2FD', '#1565C0', 'Low PCE\n(≤3%)', fontsize=7)

    # ══════════════════════════════════════════════════════════════
    # SECTION B: Regression Architecture (y ≈ 0.5–5.5)
    # ══════════════════════════════════════════════════════════════
    label(ax, 0.3, 5.5, 'B  HighPCE Regressor (Stage 2)', fontsize=10)

    # Input molecule
    box(ax, 0.3, 3.8, 1.5, 0.9, C['input'], C['input_edge'],
        'Molecule\nInput', text2='SMILES + 2D Graph', fontsize=9)

    # Split paths: upwards to graph, rightwards to fingerprint
    arrow(ax, 1.8, 4.5, 2.5, 4.5)   # to graph path
    arrow(ax, 1.8, 3.8, 2.5, 3.0)   # to FP path (angled)

    # ── Three GNN branches ──
    bw, bh = 1.5, 1.6
    bx_start = 3.0
    by = 4.0
    branches = [
        (bx_start,           'GCN Branch',  'Graph\nConvolution', C['gcn'], C['gcn_edge']),
        (bx_start + bw + 0.2, 'GAT Branch',  'Graph\nAttention',   C['gat'], C['gat_edge']),
        (bx_start + 2*(bw+0.2), 'SAGE Branch', 'GraphSAGE\nSample & Agg.', C['sage'], C['sage_edge']),
    ]
    for bx, name, desc, col, ec in branches:
        box(ax, bx, by, bw, bh, col, ec, name, text2=desc, fontsize=7.5)
        # Layer indicators
        for j in range(2):
            ly = by + bh - 0.35 - j * 0.45
            rect = FancyBboxPatch((bx + 0.12, ly - 0.15), bw - 0.24, 0.25,
                                  boxstyle="round,pad=1", facecolor='white',
                                  edgecolor='#DDD', linewidth=0.6, alpha=0.7)
            ax.add_patch(rect)
            ax.text(bx + bw/2, ly, f'Layer {j+1} (128d)', ha='center', va='center',
                    fontsize=5.5, color=C['text_light'])

        # Arrow from input to each branch
        arrow(ax, 1.8, 4.25 + 0.15*(bx - bx_start)/0.7,
              bx + 0.1, by + bh - 0.15)

        # Arrow from each branch to pooling
        arrow(ax, bx + bw/2, by - 0.05, bx + bw/2, by - 0.4)

    # ── Multi-scale pooling ──
    pool_y = by - 0.85
    for bx, _, _, col, ec in branches:
        box(ax, bx + 0.05, pool_y, bw - 0.1, 0.4, C['pool'], C['pool_edge'],
            'Multi-Scale\nPooling', fontsize=6)
        arrow(ax, bx + bw/2, pool_y - 0.05, bx + bw/2, pool_y - 0.35)

    # ── Concatenation ──
    concat_x = bx_start + 3*(bw+0.2) + 0.15
    box(ax, concat_x, 3.0, 1.0, 0.8, '#F3E5F5', '#7B1FA2',
        'Concatenate\n3 Branches\n(384-dim)', fontsize=6.5)
    for bx, _, _, _, _ in branches:
        arrow(ax, bx + bw/2, pool_y - 0.4, concat_x, 3.4,
              style='arc3,rad=0.15')

    # ── Morgan fingerprint path ──
    fp_x = 3.0
    fp_y = 1.5
    box(ax, fp_x, fp_y, 1.4, 0.55, C['fp'], C['fp_edge'],
        'Morgan Fingerprint (512-bit)', fontsize=7)
    arrow(ax, 1.8, 2.8, fp_x + 0.3, fp_y + 0.45, style='arc3,rad=-0.2')

    box(ax, fp_x, fp_y - 0.75, 1.4, 0.5, '#E3F2FD', '#1565C0',
        'MLP Encoder\n512 → 128', fontsize=7)
    arrow(ax, fp_x + 0.7, fp_y - 0.15, fp_x + 0.7, fp_y - 0.55)

    # Arrow from FP to fusion
    arrow(ax, fp_x + 1.4, fp_y - 0.3, concat_x + 1.0, fp_y - 0.1,
          style='arc3,rad=-0.15')

    # ── Fusion ──
    fusion_x = concat_x + 1.3
    box(ax, fusion_x, 1.8, 1.6, 1.6, C['fusion'], C['fusion_edge'],
        'Feature Fusion', text2='384 (graph)\n+ 128 (FP)\n= 512-dim', fontsize=8)
    arrow(ax, concat_x + 1.0, 3.4, fusion_x, 2.8, style='arc3,rad=0.1')

    # ── Regression Head ──
    reg_x = fusion_x + 1.9
    box(ax, reg_x, 1.8, 1.6, 1.6, C['reg'], C['reg_edge'],
        'Regression Head', text2='512 → 128 → 1\nDropout=0.3\nReLU', fontsize=8)
    arrow(ax, fusion_x + 1.6, 2.6, reg_x, 2.6)

    # ── Output ──
    out_x = reg_x + 2.0
    box(ax, out_x, 2.1, 1.2, 0.8, C['output'], C['output_edge'],
        'PCE\nPrediction', fontsize=9, text_color='#1A1A2E')
    arrow(ax, reg_x + 1.6, 2.6, out_x, 2.5)

    # ── Low PCE Regressor (reference box) ──
    box(ax, 0.3, 1.0, 2.0, 0.45, '#E3F2FD', '#90CAF9',
        'LowPCERegressorV2  (R²=0.114)', fontsize=6.5)
    arrow(ax, 2.3, 1.22, 5.8, 6.45, style='arc3,rad=0.3')

    # ── Stats panel ──
    stats_x = 8.5
    stats_y = 5.0
    rect = FancyBboxPatch((stats_x, stats_y - 2.8), 3.0, 3.0,
                          boxstyle="round,pad=6", facecolor='#F8F9FA',
                          edgecolor='#E0E0E0', linewidth=1.2)
    ax.add_patch(rect)
    ax.text(stats_x + 1.5, stats_y - 0.2, 'Model Statistics', ha='center', va='center',
            fontsize=9, fontweight='bold', color=C['text'])
    stats = [
        ('Parameters', '~592K'),
        ('GNN hidden dim', '128'),
        ('Layers per branch', '2'),
        ('Pooling', 'Mean / Max / Sum'),
        ('Fingerprint', 'Morgan r=2, 512-bit'),
        ('Dropout', '0.3'),
        ('Optimizer', 'Adam (lr=0.001)'),
        ('Early stopping', 'Patience=20'),
        ('Training time', '~40 min (GPU)'),
    ]
    for i, (k, v) in enumerate(stats):
        ax.text(stats_x + 0.3, stats_y - 0.55 - i * 0.28, k, fontsize=6.5,
                color=C['text_light'], va='center')
        ax.text(stats_x + 2.7, stats_y - 0.55 - i * 0.28, v, fontsize=6.5,
                color=C['text'], va='center', ha='right', fontweight='bold')

    # ── Legend ──
    legend_x = 8.5
    legend_y = 1.5
    legend_items = [
        (C['gcn'], C['gcn_edge'], 'GCN Branch'),
        (C['gat'], C['gat_edge'], 'GAT Branch'),
        (C['sage'], C['sage_edge'], 'GraphSAGE Branch'),
        (C['pool'], C['pool_edge'], 'Multi-Scale Pooling'),
        (C['fp'], C['fp_edge'], 'Morgan Fingerprint'),
        (C['fusion'], C['fusion_edge'], 'Feature Fusion'),
        (C['cls'], C['cls_edge'], 'GNN Classifier'),
    ]
    for i, (fc, ec, lbl) in enumerate(legend_items):
        rect = FancyBboxPatch((legend_x + 0.1, legend_y - 0.25 - i * 0.3), 0.3, 0.18,
                              boxstyle="round,pad=1", facecolor=fc,
                              edgecolor=ec, linewidth=1.2)
        ax.add_patch(rect)
        ax.text(legend_x + 0.55, legend_y - 0.16 - i * 0.3, lbl,
                fontsize=7, color=C['text'], va='center')

    # ── Title ──
    ax.text(6, 7.95, 'Hierarchical OPV PCE Prediction Architecture',
            ha='center', va='center', fontsize=12, fontweight='bold',
            color=C['text'])

    # ── Save ──
    path = os.path.join(OUTPUT_DIR, 'fig8_model_architecture.png')
    fig.savefig(path, bbox_inches='tight', pad_inches=0.3, facecolor='white')
    plt.close(fig)
    print(f"  ✓ Architecture diagram: {path}")
    print(f"    Size: {os.path.getsize(path)//1024}KB")


if __name__ == '__main__':
    main()
