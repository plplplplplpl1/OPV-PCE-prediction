"""
综合模型评价对比图
读取 results/baseline_metrics.json 和 results/merged_metrics.json，
生成多维度对比可视化，保存到 results/model_comparison_full.png
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

RESULTS_DIR = 'results'
BASELINE_JSON = os.path.join(RESULTS_DIR, 'baseline_metrics.json')
MERGED_JSON   = os.path.join(RESULTS_DIR, 'merged_metrics.json')
OUTPUT_PNG    = os.path.join(RESULTS_DIR, 'model_comparison_full.png')

# ── 颜色方案 ──────────────────────────────────────────────
BASELINE_COLOR = '#4C72B0'
MERGED_COLOR   = '#DD8452'
METRICS = ['accuracy', 'precision', 'recall', 'f1']
METRIC_LABELS = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

# ── 模型名称映射（英文→显示名） ──────────────────────────
MODEL_DISPLAY = {
    'Random Forest':         'Random\nForest',
    'Gradient Boosting':     'Gradient\nBoosting',
    'SVM':                   'SVM',
    'Logistic Regression':   'Logistic\nRegression',
    'MLP':                   'MLP',
    'Ensemble_AdvancedGCN':  'Ensemble\nAdvGCN',
    'AdvancedGCN':           'AdvancedGCN',
    'GCN':                   'GCN',
}


def load_json(path):
    if not os.path.exists(path):
        print(f"[警告] 文件不存在: {path}")
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_ordered_models(baseline, merged):
    """按 merged F1 降序排列，baseline-only 模型排最后"""
    all_models = list(dict.fromkeys(list(merged.keys()) + list(baseline.keys())))
    def sort_key(m):
        f1 = merged.get(m, baseline.get(m, {})).get('f1', 0)
        return -f1
    return sorted(all_models, key=sort_key)


def main():
    baseline = load_json(BASELINE_JSON)
    merged   = load_json(MERGED_JSON)

    if not baseline and not merged:
        print("没有找到任何结果文件，退出。")
        return

    models = get_ordered_models(baseline, merged)
    display_names = [MODEL_DISPLAY.get(m, m) for m in models]
    n = len(models)
    x = np.arange(n)
    w = 0.38  # bar width

    fig = plt.figure(figsize=(20, 22))
    fig.suptitle('OPV PCE 分类器综合性能对比\n（基线 1719条 vs 合并后 3018条，阈值 3%）',
                 fontsize=16, fontweight='bold', y=0.98)

    gs = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── 子图 1-4：四个指标的分组柱状图 ──────────────────
    for idx, (metric, label) in enumerate(zip(METRICS, METRIC_LABELS)):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])

        b_vals = [baseline.get(m, {}).get(metric, np.nan) for m in models]
        m_vals = [merged.get(m,   {}).get(metric, np.nan) for m in models]

        bars_b = ax.bar(x - w/2, b_vals, w, label='基线 (1719)', color=BASELINE_COLOR, alpha=0.85)
        bars_m = ax.bar(x + w/2, m_vals, w, label='合并 (3018)', color=MERGED_COLOR,   alpha=0.85)

        # 数值标签
        for bar, val in zip(bars_b, b_vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=7.5, color=BASELINE_COLOR)
        for bar, val in zip(bars_m, m_vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=7.5, color=MERGED_COLOR)

        ax.set_title(label, fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(display_names, fontsize=9)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel(label, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(0.8, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    # ── 子图 5：AUC 对比（若有） ──────────────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    b_auc = [baseline.get(m, {}).get('auc', np.nan) for m in models]
    m_auc = [merged.get(m,   {}).get('auc', np.nan) for m in models]

    has_auc = any(not np.isnan(v) for v in b_auc + m_auc)
    if has_auc:
        bars_b5 = ax5.bar(x - w/2, b_auc, w, label='基线', color=BASELINE_COLOR, alpha=0.85)
        bars_m5 = ax5.bar(x + w/2, m_auc, w, label='合并', color=MERGED_COLOR,   alpha=0.85)
        for bar, val in zip(bars_b5, b_auc):
            if not np.isnan(val):
                ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                         f'{val:.3f}', ha='center', va='bottom', fontsize=7.5, color=BASELINE_COLOR)
        for bar, val in zip(bars_m5, m_auc):
            if not np.isnan(val):
                ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                         f'{val:.3f}', ha='center', va='bottom', fontsize=7.5, color=MERGED_COLOR)
        ax5.set_ylim(0, 1.08)
        ax5.axhline(0.8, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    else:
        ax5.text(0.5, 0.5, 'AUC 数据不可用', ha='center', va='center',
                 transform=ax5.transAxes, fontsize=12, color='gray')

    ax5.set_title('AUC-ROC', fontsize=13, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(display_names, fontsize=9)
    ax5.set_ylabel('AUC', fontsize=10)
    ax5.legend(fontsize=9)
    ax5.grid(axis='y', alpha=0.3)

    # ── 子图 6：提升幅度热力图 ────────────────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    all_metrics = METRICS + (['auc'] if has_auc else [])
    all_labels  = METRIC_LABELS + (['AUC'] if has_auc else [])

    delta_matrix = []
    valid_models = []
    for m in models:
        row = []
        for met in all_metrics:
            bv = baseline.get(m, {}).get(met, np.nan)
            mv = merged.get(m,   {}).get(met, np.nan)
            if not np.isnan(bv) and not np.isnan(mv):
                row.append(mv - bv)
            else:
                row.append(np.nan)
        delta_matrix.append(row)
        valid_models.append(MODEL_DISPLAY.get(m, m))

    delta_arr = np.array(delta_matrix, dtype=float)

    # 用 0 填充 NaN 以便绘图
    delta_plot = np.where(np.isnan(delta_arr), 0, delta_arr)
    vmax = max(0.15, np.nanmax(np.abs(delta_arr)))

    im = ax6.imshow(delta_plot, cmap='RdYlGn', vmin=-vmax, vmax=vmax, aspect='auto')
    plt.colorbar(im, ax=ax6, label='Δ (合并 - 基线)')

    ax6.set_xticks(range(len(all_metrics)))
    ax6.set_xticklabels(all_labels, fontsize=9)
    ax6.set_yticks(range(len(valid_models)))
    ax6.set_yticklabels(valid_models, fontsize=9)
    ax6.set_title('性能提升热力图 (合并 - 基线)', fontsize=13, fontweight='bold')

    for i in range(len(valid_models)):
        for j in range(len(all_metrics)):
            val = delta_arr[i, j]
            if not np.isnan(val):
                color = 'white' if abs(val) > vmax * 0.6 else 'black'
                ax6.text(j, i, f'{val:+.3f}', ha='center', va='center',
                         fontsize=8, color=color, fontweight='bold')

    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches='tight')
    print(f"\n综合对比图已保存至: {OUTPUT_PNG}")

    # 打印汇总表
    print("\n" + "="*70)
    print(f"{'模型':<22} {'基线Acc':>8} {'合并Acc':>8} {'ΔAcc':>7} {'基线F1':>8} {'合并F1':>8} {'ΔF1':>7}")
    print("-"*70)
    for m in models:
        ba = baseline.get(m, {}).get('accuracy', float('nan'))
        ma = merged.get(m,   {}).get('accuracy', float('nan'))
        bf = baseline.get(m, {}).get('f1', float('nan'))
        mf = merged.get(m,   {}).get('f1', float('nan'))
        da = ma - ba if not (np.isnan(ma) or np.isnan(ba)) else float('nan')
        df = mf - bf if not (np.isnan(mf) or np.isnan(bf)) else float('nan')
        print(f"{m:<22} {ba:>8.4f} {ma:>8.4f} {da:>+7.4f} {bf:>8.4f} {mf:>8.4f} {df:>+7.4f}")


if __name__ == '__main__':
    main()
