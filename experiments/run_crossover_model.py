"""
交叉点预测模型 v2 — 基于幂律参数的解析框架
不使用稀少的交叉点数据拟合，而是从学习曲线幂律参数推导交叉点

核心公式:
  R²_xgb(n) = a_xgb - b_xgb * n^(-c_xgb)
  R²_gnn(n) = a_gnn - b_gnn * n^(-c_gnn)
  交叉点: 令两者相等，数值求解

输出: 交叉点的理论预测 + 噪声水平的连续映射
"""
import json
import numpy as np
from scipy.optimize import curve_fit, brentq
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False


def power_law(n, a, b, c):
    return a - b * n ** (-c)


def fit_pl(n_vals, r2_vals):
    """幂律拟合，带边界约束"""
    n = np.array(n_vals, dtype=float)
    y = np.array(r2_vals, dtype=float)
    a0 = min(max(y[-1] * 1.05, 0.3), 1.5)
    b0 = max(a0 - y[0], 0.1)
    c0 = 0.3
    try:
        popt, _ = curve_fit(power_law, n, y, p0=[a0, b0, c0],
                            bounds=([0.1, 0.01, 0.01], [2.0, 200, 5.0]),
                            maxfev=20000)
        resid = y - power_law(n, *popt)
        r2 = 1 - np.sum(resid**2) / np.sum((y - np.mean(y))**2)
        return popt, r2
    except Exception as e:
        return None, 0


def find_crossover(xgb_params, gnn_params, n_min=10, n_max=200000):
    """数值求解交叉点"""
    if xgb_params is None or gnn_params is None:
        return None

    def diff(n):
        return power_law(n, *xgb_params) - power_law(n, *gnn_params)

    # 在n_min到n_max范围内搜索符号变化
    n_test = np.logspace(np.log10(n_min), np.log10(n_max), 1000)
    diffs = diff(n_test)

    # 找最接近零的点
    sign_changes = np.where(np.diff(np.sign(diffs)))[0]

    if len(sign_changes) == 0:
        # 无符号变化，说明XGBoost始终领先或始终落后
        if diffs[-1] > 0:
            return {'crossover': None, 'direction': 'XGBoost always ahead'}
        else:
            return {'crossover': None, 'direction': 'GNN always ahead'}

    # 取第一个交叉点
    idx = sign_changes[0]
    try:
        n_cross = brentq(diff, n_test[idx], n_test[idx + 1])
        # 确认GNN在交叉后确实反超
        return {
            'crossover': round(n_cross, 0),
            'direction': 'XGBoost→GNN' if diffs[idx] > 0 else 'GNN→XGBoost',
        }
    except Exception:
        return None


def estimate_noise_snr(name, d):
    """
    估计任务的噪声水平和信号复杂度:
    - 噪声: 不可约误差的代理 = 1 - max( asymptotic R² )
    - SNR: max R² / (1 - max R²)
    - 线性可预测性: 最小n下XGB R² / 最大R²（信号能被简单模型提取的比例）
    """
    max_r2 = max(max(d['xgb']), max(d['gnn']))
    noise = max(1.0 - max_r2, 0.01)
    snr = max_r2 / noise if noise > 0 else 100
    min_n_r2 = d['xgb'][0]
    linear_predictability = max(min_n_r2 / max_r2, 0.0) if max_r2 > 0 else 0
    return {
        'noise': round(noise, 3),
        'snr': round(snr, 1),
        'linear_predictability': round(linear_predictability, 3),
        'max_r2': round(max_r2, 3),
    }


# ====== All datasets with learning curves ======
datasets = {
    'ESOL': {
        'n': [100, 250, 500, 1000, 1128],
        'xgb': [0.177, 0.445, 0.591, 0.676, 0.676],
        'gnn': [-2.042, 0.779, 0.857, 0.884, 0.870],
        'n_total': 1128,
        'target': 'Water solubility',
    },
    'FreeSolv': {
        'n': [100, 250, 500, 642],
        'xgb': [0.239, 0.612, 0.731, 0.731],
        'gnn': [-0.057, 0.811, 0.883, 0.871],
        'n_total': 642,
        'target': 'Hydration energy',
    },
    'Lipophilicity': {
        'n': [100, 250, 500, 1000, 2000, 4200],
        'xgb': [-0.034, 0.104, 0.213, 0.345, 0.440, 0.505],
        'gnn': [-0.195, 0.164, 0.322, 0.435, 0.574, 0.658],
        'n_total': 4200,
        'target': 'logD',
    },
    'CEPDB': {
        'n': [100, 250, 500, 1000, 1500, 2000, 3000, 5000, 10000],
        'xgb': [0.413, 0.623, 0.684, 0.736, 0.768, 0.777, 0.800, 0.831, 0.858],
        'gnn': [0.105, 0.483, 0.591, 0.696, 0.750, 0.806, 0.841, 0.883, 0.925],
        'n_total': 25000,
        'target': 'Computed PCE',
    },
    'QM9_complex': {
        'n': [100, 500, 1000, 5000, 20000, 50000],
        'xgb': [0.618, 0.740, 0.772, 0.851, 0.891, 0.904],
        'gnn': [-0.441, 0.560, 0.701, 0.853, 0.931, 0.950],
        'n_total': 133885,
        'target': 'HOMO-LUMO gap',
    },
    'NREL': {
        'n': [100, 500, 1000, 5000, 20000, 50000],
        'xgb': [0.208, 0.544, 0.639, 0.758, 0.814, 0.832],
        'gnn': [-0.334, 0.481, 0.570, 0.766, 0.814, 0.833],
        'n_total': 95004,
        'target': 'HOMO-LUMO gap (DFT)',
    },
    'OPV': {
        'n': [68, 137, 344, 689, 1033, 1378],
        'xgb': [0.511, 0.558, 0.677, 0.695, 0.725, 0.730],
        'gnn': [0.145, 0.457, 0.433, 0.542, 0.503, 0.598],
        'n_total': 1916,
        'target': 'Experimental PCE',
    },
}


# ====== Main analysis ======
results = []

plt.figure(figsize=(14, 10))

for i, (name, d) in enumerate(datasets.items()):
    n = np.array(d['n'], dtype=float)

    # 拟合
    xgb_p, xgb_r2 = fit_pl(n, d['xgb'])
    gnn_p, gnn_r2 = fit_pl(n, d['gnn'])

    # 噪声
    noise_info = estimate_noise_snr(name, d)

    # 交叉点
    cross_info = find_crossover(xgb_p, gnn_p, n_min=10, n_max=200000)

    # 在n_total处的差距
    r2_full_xgb = power_law(d['n_total'], *xgb_p) if xgb_p is not None else d['xgb'][-1]
    r2_full_gnn = power_law(d['n_total'], *gnn_p) if gnn_p is not None else d['gnn'][-1]
    delta_at_full = r2_full_gnn - r2_full_xgb

    results.append({
        'dataset': name,
        'n_total': d['n_total'],
        'noise': noise_info['noise'],
        'snr': noise_info['snr'],
        'linear_predictability': noise_info['linear_predictability'],
        'xgb_power_law': [round(v, 4) for v in xgb_p] if xgb_p is not None else None,
        'xgb_fit_r2': round(xgb_r2, 3),
        'gnn_power_law': [round(v, 4) for v in gnn_p] if gnn_p is not None else None,
        'gnn_fit_r2': round(gnn_r2, 3),
        'crossover': cross_info,
        'delta_r2_at_full': round(float(delta_at_full), 3),
    })

    # 绘图
    ax = plt.subplot(3, 3, i + 1)
    n_smooth = np.logspace(np.log10(max(n[0]*0.8, 10)), np.log10(max(n[-1]*2, 200)), 200)

    if xgb_p is not None:
        r2_xgb_s = power_law(n_smooth, *xgb_p)
        plt.plot(n_smooth, r2_xgb_s, '-', color='#E74C3C', linewidth=2, alpha=0.8)
    plt.scatter(n, d['xgb'], c='#E74C3C', s=40, zorder=5, edgecolors='black', linewidth=0.5)

    if gnn_p is not None:
        r2_gnn_s = power_law(n_smooth, *gnn_p)
        plt.plot(n_smooth, r2_gnn_s, '-', color='#3498DB', linewidth=2, alpha=0.8)
    plt.scatter(n, d['gnn'], c='#3498DB', s=40, zorder=5, edgecolors='black', linewidth=0.5)

    if cross_info and cross_info['crossover']:
        plt.axvline(x=cross_info['crossover'], color='gray', linestyle='--', alpha=0.5, linewidth=1)
        plt.text(cross_info['crossover'], plt.ylim()[0] + 0.05,
                 f' n*={cross_info["crossover"]:.0f}', fontsize=10, color='gray')

    plt.title(f'{name}\n(noise={noise_info["noise"]:.2f}, SNR={noise_info["snr"]:.0f})',
              fontsize=11)
    plt.xlabel('Training samples' if i >= 6 else '', fontsize=10)
    plt.ylabel('R²' if i % 3 == 0 else '', fontsize=10)
    plt.xscale('log')
    plt.ylim(-0.3, 1.0)
    plt.grid(True, alpha=0.3)
    if i == 0:
        plt.legend(['XGBoost', 'GNN'], fontsize=9)

plt.tight_layout()
plt.savefig('论文写作指导/论文草稿/figures/fig_crossover_analysis.png', dpi=200, bbox_inches='tight')
plt.close()

# ====== Print summary ======
print("=" * 100)
print("交叉点分析 — 基于幂律学习曲线")
print("=" * 100)
print(f"{'Dataset':<15} {'n_total':>8} {'Noise':>7} {'SNR':>6} {'LinPred':>8} {'XGB_fitR²':>9} {'GNN_fitR²':>9} {'Crossover':>10} {'ΔR²@full':>9}")
print("-" * 100)
for r in results:
    cross_str = f"{r['crossover']['crossover']:.0f}" if r['crossover'] and r['crossover']['crossover'] else \
                r['crossover']['direction'] if r['crossover'] else 'N/A'
    print(f"{r['dataset']:<15} {r['n_total']:>8d} {r['noise']:>7.3f} {r['snr']:>6.0f} "
          f"{r['linear_predictability']:>8.3f} {r['xgb_fit_r2']:>9.3f} {r['gnn_fit_r2']:>9.3f} "
          f"{cross_str:>10} {r['delta_r2_at_full']:>+9.3f}")

# ====== Noise vs Crossover scatter ======
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for r in results:
    has_cross = r['crossover'] and r['crossover']['crossover']
    color = '#E74C3C' if r['dataset'] == 'NREL' else '#3498DB' if r['dataset'] == 'OPV' else '#2ECC71'
    marker = 'o' if has_cross else '^'
    size = 200 if has_cross else 100
    label = r['dataset']
    plt.scatter(r['noise'], r['crossover']['crossover'] if has_cross else r['n_total'],
                c=color, marker=marker, s=size, edgecolors='black', linewidth=0.5, alpha=0.8)
    if has_cross:
        plt.annotate(label, (r['noise'], r['crossover']['crossover']),
                     textcoords="offset points", xytext=(5, 5), fontsize=9)
    else:
        plt.annotate(f'{label} (no cross)', (r['noise'], r['n_total']),
                     textcoords="offset points", xytext=(5, -12), fontsize=9, alpha=0.7)

plt.axhline(y=1500, color='gray', linestyle=':', alpha=0.4)
plt.axhline(y=300, color='gray', linestyle=':', alpha=0.4)
plt.text(0.01, 1550, 'Complex tasks', fontsize=10, color='gray', alpha=0.6)
plt.text(0.01, 350, 'Simple tasks', fontsize=10, color='gray', alpha=0.6)
plt.xlabel('Noise Level (1 − max R²)', fontsize=13)
plt.ylabel('Crossover Point (samples)', fontsize=13)
plt.yscale('log')
plt.title('a. Crossover point vs Task noise', fontsize=14, loc='left')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
# ΔR² at full data vs noise
for r in results:
    color = '#E74C3C' if r['dataset'] == 'NREL' else '#3498DB' if r['dataset'] == 'OPV' else '#2ECC71'
    plt.scatter(r['noise'], r['delta_r2_at_full'], c=color, s=150, edgecolors='black',
                linewidth=0.5, alpha=0.8, zorder=5)
    plt.annotate(r['dataset'], (r['noise'], r['delta_r2_at_full']),
                 textcoords="offset points", xytext=(5, 5), fontsize=9)

plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
plt.xlabel('Noise Level (1 − max R²)', fontsize=13)
plt.ylabel('ΔR² (GNN − XGBoost) at full data', fontsize=13)
plt.title('b. Performance gap vs Task noise', fontsize=14, loc='left')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('论文写作指导/论文草稿/figures/fig_noise_crossover.png', dpi=200, bbox_inches='tight')
plt.close()

print("\n\n图形已保存:")
print("  figures/fig_crossover_analysis.png")
print("  figures/fig_noise_crossover.png")

# Save JSON
with open('external_results/crossover_model.json', 'w') as f:
    json.dump(results, f, indent=2)
print("结果已保存: external_results/crossover_model.json")
