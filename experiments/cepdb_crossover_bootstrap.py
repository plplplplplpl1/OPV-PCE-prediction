"""
P1-2: 相变交叉点的 Bootstrap 置信区间
对 CEPDB 数据做有放回重采样 → 重拟合幂律 → 求解放 → 统计交叉点分布
"""
import json, os, sys
import numpy as np
from scipy.optimize import curve_fit

os.makedirs('external_results', exist_ok=True)

# ========== 1. 加载现有 CEPDB 数据 ==========
with open('external_results/cepdb_xgb.json') as f:
    cepdb_xgb = json.load(f)
with open('external_results/cepdb_gnn.json') as f:
    cepdb_gnn = json.load(f)
with open('external_results/cepdb_large_xgb.json') as f:
    cepdb_large_xgb = json.load(f)
with open('external_results/cepdb_large_gnn.json') as f:
    cepdb_large_gnn = json.load(f)
with open('external_results/cepdb_power_law.json') as f:
    power_law_orig = json.load(f)

# ========== 2. 尝试加载P1-1新生成的数据（如果已存在） ==========
try:
    with open('external_results/cepdb_gnn_n100_500.json') as f:
        cepdb_gnn_new = json.load(f)
    print(f"Loaded new GNN data: {list(cepdb_gnn_new.keys())}")
except:
    cepdb_gnn_new = {}
    print("No new GNN data yet, using only existing data")

# ========== 3. 合并所有数据 ==========
def merge_data(*dicts):
    result = {}
    for d in dicts:
        for k, v in d.items():
            result[int(k)] = {'r2_mean': v['r2_mean'], 'r2_std': v['r2_std']}
    return result

xgb_all = merge_data(cepdb_xgb, cepdb_large_xgb)
gnn_all = merge_data(cepdb_gnn, cepdb_large_gnn, cepdb_gnn_new)

print(f"\nXGBoost data points: n={sorted(xgb_all.keys())}")
print(f"GNN data points: n={sorted(gnn_all.keys())}")

# ========== 4. 找共同 n ==========
common_n = sorted(set(xgb_all.keys()) & set(gnn_all.keys()))
print(f"Common n for crossover calculation: {common_n}")

if len(common_n) < 3:
    print("ERROR: Need at least 3 common data points for fitting")
    sys.exit(1)

n_arr = np.array(common_n, dtype=float)
xgb_means = np.array([xgb_all[n]['r2_mean'] for n in common_n])
xgb_stds  = np.array([max(xgb_all[n]['r2_std'], 0.001) for n in common_n])
gnn_means = np.array([gnn_all[n]['r2_mean'] for n in common_n])
gnn_stds  = np.array([max(gnn_all[n]['r2_std'], 0.001) for n in common_n])

# ========== 5. 幂律函数 ==========
def power_law(n, a, b, c):
    return a - b * np.power(n, -c)

def find_crossover(xgb_params, gnn_params, search_range=(50, 50000)):
    """Find n where R²_xgb(n) = R²_gnn(n) by bisection"""
    def diff(n):
        return power_law(n, *xgb_params) - power_law(n, *gnn_params)
    lo, hi = search_range
    if diff(lo) * diff(hi) > 0:
        # No crossover in range, try to extrapolate
        return None
    for _ in range(60):
        mid = (lo + hi) / 2
        if diff(mid) * diff(lo) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2

# ========== 6. 原始拟合（验证） ==========
try:
    popt_xgb, _ = curve_fit(power_law, n_arr, xgb_means, p0=[0.9, 0.5, 0.3], maxfev=10000)
    popt_gnn, _ = curve_fit(power_law, n_arr, gnn_means, p0=[0.95, 0.5, 0.3], maxfev=10000)
    orig_crossover = find_crossover(popt_xgb, popt_gnn)
    print(f"\nOriginal power-law fit:")
    print(f"  XGBoost: R²(n) = {popt_xgb[0]:.4f} - {popt_xgb[1]:.4f}·n^(-{popt_xgb[2]:.4f})")
    print(f"  GNN:    R²(n) = {popt_gnn[0]:.4f} - {popt_gnn[1]:.4f}·n^(-{popt_gnn[2]:.4f})")
    print(f"  Crossover n ≈ {orig_crossover:.1f}")
except Exception as e:
    print(f"Warning: original fit failed: {e}")
    orig_crossover = None

# ========== 7. Bootstrap ==========
N_BOOTSTRAP = 2000
crossover_samples = []

np.random.seed(42)
print(f"\nRunning {N_BOOTSTRAP} bootstrap iterations...")

for i in range(N_BOOTSTRAP):
    if (i + 1) % 500 == 0:
        print(f"  {i+1}/{N_BOOTSTRAP}")

    try:
        # Resample R² values from normal distributions
        xgb_sample = np.random.normal(xgb_means, xgb_stds)
        gnn_sample = np.random.normal(gnn_means, gnn_stds)

        # Fit power laws
        popt_xgb_b, _ = curve_fit(power_law, n_arr, xgb_sample,
                                   p0=[0.9, 0.5, 0.3], maxfev=10000)
        popt_gnn_b, _ = curve_fit(power_law, n_arr, gnn_sample,
                                   p0=[0.95, 0.5, 0.3], maxfev=10000)

        # Find crossover
        cross = find_crossover(popt_xgb_b, popt_gnn_b)
        if cross is not None and 100 < cross < 50000:
            crossover_samples.append(cross)
    except:
        continue

crossover_samples = np.array(crossover_samples)
print(f"\nValid bootstrap samples: {len(crossover_samples)}/{N_BOOTSTRAP}")

if len(crossover_samples) > 50:
    mean_cross = np.mean(crossover_samples)
    median_cross = np.median(crossover_samples)
    ci_lower = np.percentile(crossover_samples, 2.5)
    ci_upper = np.percentile(crossover_samples, 97.5)
    std_cross = np.std(crossover_samples)

    print(f"\n{'='*50}")
    print(f"相变点 Bootstrap 分析结果")
    print(f"{'='*50}")
    print(f"  均值交叉点: {mean_cross:.0f}")
    print(f"  中位交叉点: {median_cross:.0f}")
    print(f"  标准差:     {std_cross:.0f}")
    print(f"  95% CI:     [{ci_lower:.0f}, {ci_upper:.0f}]")
    print(f"  数据点数:   {len(common_n)} ({common_n})")
    print(f"{'='*50}")

    # Save results
    results = {
        'n_bootstrap': N_BOOTSTRAP,
        'n_valid': int(len(crossover_samples)),
        'crossover_mean': float(mean_cross),
        'crossover_median': float(median_cross),
        'crossover_std': float(std_cross),
        'crossover_ci_95': [float(ci_lower), float(ci_upper)],
        'common_n': common_n,
        'xgb_params_original': popt_xgb.tolist() if orig_crossover else None,
        'gnn_params_original': popt_gnn.tolist() if orig_crossover else None,
        'xgb_data': {str(k): {'mean': float(xgb_all[k]['r2_mean']), 'std': float(xgb_all[k]['r2_std'])} for k in common_n},
        'gnn_data': {str(k): {'mean': float(gnn_all[k]['r2_mean']), 'std': float(gnn_all[k]['r2_std'])} for k in common_n},
    }
    with open('external_results/cepdb_crossover_bootstrap.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to external_results/cepdb_crossover_bootstrap.json")

    # Also save histogram data for visualization
    hist_bins = 30
    hist, bin_edges = np.histogram(crossover_samples, bins=hist_bins)
    hist_data = {
        'hist': hist.tolist(),
        'bin_edges': bin_edges.tolist(),
        'samples': crossover_samples[:100].tolist(),
    }
    with open('external_results/cepdb_crossover_histogram.json', 'w') as f:
        json.dump(hist_data, f, indent=2)
    print("Histogram data saved.")
else:
    print(f"ERROR: Too few valid samples ({len(crossover_samples)})")
