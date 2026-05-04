"""
学习曲线幂律拟合 + 相变阈值分析
拟合 R²(n) = a - b * n^(-c)，外推 XGBoost 与 GNN 的交叉点
"""
import numpy as np
import json

# 从论文表6的学习曲线数据
# XGBoost R² at different training sizes
n_values = np.array([68, 137, 344, 689, 1033, 1378], dtype=float)
xgb_r2 = np.array([0.511, 0.558, 0.677, 0.695, 0.725, 0.730])
gnn_r2 = np.array([0.145, 0.457, 0.433, 0.542, 0.503, 0.598])

# 幂律拟合: R²(n) = a - b * n^(-c)
from scipy.optimize import curve_fit

def power_law(n, a, b, c):
    return a - b * n ** (-c)

# 拟合 XGBoost
try:
    popt_xgb, _ = curve_fit(power_law, n_values, xgb_r2, p0=[0.8, 5, 0.5], maxfev=10000)
    print(f"XGBoost fit: R²(n) = {popt_xgb[0]:.4f} - {popt_xgb[1]:.4f} * n^(-{popt_xgb[2]:.4f})")
except Exception as e:
    print(f"XGBoost fit failed: {e}")
    popt_xgb = None

# 拟合 GNN
try:
    popt_gnn, _ = curve_fit(power_law, n_values, gnn_r2, p0=[0.8, 5, 0.5], maxfev=10000)
    print(f"GNN fit:      R²(n) = {popt_gnn[0]:.4f} - {popt_gnn[1]:.4f} * n^(-{popt_gnn[2]:.4f})")
except Exception as e:
    print(f"GNN fit failed: {e}")
    popt_gnn = None

# 外推至更大数据规模
n_extrap = np.logspace(np.log10(100), np.log10(100000), 100)

print("\n=== 外推预测 ===")
if popt_xgb is not None:
    xgb_extrap = power_law(n_extrap, *popt_xgb)
    print(f"XGBoost @ 2500 samples: R²={power_law(2500, *popt_xgb):.4f}")
    print(f"XGBoost @ 5000 samples: R²={power_law(5000, *popt_xgb):.4f}")
    print(f"XGBoost @ 10000 samples: R²={power_law(10000, *popt_xgb):.4f}")
    print(f"XGBoost asymptotic: R²={popt_xgb[0]:.4f}")

if popt_gnn is not None:
    gnn_extrap = power_law(n_extrap, *popt_gnn)
    print(f"GNN @ 2500 samples: R²={power_law(2500, *popt_gnn):.4f}")
    print(f"GNN @ 5000 samples: R²={power_law(5000, *popt_gnn):.4f}")
    print(f"GNN @ 10000 samples: R²={power_law(10000, *popt_gnn):.4f}")
    print(f"GNN asymptotic: R²={popt_gnn[0]:.4f}")

# 寻找交叉点 (XGBoost == GNN)
if popt_xgb is not None and popt_gnn is not None:
    def gap(n):
        return power_law(n, *popt_xgb) - power_law(n, *popt_gnn)

    # Binary search for cross-over
    lo, hi = 100, 1000000
    for _ in range(50):
        mid = (lo + hi) / 2
        if gap(mid) > 0:
            lo = mid
        else:
            hi = mid
    cross_point = (lo + hi) / 2

    print(f"\n=== 相变分析 ===")
    print(f"估计交叉点: n ≈ {cross_point:.0f} 个训练样本")
    print(f"在交叉点处 R² ≈ {power_law(cross_point, *popt_xgb):.4f}")

    # 判断可行性
    if cross_point > 100000:
        print(f"结论：在实验可及的数据范围内，XGBoost 将持续优于 GNN")
    elif cross_point > 10000:
        print(f"结论：需要约 {cross_point:.0f} 个样本 GNN 才可能追平 XGBoost，已接近但超出当前 OPV 数据集规模")
    else:
        print(f"结论：在约 {cross_point:.0f} 个样本时 GNN 可能追平 XGBoost，这是一个可行的实验目标")

# 保存结果
results = {
    'xgb_fit': {'a': float(popt_xgb[0]), 'b': float(popt_xgb[1]), 'c': float(popt_xgb[2])} if popt_xgb is not None else None,
    'gnn_fit': {'a': float(popt_gnn[0]), 'b': float(popt_gnn[1]), 'c': float(popt_gnn[2])} if popt_gnn is not None else None,
    'cross_point_samples': float(cross_point) if (popt_xgb is not None and popt_gnn is not None) else None,
    'xgb_asymptotic_r2': float(popt_xgb[0]) if popt_xgb is not None else None,
    'gnn_asymptotic_r2': float(popt_gnn[0]) if popt_gnn is not None else None,
}
with open('external_results/learning_curve_power_law.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nResults saved to external_results/learning_curve_power_law.json")

# 输出用于论文表格的数据
print("\n=== 用于论文的数据 ===")
print("| n | XGBoost (观测) | GNN (观测) | XGBoost (拟合) | GNN (拟合) |")
print("|---|---------------|-----------|---------------|-----------|")
for n, x, g in zip(n_values, xgb_r2, gnn_r2):
    xf = power_law(n, *popt_xgb) if popt_xgb is not None else 0
    gf = power_law(n, *popt_gnn) if popt_gnn is not None else 0
    print(f"| {int(n):4d} | {x:.4f} | {g:.4f} | {xf:.4f} | {gf:.4f} |")
