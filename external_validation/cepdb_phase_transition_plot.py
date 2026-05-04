"""
CEPDB Phase Transition Plot + Updated Power Law Fit
Generates Figure X: XGBoost vs GNN R² across training sizes 100-10000
with crossover point clearly marked. Also re-fits power law on full dataset.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ===== Data from original + large extension =====
data = [
    (100,  0.413, None),
    (250,  0.623, 0.483),
    (500,  0.684, None),
    (1000, 0.736, 0.696),
    (1500, 0.768, 0.750),
    (2000, 0.777, 0.806),
    (3000, 0.800, 0.841),
    (5000, 0.831, 0.883),
    (10000,0.858, 0.925),
]
# Std from the JSON files
xgb_std = {
    100: 0.076, 250: 0.028, 500: 0.023, 1000: 0.019, 1500: 0.018,
    2000: 0.005, 3000: 0.002, 5000: 0.003, 10000: 0.004
}
gnn_std = {
    250: 0.037, 1000: 0.052, 1500: 0.046,
    2000: 0.005, 3000: 0.007, 5000: 0.004, 10000: 0.004
}

sizes = np.array([d[0] for d in data])
xgb_r2 = np.array([d[1] for d in data])
gnn_r2 = np.array([d[2] if d[2] is not None else np.nan for d in data])

xgb_err = np.array([xgb_std[s] for s in sizes])
gnn_err = np.array([gnn_std[s] if s in gnn_std else 0 for s in sizes])

# ===== Power law fit: R²(n) = a - b * n^(-c) =====
def power_law(n, a, b, c):
    return a - b * np.power(n, -c)

# XGBoost fit (all 9 points)
valid_xgb = ~np.isnan(xgb_r2)
popt_xgb, pcov_xgb = curve_fit(power_law, sizes[valid_xgb], xgb_r2[valid_xgb],
                                 p0=[0.9, 0.5, 0.3], maxfev=10000)
a_xgb, b_xgb, c_xgb = popt_xgb
a_xgb_err = np.sqrt(np.diag(pcov_xgb))[0]
xgb_r2_asymp = a_xgb
xgb_r2_10k = power_law(10000, *popt_xgb)

print(f"XGBoost power law: R²(n) = {a_xgb:.4f} - {b_xgb:.4f} * n^(-{c_xgb:.4f})")
print(f"  Asymptotic R² = {xgb_r2_asymp:.4f} ± {a_xgb_err:.4f}")
print(f"  R²(10,000) = {xgb_r2_10k:.4f}")

# GNN fit (all 7 points with data)
valid_gnn = ~np.isnan(gnn_r2)
popt_gnn, pcov_gnn = curve_fit(power_law, sizes[valid_gnn], gnn_r2[valid_gnn],
                                 p0=[0.95, 0.5, 0.3], maxfev=10000)
a_gnn, b_gnn, c_gnn = popt_gnn
a_gnn_err = np.sqrt(np.diag(pcov_gnn))[0]
gnn_r2_asymp = a_gnn
gnn_r2_10k = power_law(10000, *popt_gnn)

print(f"\nGNN power law: R²(n) = {a_gnn:.4f} - {b_gnn:.4f} * n^(-{c_gnn:.4f})")
print(f"  Asymptotic R² = {gnn_r2_asymp:.4f} ± {a_gnn_err:.4f}")
print(f"  R²(10,000) = {gnn_r2_10k:.4f}")

# Find crossover point (where XGBoost and GNN curves intersect)
def diff(n):
    return power_law(n, *popt_xgb) - power_law(n, *popt_gnn)

# Binary search for crossover
lo, hi = 100, 50000
for _ in range(50):
    mid = (lo + hi) / 2
    if diff(mid) > 0:
        lo = mid
    else:
        hi = mid
crossover_n = (lo + hi) / 2
print(f"\nCrossover point (fitted): n ≈ {crossover_n:.0f}")

# Extrapolate to larger range
n_extrap = np.logspace(2, 5, 200)
xgb_extrap = power_law(n_extrap, *popt_xgb)
gnn_extrap = power_law(n_extrap, *popt_gnn)

# Save results
results = {
    'xgb': {'a': float(a_xgb), 'b': float(b_xgb), 'c': float(c_xgb),
            'a_err': float(a_xgb_err), 'r2_asymptotic': float(xgb_r2_asymp)},
    'gnn': {'a': float(a_gnn), 'b': float(b_gnn), 'c': float(c_gnn),
            'a_err': float(a_gnn_err), 'r2_asymptotic': float(gnn_r2_asymp)},
    'crossover_n': float(crossover_n),
}
os.makedirs('external_results', exist_ok=True)
with open('external_results/cepdb_power_law.json', 'w') as f:
    json.dump(results, f, indent=2)

# ===== Figure =====
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left panel: CEPDB trajectory
ax1.errorbar(sizes, xgb_r2, yerr=xgb_err, fmt='o-', color='#E74C3C',
             capsize=4, capthick=1.5, markersize=8, label='XGBoost', linewidth=2)
ax1.errorbar(sizes, gnn_r2, yerr=gnn_err, fmt='s-', color='#3498DB',
             capsize=4, capthick=1.5, markersize=8, label='GNN (HighPCERegressorV3)', linewidth=2)

# Extrapolation curves
ax1.plot(n_extrap, xgb_extrap, '--', color='#E74C3C', alpha=0.4, linewidth=1,
         label=f'XGBoost extrap. (R²asymp={xgb_r2_asymp:.3f})')
ax1.plot(n_extrap, gnn_extrap, '--', color='#3498DB', alpha=0.4, linewidth=1,
         label=f'GNN extrap. (R²asymp={gnn_r2_asymp:.3f})')

# Crossover marker
ax1.axvline(x=crossover_n, color='green', linestyle=':', linewidth=1.5, alpha=0.8)
ax1.annotate(f'Crossover\nn≈{crossover_n:.0f}',
             xy=(crossover_n, power_law(crossover_n, *popt_xgb)),
             xytext=(crossover_n*1.5, power_law(crossover_n, *popt_xgb)-0.08),
             fontsize=10, color='green', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='green', lw=1.5))

# Phase transition regions
ax1.axvspan(100, crossover_n, alpha=0.06, color='#E74C3C', label='XGBoost-dominant regime')
ax1.axvspan(crossover_n, 50000, alpha=0.06, color='#3498DB', label='GNN-dominant regime')

ax1.set_xscale('log')
ax1.set_xlabel('Training Samples (n)', fontsize=12)
ax1.set_ylabel('R²', fontsize=12)
ax1.set_title('CEPDB: XGBoost vs GNN Performance', fontsize=13, fontweight='bold')
ax1.set_ylim(0.3, 1.0)
ax1.legend(fontsize=9, loc='lower right')
ax1.grid(True, alpha=0.3)
ax1.tick_params(labelsize=10)

# Right panel: ΔR² trajectory
deltas = []
delta_errs = []
delta_sizes = []
for i, n in enumerate(sizes):
    xgb_v = xgb_r2[i]
    gnn_v = gnn_r2[i]
    if not np.isnan(xgb_v) and not np.isnan(gnn_v):
        deltas.append(xgb_v - gnn_v)
        delta_errs.append(np.sqrt(xgb_err[i]**2 + gnn_err[i]**2))
        delta_sizes.append(n)

deltas = np.array(deltas)
delta_errs = np.array(delta_errs)
delta_sizes = np.array(delta_sizes)

colors = ['#E74C3C' if d > 0 else '#3498DB' for d in deltas]
ax2.bar(range(len(deltas)), deltas, yerr=delta_errs, color=colors,
        capsize=4, edgecolor='white', linewidth=1.2, width=0.6)
ax2.axhline(y=0, color='black', linewidth=1)
ax2.set_xticks(range(len(deltas)))
ax2.set_xticklabels([f'n={int(s)}' for s in delta_sizes], fontsize=9)
ax2.set_ylabel('ΔR² (XGBoost − GNN)', fontsize=12)
ax2.set_title('Performance Gap Trajectory', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, d in enumerate(deltas):
    label = f'{d:+.3f}'
    y_pos = d + (0.015 if d > 0 else -0.035)
    ax2.text(i, y_pos, label, ha='center', fontsize=8, fontweight='bold',
             color='#E74C3C' if d > 0 else '#3498DB')

ax2.text(4.5, 0.05, 'XGBoost\nsuperior', ha='center', fontsize=9, color='#E74C3C', fontweight='bold')
ax2.text(4.5, -0.05, 'GNN\nsuperior', ha='center', fontsize=9, color='#3498DB', fontweight='bold')

plt.tight_layout()
plt.savefig('/root/第四版r2=0.72/最小版本/论文写作指导/论文草稿/figures/cepdb_phase_transition.png', dpi=200, bbox_inches='tight')
plt.savefig('/root/第四版r2=0.72/最小版本/论文写作指导/论文草稿/figures/cepdb_phase_transition.pdf', bbox_inches='tight')
print(f"\nFigure saved.")

# Summary
print("\n" + "="*60)
print("CEPDB Power Law Re-fit Summary")
print("="*60)
print(f"  XGBoost: R²(n) = {a_xgb:.4f} - {b_xgb:.4f}·n^(-{c_xgb:.4f})")
print(f"    Asymptotic: R²={xgb_r2_asymp:.3f}")
print(f"    R²(10,000) = {xgb_r2_10k:.3f}")
print(f"  GNN: R²(n) = {a_gnn:.4f} - {b_gnn:.4f}·n^(-{c_gnn:.4f})")
print(f"    Asymptotic: R²={gnn_r2_asymp:.3f}")
print(f"    R²(10,000) = {gnn_r2_10k:.3f}")
print(f"  Crossover: n ≈ {crossover_n:.0f}")
