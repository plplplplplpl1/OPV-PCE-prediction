"""
Shared style definitions for all paper figures.
Centralizes colors, fonts, sizes for consistency.
"""
import matplotlib.pyplot as plt

# ---- Color palette ----
XGB_COLOR = '#C23B22'       # XGBoost red
XGB_LIGHT = '#E8836D'
GNN_COLOR = '#2E5A88'       # GNN blue
GNN_LIGHT = '#6B93C4'
GPS_COLOR = '#27AE60'       # GraphGPS green
GPS_LIGHT = '#6FCF97'

# For comparison / delta charts
POS_COLOR = '#C23B22'       # XGBoost superior
NEG_COLOR = '#2E5A88'       # GNN superior
REF_COLOR = '#7F8C8D'       # Neutral / reference
ACCENT_COLOR = '#E67E22'    # Crossover / key point

# ---- Font settings ----
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 200,
})

# ---- Figure sizes ----
FIG_WIDE = (8, 5)       # Standard wide figure
FIG_SQUARE = (6, 5)     # Square figure
FIG_DOUBLE = (14, 5.5)  # Two-panel figure
FIG_TRIPLE = (18, 5.5)  # Three-panel figure
FIG_TALL = (6, 6)       # Tall figure

# ---- Grid style ----
GRID_KWARGS = dict(alpha=0.25, linewidth=0.5)
