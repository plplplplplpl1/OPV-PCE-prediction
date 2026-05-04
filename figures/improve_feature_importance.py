#!/usr/bin/env python3
"""
Feature importance analysis: map Morgan fingerprint bits to chemical substructures.
Generates an improved fig3 with chemically meaningful labels.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
import xgboost as xgb
import os, warnings, re
from collections import Counter
warnings.filterwarnings('ignore')

BASE_DIR    = "/root/第四版r2=0.72/最小版本"
DATA_PATH   = os.path.join(BASE_DIR, "data/data.csv")
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8, 'axes.linewidth': 0.6,
    'figure.dpi': 300, 'savefig.dpi': 600,
})

# Load data
df = pd.read_csv(DATA_PATH, encoding='latin-1')
df.columns = df.columns.str.strip()
for c in df.columns:
    if 'pce' in c.lower(): df = df.rename(columns={c: 'PCE'})
    if 'smiles' in c.lower(): df = df.rename(columns={c: 'SMILES'})
df_high = df[df['PCE'] > 3].copy()

def morgan_fp(smi, nbits=4096, radius=2):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None, None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    # Get bit info for substructure mapping
    info = {}
    AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits, bitInfo=info)
    return np.array(fp, dtype=np.float32), info

# Build dataset
X, y, all_info = [], [], []
for _, r in df_high.iterrows():
    f, info = morgan_fp(r['SMILES'])
    if f is not None:
        X.append(f)
        y.append(r['PCE'])
        all_info.append(info)
X, y = np.array(X), np.array(y)

# Train XGBoost
model = xgb.XGBRegressor(n_estimators=2000, learning_rate=0.03, max_depth=6,
                         min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                         reg_lambda=5, gamma=0.1, random_state=42, verbosity=0,
                         tree_method='hist')
model.fit(X, y, verbose=False)

# Get top-20 feature importance
imp = model.feature_importances_
top20_idx = np.argsort(imp)[-20:][::-1]
top20_vals = imp[top20_idx]

print("Top 20 Morgan fingerprint bits:")
print(f"{'Bit #':<8} {'Importance':<12} {'Frequency':<10} {'Top SMARTS/Substructure'}")
print("-" * 80)

# For each important bit, find what substructures activate it
smiles_list = df_high['SMILES'].tolist()
bit_substructs = {}
bit_frequencies = {}

for bit in top20_idx:
    # Collect SMARTS for this bit across all molecules
    smarts_list = []
    freq = 0
    for i, info in enumerate(all_info):
        if bit in info:
            freq += 1
            # Get atoms that activate this bit
            mol = Chem.MolFromSmiles(smiles_list[i])
            if mol is None: continue
            for atom_idx, radius in info[bit]:
                if atom_idx >= mol.GetNumAtoms():
                    continue
                try:
                    env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
                    amap = {}
                    submol = Chem.PathToSubmol(mol, env, atomMap=amap)
                    if submol.GetNumAtoms() > 0:
                        smarts = Chem.MolToSmarts(submol)
                        smarts_list.append(smarts)
                except:
                    continue

    bit_frequencies[bit] = freq

    # Find most common SMARTS pattern
    if smarts_list:
        common_smarts = Counter(smarts_list).most_common(3)
        top_smarts = common_smarts[0][0]
        # Try to get a readable name
        try:
            submol = Chem.MolFromSmarts(top_smarts)
            if submol and submol.GetNumAtoms() <= 10:
                smarts_display = top_smarts
            else:
                smarts_display = top_smarts[:40] + '...' if len(top_smarts) > 40 else top_smarts
        except:
            smarts_display = '(complex)'
    else:
        smarts_display = '(singleton)'

    bit_substructs[bit] = smarts_display
    print(f"Bit {bit:<5} {imp[bit]:.6f}     {freq:<9} {smarts_display}")

# Create human-readable labels
def make_label(bit):
    """Create a short chemical description from SMARTS."""
    smarts = bit_substructs[bit]
    freq = bit_frequencies[bit]
    # Map common SMARTS patterns to readable names
    pattern_map = {
        '[#6]/[#6]=[#6](\[#6](=[#8])-[#6])-[#6](-[#6])=[#6]': 'Carbonyl conj. system',
        '[#6]-[#6](=[#8])-[#6]': 'Methyl ketone',
        '[#6]=[#6](-[#6])-[#6]': 'Alkene (C=C)',
        '[#6]:[#6](:[#6](:[#16]:[#6]):[#6](:[#6]):[#6]):[#7]': 'Thiophene-pyridine',
        '[#6]:[#6](-[#6]):[#16]:[#6](:[#6]):[#6]': 'Thiophene ring',
        '[#6]=[#6](-[#6]#[#7])-[#6]': 'Acrylonitrile (C=C-CN)',
        '[#6]-[#6]-[#6]': 'Propane chain',
        '[#6]-[#6]:[#6]:[#6]': 'Alkyl-benzene',
        '[#6]/[#6]=[#6](/[#16]:[#6]):[#6](:[#7])=[#8]': 'Thio-carbonylamide',
        '[#6]-[#6]-[#6]-[#6]-[#6]': 'Pentane chain',
        '[#6]-[#6](:[#6]:[#6]:[#6]):[#16]': 'Thioether-aryl',
        '[#6]-[#6]=[#6]': 'Allyl group',
        '[#6]:[#6](:[#6]):[#6](:[#6](:[#6]):[#6]):[#6](:[#6]):[#6]': 'Naphthalene core',
        '[#6]-[#6](-[#6])=[#6]': 'Methyl-alkene',
        '[#6]-[#6](-[#6])(-[#6]': 'Quaternary C',
        'c1ccccc1': 'Benzene ring',
        'c1ccc2ccccc2c1': 'Naphthalene',
        'c1ccsc1': 'Thiophene',
        'O': 'Carbonyl (C=O)',
        '[#7]': 'N atom',
        '[#8]': 'O atom',
        '[#16]': 'S atom (thiophene)',
        '[#9]': 'F atom',
        '[#17]': 'Cl atom',
        '[#35]': 'Br atom',
        'C=O': 'Carbonyl',
        'CO': 'Methoxy',
        '[#6]-[#8]': 'C-O bond',
        '[#6]=O': 'C=O carbonyl',
        'F': 'Fluorine',
        'Cl': 'Chlorine',
        '[#6]-[#6](-[#6])(-[#6])': 'quat. carbon',
    }

    # Try to find a matching pattern
    for pattern, name in pattern_map.items():
        if pattern in smarts:
            return f'{name}'

    # Fall back to abbreviated SMARTS
    if smarts.startswith('('):
        smarts_abbr = smarts[:20]
    else:
        smarts_abbr = smarts[:25]
    return f'{smarts_abbr}'

labels = [f'Bit {i}\n({make_label(i)})' for i in top20_idx]

# Plot improved figure
fig, ax = plt.subplots(figsize=(6.5, 5.5))
colors = plt.cm.Blues(np.linspace(0.4, 0.85, len(top20_idx)))[::-1]

bars = ax.barh(range(len(top20_idx)), top20_vals, color=colors, edgecolor='#2C3E50',
               linewidth=0.4, height=0.7)
ax.set_yticks(range(len(top20_idx)))
ax.set_yticklabels(labels, fontsize=6.5)
ax.invert_yaxis()
ax.set_xlabel('Feature Importance (Gain)', fontsize=9)
ax.set_title('Top 20 Morgan Fingerprint Bits — Chemical Substructure Mapping',
             fontweight='bold', fontsize=10)

# Add value labels
for i, (v, bit) in enumerate(zip(top20_vals, top20_idx)):
    ax.text(v + 0.0003, i, f'{v:.4f}', va='center', fontsize=6.5, color='#555')

ax.set_xlim(0, max(top20_vals) * 1.25)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
path = os.path.join(OUTPUT_DIR, 'fig3_feature_importance_improved.png')
fig.savefig(path, dpi=600, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"\nSaved: {path}")
print(f"Size: {os.path.getsize(path)//1024}KB")

# Also generate a table of chemical interpretations
print("\n\nChemical interpretation of top-20 bits:")
print("=" * 70)
for i, bit in enumerate(top20_idx):
    freq_pct = bit_frequencies[bit] / len(smiles_list) * 100
    print(f"{i+1:2d}. Bit {bit:4d} (imp={imp[bit]:.4f}, freq={freq_pct:.1f}%) → {bit_substructs[bit]}")
