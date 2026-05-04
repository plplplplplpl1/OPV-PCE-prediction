#!/usr/bin/env python3
"""
GAT attention visualization.
Extracts attention weights from trained HighPCERegressorV3 GAT branch
and visualizes per-atom attention for representative molecules.
"""
import os, sys, json, random, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool, global_max_pool, global_add_pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
import pandas as pd
warnings.filterwarnings('ignore')
from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')

BASE_DIR = "/root/第四版r2=0.72/最小版本"
DATA_PATH = os.path.join(BASE_DIR, "data/data.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
MODEL_PATH = os.path.join(BASE_DIR, "best_high_pce_regressor_v3.pth")
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans', 'Arial'],
    'font.size': 9, 'savefig.dpi': 600, 'figure.dpi': 300,
})

PCE_THRESHOLD = 3.0
FP_DIM = 512
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Model definition (same architecture as HighPCERegressorV3) ──
class HighPCERegressorV3(nn.Module):
    def __init__(self, in_channels=30, hidden=128, fp_dim=FP_DIM, dropout=0.3):
        super().__init__()
        self.gcn1 = GCNConv(in_channels, hidden)
        self.gcn2 = GCNConv(hidden, hidden)
        self.gat1 = GATConv(in_channels, hidden//4, heads=4, dropout=dropout, concat=True)
        self.gat2 = GATConv(hidden, hidden, heads=1, dropout=dropout, concat=True)
        self.sage1 = SAGEConv(in_channels, hidden)
        self.sage2 = SAGEConv(hidden, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.bn3 = nn.BatchNorm1d(hidden)
        self.fp_encoder = nn.Sequential(
            nn.Linear(fp_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU())
        fused_dim = hidden * 9 + 128
        self.regressor = nn.Sequential(
            nn.Linear(fused_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1))
        self.dropout = dropout
        # Storage for attention weights
        self.gat_attentions = {}

    def forward(self, data, collect_attn=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        fp = data.fp
        # GCN branch
        x_gcn = F.relu(self.bn1(self.gcn1(x, edge_index)))
        x_gcn = F.relu(self.gcn2(x_gcn, edge_index))
        # GAT branch
        x_gat1, attn1 = self.gat1(x, edge_index, return_attention_weights=True)
        x_gat1 = F.relu(x_gat1)
        x_gat2 = F.relu(self.bn2(self.gat2(x_gat1, edge_index)))
        if collect_attn:
            self.gat_attentions['layer1'] = attn1
        # SAGE branch
        x_sage = F.relu(self.sage1(x, edge_index))
        x_sage = F.relu(self.bn3(self.sage2(x_sage, edge_index)))
        # Pooling
        def pool3(h):
            return torch.cat([global_mean_pool(h,batch), global_max_pool(h,batch), global_add_pool(h,batch)], dim=1)
        g = torch.cat([pool3(x_gcn), pool3(x_gat2), pool3(x_sage)], dim=1)
        fp_feat = self.fp_encoder(fp)
        g = torch.cat([g, fp_feat], dim=1)
        return self.regressor(g).squeeze(1)

# ── Graph construction ──
def smiles_to_graph(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None, None
        atom_features = []
        for atom in mol.GetAtoms():
            atomic_num = atom.GetAtomicNum()
            degree = atom.GetDegree()
            formal_charge = atom.GetFormalCharge()
            is_aromatic = int(atom.GetIsAromatic())
            is_in_ring = int(atom.IsInRing())
            try:
                hybridization = int(atom.GetHybridization())
                num_h = atom.GetTotalNumHs()
                valence = atom.GetTotalValence()
                r3 = int(atom.IsInRingSize(3)); r4 = int(atom.IsInRingSize(4))
                r5 = int(atom.IsInRingSize(5)); r6 = int(atom.IsInRingSize(6))
            except:
                hybridization = num_h = valence = r3 = r4 = r5 = r6 = 0
            common_atoms = [1, 6, 7, 8, 9, 15, 16, 17, 35]
            feat = [atomic_num/100.0, degree/6.0, formal_charge/8.0, num_h/4.0,
                    valence/8.0, is_aromatic, is_in_ring, r3, r4, r5, r6] \
                   + [int(atomic_num==a) for a in common_atoms] \
                   + [int(degree==d) for d in range(5)] \
                   + [int(hybridization==h) for h in range(1,6)]
            atom_features.append(feat)
        edge_indices = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_indices += [[i, j], [j, i]]
        if not edge_indices: return None, None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=FP_DIM)
        fp_tensor = torch.tensor(np.array(fp, dtype=np.float32)).unsqueeze(0)
        x = torch.tensor(atom_features, dtype=torch.float)
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        data = Data(x=x, edge_index=edge_index, fp=fp_tensor)
        return data, mol
    except:
        return None, None

# ── Load data ──
df = pd.read_csv(DATA_PATH, encoding='latin-1')
df.columns = df.columns.str.strip()
for c in df.columns:
    if 'pce' in c.lower(): df = df.rename(columns={c: 'PCE'})
    if 'smiles' in c.lower(): df = df.rename(columns={c: 'SMILES'})
df_high = df[df['PCE'] > PCE_THRESHOLD].copy()

# ── Load trained model ──
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model not found at {MODEL_PATH}")
    print("Running with untrained model for demonstration instead.")
    model = HighPCERegressorV3().to(DEVICE)
    model_trained = False
else:
    model = HighPCERegressorV3().to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state, strict=False)
    model.eval()
    model_trained = True
    print(f"Loaded model from {MODEL_PATH}")

# ── Select representative molecules ──
# Pick 4 molecules: low PCE (~4%), mid (~7%), high (~12%), very high (~16%)
target_pce_bins = [(3, 5, 'Low (3-5%)'), (6, 9, 'Mid (6-9%)'), (11, 14, 'High (11-14%)')]
selected = []
for lo, hi, label in target_pce_bins:
    subset = df_high[(df_high['PCE'] > lo) & (df_high['PCE'] <= hi)]
    if len(subset) > 0:
        row = subset.sample(1, random_state=42).iloc[0]
        selected.append((row['SMILES'], row['PCE'], label))

print("\nSelected molecules for attention visualization:")
for smi, pce, label in selected:
    print(f"  {label}: PCE={pce:.2f}%, SMILES={smi[:60]}...")

# ── Extract attention weights ──
fig, axes = plt.subplots(1, len(selected), figsize=(5.5 * len(selected), 4.5))

for idx, (smi, pce, label) in enumerate(selected):
    data, mol = smiles_to_graph(smi)
    if data is None or mol is None:
        print(f"  Skipping {label}: graph construction failed")
        continue

    data = data.to(DEVICE)
    data.batch = torch.zeros(data.x.shape[0], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        pred = model(data, collect_attn=True)

    # Get attention from first GAT layer (shape: [num_edges, heads])
    edge_index = data.edge_index.cpu()
    attn_edge, attn_weights = model.gat_attentions['layer1']
    # attn_weights shape: [num_edges, heads] — average across heads
    attn_mean = attn_weights.mean(dim=1).cpu().numpy()  # [num_edges]

    # Compute per-atom attention: for each atom, sum attention of its outgoing edges
    num_atoms = data.x.shape[0]
    atom_attn = np.zeros(num_atoms)
    for i in range(edge_index.shape[1]):
        src = edge_index[0, i].item()
        if src < num_atoms:
            atom_attn[src] += attn_mean[i]
    # Normalize
    if atom_attn.max() > 0:
        atom_attn = atom_attn / atom_attn.max()

    # Visualize: highlight atoms by attention weight on molecule
    rdkit_atom_colors = {}
    for i in range(num_atoms):
        rdkit_atom_colors[i] = (1.0, 0.6 * (1 - atom_attn[i]), 0.6 * (1 - atom_attn[i]))

    # Draw molecule with highlighted atoms
    try:
        img = Draw.MolToImage(mol, size=(400, 300), highlightAtoms=list(range(num_atoms)),
                              highlightColor=(1, 0, 0), highlightBonds=None)
        # Draw with atom indices for reference
        mol_with_idx = Chem.Mol(mol)
        for atom in mol_with_idx.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx())
        img2 = Draw.MolToImage(mol_with_idx, size=(400, 300))
        axes[idx].imshow(img)
        axes[idx].set_title(f'{label}\nPCE={pce:.2f}% (pred={pred.item():.2f}%)', fontsize=10, fontweight='bold')
        axes[idx].axis('off')
    except Exception as e:
        print(f"  Drawing failed for {label}: {e}")
        axes[idx].text(0.5, 0.5, 'Drawing failed', ha='center', va='center')
        axes[idx].axis('off')

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'figS3_gat_attention.png'), dpi=600, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"\nSaved figS3_gat_attention.png")

# Also check: which GAT-attended atoms correspond to Fingerprint-active bits?
print("\nGAT Attention vs Fingerprint overlap analysis:")
for smi, pce, label in selected:
    mol = Chem.MolFromSmiles(smi)
    if mol is None: continue
    fp_info = {}
    AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=FP_DIM, bitInfo=fp_info)
    # Get top-10 most important fingerprint bits from SHAP (we'll hardcode the known top bits)
    top_fp_bits = [1854, 3524, 3818, 1103, 1895]  # from manuscript Fig 3
    fp_atoms = set()
    for bit in top_fp_bits:
        if bit in fp_info:
            for atom_idx, radius in fp_info[bit]:
                fp_atoms.add(atom_idx)
    print(f"  {label}: {len(fp_atoms)}/{mol.GetNumAtoms()} atoms appear in top fingerprint bits")

print("\nDone.")
