"""
GraphGPS Baseline for OPV High-PCE Regression
Uses GPSConv (Graph Transformer + local MPNN) as a SOTA GNN baseline.
Compares against XGBoost and the custom HighPCERegressorV3 on the same data splits.
"""
import os, sys, argparse, random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

os.environ['RDKIT_SILENCE'] = '1'
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import AllChem

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import GPSConv, GCNConv, global_mean_pool, global_max_pool, global_add_pool

DATA_CSV = 'data/data_merged.csv' if os.path.exists('data/data_merged.csv') else 'data/data.csv'
PCE_THRESHOLD = 3.0
EPOCHS = 200
PATIENCE = 25
BATCH_SIZE = 64  # Larger batch for GPS (more memory efficient with attention)
LR = 0.0005
HIDDEN = 128
NUM_LAYERS = 4
NUM_HEADS = 4
DROPOUT = 0.2

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def smiles_to_graph(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
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
                r3 = int(atom.IsInRingSize(3))
                r4 = int(atom.IsInRingSize(4))
                r5 = int(atom.IsInRingSize(5))
                r6 = int(atom.IsInRingSize(6))
            except:
                hybridization = num_h = valence = r3 = r4 = r5 = r6 = 0
            common_atoms = [1, 6, 7, 8, 9, 15, 16, 17, 35]
            feat = [
                atomic_num / 100.0, degree / 6.0, formal_charge / 8.0,
                num_h / 4.0, valence / 8.0, is_aromatic, is_in_ring,
                r3, r4, r5, r6,
            ] + [int(atomic_num == a) for a in common_atoms] \
              + [int(degree == d) for d in range(5)] \
              + [int(hybridization == h) for h in range(1, 6)]
            atom_features.append(feat)
        edge_indices = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_indices += [[i, j], [j, i]]
        if not edge_indices: return None
        x = torch.tensor(atom_features, dtype=torch.float)
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        return Data(x=x, edge_index=edge_index)
    except:
        return None


class GraphGPSRegressor(nn.Module):
    """
    GraphGPS: Local MPNN (GCN) + Global Attention (Transformer) via GPSConv.
    Stacked GPSConv layers with pre-norm, residual connections, and global pooling.
    """
    def __init__(self, in_channels=30, hidden=HIDDEN, num_layers=NUM_LAYERS,
                 num_heads=NUM_HEADS, dropout=DROPOUT):
        super().__init__()
        self.node_encoder = nn.Linear(in_channels, hidden)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            local_mpnn = GCNConv(hidden, hidden, improved=True)
            self.convs.append(GPSConv(hidden, local_mpnn, heads=num_heads, dropout=dropout))
            self.norms.append(nn.BatchNorm1d(hidden))

        self.pool = global_mean_pool
        self.output = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.node_encoder(x)

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, batch)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.pool(x, batch)
        return self.output(x).squeeze(-1)


def load_data():
    df = pd.read_csv(DATA_CSV, encoding='latin-1')
    pce_col = df.columns[2]
    smiles_col = df.columns[-1]
    df[pce_col] = pd.to_numeric(df[pce_col], errors='coerce')
    df[smiles_col] = df[smiles_col].astype(str).str.strip()
    df = df.dropna(subset=[pce_col, smiles_col])
    df = df[df[smiles_col] != 'nan'].reset_index(drop=True)
    df_high = df[df[pce_col] > PCE_THRESHOLD].copy().reset_index(drop=True)
    graphs, pce_vals, failed = [], [], 0
    for _, row in df_high.iterrows():
        g = smiles_to_graph(row[smiles_col])
        if g is not None:
            g.y = torch.tensor([float(row[pce_col])], dtype=torch.float)
            graphs.append(g)
            pce_vals.append(float(row[pce_col]))
        else:
            failed += 1
    return graphs, np.array(pce_vals)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        preds.extend(pred.cpu().numpy().tolist())
        targets.extend(batch.y.view(-1).cpu().numpy().tolist())
    preds, targets = np.array(preds), np.array(targets)
    mae = mean_absolute_error(targets, preds)
    rmse = float(np.sqrt(np.mean((targets - preds) ** 2)))
    r2 = r2_score(targets, preds)
    return mae, rmse, r2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | seed={args.seed}")

    graphs, pce_values = load_data()
    print(f"Loaded {len(graphs)} high-PCE graphs")

    indices = list(range(len(graphs)))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=args.seed)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=args.seed)

    train_graphs = [graphs[i] for i in train_idx]
    val_graphs   = [graphs[i] for i in val_idx]
    test_graphs  = [graphs[i] for i in test_idx]
    print(f"Split: train={len(train_graphs)}, val={len(val_graphs)}, test={len(test_graphs)}")

    train_loader = GeoDataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = GeoDataLoader(val_graphs, batch_size=BATCH_SIZE)
    test_loader  = GeoDataLoader(test_graphs, batch_size=BATCH_SIZE)

    in_dim = graphs[0].x.shape[1]
    model = GraphGPSRegressor(in_channels=in_dim).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.HuberLoss(delta=1.0)

    best_val_mae = float('inf')
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = criterion(pred, batch.y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        scheduler.step()

        val_mae, val_rmse, val_r2 = evaluate(model, val_loader, device)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            save_path = f'external_results/gps_seed{args.seed}.pth'
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | loss={total_loss/len(train_loader.dataset):.4f} | "
                  f"val_MAE={val_mae:.4f} | val_R²={val_r2:.4f} | patience={patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

    # Test
    model.load_state_dict(torch.load(f'external_results/gps_seed{args.seed}.pth', weights_only=True))
    test_mae, test_rmse, test_r2 = evaluate(model, test_loader, device)
    print(f"\n>>> GraphGPS seed={args.seed}: R²={test_r2:.4f} MAE={test_mae:.4f} RMSE={test_rmse:.4f}")

    # Save result
    result = {'seed': args.seed, 'r2': test_r2, 'mae': test_mae, 'rmse': test_rmse}
    import json
    with open(f'external_results/gps_seed{args.seed}.json', 'w') as f:
        json.dump(result, f, indent=2)


if __name__ == '__main__':
    main()
