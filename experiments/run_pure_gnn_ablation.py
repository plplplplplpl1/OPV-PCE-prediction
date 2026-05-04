#!/usr/bin/env python3
"""
Pure Graph GNN Ablation — No Morgan Fingerprint Fusion
Compares: pure GNN (graph only) vs GNN+FP vs XGBoost
Runs 4 seeds on high-PCE subset
"""

import os, sys, argparse, random, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool, global_max_pool, global_add_pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')

DATA_CSV = 'data/data_merged.csv' if os.path.exists('data/data_merged.csv') else 'data/data.csv'
PCE_THRESHOLD = 3.0
EPOCHS = 200
PATIENCE = 20
BATCH_SIZE = 32
LR = 0.001

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def smiles_to_graph(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        atom_features = []
        for atom in mol.GetAtoms():
            an = atom.GetAtomicNum()
            feat = [an/100.0, atom.GetDegree()/6.0, atom.GetFormalCharge()/8.0,
                    int(atom.GetIsAromatic()), int(atom.IsInRing()),
                    atom.GetHybridization()/10.0, atom.GetTotalNumHs()/6.0,
                    atom.GetTotalValence()/10.0]
            atom_features.append(feat)
        edge_indices = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_indices.extend([(i, j), (j, i)])
        x = torch.tensor(atom_features, dtype=torch.float)
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        return Data(x=x, edge_index=edge_index)
    except: return None

class PureGNN(nn.Module):
    """GNN without fingerprint — pure graph message passing"""
    def __init__(self, in_channels=30, hidden=128, dropout=0.3):
        super().__init__()
        self.gcn1 = GCNConv(in_channels, hidden)
        self.gcn2 = GCNConv(hidden, hidden)
        self.gat1 = GATConv(in_channels, hidden//4, heads=4, dropout=dropout)
        self.gat2 = GATConv(hidden, hidden, heads=1, dropout=dropout)
        self.sage1 = SAGEConv(in_channels, hidden)
        self.sage2 = SAGEConv(hidden, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.bn3 = nn.BatchNorm1d(hidden)
        fused_dim = hidden * 9  # 3 branches × 3 pooling — NO fingerprint
        self.regressor = nn.Sequential(
            nn.Linear(fused_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1))
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x_gcn = F.relu(self.bn1(self.gcn1(x, edge_index)))
        x_gcn = F.relu(self.gcn2(x_gcn, edge_index))
        x_gat = F.relu(self.gat1(x, edge_index))
        x_gat = F.relu(self.bn2(self.gat2(x_gat, edge_index)))
        x_sage = F.relu(self.sage1(x, edge_index))
        x_sage = F.relu(self.bn3(self.sage2(x_sage, edge_index)))
        def pool3(h):
            return torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch), global_add_pool(h, batch)], dim=1)
        g = torch.cat([pool3(x_gcn), pool3(x_gat), pool3(x_sage)], dim=1)
        return self.regressor(g).squeeze(1)

def load_data():
    df = pd.read_csv(DATA_CSV, encoding='latin-1')
    pce_col = df.columns[2]; smiles_col = df.columns[-1]
    df[pce_col] = pd.to_numeric(df[pce_col], errors='coerce')
    df[smiles_col] = df[smiles_col].astype(str).str.strip()
    df = df.dropna(subset=[pce_col, smiles_col])
    df = df[df[smiles_col] != 'nan'].reset_index(drop=True)
    df_high = df[df[pce_col] > PCE_THRESHOLD].copy().reset_index(drop=True)
    graphs, pce_values = [], []
    failed = 0
    for _, row in df_high.iterrows():
        g = smiles_to_graph(row[smiles_col])
        if g is not None:
            g.y = torch.tensor([float(row[pce_col])], dtype=torch.float)
            graphs.append(g)
            pce_values.append(float(row[pce_col]))
        else:
            failed += 1
    print(f"  Graphs: {len(graphs)}, failed: {failed}")
    return graphs, np.array(pce_values)

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        loss = criterion(pred, batch.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        preds.extend(pred.cpu().tolist())
        targets.extend(batch.y.view(-1).cpu().tolist())
    preds = np.array(preds); targets = np.array(targets)
    return mean_absolute_error(targets, preds), float(np.sqrt(np.mean((targets-preds)**2))), r2_score(targets, preds)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="results/pure_gnn")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"pure_gnn_seed{args.seed}.pth")

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*50}")
    print(f"Pure GNN (no fingerprint) — seed={args.seed}, device={device}")

    graphs, pce_values = load_data()
    indices = list(range(len(graphs)))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=args.seed)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=args.seed)

    train_g = [graphs[i] for i in train_idx]
    val_g   = [graphs[i] for i in val_idx]
    test_g  = [graphs[i] for i in test_idx]
    print(f"  Train: {len(train_g)}, Val: {len(val_g)}, Test: {len(test_g)}")

    train_loader = GeoDataLoader(train_g, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = GeoDataLoader(val_g,   batch_size=BATCH_SIZE)
    test_loader  = GeoDataLoader(test_g,  batch_size=BATCH_SIZE)

    in_dim = graphs[0].x.shape[1]
    model = PureGNN(in_channels=in_dim).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.HuberLoss(delta=1.0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    best_val_mae = float('inf')
    patience_counter = 0
    start = time.time()

    for epoch in range(1, EPOCHS+1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_mae, val_rmse, val_r2 = evaluate(model, val_loader, device)
        scheduler.step(val_mae)
        if val_mae < best_val_mae:
            best_val_mae = val_mae; patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | loss={train_loss:.4f} | val_MAE={val_mae:.4f} | val_R2={val_r2:.4f}")
        if patience_counter >= PATIENCE:
            print(f"  Early stop at epoch {epoch}")
            break

    model.load_state_dict(torch.load(save_path, weights_only=True))
    test_mae, test_rmse, test_r2 = evaluate(model, test_loader, device)
    elapsed = time.time() - start
    print(f"\n  Results (seed={args.seed}): MAE={test_mae:.4f}, RMSE={test_rmse:.4f}, R²={test_r2:.4f} ({elapsed:.0f}s)")

    result = {"seed": args.seed, "MAE": test_mae, "RMSE": test_rmse, "R2": test_r2,
              "elapsed_s": elapsed, "params": sum(p.numel() for p in model.parameters())}
    with open(os.path.join(args.save_dir, f"results_seed{args.seed}.json"), 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  Saved to {args.save_dir}/results_seed{args.seed}.json")

if __name__ == '__main__':
    main()
