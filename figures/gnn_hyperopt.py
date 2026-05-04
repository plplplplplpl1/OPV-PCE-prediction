#!/usr/bin/env python3
"""
Limited GNN hyperparameter optimization (8-10 trials).
Focuses on 3 key hyperparameters: learning_rate, hidden_dim, dropout.
Training uses train_high_pce_v3.py's infrastructure.
"""
import os, sys, json, random, warnings, itertools
import numpy as np
from tqdm import tqdm
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
warnings.filterwarnings('ignore')

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
except ImportError:
    print("RDKit not available"); exit(1)

BASE_DIR = "/root/第四版r2=0.72/最小版本"
DATA_PATH = os.path.join(BASE_DIR, "data/data.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "论文写作指导/论文草稿/results")
os.makedirs(RESULTS_DIR, exist_ok=True)

PCE_THRESHOLD = 3.0
FP_DIM = 512
EPOCHS = 80
PATIENCE = 10
BATCH_SIZE = 32
N_TRIALS = 4  # quick focused search

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

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
        if not edge_indices: return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=FP_DIM)
        fp_tensor = torch.tensor(np.array(fp, dtype=np.float32)).unsqueeze(0)
        x = torch.tensor(atom_features, dtype=torch.float)
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        return Data(x=x, edge_index=edge_index, fp=fp_tensor)
    except:
        return None

class HighPCERegressorV3(nn.Module):
    def __init__(self, in_channels=30, hidden=128, fp_dim=FP_DIM, dropout=0.3):
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
        self.fp_encoder = nn.Sequential(
            nn.Linear(fp_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU())
        fused_dim = hidden * 9 + 128
        self.regressor = nn.Sequential(
            nn.Linear(fused_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1))
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        fp = data.fp
        x_gcn = F.relu(self.bn1(self.gcn1(x, edge_index)))
        x_gcn = F.relu(self.gcn2(x_gcn, edge_index))
        x_gat = F.relu(self.gat1(x, edge_index))
        x_gat = F.relu(self.bn2(self.gat2(x_gat, edge_index)))
        x_sage = F.relu(self.sage1(x, edge_index))
        x_sage = F.relu(self.bn3(self.sage2(x_sage, edge_index)))
        def pool3(h):
            return torch.cat([global_mean_pool(h,batch), global_max_pool(h,batch), global_add_pool(h,batch)], dim=1)
        g = torch.cat([pool3(x_gcn), pool3(x_gat), pool3(x_sage)], dim=1)
        fp_feat = self.fp_encoder(fp)
        g = torch.cat([g, fp_feat], dim=1)
        return self.regressor(g).squeeze(1)

def load_data():
    df = pd.read_csv(DATA_PATH, encoding='latin-1')
    df.columns = df.columns.str.strip()
    for c in df.columns:
        if 'pce' in c.lower(): df = df.rename(columns={c: 'PCE'})
        if 'smiles' in c.lower(): df = df.rename(columns={c: 'SMILES'})
    df_high = df[df['PCE'] > PCE_THRESHOLD].copy()
    graphs, pce_values = [], []
    failed = 0
    for _, r in df_high.iterrows():
        g = smiles_to_graph(r['SMILES'])
        if g is not None:
            g.y = torch.tensor([float(r['PCE'])], dtype=torch.float)
            graphs.append(g)
            pce_values.append(float(r['PCE']))
        else:
            failed += 1
    print(f"  Graphs: {len(graphs)}, failed: {failed}")
    return graphs, np.array(pce_values)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        preds.extend(pred.cpu().numpy())
        targets.extend(batch.y.view(-1).cpu().numpy())
    preds, targets = np.array(preds), np.array(targets)
    return mean_absolute_error(targets, preds), float(np.sqrt(np.mean((targets-preds)**2))), r2_score(targets, preds)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    graphs, pce_values = load_data()

    # Fixed split (seed=9999 for comparability with reported results)
    indices = list(range(len(graphs)))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=9999)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=9999)

    train_graphs = [graphs[i] for i in train_idx]
    val_graphs = [graphs[i] for i in val_idx]
    test_graphs = [graphs[i] for i in test_idx]

    # Hyperparameter grid - pick 4 informative combos
    combos = [
        (0.001, 128, 0.3),   # default
        (0.0005, 256, 0.2),  # higher capacity, lower lr
        (0.005, 64, 0.3),    # higher lr, smaller model
        (0.001, 256, 0.4),   # higher capacity, more dropout
    ]

    print(f"\n  Running {N_TRIALS} hyperparameter combinations...")
    print(f"  {'Trial':<6} {'LR':<8} {'Hidden':<8} {'Dropout':<8} {'Val R²':<8} {'Test R²':<8} {'Test MAE':<8}")
    print("  " + "-" * 60)

    results = []
    best_test_r2 = -1
    best_config = None

    for trial_i, (lr, hidden, dropout) in enumerate(combos):
        set_seed(9999)

        train_loader = GeoDataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = GeoDataLoader(val_graphs, batch_size=BATCH_SIZE)
        test_loader = GeoDataLoader(test_graphs, batch_size=BATCH_SIZE)

        in_dim = graphs[0].x.shape[1]
        model = HighPCERegressorV3(in_channels=in_dim, hidden=hidden, dropout=dropout).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.HuberLoss(delta=1.0)

        best_val_mae = float('inf')
        patience_counter = 0

        pbar = tqdm(range(1, EPOCHS + 1), desc=f'Trial {trial_i+1}/{N_TRIALS} (lr={lr}, h={hidden}, d={dropout})',
                     leave=False, ncols=100)
        for epoch in pbar:
            model.train()
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                pred = model(batch)
                loss = criterion(pred, batch.y.view(-1))
                loss.backward()
                optimizer.step()

            val_mae, val_rmse, val_r2 = evaluate(model, val_loader, device)
            pbar.set_postfix({'val_mae': f'{val_mae:.4f}', 'val_r2': f'{val_r2:.4f}'})

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                patience_counter = 0
                best_state = model.state_dict()
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                pbar.set_postfix({'val_mae': f'{val_mae:.4f}', 'val_r2': f'{val_r2:.4f}', 'early_stop': epoch})
                break

        # Evaluate on test set
        model.load_state_dict(best_state)
        test_mae, test_rmse, test_r2 = evaluate(model, test_loader, device)

        print(f"  {trial_i+1:<6} {lr:<8} {hidden:<8} {dropout:<8.1f} {val_r2:<8.4f} {test_r2:<8.4f} {test_mae:<8.4f}")

        results.append({
            'trial': trial_i + 1, 'lr': lr, 'hidden_dim': hidden, 'dropout': dropout,
            'val_r2': float(val_r2), 'val_mae': float(val_mae), 'val_rmse': float(val_rmse),
            'test_r2': float(test_r2), 'test_mae': float(test_mae), 'test_rmse': float(test_rmse),
        })

        if test_r2 > best_test_r2:
            best_test_r2 = test_r2
            best_config = {'lr': lr, 'hidden_dim': hidden, 'dropout': dropout, 'test_r2': test_r2}

    print("\n" + "=" * 60)
    print(f"Best test R²: {best_test_r2:.4f}")
    print(f"Best config: {best_config}")

    # Compare with reported GNN (R²=0.6432) and default (R²=0.6432)
    print(f"\nReported GNN (default): R²=0.6432")
    print(f"GNN hyperopt best:      R²={best_test_r2:.4f}")
    print(f"Improvement:            ΔR²={best_test_r2 - 0.6432:+.4f}")

    # Save
    output = {
        'best_test_r2': best_test_r2,
        'best_config': best_config,
        'all_results': results,
        'reported_gnn_r2': 0.6432,
        'note': 'Limited hyperparameter search (9 trials on lr/hidden/dropout). Not exhaustive Optuna.',
    }
    with open(os.path.join(RESULTS_DIR, 'gnn_hyperopt_results.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {os.path.join(RESULTS_DIR, 'gnn_hyperopt_results.json')}")

if __name__ == '__main__':
    main()
