"""
简单2层GCN在OPV高PCE数据集上的对比实验
用于验证"架构复杂度推后交叉点"假说

架构: 2层GCN, 64维隐藏层, 无指纹融合
与 QM9 简单GCN实验保持一致
"""
import os
import sys
import json
import argparse
import random
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

DATA_CSV = 'data/data_merged.csv' if os.path.exists('data/data_merged.csv') else 'data/data.csv'
PCE_THRESHOLD = 3.0
EPOCHS = 200
PATIENCE = 20
BATCH_SIZE = 32
LR = 0.001


def set_seed(seed):
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
        if mol is None:
            return None

        atom_features = []
        for atom in mol.GetAtoms():
            atomic_num = atom.GetAtomicNum()
            degree = atom.GetDegree()
            formal_charge = atom.GetFormalCharge()
            is_aromatic = int(atom.GetIsAromatic())
            try:
                hybridization = int(atom.GetHybridization())
                num_h = atom.GetTotalNumHs()
                valence = atom.GetTotalValence()
            except Exception:
                hybridization = num_h = valence = 0

            common_atoms = [1, 6, 7, 8, 9, 15, 16, 17, 35]
            feat = [
                atomic_num / 100.0, degree / 6.0, formal_charge / 8.0,
                num_h / 4.0, valence / 8.0, is_aromatic,
            ] + [int(atomic_num == a) for a in common_atoms] \
              + [int(degree == d) for d in range(5)] \
              + [int(hybridization == h) for h in range(1, 6)]
            atom_features.append(feat)

        edge_indices = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_indices += [[i, j], [j, i]]

        if not edge_indices:
            return None

        x = torch.tensor(atom_features, dtype=torch.float)
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        return Data(x=x, edge_index=edge_index)
    except Exception:
        return None


class SimpleGCN(nn.Module):
    """简单2层GCN, 64维隐藏层, 单池化, 无指纹融合"""

    def __init__(self, in_channels=30, hidden=64, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = global_mean_pool(x, batch)
        return self.regressor(x).squeeze(1)


def load_data():
    df = pd.read_csv(DATA_CSV, encoding='latin-1')
    pce_col = df.columns[2]
    smiles_col = df.columns[-1]

    df[pce_col] = pd.to_numeric(df[pce_col], errors='coerce')
    df[smiles_col] = df[smiles_col].astype(str).str.strip()
    df = df.dropna(subset=[pce_col, smiles_col])
    df = df[df[smiles_col] != 'nan'].reset_index(drop=True)

    df_high = df[df[pce_col] > PCE_THRESHOLD].copy().reset_index(drop=True)
    print(f"高PCE样本: {len(df_high)}, 范围: {df_high[pce_col].min():.2f}%~{df_high[pce_col].max():.2f}%")

    graphs, failed = [], 0
    for _, row in df_high.iterrows():
        g = smiles_to_graph(row[smiles_col])
        if g is not None:
            g.y = torch.tensor([float(row[pce_col])], dtype=torch.float)
            graphs.append(g)
        else:
            failed += 1
    print(f"图转换: {len(graphs)}成功, {failed}失败")
    return graphs


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
        preds.extend(model(batch).cpu().numpy().tolist())
        targets.extend(batch.y.view(-1).cpu().numpy().tolist())
    preds, targets = np.array(preds), np.array(targets)
    return mean_absolute_error(targets, preds), float(np.sqrt(np.mean((targets - preds) ** 2))), r2_score(targets, preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 333, 9999])
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--output', type=str, default='external_results/simple_gcn_opv.json')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device} | 隐藏维: {args.hidden} | 种子: {args.seeds}")

    graphs = load_data()
    in_dim = graphs[0].x.shape[1]
    print(f"节点特征维度: {in_dim}")

    all_results = []
    for seed in args.seeds:
        print(f"\n{'='*50}\n训练 seed={seed}\n{'='*50}")
        set_seed(seed)

        indices = list(range(len(graphs)))
        train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=seed)
        train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=seed)

        train_g = [graphs[i] for i in train_idx]
        val_g   = [graphs[i] for i in val_idx]
        test_g  = [graphs[i] for i in test_idx]
        print(f"划分: 训练{len(train_g)} 验证{len(val_g)} 测试{len(test_g)}")

        train_loader = GeoDataLoader(train_g, BATCH_SIZE, shuffle=True)
        val_loader   = GeoDataLoader(val_g,   BATCH_SIZE)
        test_loader  = GeoDataLoader(test_g,  BATCH_SIZE)

        model = SimpleGCN(in_channels=in_dim, hidden=args.hidden, dropout=args.dropout).to(device)
        params = sum(p.numel() for p in model.parameters())
        print(f"参数量: {params:,}")

        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        criterion = nn.HuberLoss(delta=1.0)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

        best_val_mae = float('inf')
        patience_counter = 0
        save_path = f'保存的模型/simple_gcn_seed{seed}.pth'

        for epoch in range(1, args.epochs + 1):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_mae, val_rmse, val_r2 = evaluate(model, val_loader, device)
            scheduler.step(val_mae)

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                patience_counter = 0
                os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
                torch.save(model.state_dict(), save_path)
            else:
                patience_counter += 1

            if epoch % 20 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d} | loss={train_loss:.4f} | val_MAE={val_mae:.4f} | val_R²={val_r2:.4f}")

            if patience_counter >= args.patience:
                print(f"早停 (epoch {epoch})")
                break

        model.load_state_dict(torch.load(save_path, weights_only=True))
        mae, rmse, r2 = evaluate(model, test_loader, device)
        print(f"Seed {seed} 测试集: MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")
        all_results.append({'seed': seed, 'r2': r2, 'mae': mae, 'rmse': rmse, 'params': params})

    r2_vals = [r['r2'] for r in all_results]
    mae_vals = [r['mae'] for r in all_results]
    rmse_vals = [r['rmse'] for r in all_results]

    summary = {
        'model': 'SimpleGCN (2-layer, 64-dim, no fingerprint)',
        'seeds': args.seeds,
        'hidden_dim': args.hidden,
        'per_seed': all_results,
        'mean_r2': float(np.mean(r2_vals)),
        'std_r2': float(np.std(r2_vals)),
        'mean_mae': float(np.mean(mae_vals)),
        'std_mae': float(np.std(mae_vals)),
        'mean_rmse': float(np.mean(rmse_vals)),
        'std_rmse': float(np.std(rmse_vals)),
        'comparison': {
            'xgb_optuna_r2': 0.7360,
            'xgb_multiseed_r2': '0.686±0.026',
            'gnn_v3_multiseed_r2': '0.635±0.039',
            'gps_multiseed_r2': '0.616±0.051',
        }
    }

    print(f"\n{'='*50}")
    print(f"简单GCN (2层, {args.hidden}维) 多种子结果:")
    print(f"  R²  = {np.mean(r2_vals):.4f} ± {np.std(r2_vals):.4f}")
    print(f"  MAE = {np.mean(mae_vals):.4f} ± {np.std(mae_vals):.4f}")
    print(f"  RMSE= {np.mean(rmse_vals):.4f} ± {np.std(rmse_vals):.4f}")
    print(f"\n对比: V3 GNN多分支: 0.635±0.039")
    print(f"      GraphGPS:    0.616±0.051")
    print(f"      XGBoost多种子: 0.686±0.026")
    print(f"      XGBoost最优:  0.7360")

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n结果已保存: {args.output}")


if __name__ == '__main__':
    main()
