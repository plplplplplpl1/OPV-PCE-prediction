"""
低PCE回归器 v2
改进：过滤极端异常值（PCE < 0.01%），使用 log1p 变换
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import (
    GCNConv, GATConv, SAGEConv,
    global_mean_pool, global_max_pool, global_add_pool
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
except ImportError:
    print("错误：RDKit 未安装。")
    exit(1)

DATA_CSV = 'data/data_merged.csv' if os.path.exists('data/data_merged.csv') else 'data/data.csv'
PCE_THRESHOLD = 3.0
SAVE_PATH = 'best_low_pce_regressor_v2.pth'
MIN_PCE = 0.01  # 过滤极端异常值
EPOCHS = 200
PATIENCE = 20
BATCH_SIZE = 32
LR = 0.001
FP_DIM = 512  # 更大的指纹维度


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
            is_in_ring = int(atom.IsInRing())
            try:
                hybridization = int(atom.GetHybridization())
                num_h = atom.GetTotalNumHs()
                valence = atom.GetTotalValence()
                r3 = int(atom.IsInRingSize(3))
                r4 = int(atom.IsInRingSize(4))
                r5 = int(atom.IsInRingSize(5))
                r6 = int(atom.IsInRingSize(6))
            except Exception:
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

        if not edge_indices:
            return None

        # Morgan 指纹（radius=2, 512-bit）
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=FP_DIM)
        fp_tensor = torch.tensor(np.array(fp, dtype=np.float32)).unsqueeze(0)

        x = torch.tensor(atom_features, dtype=torch.float)
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        return Data(x=x, edge_index=edge_index, fp=fp_tensor)
    except Exception:
        return None


class LowPCERegressor(nn.Module):
    """低PCE回归器，与高PCE v3相同架构"""

    def __init__(self, in_channels=30, hidden=128, fp_dim=FP_DIM, dropout=0.3):
        super().__init__()
        # GCN 分支（2层，与 v1 相同）
        self.gcn1 = GCNConv(in_channels, hidden)
        self.gcn2 = GCNConv(hidden, hidden)
        # GAT 分支
        self.gat1 = GATConv(in_channels, hidden // 4, heads=4, dropout=dropout)
        self.gat2 = GATConv(hidden, hidden, heads=1, dropout=dropout)
        # SAGE 分支
        self.sage1 = SAGEConv(in_channels, hidden)
        self.sage2 = SAGEConv(hidden, hidden)

        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.bn3 = nn.BatchNorm1d(hidden)

        # Morgan 指纹编码器
        self.fp_encoder = nn.Sequential(
            nn.Linear(fp_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # 融合维度：3分支 × 3池化 × hidden + 128（指纹）
        fused_dim = hidden * 9 + 128

        self.regressor = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
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
            return torch.cat([
                global_mean_pool(h, batch),
                global_max_pool(h, batch),
                global_add_pool(h, batch),
            ], dim=1)

        g = torch.cat([pool3(x_gcn), pool3(x_gat), pool3(x_sage)], dim=1)
        fp_feat = self.fp_encoder(fp)
        g = torch.cat([g, fp_feat], dim=1)
        return self.regressor(g).squeeze(1)


def load_data():
    print("读取数据 ...")
    df = pd.read_csv(DATA_CSV, encoding='latin-1')
    pce_col = df.columns[2]
    smiles_col = df.columns[-1]

    df[pce_col] = pd.to_numeric(df[pce_col], errors='coerce')
    df[smiles_col] = df[smiles_col].astype(str).str.strip()
    df = df.dropna(subset=[pce_col, smiles_col])
    df = df[df[smiles_col] != 'nan'].reset_index(drop=True)

    # 过滤 PCE ≤ 3% 且 >= 0.01% 的样本（去除极端异常值）
    df_low = df[(df[pce_col] <= PCE_THRESHOLD) & (df[pce_col] >= MIN_PCE)].copy().reset_index(drop=True)
    print(f"  低PCE样本数 ({MIN_PCE}% ≤ PCE ≤ {PCE_THRESHOLD}%): {len(df_low)}")
    print(f"  PCE 范围: {df_low[pce_col].min():.4f}% ~ {df_low[pce_col].max():.2f}%")

    graphs, pce_values, failed = [], [], 0
    for _, row in df_low.iterrows():
        g = smiles_to_graph(row[smiles_col])
        if g is not None:
            pce = float(row[pce_col])
            g.y = torch.tensor([np.log1p(pce)], dtype=torch.float)  # log1p 变换
            graphs.append(g)
            pce_values.append(pce)
        else:
            failed += 1

    print(f"  图转换成功: {len(graphs)}, 失败: {failed}")
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
        # 还原 log1p 变换
        preds.extend(np.expm1(pred.cpu().numpy()).tolist())
        targets.extend(np.expm1(batch.y.view(-1).cpu().numpy()).tolist())
    preds = np.array(preds)
    targets = np.array(targets)
    mae = mean_absolute_error(targets, preds)
    rmse = float(np.sqrt(np.mean((targets - preds) ** 2)))
    r2 = r2_score(targets, preds)
    return mae, rmse, r2


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    graphs, pce_values = load_data()

    indices = list(range(len(graphs)))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=42)

    train_graphs = [graphs[i] for i in train_idx]
    val_graphs   = [graphs[i] for i in val_idx]
    test_graphs  = [graphs[i] for i in test_idx]

    print(f"\n数据集划分: 训练 {len(train_graphs)}, 验证 {len(val_graphs)}, 测试 {len(test_graphs)}")

    train_loader = GeoDataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = GeoDataLoader(val_graphs,   batch_size=BATCH_SIZE)
    test_loader  = GeoDataLoader(test_graphs,  batch_size=BATCH_SIZE)

    in_dim = graphs[0].x.shape[1]
    model = LowPCERegressor(in_channels=in_dim).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.HuberLoss(delta=1.0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    best_val_mae = float('inf')
    patience_counter = 0

    print(f"\n开始训练（最多 {EPOCHS} 轮，早停 patience={PATIENCE}）...")
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_mae, val_rmse, val_r2 = evaluate(model, val_loader, device)
        scheduler.step(val_mae)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            torch.save(model.state_dict(), SAVE_PATH)
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | train_loss={train_loss:.4f} | "
                  f"val_MAE={val_mae:.4f} | val_RMSE={val_rmse:.4f} | val_R2={val_r2:.4f} | "
                  f"patience={patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print(f"\n早停触发（epoch {epoch}）")
            break

    model.load_state_dict(torch.load(SAVE_PATH, weights_only=True))
    test_mae, test_rmse, test_r2 = evaluate(model, test_loader, device)
    print(f"\n测试集结果 ({MIN_PCE}% ≤ PCE ≤ {PCE_THRESHOLD}%):")
    print(f"  MAE  = {test_mae:.4f} %")
    print(f"  RMSE = {test_rmse:.4f} %")
    print(f"  R2   = {test_r2:.4f}")
    print(f"\n最佳模型已保存至: {SAVE_PATH}")


if __name__ == '__main__':
    main()
