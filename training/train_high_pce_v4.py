"""
高PCE回归器 V4 - 升级版
主要改进:
1. Morgan指纹维度: 512 -> 4096 (与XGBoost一致)
2. 融合全量RDKit描述符 (217个)
3. 优化的架构设计
"""

import os
import argparse
import random
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
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
except ImportError:
    print("错误：RDKit 未安装。")
    exit(1)

DATA_CSV = 'data/data_merged.csv' if os.path.exists('data/data_merged.csv') else 'data/data.csv'
PCE_THRESHOLD = 3.0
SAVE_PATH = 'best_high_pce_regressor_v4.pth'
EPOCHS = 200
PATIENCE = 20
BATCH_SIZE = 32
LR = 0.001
FP_DIM = 4096  # 升级到4096位指纹

# RDKit描述符列表
_ALL_DESC_FUNCS: list = []
try:
    _ALL_DESC_FUNCS = list(Descriptors._descList)
except Exception:
    _ALL_DESC_FUNCS = []


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_rdkit_descriptors(mol):
    """提取全量RDKit描述符"""
    desc = []
    for _, fn in _ALL_DESC_FUNCS:
        try:
            v = fn(mol)
            if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                v = 0.0
        except Exception:
            v = 0.0
        desc.append(float(v))

    # 处理极端值
    desc_clean = []
    for v in desc:
        if np.isinf(v) or np.isnan(v):
            desc_clean.append(0.0)
        elif abs(v) > 1e10:
            desc_clean.append(np.sign(v) * 1e10)
        else:
            desc_clean.append(v)

    return np.array(desc_clean, dtype=np.float32)


def smiles_to_graph(smiles):
    """将SMILES转换为图数据，包含指纹和描述符"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # 原子特征
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

        # 边索引
        edge_indices = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_indices += [[i, j], [j, i]]

        if not edge_indices:
            return None

        # Morgan 指纹（4096-bit）
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=FP_DIM)
        fp_tensor = torch.tensor(np.array(fp, dtype=np.float32)).unsqueeze(0)

        # RDKit描述符
        rdkit_desc = get_rdkit_descriptors(mol)
        desc_tensor = torch.tensor(rdkit_desc, dtype=torch.float32).unsqueeze(0)

        x = torch.tensor(atom_features, dtype=torch.float)
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        return Data(x=x, edge_index=edge_index, fp=fp_tensor, desc=desc_tensor)
    except Exception:
        return None


class HighPCERegressorV4(nn.Module):
    """V4架构: 4096位指纹 + RDKit描述符 + 优化的多分支GNN"""

    def __init__(self, in_channels=30, hidden=128, fp_dim=FP_DIM, desc_dim=None, dropout=0.3):
        super().__init__()
        if desc_dim is None:
            desc_dim = len(_ALL_DESC_FUNCS) if _ALL_DESC_FUNCS else 12

        # GCN 分支
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

        # Morgan指纹编码器 (4096 -> 256 -> 128)
        self.fp_encoder = nn.Sequential(
            nn.Linear(fp_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
        )

        # RDKit描述符编码器
        self.desc_encoder = nn.Sequential(
            nn.Linear(desc_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
        )

        # 融合维度: 3分支 × 3池化 × hidden + 128(指纹) + 64(描述符)
        fused_dim = hidden * 9 + 128 + 64

        # 回归器
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
        desc = data.desc

        # GCN分支
        x_gcn = F.relu(self.bn1(self.gcn1(x, edge_index)))
        x_gcn = F.relu(self.gcn2(x_gcn, edge_index))

        # GAT分支
        x_gat = F.relu(self.gat1(x, edge_index))
        x_gat = F.relu(self.bn2(self.gat2(x_gat, edge_index)))

        # SAGE分支
        x_sage = F.relu(self.sage1(x, edge_index))
        x_sage = F.relu(self.bn3(self.sage2(x_sage, edge_index)))

        # 多尺度池化
        def pool3(h):
            return torch.cat([
                global_mean_pool(h, batch),
                global_max_pool(h, batch),
                global_add_pool(h, batch),
            ], dim=1)

        # 图特征
        g = torch.cat([pool3(x_gcn), pool3(x_gat), pool3(x_sage)], dim=1)

        # 指纹特征
        fp_feat = self.fp_encoder(fp)

        # 描述符特征
        desc_feat = self.desc_encoder(desc)

        # 融合所有特征
        g = torch.cat([g, fp_feat, desc_feat], dim=1)
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

    df_high = df[df[pce_col] > PCE_THRESHOLD].copy().reset_index(drop=True)
    print(f"  高PCE样本数 (PCE > {PCE_THRESHOLD}%): {len(df_high)}")
    print(f"  PCE 范围: {df_high[pce_col].min():.2f}% ~ {df_high[pce_col].max():.2f}%")

    graphs, pce_values, failed = [], [], 0
    for _, row in df_high.iterrows():
        g = smiles_to_graph(row[smiles_col])
        if g is not None:
            g.y = torch.tensor([float(row[pce_col])], dtype=torch.float)
            graphs.append(g)
            pce_values.append(float(row[pce_col]))
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
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
        preds.extend(pred.cpu().numpy().tolist())
        targets.extend(batch.y.view(-1).cpu().numpy().tolist())
    preds = np.array(preds)
    targets = np.array(targets)
    mae = mean_absolute_error(targets, preds)
    rmse = float(np.sqrt(mean_squared_error(targets, preds)))
    r2 = r2_score(targets, preds)
    return mae, rmse, r2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp-dim", type=int, default=4096, help="指纹维度")
    parser.add_argument("--hidden", type=int, default=128, help="隐藏层维度")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout率")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    parser.add_argument("--batch-size", type=int, default=32, help="批次大小")
    parser.add_argument("--epochs", type=int, default=200, help="最大训练轮数")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device} | seed={args.seed}")
    print(f"指纹维度: {args.fp_dim} | 隐藏层: {args.hidden} | Dropout: {args.dropout}")

    graphs, pce_values = load_data()

    indices = list(range(len(graphs)))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=args.seed)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=args.seed)

    train_graphs = [graphs[i] for i in train_idx]
    val_graphs   = [graphs[i] for i in val_idx]
    test_graphs  = [graphs[i] for i in test_idx]

    print(f"\n数据集划分: 训练 {len(train_graphs)}, 验证 {len(val_graphs)}, 测试 {len(test_graphs)}")

    train_loader = GeoDataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader   = GeoDataLoader(val_graphs,   batch_size=args.batch_size)
    test_loader  = GeoDataLoader(test_graphs,  batch_size=args.batch_size)

    in_dim = graphs[0].x.shape[1]
    desc_dim = len(_ALL_DESC_FUNCS) if _ALL_DESC_FUNCS else 12

    model = HighPCERegressorV4(
        in_channels=in_dim,
        hidden=args.hidden,
        fp_dim=args.fp_dim,
        desc_dim=desc_dim,
        dropout=args.dropout
    ).to(device)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.HuberLoss(delta=1.0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    best_val_mae = float('inf')
    patience_counter = 0

    print(f"\n开始训练（最多 {args.epochs} 轮，早停 patience={PATIENCE}）...")
    for epoch in range(1, args.epochs + 1):
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
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:3d} | train_loss={train_loss:.4f} | lr={current_lr:.6f} | "
                  f"val_MAE={val_mae:.4f} | val_RMSE={val_rmse:.4f} | val_R2={val_r2:.4f} | "
                  f"patience={patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print(f"\n早停触发（epoch {epoch}）")
            break

    model.load_state_dict(torch.load(SAVE_PATH, weights_only=True))
    test_mae, test_rmse, test_r2 = evaluate(model, test_loader, device)
    print(f"\n{'='*60}")
    print(f"测试集结果（PCE > {PCE_THRESHOLD}% 子集）:")
    print(f"  MAE  = {test_mae:.4f} %")
    print(f"  RMSE = {test_rmse:.4f} %")
    print(f"  R2   = {test_r2:.4f}")
    print(f"{'='*60}")
    print(f"\n最佳模型已保存至: {SAVE_PATH}")

    # 检查是否达到目标
    if test_r2 >= 0.75:
        print(f"\n🎉 目标达成！R² = {test_r2:.4f} >= 0.75")
    else:
        print(f"\n⚠️  尚未达到目标。当前R² = {test_r2:.4f}, 目标 = 0.75")


if __name__ == '__main__':
    main()
