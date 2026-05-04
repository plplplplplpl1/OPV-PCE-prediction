"""
High PCE Regressor v3 (PyG) + SMILES Augmentation + Seed control

目标：更“论文式”的训练策略（数据增强/集成）来提升泛化。
"""

import os
import argparse
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import (
    GCNConv,
    GATConv,
    SAGEConv,
    global_mean_pool,
    global_max_pool,
    global_add_pool,
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
except ImportError:
    raise SystemExit("错误：RDKit 未安装。")


PCE_THRESHOLD = 3.0
FP_DIM = 512


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def randomize_smiles(smiles: str, seed: int) -> str | None:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        # 使用受控 seed 的随机 SMILES
        rng = random.Random(seed)
        # RDKit 的 doRandom 依赖全局随机；这里用打乱原子顺序的方式近似可复现
        atom_order = list(range(mol.GetNumAtoms()))
        rng.shuffle(atom_order)
        mol2 = Chem.RenumberAtoms(mol, atom_order)
        return Chem.MolToSmiles(mol2, canonical=False)
    except Exception:
        return None


def smiles_to_graph(smiles: str):
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
                atomic_num / 100.0,
                degree / 6.0,
                formal_charge / 8.0,
                num_h / 4.0,
                valence / 8.0,
                is_aromatic,
                is_in_ring,
                r3,
                r4,
                r5,
                r6,
            ] + [int(atomic_num == a) for a in common_atoms] + [int(degree == d) for d in range(5)] + [
                int(hybridization == h) for h in range(1, 6)
            ]
            atom_features.append(feat)

        edge_indices = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_indices += [[i, j], [j, i]]
        if not edge_indices:
            return None

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=FP_DIM)
        fp_tensor = torch.tensor(np.array(fp, dtype=np.float32)).unsqueeze(0)

        x = torch.tensor(atom_features, dtype=torch.float)
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        return Data(x=x, edge_index=edge_index, fp=fp_tensor)
    except Exception:
        return None


class HighPCERegressorV3(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 128, fp_dim: int = FP_DIM, dropout: float = 0.3):
        super().__init__()
        self.gcn1 = GCNConv(in_channels, hidden)
        self.gcn2 = GCNConv(hidden, hidden)

        self.gat1 = GATConv(in_channels, hidden // 4, heads=4, dropout=dropout)
        self.gat2 = GATConv(hidden, hidden, heads=1, dropout=dropout)

        self.sage1 = SAGEConv(in_channels, hidden)
        self.sage2 = SAGEConv(hidden, hidden)

        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.bn3 = nn.BatchNorm1d(hidden)

        self.fp_encoder = nn.Sequential(
            nn.Linear(fp_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

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
            return torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch), global_add_pool(h, batch)], dim=1)

        g = torch.cat([pool3(x_gcn), pool3(x_gat), pool3(x_sage)], dim=1)
        fp_feat = self.fp_encoder(fp)
        g = torch.cat([g, fp_feat], dim=1)
        return self.regressor(g).squeeze(1)


def load_high_pce_rows():
    data_csv = "data/data_merged.csv" if os.path.exists("data/data_merged.csv") else "data/data.csv"
    df = pd.read_csv(data_csv, encoding="latin-1")
    pce_col = df.columns[2]
    smiles_col = df.columns[-1]

    df[pce_col] = pd.to_numeric(df[pce_col], errors="coerce")
    df[smiles_col] = df[smiles_col].astype(str).str.strip()
    df = df.dropna(subset=[pce_col, smiles_col])
    df = df[df[smiles_col] != "nan"].reset_index(drop=True)

    df_high = df[df[pce_col] > PCE_THRESHOLD].copy().reset_index(drop=True)
    return data_csv, df_high, pce_col, smiles_col


def build_graphs(smiles_list: list[str], pce_list: list[float]):
    graphs = []
    failed = 0
    for smi, pce in zip(smiles_list, pce_list):
        g = smiles_to_graph(smi)
        if g is None:
            failed += 1
            continue
        g.y = torch.tensor([float(pce)], dtype=torch.float)
        graphs.append(g)
    return graphs, failed


@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    preds, targets = [], []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        preds.extend(pred.detach().cpu().numpy().tolist())
        targets.extend(batch.y.view(-1).detach().cpu().numpy().tolist())
    preds = np.array(preds, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)
    mae = float(mean_absolute_error(targets, preds))
    rmse = float(np.sqrt(np.mean((targets - preds) ** 2)))
    r2 = float(r2_score(targets, preds))
    return mae, rmse, r2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--n-augment", type=int, default=2, help="每个训练样本生成多少个随机 SMILES 变体")
    parser.add_argument("--save", type=str, default="best_high_pce_regressor_v3_aug.pth")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device} | seed={args.seed} | n_augment={args.n_augment}")

    data_csv, df_high, pce_col, smiles_col = load_high_pce_rows()
    smiles = df_high[smiles_col].tolist()
    pce = df_high[pce_col].tolist()
    print(f"数据文件: {data_csv}")
    print(f"高PCE样本数 (PCE > {PCE_THRESHOLD}%): {len(df_high)}")

    idx = np.arange(len(df_high))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=args.seed, shuffle=True)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=args.seed, shuffle=True)

    train_smiles = [smiles[i] for i in train_idx]
    train_pce = [pce[i] for i in train_idx]
    val_smiles = [smiles[i] for i in val_idx]
    val_pce = [pce[i] for i in val_idx]
    test_smiles = [smiles[i] for i in test_idx]
    test_pce = [pce[i] for i in test_idx]

    # SMILES augmentation (training only)
    if args.n_augment > 0:
        aug_smiles = []
        aug_pce = []
        for i, (smi, y) in enumerate(zip(train_smiles, train_pce)):
            for k in range(args.n_augment):
                smi2 = randomize_smiles(smi, seed=args.seed + i * 100 + k)
                if smi2:
                    aug_smiles.append(smi2)
                    aug_pce.append(y)
        train_smiles = train_smiles + aug_smiles
        train_pce = train_pce + aug_pce
        print(f"训练集增强后样本数: {len(train_smiles)}（新增 {len(aug_smiles)}）")

    train_graphs, train_failed = build_graphs(train_smiles, train_pce)
    val_graphs, val_failed = build_graphs(val_smiles, val_pce)
    test_graphs, test_failed = build_graphs(test_smiles, test_pce)
    print(f"图转换失败: train={train_failed}, val={val_failed}, test={test_failed}")
    print(f"数据集划分(有效图): train={len(train_graphs)}, val={len(val_graphs)}, test={len(test_graphs)}")

    train_loader = GeoDataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = GeoDataLoader(val_graphs, batch_size=args.batch_size)
    test_loader = GeoDataLoader(test_graphs, batch_size=args.batch_size)

    in_dim = train_graphs[0].x.shape[1]
    model = HighPCERegressorV3(in_channels=in_dim).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.HuberLoss(delta=1.0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    best_val_mae = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = criterion(pred, batch.y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs

        train_loss = total_loss / max(1, len(train_loader.dataset))
        val_mae, val_rmse, val_r2 = eval_model(model, val_loader, device)
        scheduler.step(val_mae)

        improved = val_mae < best_val_mae
        if improved:
            best_val_mae = val_mae
            patience_counter = 0
            torch.save(model.state_dict(), args.save)
        else:
            patience_counter += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | "
                f"val_MAE={val_mae:.4f} | val_RMSE={val_rmse:.4f} | val_R2={val_r2:.4f} | "
                f"patience={patience_counter}/{args.patience}"
            )

        if patience_counter >= args.patience:
            print(f"早停触发（epoch {epoch}）")
            break

    model.load_state_dict(torch.load(args.save, weights_only=True))
    test_mae, test_rmse, test_r2 = eval_model(model, test_loader, device)
    print("\n测试集结果（PCE > 3% 子集）:")
    print(f"  MAE  = {test_mae:.4f} %")
    print(f"  RMSE = {test_rmse:.4f} %")
    print(f"  R2   = {test_r2:.4f}")
    print(f"\n已保存最佳模型至: {args.save}")


if __name__ == "__main__":
    main()

