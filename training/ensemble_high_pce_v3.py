"""
Ensemble runner for HighPCERegressorV3 (PyG).

核心思路（更像论文复现）：
- 固定一次数据划分（split_seed），所有模型使用同一 train/val/test
- 改变训练随机种子（model_seed），训练多个模型
- 在 test 上做预测平均，报告 ensemble 的 MAE/RMSE/R2
"""

import os
import argparse
import random
from dataclasses import dataclass

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


def data_path() -> str:
    return "data/data_merged.csv" if os.path.exists("data/data_merged.csv") else "data/data.csv"


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
    def __init__(self, in_channels=30, hidden=128, fp_dim=FP_DIM, dropout=0.3):
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
            return torch.cat(
                [global_mean_pool(h, batch), global_max_pool(h, batch), global_add_pool(h, batch)],
                dim=1,
            )

        g = torch.cat([pool3(x_gcn), pool3(x_gat), pool3(x_sage)], dim=1)
        fp_feat = self.fp_encoder(fp)
        g = torch.cat([g, fp_feat], dim=1)
        return self.regressor(g).squeeze(1)


@dataclass(frozen=True)
class DatasetSplit:
    train: list[int]
    val: list[int]
    test: list[int]


def load_high_pce_graphs():
    csv_path = data_path()
    df = pd.read_csv(csv_path, encoding="latin-1")
    pce_col = df.columns[2]
    smiles_col = df.columns[-1]

    df[pce_col] = pd.to_numeric(df[pce_col], errors="coerce")
    df[smiles_col] = df[smiles_col].astype(str).str.strip()
    df = df.dropna(subset=[pce_col, smiles_col])
    df = df[df[smiles_col] != "nan"].reset_index(drop=True)
    df_high = df[df[pce_col] > PCE_THRESHOLD].copy().reset_index(drop=True)

    graphs = []
    failed = 0
    for smi, pce in zip(df_high[smiles_col].tolist(), df_high[pce_col].tolist()):
        g = smiles_to_graph(smi)
        if g is None:
            failed += 1
            continue
        g.y = torch.tensor([float(pce)], dtype=torch.float)
        graphs.append(g)

    if not graphs:
        raise RuntimeError("没有成功构建任何图")

    print(f"数据文件: {csv_path}")
    print(f"高PCE样本数 (PCE > {PCE_THRESHOLD}%): {len(df_high)} | 图成功: {len(graphs)} | 失败: {failed}")
    return graphs


def make_split(n: int, split_seed: int) -> DatasetSplit:
    idx = list(range(n))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=split_seed, shuffle=True)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=split_seed, shuffle=True)
    return DatasetSplit(train=train_idx, val=val_idx, test=test_idx)


def train_one_model(
    graphs: list[Data],
    split: DatasetSplit,
    model_seed: int,
    device: torch.device,
    epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
):
    set_seed(model_seed)

    train_graphs = [graphs[i] for i in split.train]
    val_graphs = [graphs[i] for i in split.val]
    test_graphs = [graphs[i] for i in split.test]

    train_loader = GeoDataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = GeoDataLoader(val_graphs, batch_size=batch_size)
    test_loader = GeoDataLoader(test_graphs, batch_size=batch_size)

    in_dim = train_graphs[0].x.shape[1]
    model = HighPCERegressorV3(in_channels=in_dim).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.HuberLoss(delta=1.0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    best_val_mae = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
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

        model.eval()
        with torch.no_grad():
            val_preds, val_targets = [], []
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch)
                val_preds.extend(pred.detach().cpu().numpy().tolist())
                val_targets.extend(batch.y.view(-1).detach().cpu().numpy().tolist())
            val_preds = np.array(val_preds, dtype=np.float32)
            val_targets = np.array(val_targets, dtype=np.float32)
            val_mae = float(mean_absolute_error(val_targets, val_preds))
            val_rmse = float(np.sqrt(np.mean((val_targets - val_preds) ** 2)))
            val_r2 = float(r2_score(val_targets, val_preds))

        scheduler.step(val_mae)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"  seed={model_seed} | epoch {epoch:3d} | train_loss={train_loss:.4f} | "
                f"val_MAE={val_mae:.4f} | val_RMSE={val_rmse:.4f} | val_R2={val_r2:.4f} | "
                f"patience={patience_counter}/{patience}"
            )

        if patience_counter >= patience:
            break

    if best_state is None:
        best_state = model.state_dict()
    model.load_state_dict(best_state, strict=True)
    model.to(device)
    model.eval()

    with torch.no_grad():
        test_preds, test_targets = [], []
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch)
            test_preds.extend(pred.detach().cpu().numpy().tolist())
            test_targets.extend(batch.y.view(-1).detach().cpu().numpy().tolist())

    test_preds = np.array(test_preds, dtype=np.float32)
    test_targets = np.array(test_targets, dtype=np.float32)
    mae = float(mean_absolute_error(test_targets, test_preds))
    rmse = float(np.sqrt(np.mean((test_targets - test_preds) ** 2)))
    r2 = float(r2_score(test_targets, test_preds))
    return test_preds, test_targets, {"seed": model_seed, "mae": mae, "rmse": rmse, "r2": r2}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-seed", type=int, default=9999)
    parser.add_argument("--model-seeds", type=str, default="9999,123,42")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    print(f"split_seed={args.split_seed} | model_seeds={args.model_seeds}")

    graphs = load_high_pce_graphs()
    split = make_split(len(graphs), split_seed=args.split_seed)
    print(f"划分(固定): train={len(split.train)}, val={len(split.val)}, test={len(split.test)}")

    seeds = [int(s.strip()) for s in args.model_seeds.split(",") if s.strip()]
    all_preds = []
    targets_ref = None
    single_results = []

    for s in seeds:
        print(f"\n训练模型: model_seed={s}")
        preds, targets, metrics = train_one_model(
            graphs=graphs,
            split=split,
            model_seed=s,
            device=device,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        single_results.append(metrics)
        all_preds.append(preds)
        if targets_ref is None:
            targets_ref = targets

        print(f"  单模型 test: MAE={metrics['mae']:.4f}% | RMSE={metrics['rmse']:.4f}% | R2={metrics['r2']:.4f}")

    assert targets_ref is not None
    ens_pred = np.mean(np.stack(all_preds, axis=0), axis=0)
    ens_mae = float(mean_absolute_error(targets_ref, ens_pred))
    ens_rmse = float(np.sqrt(np.mean((targets_ref - ens_pred) ** 2)))
    ens_r2 = float(r2_score(targets_ref, ens_pred))

    print("\n================ Ensemble Result ================")
    for r in single_results:
        print(f"single seed={r['seed']} | R2={r['r2']:.4f} | MAE={r['mae']:.4f}% | RMSE={r['rmse']:.4f}%")
    print(f"ENSEMBLE (avg) | R2={ens_r2:.4f} | MAE={ens_mae:.4f}% | RMSE={ens_rmse:.4f}%")


if __name__ == "__main__":
    main()

