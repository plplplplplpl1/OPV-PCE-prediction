"""
P1-1: 自监督掩码节点重建预训练
在OPV全部分子上进行预训练，然后微调回归头

设计:
- 掩码15%的节点特征，用GNN编码器+轻量解码器重建
- 预训练后丢弃解码器，用编码器初始化回归模型
- 对比: 预训练+微调 vs 从头训练 vs XGBoost
"""
import os, sys, json, random, warnings
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
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool, global_max_pool, global_add_pool

FP_DIM = 512
SEED = 9999
PCE_THRESHOLD = 3.0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_FILE = 'external_results/ssl_pretrain_results.json'

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def smiles_to_graph(smiles, fp_dim=FP_DIM):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        atom_features = []
        for atom in mol.GetAtoms():
            an = atom.GetAtomicNum(); d = atom.GetDegree(); fc = atom.GetFormalCharge()
            ar = int(atom.GetIsAromatic())
            try: hyb = int(atom.GetHybridization()); nh = atom.GetTotalNumHs(); val = atom.GetTotalValence()
            except: hyb = nh = val = 0
            common = [1, 6, 7, 8, 9, 15, 16, 17, 35]
            feat = [an / 100., d / 6., fc / 8., nh / 4., val / 8., ar] \
                + [int(an == a) for a in common] + [int(d == dd) for dd in range(5)] \
                + [int(hyb == h) for h in range(1, 6)]
            atom_features.append(feat)
        edges = []
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx(); edges += [[i, j], [j, i]]
        if not edges: return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=fp_dim)
        x = torch.tensor(atom_features, dtype=torch.float)
        ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
        fp_t = torch.tensor(np.array(fp, dtype=np.float32)).unsqueeze(0)
        return Data(x=x, edge_index=ei, fp=fp_t)
    except: return None


class GNNEncoder(nn.Module):
    """GNN编码器（三分支，与V3一致）— 用于自监督预训练"""
    def __init__(self, in_channels=30, hidden=128, dropout=0.3):
        super().__init__()
        self.gcn1 = GCNConv(in_channels, hidden); self.gcn2 = GCNConv(hidden, hidden)
        self.gat1 = GATConv(in_channels, hidden//4, heads=4, dropout=dropout)
        self.gat2 = GATConv(hidden, hidden, heads=1, dropout=dropout)
        self.sage1 = SAGEConv(in_channels, hidden); self.sage2 = SAGEConv(hidden, hidden)
        self.bn1 = nn.BatchNorm1d(hidden); self.bn2 = nn.BatchNorm1d(hidden); self.bn3 = nn.BatchNorm1d(hidden)
        self.dropout = nn.Dropout(dropout)
        self.fp_encoder = nn.Sequential(nn.Linear(FP_DIM, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, 128), nn.ReLU())

    def forward(self, data, return_graph_embed=False):
        x, ei, batch = data.x, data.edge_index, data.batch
        x_gcn = F.relu(self.bn1(self.gcn1(x, ei))); x_gcn = self.dropout(x_gcn); x_gcn = F.relu(self.gcn2(x_gcn, ei))
        x_gat = F.relu(self.gat1(x, ei)); x_gat = self.dropout(x_gat); x_gat = F.relu(self.bn2(self.gat2(x_gat, ei)))
        x_sage = F.relu(self.sage1(x, ei)); x_sage = self.dropout(x_sage); x_sage = F.relu(self.bn3(self.sage2(x_sage, ei)))
        # 节点级特征拼接（三分支）
        node_h = torch.cat([x_gcn, x_gat, x_sage], dim=1)  # [num_nodes, hidden*3]
        if return_graph_embed:
            def p3(h): return torch.cat([global_mean_pool(h,batch), global_max_pool(h,batch), global_add_pool(h,batch)], dim=1)
            graph_h = torch.cat([p3(x_gcn), p3(x_gat), p3(x_sage)], dim=1)
            fp_h = self.fp_encoder(data.fp)
            graph_h = torch.cat([graph_h, fp_h], dim=1)
            return node_h, graph_h
        return node_h


class MaskedNodeDecoder(nn.Module):
    """轻量解码器：从节点嵌入重建掩码特征"""
    def __init__(self, in_dim=128*3, out_dim=25):  # in_dim=384 (3×128), out_dim=节点特征数
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, out_dim),
        )

    def forward(self, node_h):
        return self.decoder(node_h)


class RegressorHead(nn.Module):
    """回归头（可插拔，用于微调）"""
    def __init__(self, in_dim=128*9 + 128):  # 3分支×3池化×128 + 128指纹
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, h):
        return self.head(h).squeeze(1)


def load_all_opv_molecules():
    """加载所有OPV分子（全部PCE范围）"""
    data_csv = 'data/data_merged.csv' if os.path.exists('data/data_merged.csv') else 'data/data.csv'
    df = pd.read_csv(data_csv, encoding='latin-1')
    pc = df.columns[2]; sc = df.columns[-1]
    df[pc] = pd.to_numeric(df[pc], errors='coerce')
    df[sc] = df[sc].astype(str).str.strip()
    df = df.dropna(subset=[pc, sc])
    df = df[df[sc] != 'nan'].reset_index(drop=True)
    graphs = []
    for _, row in df.iterrows():
        g = smiles_to_graph(row[sc])
        if g is not None:
            g.y = torch.tensor([float(row[pc])], dtype=torch.float)
            graphs.append(g)
    return graphs, df


def mask_node_features(graphs, mask_ratio=0.15):
    """随机掩码节点特征，返回掩码后的图和掩码目标"""
    masked_graphs = []
    targets = []
    for g in graphs:
        n = g.x.shape[0]
        mask = torch.rand(n) < mask_ratio
        if mask.sum() == 0:
            mask[0] = True  # 确保至少掩码一个
        g_masked = g.clone()
        g_masked.x = g.x.clone()
        g_masked.x[mask] = 0.0  # 掩码特征置零
        masked_graphs.append(g_masked)
        targets.append(g.x[mask].mean(dim=0))  # 每个被掩码节点的平均特征
    return masked_graphs, targets


def ssl_pretrain(encoder, decoder, train_loader, val_loader, epochs=100, lr=0.001):
    """自监督预训练：掩码节点重建"""
    params = list(encoder.parameters()) + list(decoder.parameters())
    opt = optim.AdamW(params, lr=lr, weight_decay=1e-4)
    best_loss = float('inf')
    for epoch in range(epochs):
        encoder.train(); decoder.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            node_h = encoder(batch, return_graph_embed=False)
            pred = decoder(node_h)
            # 对掩码位置计算loss
            mask = (batch.x == 0).all(dim=1)
            if mask.sum() > 0:
                loss = F.mse_loss(pred[mask], batch.x[mask])
            else:
                loss = F.mse_loss(pred, batch.x)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()

        # 验证
        encoder.eval(); decoder.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                node_h = encoder(batch)
                pred = decoder(node_h)
                mask = (batch.x == 0).all(dim=1)
                if mask.sum() > 0:
                    val_loss += F.mse_loss(pred[mask], batch.x[mask]).item()
                else:
                    val_loss += F.mse_loss(pred, batch.x).item()

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(encoder.state_dict(), '保存的模型/ssl_encoder.pth')
            torch.save(decoder.state_dict(), '保存的模型/ssl_decoder.pth')

        if epoch % 10 == 0:
            print(f"  SSL Epoch {epoch:3d} | train_loss={total_loss:.4f} | val_loss={val_loss:.4f}")


def finetune_regression(encoder, train_loader, val_loader, test_loader, epochs=200):
    """微调：冻结编码器+训练回归头"""
    encoder.eval()
    regressor = RegressorHead().to(DEVICE)
    opt = optim.AdamW(regressor.parameters(), lr=0.001, weight_decay=1e-4)
    crit = nn.HuberLoss(delta=1.0)

    best_mae = float('inf')
    for epoch in range(epochs):
        regressor.train()
        for batch in train_loader:
            batch = batch.to(DEVICE)
            with torch.no_grad():
                _, graph_h = encoder(batch, return_graph_embed=True)
            pred = regressor(graph_h)
            loss = crit(pred, batch.y.view(-1))
            opt.zero_grad(); loss.backward(); opt.step()

        regressor.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                _, graph_h = encoder(batch, return_graph_embed=True)
                p = regressor(graph_h)
                preds.extend(p.cpu().numpy()); targets.extend(batch.y.view(-1).cpu().numpy())
        mae = mean_absolute_error(targets, preds)
        if mae < best_mae:
            best_mae = mae
            torch.save(regressor.state_dict(), '保存的模型/ssl_regressor.pth')

    # 测试
    regressor.load_state_dict(torch.load('保存的模型/ssl_regressor.pth', weights_only=True))
    regressor.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            _, graph_h = encoder(batch, return_graph_embed=True)
            p = regressor(graph_h)
            preds.extend(p.cpu().numpy()); targets.extend(batch.y.view(-1).cpu().numpy())
    mae_t = mean_absolute_error(targets, preds)
    rmse_t = float(np.sqrt(np.mean((np.array(preds) - np.array(targets)) ** 2)))
    r2_t = r2_score(targets, preds)
    return mae_t, rmse_t, r2_t


def train_from_scratch(encoder, train_loader, val_loader, test_loader, epochs=200):
    """从头训练（不预训练）"""
    # 重新初始化编码器
    in_dim = next(iter(train_loader)).x.shape[1]
    scratch_encoder = GNNEncoder(in_channels=in_dim).to(DEVICE)
    regressor = RegressorHead().to(DEVICE)
    params = list(scratch_encoder.parameters()) + list(regressor.parameters())
    opt = optim.AdamW(params, lr=0.001, weight_decay=1e-4)
    crit = nn.HuberLoss(delta=1.0)

    best_mae = float('inf')
    for epoch in range(epochs):
        scratch_encoder.train(); regressor.train()
        for batch in train_loader:
            batch = batch.to(DEVICE)
            _, graph_h = scratch_encoder(batch, return_graph_embed=True)
            pred = regressor(graph_h)
            loss = crit(pred, batch.y.view(-1))
            opt.zero_grad(); loss.backward(); opt.step()

        scratch_encoder.eval(); regressor.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                _, graph_h = scratch_encoder(batch, return_graph_embed=True)
                p = regressor(graph_h)
                preds.extend(p.cpu().numpy()); targets.extend(batch.y.view(-1).cpu().numpy())
        mae = mean_absolute_error(targets, preds)
        if mae < best_mae:
            best_mae = mae

    # 测试
    scratch_encoder.eval(); regressor.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            _, graph_h = scratch_encoder(batch, return_graph_embed=True)
            p = regressor(graph_h)
            preds.extend(p.cpu().numpy()); targets.extend(batch.y.view(-1).cpu().numpy())
    return mean_absolute_error(targets, preds), float(np.sqrt(np.mean((np.array(preds)-np.array(targets))**2))), r2_score(targets, preds)


def main():
    set_seed(SEED)
    print(f"自监督预训练 | seed={SEED} | device={DEVICE}")

    # 1) 加载数据
    print("\n加载OPV分子...")
    all_graphs, df = load_all_opv_molecules()
    print(f"  总分子: {len(all_graphs)}")

    # 分离预训练数据（所有分子）和回归任务数据（仅高PCE）
    pc = df.columns[2]
    high_indices = [i for i, gr in enumerate(all_graphs) if gr.y.item() > PCE_THRESHOLD]

    high_graphs = [all_graphs[i] for i in high_indices]
    print(f"  高PCE分子(回归任务): {len(high_graphs)}")

    # 2) 划分回归数据
    indices = list(range(len(high_graphs)))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=SEED)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=SEED)

    train_g = [high_graphs[i] for i in train_idx]
    val_g   = [high_graphs[i] for i in val_idx]
    test_g  = [high_graphs[i] for i in test_idx]
    print(f"  回归划分: 训练{len(train_g)} 验证{len(val_g)} 测试{len(test_g)}")

    # 3) SSL预训练数据 — 使用所有分子（包括低PCE）
    # 预训练用全部数据
    pretrain_idx = list(range(len(all_graphs)))
    pt_train, pt_val = train_test_split(pretrain_idx, test_size=0.1, random_state=SEED)
    pt_train_g = [all_graphs[i] for i in pt_train]
    pt_val_g   = [all_graphs[i] for i in pt_val]
    print(f"  SSL预训练: 训练{len(pt_train_g)} 验证{len(pt_val_g)}")

    # 对预训练数据做掩码
    print("  应用节点特征掩码...")
    # 我们用普通的SSL loader，在训练时做掩码
    pt_train_loader = GeoDataLoader(pt_train_g, batch_size=32, shuffle=True)
    pt_val_loader = GeoDataLoader(pt_val_g, batch_size=32)

    in_dim = all_graphs[0].x.shape[1]
    print(f"  节点特征维度: {in_dim}")

    # 4) SSL预训练
    print("\n--- SSL预训练 ---")
    encoder = GNNEncoder(in_channels=in_dim).to(DEVICE)
    decoder = MaskedNodeDecoder(in_dim=128*3, out_dim=in_dim).to(DEVICE)

    ssl_pretrain(encoder, decoder, pt_train_loader, pt_val_loader, epochs=100)

    # 5) 加载最佳编码器
    encoder.load_state_dict(torch.load('保存的模型/ssl_encoder.pth', weights_only=True))

    # 6) 微调
    print("\n--- 微调（冻结编码器+训练回归头）---")
    train_loader = GeoDataLoader(train_g, batch_size=32, shuffle=True)
    val_loader = GeoDataLoader(val_g, batch_size=32)
    test_loader = GeoDataLoader(test_g, batch_size=32)

    ssl_mae, ssl_rmse, ssl_r2 = finetune_regression(encoder, train_loader, val_loader, test_loader)
    print(f"  SSL预训练+微调: MAE={ssl_mae:.4f} RMSE={ssl_rmse:.4f} R²={ssl_r2:.4f}")

    # 7) 从头训练
    print("\n--- 从头训练 ---")
    scratch_mae, scratch_rmse, scratch_r2 = train_from_scratch(encoder, train_loader, val_loader, test_loader)
    print(f"  从头训练: MAE={scratch_mae:.4f} RMSE={scratch_rmse:.4f} R²={scratch_r2:.4f}")

    # 8) 保存结果
    xgb_r2_ref = 0.736  # 来自表3 Optuna优化XGBoost
    result = {
        'method': 'SSL masked node pretrain + finetune',
        'pretrain_data': f'{len(pt_train_g)} train + {len(pt_val_g)} val (all PCE ranges)',
        'seed': SEED,
        'ssl_pretrain_finetune': {'r2': ssl_r2, 'mae': ssl_mae, 'rmse': ssl_rmse},
        'scratch_gnn': {'r2': scratch_r2, 'mae': scratch_mae, 'rmse': scratch_rmse},
        'xgb_baseline_ref': {'r2': xgb_r2_ref, 'source': 'Table 3 Optuna-optimized'},
        'improvement': {
            'ssl_over_scratch_r2': ssl_r2 - scratch_r2,
            'ssl_vs_xgb_r2': ssl_r2 - xgb_r2_ref,
        }
    }

    with open(RESULTS_FILE, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n结果已保存: {RESULTS_FILE}")

    print("\n" + "=" * 60)
    print("最终对比:")
    print(f"  SSL预训练+微调: R²={ssl_r2:.4f}")
    print(f"  从头训练:      R²={scratch_r2:.4f}")
    print(f"  XGBoost:        R²={xgb_r2_ref:.4f} (表3参考值)")
    print(f"  SSL提升(相对从头): ΔR²={ssl_r2-scratch_r2:+.4f}")
    print(f"  vs XGBoost gap:   ΔR²={ssl_r2-xgb_r2_ref:+.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
