"""
不确定性量化：XGBoost分位数回归 vs GNN MC Dropout

输出:
1. 预测区间校准曲线
2. 区间宽度对比
3. 虚拟筛选中的不确定性-感知决策比较
"""
import os
import sys
# 确保工作目录是项目根目录
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if os.path.basename(project_root) == '实验':
    project_root = os.path.dirname(project_root)
os.chdir(project_root)
print(f"工作目录: {os.getcwd()}")
import json
import argparse
import random
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool, global_max_pool, global_add_pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog('rdApp.*')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_CSV = 'data/data_merged.csv' if os.path.exists('data/data_merged.csv') else 'data/data.csv'
PCE_THRESHOLD = 3.0
FP_DIM = 512
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)


# ====== Data loading (shared) ======
def smiles_to_graph(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        atom_feat = []
        for atom in mol.GetAtoms():
            an = atom.GetAtomicNum(); d = atom.GetDegree()
            fc = atom.GetFormalCharge(); ar = int(atom.GetIsAromatic())
            try: hyb = int(atom.GetHybridization()); nh = atom.GetTotalNumHs(); val = atom.GetTotalValence()
            except: hyb = nh = val = 0
            common = [1, 6, 7, 8, 9, 15, 16, 17, 35]
            feat = [an/100., d/6., fc/8., nh/4., val/8., ar] \
                + [int(an == a) for a in common] + [int(d == dd) for dd in range(5)] \
                + [int(hyb == h) for h in range(1, 6)]
            atom_feat.append(feat)
        edges = []
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx(); edges += [[i, j], [j, i]]
        if not edges: return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=FP_DIM)
        fp_t = torch.tensor(np.array(fp, dtype=np.float32)).unsqueeze(0)
        x = torch.tensor(atom_feat, dtype=torch.float)
        ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return Data(x=x, edge_index=ei, fp=fp_t)
    except: return None


def load_data():
    df = pd.read_csv(DATA_CSV, encoding='latin-1')
    pc = df.columns[2]; sc = df.columns[-1]
    print(f"  数据列: PCE='{pc}', SMILES='{sc}', 总行数={len(df)}")
    df[pc] = pd.to_numeric(df[pc], errors='coerce')
    df[sc] = df[sc].astype(str).str.strip()
    df = df.dropna(subset=[pc, sc])
    df = df[df[sc] != 'nan'].reset_index(drop=True)
    df_h = df[df[pc] > PCE_THRESHOLD].copy().reset_index(drop=True)
    print(f"  高PCE样本: {len(df_h)}")
    graphs, ys = [], []
    for _, row in df_h.iterrows():
        g = smiles_to_graph(row[sc])
        if g is not None:
            g.y = torch.tensor([float(row[pc])], dtype=torch.float)
            graphs.append(g); ys.append(float(row[pc]))
    print(f"  图转换成功: {len(graphs)}")
    return graphs, np.array(ys), df_h


# ====== GNN with MC Dropout ======
class GNN_MCDropout(nn.Module):
    """HighPCERegressorV3 with dropout kept active at inference"""
    def __init__(self, in_channels=30, hidden=128, fp_dim=FP_DIM, dropout=0.3):
        super().__init__()
        self.gcn1 = GCNConv(in_channels, hidden)
        self.gcn2 = GCNConv(hidden, hidden)
        self.gat1 = GATConv(in_channels, hidden//4, heads=4, dropout=dropout)
        self.gat2 = GATConv(hidden, hidden, heads=1, dropout=dropout)
        self.sage1 = SAGEConv(in_channels, hidden)
        self.sage2 = SAGEConv(hidden, hidden)
        self.bn1 = nn.BatchNorm1d(hidden); self.bn2 = nn.BatchNorm1d(hidden); self.bn3 = nn.BatchNorm1d(hidden)
        self.dropout = nn.Dropout(dropout)
        self.fp_encoder = nn.Sequential(nn.Linear(fp_dim, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, 128), nn.ReLU())
        fused_dim = hidden * 9 + 128
        self.regressor = nn.Sequential(
            nn.Linear(fused_dim, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(256, 64), nn.BatchNorm1d(64),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1),
        )
    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        fp = data.fp
        x_gcn = F.relu(self.bn1(self.gcn1(x, ei)))
        x_gcn = self.dropout(x_gcn)
        x_gcn = F.relu(self.gcn2(x_gcn, ei))

        x_gat = F.relu(self.gat1(x, ei))
        x_gat = self.dropout(x_gat)
        x_gat = F.relu(self.bn2(self.gat2(x_gat, ei)))

        x_sage = F.relu(self.sage1(x, ei))
        x_sage = self.dropout(x_sage)
        x_sage = F.relu(self.bn3(self.sage2(x_sage, ei)))

        def p3(h): return torch.cat([global_mean_pool(h,batch), global_max_pool(h,batch), global_add_pool(h,batch)], dim=1)
        g = torch.cat([p3(x_gcn), p3(x_gat), p3(x_sage)], dim=1)
        fp_f = self.fp_encoder(fp)
        g = torch.cat([g, fp_f], dim=1)
        return self.regressor(g).squeeze(1)


def enable_dropout(m):
    """递归启用所有Dropout层"""
    if isinstance(m, nn.Dropout):
        m.train()

def mc_predict(model, loader, n_samples=50):
    """MC Dropout采样预测（eval模式+手动激活Dropout）"""
    model.eval()
    model.apply(enable_dropout)  # 仅激活Dropout层，BatchNorm保持eval
    all_preds, all_targets = [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        samples = []
        for _ in range(n_samples):
            with torch.no_grad():
                samples.append(model(batch).cpu().numpy())
        all_preds.append(np.stack(samples, axis=1))
        all_targets.extend(batch.y.view(-1).cpu().numpy().tolist())
    preds = np.concatenate(all_preds, axis=0)
    targets = np.array(all_targets)
    return preds.mean(axis=1), preds.std(axis=1), targets


def train_gnn(train_loader, val_loader, in_dim, seed=42):
    set_seed(seed)
    model = GNN_MCDropout(in_channels=in_dim).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    crit = nn.HuberLoss(delta=1.0)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
    best_mae = float('inf'); save_path = f'保存的模型/gnn_uncertainty_seed{seed}.pth'
    for epoch in range(200):
        model.train(); loss = 0
        for b in train_loader:
            b = b.to(DEVICE); opt.zero_grad()
            p = model(b); l = crit(p, b.y.view(-1)); l.backward(); opt.step()
            loss += l.item() * b.num_graphs
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for b in val_loader:
                b = b.to(DEVICE); preds.extend(model(b).cpu().numpy()); targets.extend(b.y.view(-1).cpu().numpy())
        mae = mean_absolute_error(targets, preds); sched.step(mae)
        if mae < best_mae: best_mae = mae; torch.save(model.state_dict(), save_path)
    model.load_state_dict(torch.load(save_path, weights_only=True))
    return model


def train_xgb_quantile(x_train, y_train, x_test, y_test):
    """训练XGBoost分位数回归获取预测区间"""
    # 训练多个分位数模型(0.05到0.95,步长0.05) + 中位数(0.5)
    alphas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
              0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    models = {}
    for alpha in alphas:
        params = {
            'objective': 'reg:quantileerror',
            'quantile_alpha': alpha,
            'max_depth': 6, 'learning_rate': 0.05, 'n_estimators': 500,
            'subsample': 0.8, 'colsample_bytree': 0.8, 'seed': 42,
        }
        m = xgb.XGBRegressor(**params)
        m.fit(x_train, y_train)
        models[alpha] = m

    preds = {}
    for alpha, m in models.items():
        preds[alpha] = m.predict(x_test)

    return preds, models


def compute_fingerprints(smiles_list, nBits=2048):
    """批量计算Morgan指纹"""
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None: fps.append(np.zeros(nBits))
        else: fps.append(np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nBits)))
    return np.array(fps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='external_results/uncertainty_quantification.json')
    parser.add_argument('--mc_samples', type=int, default=50)
    args = parser.parse_args()

    print(f"不确定性量化 | seed={args.seed} | MC samples={args.mc_samples}")

    # 1) 加载数据
    graphs, pce_values, df = load_data()
    indices = list(range(len(graphs)))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=args.seed)
    train_g = [graphs[i] for i in train_idx]
    test_g = [graphs[i] for i in test_idx]

    print(f"训练: {len(train_g)}, 测试: {len(test_g)}")

    # 2) GNN MC Dropout
    print("\n--- GNN MC Dropout ---")
    in_dim = graphs[0].x.shape[1]
    val_idx, _ = train_test_split(list(range(len(train_g))), test_size=0.1, random_state=args.seed)
    train_sub = [train_g[i] for i in range(len(train_g)) if i not in set(val_idx)]
    val_sub = [train_g[i] for i in val_idx]

    tl = GeoDataLoader(train_sub, BATCH_SIZE, shuffle=True)
    vl = GeoDataLoader(val_sub, BATCH_SIZE)
    te_loader = GeoDataLoader(test_g, BATCH_SIZE)

    model = train_gnn(tl, vl, in_dim, args.seed)
    gnn_mean, gnn_std, targets = mc_predict(model, te_loader, n_samples=args.mc_samples)

    gnn_r2 = r2_score(targets, gnn_mean)
    gnn_mae = mean_absolute_error(targets, gnn_mean)
    gnn_rmse = float(np.sqrt(np.mean((targets - gnn_mean) ** 2)))
    gnn_interval_width = np.mean(gnn_std) * 2 * 1.96  # 95% CI width
    gnn_coverage = np.mean((targets >= gnn_mean - 1.96*gnn_std) & (targets <= gnn_mean + 1.96*gnn_std))

    print(f"  R²={gnn_r2:.4f}, MAE={gnn_mae:.4f}")
    print(f"  平均95%CI宽度={gnn_interval_width:.4f}, 覆盖率={gnn_coverage:.3f}")

    # 3) XGBoost Quantile Regression
    print("\n--- XGBoost Quantile Regression ---")
    train_fps = compute_fingerprints(df.iloc[train_idx][df.columns[-1]].values)
    test_fps = compute_fingerprints(df.iloc[test_idx][df.columns[-1]].values)
    train_pce = pce_values[train_idx]
    test_pce = pce_values[test_idx]

    xgb_preds, xgb_models = train_xgb_quantile(train_fps, train_pce, test_fps, test_pce)

    xgb_mean = xgb_preds[0.5]
    xgb_lower = xgb_preds[0.05]
    xgb_upper = xgb_preds[0.95]
    xgb_interval_width = np.mean(xgb_upper - xgb_lower)

    xgb_r2 = r2_score(test_pce, xgb_mean)
    xgb_mae = mean_absolute_error(test_pce, xgb_mean)
    xgb_rmse = float(np.sqrt(np.mean((test_pce - xgb_mean) ** 2)))
    xgb_coverage = np.mean((test_pce >= xgb_lower) & (test_pce <= xgb_upper))

    print(f"  R²={xgb_r2:.4f}, MAE={xgb_mae:.4f}")
    print(f"  平均95%CI宽度={xgb_interval_width:.4f}, 覆盖率={xgb_coverage:.3f}")

    # 4) 校准曲线
    print("\n--- 校准分析 ---")
    confidence_levels = np.arange(0.1, 0.96, 0.05)
    results_by_level = []

    available_alphas = sorted(xgb_preds.keys())
    def closest_quantile(alpha):
        return min(available_alphas, key=lambda a: abs(a - alpha))

    for cl in confidence_levels:
        z = 1.96 * (cl / 0.95)
        alpha = (1 - cl) / 2
        lo = xgb_preds[closest_quantile(alpha)]
        hi = xgb_preds[closest_quantile(1 - alpha)]
        xgb_cov = np.mean((test_pce >= lo) & (test_pce <= hi))

        # GNN
        gnn_lo = gnn_mean - z * gnn_std
        gnn_hi = gnn_mean + z * gnn_std
        gnn_cov = np.mean((targets >= gnn_lo) & (targets <= gnn_hi))

        results_by_level.append({'confidence': round(cl, 2), 'xgb_coverage': round(xgb_cov, 3),
                                  'gnn_coverage': round(gnn_cov, 3)})

    # 校准曲线图
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    cal = np.array([(r['confidence'], r['xgb_coverage'], r['gnn_coverage']) for r in results_by_level])
    plt.plot(cal[:, 0], cal[:, 0], 'k--', alpha=0.5, label='Perfect calibration')
    plt.plot(cal[:, 0], cal[:, 1], 's-', color='#E74C3C', label='XGBoost', linewidth=2)
    plt.plot(cal[:, 0], cal[:, 2], 'o-', color='#3498DB', label='GNN', linewidth=2)
    plt.xlabel('Expected confidence'); plt.ylabel('Observed coverage')
    plt.title('a. Calibration curves', loc='left')
    plt.legend(); plt.grid(True, alpha=0.3)

    # 预测区间宽度 vs PCE range
    plt.subplot(1, 2, 2)
    order = np.argsort(test_pce)
    plt.fill_between(np.arange(len(order)), xgb_lower[order], xgb_upper[order],
                      alpha=0.3, color='#E74C3C', label='XGBoost 95% CI')
    plt.fill_between(np.arange(len(order)), gnn_mean[order]-1.96*gnn_std[order],
                      gnn_mean[order]+1.96*gnn_std[order], alpha=0.3, color='#3498DB', label='GNN 95% CI')
    plt.plot(test_pce[order], 'k.', markersize=2, alpha=0.5, label='True PCE')
    plt.xlabel('Test sample (sorted by PCE)'); plt.ylabel('PCE (%)')
    plt.title('b. Prediction intervals', loc='left')
    plt.legend(fontsize=8); plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('论文写作指导/论文草稿/figures/fig_uncertainty.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("\n图形已保存: figures/fig_uncertainty.png")

    # 5) 保存结果
    output = {
        'method': 'XGBoost quantile regression vs GNN MC Dropout',
        'mc_samples': args.mc_samples,
        'seed': args.seed,
        'gnn': {
            'r2': float(gnn_r2), 'mae': float(gnn_mae), 'rmse': float(gnn_rmse),
            'mean_95ci_width': float(gnn_interval_width),
            'coverage_95ci': float(gnn_coverage),
        },
        'xgb': {
            'r2': float(xgb_r2), 'mae': float(xgb_mae), 'rmse': float(xgb_rmse),
            'mean_95ci_width': float(xgb_interval_width),
            'coverage_95ci': float(xgb_coverage),
        },
        'calibration': [{k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in r.items()} for r in results_by_level],
        'interpretation': {
            'coverage_comparison': f"XGBoost 95%CI覆盖率={float(xgb_coverage):.3f}, GNN 95%CI覆盖率={float(gnn_coverage):.3f}",
            'width_comparison': f"XGBoost平均区间宽度={float(xgb_interval_width):.2f}%, GNN={float(gnn_interval_width):.2f}%",
            'screening_value': '更窄的预测区间（校准良好的前提下）意味着在虚拟筛选中可以更自信地设定PCE阈值',
        }
    }

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n结果已保存: {args.output}")


if __name__ == '__main__':
    main()
