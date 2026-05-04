#!/usr/bin/env python3
"""
混合架构突破方案
结合XGBoost和神经网络，学习特征交互
"""

import os
import numpy as np
import pandas as pd
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except Exception as e:
    raise SystemExit(f"RDKit 未安装: {e}")

try:
    import xgboost as xgb
except Exception as e:
    raise SystemExit(f"xgboost 未安装: {e}")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except:
    TORCH_AVAILABLE = False

PCE_THRESHOLD = 3.0
SEED = 123
TARGET_R2 = 0.75

BEST_PARAMS = {
    "learning_rate": 0.01182027211805136,
    "max_depth": 6,
    "min_child_weight": 5,
    "subsample": 0.553293647018485,
    "colsample_bytree": 0.8163752824064078,
    "reg_alpha": 8.185079057443645e-06,
    "reg_lambda": 6.550633679214373e-05,
    "gamma": 0.0005908795335845525,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
}


class FeatureInteractionNet(nn.Module):
    """神经网络学习XGBoost预测和原始特征的交互"""

    def __init__(self, xgb_pred_dim, raw_feat_dim, hidden_dim=512):
        super().__init__()

        # 特征交互层
        self.interaction = nn.Sequential(
            nn.Linear(xgb_pred_dim + raw_feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim // 4, 1),
        )

        # 残差连接（直接使用XGBoost预测）
        self.residual_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, xgb_pred, raw_features):
        # 拼接XGBoost预测和原始特征
        combined = torch.cat([xgb_pred, raw_features], dim=1)

        # 通过交互网络
        interaction_out = self.interaction(combined)

        # 残差连接
        residual = self.residual_weight * xgb_pred
        output = interaction_out + residual

        return output.squeeze()


def rdkit_features(smiles: str):
    """提取RDKit特征（4096指纹 + 217描述符）"""
    smiles = str(smiles).strip()
    if not smiles or smiles == "nan":
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=4096)
    fp_arr = np.frombuffer(fp.ToBitString().encode("ascii"), dtype=np.uint8) - ord("0")

    desc = []
    for _, fn in list(Descriptors._descList):
        try:
            v = fn(mol)
            desc.append(float(v) if v is not None and not (isinstance(v, float) and (np.isnan(v) or np.isinf(v))) else 0.0)
        except:
            desc.append(0.0)

    desc_clean = [0.0 if np.isinf(v) or abs(v) > 1e10 else v for v in desc]
    return np.concatenate([fp_arr.astype(np.float64), np.array(desc_clean, dtype=np.float64)])


def load_data():
    """加载数据"""
    df = pd.read_csv("data/data.csv", encoding="latin-1")
    pce_col, smiles_col = df.columns[2], df.columns[-1]

    df[pce_col] = pd.to_numeric(df[pce_col], errors="coerce")
    df[smiles_col] = df[smiles_col].astype(str).str.strip()
    df = df.dropna(subset=[pce_col, smiles_col])
    df = df[df[smiles_col] != "nan"].reset_index(drop=True)
    df = df[df[pce_col] > PCE_THRESHOLD].reset_index(drop=True)

    feats, y = [], []
    for smi, pce in zip(df[smiles_col].tolist(), df[pce_col].tolist()):
        v = rdkit_features(smi)
        if v is not None:
            feats.append(v)
            y.append(float(pce))

    X = np.stack(feats, axis=0)
    y = np.array(y, dtype=np.float32)
    return X, y


def select_important_features(X, y, top_k=500):
    """选择最重要的特征（基于特征重要性）"""
    print(f"特征选择: {X.shape[1]} -> {top_k}")

    # 训练一个临时XGBoost获取特征重要性
    idx = np.arange(len(y))
    train_idx, _ = train_test_split(idx, test_size=0.2, random_state=SEED, shuffle=True)

    X_train = X[train_idx]
    y_train = y[train_idx]

    dtrain = xgb.DMatrix(X_train, label=y_train)

    params = {
        "learning_rate": 0.1,
        "max_depth": 6,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "seed": SEED,
    }

    temp_model = xgb.train(params, dtrain, num_boost_round=200, verbose_eval=False)

    # 获取特征重要性
    importance = temp_model.get_score(importance_type='gain')

    # 排序并选择top-k
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    top_features = [int(f[1:].split('}')[0]) for f, _ in sorted_features[:top_k]]

    return top_features


def train_xgb_ensemble(X_train, y_train, X_val, y_val, n_models=20):
    """训练XGBoost集成"""
    print(f"训练{n_models}个XGBoost模型...")

    base_params = {**BEST_PARAMS}

    try:
        import torch
        if torch.cuda.is_available():
            base_params["device"] = "cuda"
    except:
        pass

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    models = []
    val_preds = []

    for i in range(n_models):
        params = dict(base_params)
        params["seed"] = SEED + i * 100

        # 参数多样性
        if i % 3 == 0:
            params["learning_rate"] = BEST_PARAMS["learning_rate"] * 0.95
        elif i % 3 == 1:
            params["learning_rate"] = BEST_PARAMS["learning_rate"] * 1.05

        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=8000,
            evals=[(dval, "val")],
            verbose_eval=False,
            early_stopping_rounds=300,
        )

        models.append(model)
        val_preds.append(model.predict(dval))

        if (i + 1) % 5 == 0:
            avg_pred = np.mean(val_preds, axis=0)
            current_r2 = r2_score(y_val, avg_pred)
            print(f"  [{i+1}/{n_models}] 当前R²={current_r2:.4f}")

    return models


def main():
    print(f"\n{'='*70}")
    print(f"混合架构突破方案")
    print(f"目标: R² > {TARGET_R2}")
    print(f"{'='*70}\n")

    if not TORCH_AVAILABLE:
        print("PyTorch不可用，无法使用神经网络")
        return

    X, y = load_data()
    print(f"数据: {len(X)}样本, {X.shape[1]}特征\n")

    # 数据划分
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=SEED, shuffle=True)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=SEED, shuffle=True)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"划分: 训练{len(X_train)} | 验证{len(X_val)} | 测试{len(X_test)}\n")

    # 使用PyTorch设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}\n")

    # ============================================
    # 阶段1: 训练XGBoost集成
    # ============================================
    print(f"{'='*70}")
    print(f"阶段1: 训练XGBoost集成")
    print(f"{'='*70}\n")

    xgb_models = train_xgb_ensemble(X_train, y_train, X_val, y_val, n_models=20)

    # 获取XGBoost预测
    dval = xgb.DMatrix(X_val)
    dtest = xgb.DMatrix(X_test)

    xgb_val_preds = np.stack([m.predict(dval) for m in xgb_models], axis=1)
    xgb_test_preds = np.stack([m.predict(dtest) for m in xgb_models], axis=1)

    # XGBoost简单平均
    xgb_val_avg = np.mean(xgb_val_preds, axis=1)
    xgb_test_avg = np.mean(xgb_test_preds, axis=1)

    xgb_val_r2 = r2_score(y_val, xgb_val_avg)
    xgb_test_r2 = r2_score(y_test, xgb_test_avg)

    print(f"\nXGBoost集成结果:")
    print(f"  验证集 R² = {xgb_val_r2:.4f}")
    print(f"  测试集 R² = {xgb_test_r2:.4f}")
    print(f"  测试集 MAE = {mean_absolute_error(y_test, xgb_test_avg):.4f}%")

    # ============================================
    # 阶段2: 特征选择和降维
    # ============================================
    print(f"\n{'='*70}")
    print(f"阶段2: 特征选择")
    print(f"{'='*70}\n")

    # 选择最重要的特征（用于神经网络输入）
    top_features = select_important_features(X_train, y_train, top_k=800)

    X_train_reduced = X_train[:, top_features]
    X_val_reduced = X_val[:, top_features]
    X_test_reduced = X_test[:, top_features]

    print(f"降维后: {X_train_reduced.shape[1]}特征")

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_reduced)
    X_val_scaled = scaler.transform(X_val_reduced)
    X_test_scaled = scaler.transform(X_test_reduced)

    # ============================================
    # 阶段3: 训练混合神经网络
    # ============================================
    print(f"\n{'='*70}")
    print(f"阶段3: 训练混合神经网络")
    print(f"{'='*70}\n")

    # 准备PyTorch数据
    # 输入: XGBoost预测 + 降维后的原始特征
    train_xgb_pred = torch.FloatTensor(xgb_val_avg[:len(X_train)]).to(device)  # 使用训练集的XGBoost预测
    val_xgb_pred = torch.FloatTensor(xgb_val_avg).to(device)
    test_xgb_pred = torch.FloatTensor(xgb_test_avg).to(device)

    # 注意：需要重新计算训练集的XGBoost预测
    dtrain2 = xgb.DMatrix(X_train)
    xgb_train_preds = np.stack([m.predict(dtrain2) for m in xgb_models], axis=1)
    train_xgb_pred = torch.FloatTensor(np.mean(xgb_train_preds, axis=1)).to(device)

    train_feat = torch.FloatTensor(X_train_scaled).to(device)
    val_feat = torch.FloatTensor(X_val_scaled).to(device)
    test_feat = torch.FloatTensor(X_test_scaled).to(device)

    train_y = torch.FloatTensor(y_train).to(device)
    val_y = torch.FloatTensor(y_val).to(device)
    test_y = torch.FloatTensor(y_test).to(device)

    # 创建模型
    model = FeatureInteractionNet(xgb_pred_dim=1, raw_feat_dim=X_train_scaled.shape[1]).to(device)

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # 损失函数
    criterion = nn.HuberLoss(delta=1.0)

    # 训练
    best_val_loss = float('inf')
    best_val_r2 = -1
    patience_counter = 0
    max_patience = 50

    epochs = 500
    batch_size = 64

    for epoch in range(epochs):
        model.train()

        # 小批量训练
        indices = np.random.permutation(len(X_train_scaled))
        train_loss = 0.0

        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i+batch_size]

            batch_xgb = train_xgb_pred[batch_idx].unsqueeze(1)
            batch_feat = train_feat[batch_idx]
            batch_y = train_y[batch_idx]

            optimizer.zero_grad()
            outputs = model(batch_xgb, batch_feat)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * len(batch_idx)

        train_loss /= len(X_train_scaled)

        # 验证
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_xgb_pred.unsqueeze(1), val_feat)
            val_loss = criterion(val_outputs, val_y).item()

            val_pred_np = val_outputs.cpu().numpy()
            val_r2 = r2_score(y_val, val_pred_np)

        # 学习率调度
        scheduler.step(val_loss)

        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_r2 = val_r2
            patience_counter = 0

            # 保存最佳模型
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if (epoch + 1) % 50 == 0:
            test_outputs = model(test_xgb_pred.unsqueeze(1), test_feat)
            test_pred_np = test_outputs.cpu().numpy()
            test_r2 = r2_score(y_test, test_pred_np)

            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  训练损失: {train_loss:.4f}")
            print(f"  验证损失: {val_loss:.4f}")
            print(f"  验证R²:   {val_r2:.4f}")
            print(f"  测试R²:   {test_r2:.4f}")

        if patience_counter >= max_patience:
            print(f"\n早停于epoch {epoch+1}")
            break

    # 加载最佳模型
    model.load_state_dict(best_model_state)

    # ============================================
    # 阶段4: 最终评估
    # ============================================
    print(f"\n{'='*70}")
    print(f"阶段4: 最终评估")
    print(f"{'='*70}\n")

    model.eval()
    with torch.no_grad():
        test_outputs = model(test_xgb_pred.unsqueeze(1), test_feat)
        hybrid_pred = test_outputs.cpu().numpy()

    hybrid_r2 = r2_score(y_test, hybrid_pred)
    hybrid_mae = mean_absolute_error(y_test, hybrid_pred)
    hybrid_rmse = np.sqrt(mean_squared_error(y_test, hybrid_pred))

    print(f"混合模型结果:")
    print(f"  R²  = {hybrid_r2:.4f}")
    print(f"  MAE = {hybrid_mae:.4f}%")
    print(f"  RMSE = {hybrid_rmse:.4f}%")

    print(f"\n对比:")
    print(f"  XGBoost集成: R² = {xgb_test_r2:.4f}")
    print(f"  混合模型:    R² = {hybrid_r2:.4f}")
    print(f"  提升:        ΔR² = {hybrid_r2 - xgb_test_r2:.4f}")

    if hybrid_r2 >= TARGET_R2:
        print(f"\n🎉 成功！R² = {hybrid_r2:.4f} >= {TARGET_R2}")
    else:
        gap = TARGET_R2 - hybrid_r2
        print(f"\n差距: {gap:.4f}")

    # 保存
    os.makedirs("results/best_models", exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    result_data = {
        "created_at": stamp,
        "n_xgb_models": len(xgb_models),
        "selected_features": len(top_features),
        "xgb_result": {
            "r2": float(xgb_test_r2),
            "mae": float(mean_absolute_error(y_test, xgb_test_avg)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, xgb_test_avg)))
        },
        "hybrid_result": {
            "r2": float(hybrid_r2),
            "mae": float(hybrid_mae),
            "rmse": float(hybrid_rmse)
        },
        "improvement": float(hybrid_r2 - xgb_test_r2),
        "target_reached": hybrid_r2 >= TARGET_R2,
    }

    with open(f"results/best_models/hybrid_075_{stamp}.json", "w") as f:
        json.dump(result_data, f, indent=2)

    print(f"已保存: results/best_models/hybrid_075_{stamp}.json")

    return hybrid_r2


if __name__ == "__main__":
    main()
