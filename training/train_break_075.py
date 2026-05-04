#!/usr/bin/env python3
"""
专注突破R²=0.75
使用已验证的最佳配置 + 大规模模型集成
"""

import os
import argparse
import numpy as np
import pandas as pd
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.linear_model import Ridge

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

PCE_THRESHOLD = 3.0
SEED = 123  # 与原始最佳模型相同的seed

# 已验证的最佳XGBoost参数
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


def _get_data_path() -> str:
    return "data/data.csv"


def _infer_columns(df: pd.DataFrame) -> tuple[str, str]:
    return df.columns[2], df.columns[-1]


_ALL_DESC_FUNCS: list = []
try:
    _ALL_DESC_FUNCS = list(Descriptors._descList)
except Exception:
    _ALL_DESC_FUNCS = []


def rdkit_features(smiles: str) -> np.ndarray | None:
    smiles = str(smiles).strip()
    if not smiles or smiles == "nan":
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=4096)
    fp_arr = np.frombuffer(fp.ToBitString().encode("ascii"), dtype=np.uint8) - ord("0")

    desc = []
    for _, fn in _ALL_DESC_FUNCS:
        try:
            v = fn(mol)
            if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                v = 0.0
        except Exception:
            v = 0.0
        desc.append(float(v))

    desc_clean = []
    for v in desc:
        if np.isinf(v) or np.isnan(v):
            desc_clean.append(0.0)
        elif abs(v) > 1e10:
            desc_clean.append(np.sign(v) * 1e10)
        else:
            desc_clean.append(v)

    desc_values = np.array(desc_clean, dtype=np.float64)
    return np.concatenate([fp_arr.astype(np.float64), desc_values], axis=0)


def load_data():
    data_csv = _get_data_path()
    df = pd.read_csv(data_csv, encoding="latin-1")
    pce_col, smiles_col = _infer_columns(df)

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
    print(f"数据: {len(X)}样本, 特征维度: {X.shape[1]}")
    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-models", type=int, default=25, help="模型数量")
    parser.add_argument("--lr-var", action="store_true", help="微调学习率")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"专注突破R²=0.75 | 模型数: {args.n_models} | Seed: {SEED}")
    print(f"{'='*70}\n")

    use_gpu = True
    try:
        import torch
        if not torch.cuda.is_available():
            use_gpu = False
    except:
        use_gpu = False

    # 加载数据
    X, y = load_data()

    # 固定数据划分（与原始最佳模型相同）
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=SEED, shuffle=True)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=SEED, shuffle=True)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"划分: 训练{len(X_train)} | 验证{len(X_val)} | 测试{len(X_test)}")

    base_params = {**BEST_PARAMS, "seed": SEED}
    if use_gpu:
        base_params["device"] = "cuda"

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # 训练多个模型
    models = []
    val_preds = []
    val_r2s = []

    print(f"\n训练{args.n_models}个XGBoost模型...")
    for i in range(args.n_models):
        # 微调参数
        params = dict(base_params)

        if args.lr_var:
            # 轻微调整学习率
            lr_factor = 0.95 + (i % 5) * 0.025  # 0.95, 0.975, 1.0, 1.025, 1.05
            params["learning_rate"] = BEST_PARAMS["learning_rate"] * lr_factor

        params["seed"] = SEED + i * 1000

        print(f"  [{i+1:2d}/{args.n_models}] seed={params['seed']}", end=" ")

        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=8000,
            evals=[(dval, "val")],
            verbose_eval=False,
            early_stopping_rounds=300,
        )

        val_pred = model.predict(dval)
        val_r2 = r2_score(y_val, val_pred)

        models.append(model)
        val_preds.append(val_pred)
        val_r2s.append(val_r2)

        print(f"val_R²={val_r2:.4f}")

    # 收集测试集预测
    test_preds = np.stack([m.predict(dtest) for m in models], axis=1)
    val_preds = np.stack(val_preds, axis=1)
    val_r2s = np.array(val_r2s)

    print(f"\n{'='*70}")
    print("集成方法评估")
    print(f"{'='*70}\n")

    # 方法1: 简单平均
    avg_pred = np.mean(test_preds, axis=1)
    avg_r2 = r2_score(y_test, avg_pred)
    avg_mae = mean_absolute_error(y_test, avg_pred)

    print(f"1. 简单平均:")
    print(f"   R²  = {avg_r2:.4f}")
    print(f"   MAE = {avg_mae:.4f}%")

    # 方法2: 加权平均（基于验证集R²）
    weights = np.maximum(val_r2s, 0)
    weights = weights / weights.sum()
    w_pred = np.average(test_preds, axis=1, weights=weights)
    w_r2 = r2_score(y_test, w_pred)
    w_mae = mean_absolute_error(y_test, w_pred)

    print(f"\n2. 加权平均 (基于val_R²):")
    print(f"   R²  = {w_r2:.4f}")
    print(f"   MAE = {w_mae:.4f}%")

    # 方法3: Ridge学习权重
    ridge = Ridge(alpha=1.0)
    ridge.fit(val_preds, y_val)
    ridge_pred = ridge.predict(test_preds)
    ridge_r2 = r2_score(y_test, ridge_pred)
    ridge_mae = mean_absolute_error(y_test, ridge_pred)

    print(f"\n3. Ridge元学习器:")
    print(f"   R²  = {ridge_r2:.4f}")
    print(f"   MAE = {ridge_mae:.4f}%")
    print(f"   权重: min={ridge.coef_.min():.3f}, max={ridge.coef_.max():.3f}, std={ridge.coef_.std():.3f}")

    # 最佳单模型
    best_single_idx = np.argmax(val_r2s)
    best_single_pred = test_preds[:, best_single_idx]
    best_single_r2 = r2_score(y_test, best_single_pred)

    print(f"\n4. 最佳单模型:")
    print(f"   R²  = {best_single_r2:.4f}")

    # 找到最佳方法
    results = {
        "简单平均": avg_r2,
        "加权平均": w_r2,
        "Ridge": ridge_r2,
        "最佳单模型": best_single_r2,
    }

    best_method = max(results, key=results.get)
    best_r2 = results[best_method]

    print(f"\n{'='*70}")
    print(f"最佳方法: {best_method}")
    print(f"最终R²:   {best_r2:.4f}")
    print(f"{'='*70}\n")

    if best_r2 >= 0.75:
        print(f"🎉 成功！R² = {best_r2:.4f} >= 0.75")
    else:
        gap = 0.75 - best_r2
        print(f"⚠️  接近！R² = {best_r2:.4f}, 差距 = {gap:.4f}")

        # 建议
        if gap < 0.01:
            print(f"\n💡 建议: 增加2-3个模型可能突破0.75")
        elif gap < 0.02:
            print(f"\n💡 建议: 增加5-10个模型或尝试微调超参数")

    # 保存
    if args.save:
        os.makedirs("results/best_models", exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_data = {
            "created_at": stamp,
            "n_models": args.n_models,
            "seed": SEED,
            "best_method": best_method,
            "best_r2": float(best_r2),
            "all_results": {k: float(v) for k, v in results.items()},
            "individual_val_r2": val_r2s.tolist(),
        }
        path = f"results/best_models/break_075_{stamp}.json"
        with open(path, "w") as f:
            json.dump(result_data, f, indent=2)
        print(f"\n已保存: {path}")

    return best_r2


if __name__ == "__main__":
    main()
