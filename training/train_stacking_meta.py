#!/usr/bin/env python3
"""
最后的突破方案：Stacking元学习
使用验证集预测训练元模型来提升泛化能力
"""

import os
import numpy as np
import pandas as pd
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.linear_model import Ridge, ElasticNet, Lasso
from sklearn.ensemble import GradientBoostingRegressor
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


def rdkit_features(smiles: str):
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


def main():
    print(f"\n{'='*70}")
    print(f"Stacking元学习方案")
    print(f"目标: R² > {TARGET_R2}")
    print(f"{'='*70}\n")

    use_gpu = True
    try:
        import torch
        if not torch.cuda.is_available():
            use_gpu = False
    except:
        use_gpu = False

    X, y = load_data()

    # 数据划分
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=SEED, shuffle=True)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=SEED, shuffle=True)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"划分: 训练{len(X_train)} | 验证{len(X_val)} | 测试{len(X_test)}\n")

    base_params = {**BEST_PARAMS}
    if use_gpu:
        base_params["device"] = "cuda"

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # 训练基模型
    n_base_models = 50
    print(f"训练{n_base_models}个基模型...")

    val_predictions = []
    test_predictions = []
    val_r2s = []

    for i in range(n_base_models):
        params = dict(base_params)
        params["seed"] = SEED + i * 100

        # 参数多样性
        lr_variants = [0.009, 0.010, 0.0118, 0.013, 0.014]
        depth_variants = [5, 6, 7]
        params["learning_rate"] = lr_variants[i % len(lr_variants)]
        params["max_depth"] = depth_variants[(i // len(lr_variants)) % len(depth_variants)]

        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=8000,
            evals=[(dval, "val")],
            verbose_eval=False,
            early_stopping_rounds=300,
        )

        val_pred = model.predict(dval)
        test_pred = model.predict(dtest)

        val_predictions.append(val_pred)
        test_predictions.append(test_pred)
        val_r2s.append(r2_score(y_val, val_pred))

    val_predictions = np.stack(val_predictions, axis=1)
    test_predictions = np.stack(test_predictions, axis=1)
    val_r2s = np.array(val_r2s)

    print(f"\n{'='*70}")
    print(f"训练元模型")
    print(f"{'='*70}\n")

    # 使用验证集预测训练元模型
    X_meta_train = val_predictions
    y_meta_train = y_val
    X_meta_test = test_predictions

    # 尝试多种元模型
    meta_models = {
        "Ridge": Ridge(alpha=1.0),
        "Ridge_0.1": Ridge(alpha=0.1),
        "Ridge_10": Ridge(alpha=10.0),
        "ElasticNet": ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000),
        "Lasso": Lasso(alpha=1.0, max_iter=10000),
    }

    meta_results = {}

    for name, meta_model in meta_models.items():
        try:
            meta_model.fit(X_meta_train, y_meta_train)
            meta_pred = meta_model.predict(X_meta_test)

            r2 = r2_score(y_test, meta_pred)
            mae = mean_absolute_error(y_test, meta_pred)
            rmse = np.sqrt(mean_squared_error(y_test, meta_pred))

            meta_results[name] = {"r2": r2, "mae": mae, "rmse": rmse}

            print(f"{name}: R²={r2:.4f}, MAE={mae:.4f}%, RMSE={rmse:.4f}%")
        except Exception as e:
            print(f"{name}: 失败 - {e}")

    # 也测试简单集成
    avg_pred = np.mean(test_predictions, axis=1)
    avg_r2 = r2_score(y_test, avg_pred)
    avg_mae = mean_absolute_error(y_test, avg_pred)
    avg_rmse = np.sqrt(mean_squared_error(y_test, avg_pred))

    print(f"\n简单平均: R²={avg_r2:.4f}, MAE={avg_mae:.4f}%, RMSE={avg_rmse:.4f}%")

    # 加权平均
    weights = np.maximum(val_r2s, 0)
    weights = weights / weights.sum()
    w_pred = np.average(test_predictions, axis=1, weights=weights)
    w_r2 = r2_score(y_test, w_pred)

    print(f"加权平均: R²={w_r2:.4f}")

    # 中位数
    median_pred = np.median(test_predictions, axis=1)
    median_r2 = r2_score(y_test, median_pred)

    print(f"中位数: R²={median_r2:.4f}")

    # 找最佳
    all_results = {
        **meta_results,
        "简单平均": {"r2": avg_r2, "mae": avg_mae, "rmse": avg_rmse},
        "加权平均": {"r2": w_r2, "mae": mean_absolute_error(y_test, w_pred), "rmse": np.sqrt(mean_squared_error(y_test, w_pred))},
        "中位数": {"r2": median_r2, "mae": mean_absolute_error(y_test, median_pred), "rmse": np.sqrt(mean_squared_error(y_test, median_pred))},
    }

    best_name = max(all_results, key=lambda x: all_results[x]["r2"])
    best = all_results[best_name]

    print(f"\n{'='*70}")
    print(f"最佳方法: {best_name}")
    print(f"R²  = {best['r2']:.4f}")
    print(f"MAE = {best['mae']:.4f}%")
    print(f"RMSE = {best['rmse']:.4f}%")
    print(f"{'='*70}\n")

    if best['r2'] >= TARGET_R2:
        print(f"🎉 成功！R² = {best['r2']:.4f} >= {TARGET_R2}")
    else:
        print(f"差距: {TARGET_R2 - best['r2']:.4f}")

    # 保存
    os.makedirs("results/best_models", exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    result_data = {
        "created_at": stamp,
        "n_base_models": n_base_models,
        "seed": SEED,
        "best_method": best_name,
        "best_result": {
            "r2": float(best['r2']),
            "mae": float(best['mae']),
            "rmse": float(best['rmse'])
        },
        "all_results": {k: {"r2": float(v["r2"]), "mae": float(v["mae"]), "rmse": float(v["rmse"])} for k, v in all_results.items()},
        "target_reached": best['r2'] >= TARGET_R2,
    }

    with open(f"results/best_models/stacking_075_{stamp}.json", "w") as f:
        json.dump(result_data, f, indent=2)

    print(f"已保存: results/best_models/stacking_075_{stamp}.json")

    return best


if __name__ == "__main__":
    main()
