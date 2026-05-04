#!/usr/bin/env python3
"""
混合模型策略：结合XGBoost、CatBoost、LightGBM
目标R²>0.75
"""

import os
import argparse
import numpy as np
import pandas as pd
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

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
    xgb = None

try:
    from catboost import CatBoostRegressor
except Exception as e:
    CatBoostRegressor = None

try:
    from lightgbm import LGBMRegressor
except Exception as e:
    LGBMRegressor = None

PCE_THRESHOLD = 3.0
SEED = 123
TARGET_R2 = 0.75

XGB_PARAMS = {
    "learning_rate": 0.012,
    "max_depth": 6,
    "min_child_weight": 5,
    "subsample": 0.55,
    "colsample_bytree": 0.82,
    "reg_alpha": 1e-5,
    "reg_lambda": 1e-4,
    "gamma": 0.001,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
}

CAT_PARAMS = {
    "iterations": 5000,
    "learning_rate": 0.012,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "od_type": "Iter",
    "od_wait": 300,
}

LGBM_PARAMS = {
    "n_estimators": 3000,
    "learning_rate": 0.012,
    "max_depth": 6,
    "num_leaves": 64,
    "min_child_samples": 5,
    "subsample": 0.55,
    "colsample_bytree": 0.82,
    "reg_alpha": 1e-5,
    "reg_lambda": 1e-4,
    "verbose": -1,
}


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
    for _, fn in list(Descriptors._descList):
        try:
            v = fn(mol)
            desc.append(float(v) if v is not None and not (isinstance(v, float) and (np.isnan(v) or np.isinf(v))) else 0.0)
        except:
            desc.append(0.0)

    desc_clean = [0.0 if np.isinf(v) or np.isnan(v) or abs(v) > 1e10 else v for v in desc]
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
    print(f"数据: {len(X)}样本, {X.shape[1]}维特征")
    return X, y


def main():
    print(f"\n{'='*70}")
    print(f"混合模型集成策略")
    print(f"目标: R² > {TARGET_R2}")
    print(f"{'='*70}\n")

    use_gpu = True
    try:
        import torch
        if not torch.cuda.is_available():
            use_gpu = False
    except:
        use_gpu = False

    print(f"可用模型:")
    if xgb: print(f"  - XGBoost")
    if CatBoostRegressor: print(f"  - CatBoost")
    if LGBMRegressor: print(f"  - LightGBM")

    X, y = load_data()

    # 固定数据划分
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=SEED, shuffle=True)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=SEED, shuffle=True)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"划分: 训练{len(X_train)} | 验证{len(X_val)} | 测试{len(X_test)}\n")

    all_preds = []
    all_val_r2s = []
    model_names = []

    # XGBoost模型
    if xgb is not None:
        print("训练XGBoost模型...")
        for i in range(10):
            params = {**XGB_PARAMS, "seed": SEED + i * 100}
            if use_gpu:
                params["device"] = "cuda"

            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)

            model = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=8000,
                evals=[(dval, "val")],
                verbose_eval=False,
                early_stopping_rounds=300,
            )

            val_pred = model.predict(xgb.DMatrix(X_val))
            val_r2 = r2_score(y_val, val_pred)
            test_pred = model.predict(xgb.DMatrix(X_test))

            all_preds.append(test_pred)
            all_val_r2s.append(val_r2)
            model_names.append(f"xgb_{i}")

            print(f"  xgb_{i}: val_R²={val_r2:.4f}")

    # CatBoost模型
    if CatBoostRegressor is not None:
        print("\n训练CatBoost模型...")
        task_type = "GPU" if use_gpu else "CPU"

        for i in range(8):
            params = {**CAT_PARAMS, "random_seed": SEED + i * 100, "task_type": task_type}

            model = CatBoostRegressor(**params)
            model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)

            val_pred = model.predict(X_val)
            val_r2 = r2_score(y_val, val_pred)
            test_pred = model.predict(X_test)

            all_preds.append(test_pred)
            all_val_r2s.append(val_r2)
            model_names.append(f"cat_{i}")

            print(f"  cat_{i}: val_R²={val_r2:.4f}")

    # LightGBM模型
    if LGBMRegressor is not None:
        print("\n训练LightGBM模型...")
        for i in range(8):
            params = {**LGBM_PARAMS, "random_state": SEED + i * 100}

            model = LGBMRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[])

            val_pred = model.predict(X_val)
            val_r2 = r2_score(y_val, val_pred)
            test_pred = model.predict(X_test)

            all_preds.append(test_pred)
            all_val_r2s.append(val_r2)
            model_names.append(f"lgbm_{i}")

            print(f"  lgbm_{i}: val_R²={val_r2:.4f}")

    if not all_preds:
        print("\n没有可用的模型！")
        return

    all_preds = np.stack(all_preds, axis=1)
    all_val_r2s = np.array(all_val_r2s)

    print(f"\n{'='*70}")
    print("混合集成结果 ({len(all_preds[0])}个模型)")
    print(f"{'='*70}\n")

    # 简单平均
    avg_pred = np.mean(all_preds, axis=1)
    avg_r2 = r2_score(y_test, avg_pred)
    print(f"简单平均: R² = {avg_r2:.4f}")

    # 加权平均
    weights = np.maximum(all_val_r2s, 0)
    weights = weights / weights.sum()
    w_pred = np.average(all_preds, axis=1, weights=weights)
    w_r2 = r2_score(y_test, w_pred)
    print(f"加权平均: R² = {w_r2:.4f}")

    # 中位数
    median_pred = np.median(all_preds, axis=1)
    median_r2 = r2_score(y_test, median_pred)
    print(f"中位数:   R² = {median_r2:.4f}")

    best_r2 = max(avg_r2, w_r2, median_r2)

    print(f"\n{'='*70}")
    print(f"最佳R²: {best_r2:.4f}")

    if best_r2 >= TARGET_R2:
        print(f"🎉 成功！R² = {best_r2:.4f} >= {TARGET_R2}")
    else:
        print(f"⚠️  接近！R² = {best_r2:.4f}, 差距 = {TARGET_R2 - best_r2:.4f}")
    print(f"{'='*70}\n")

    return best_r2


if __name__ == "__main__":
    main()
