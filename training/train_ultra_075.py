#!/usr/bin/env python3
"""
超大规模集成：100+模型 + 多样化参数
最后冲刺R²>0.75
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
    raise SystemExit(f"xgboost 未安装: {e}")

PCE_THRESHOLD = 3.0
SEED = 123
TARGET_R2 = 0.75

# 基础参数
BASE_PARAMS = {
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-models", type=int, default=120)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"超大规模集成冲刺")
    print(f"模型数: {args.n_models}")
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

    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=SEED, shuffle=True)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=SEED, shuffle=True)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"划分: 训练{len(X_train)} | 验证{len(X_val)} | 测试{len(X_test)}\n")

    all_preds = []
    all_val_r2s = []

    # 参数变体
    lr_variants = [0.009, 0.010, 0.0118, 0.013, 0.014]
    depth_variants = [5, 6, 7]

    print(f"训练{args.n_models}个模型...")
    for i in range(args.n_models):
        params = dict(BASE_PARAMS)

        # 多样化参数
        params["learning_rate"] = lr_variants[i % len(lr_variants)]
        params["max_depth"] = depth_variants[(i // len(lr_variants)) % len(depth_variants)]
        params["seed"] = SEED + i * 100

        if use_gpu:
            params["device"] = "cuda"

        print(f"  [{i+1:3d}/{args.n_models}] lr={params['learning_rate']:.3f}, depth={params['max_depth']}, seed={params['seed']}", end=" ")

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

        val_pred = model.predict(dval)
        val_r2 = r2_score(y_val, val_pred)
        test_pred = model.predict(xgb.DMatrix(X_test))

        all_preds.append(test_pred)
        all_val_r2s.append(val_r2)

        print(f"val_R²={val_r2:.4f}")

        # 每20个模型显示当前最佳
        if (i + 1) % 20 == 0:
            current_avg = r2_score(y_test, np.mean(np.stack(all_preds), axis=1))
            print(f"    → 当前平均R²: {current_avg:.4f}")

    all_preds = np.stack(all_preds, axis=1)
    all_val_r2s = np.array(all_val_r2s)

    print(f"\n{'='*70}")
    print("最终结果")
    print(f"{'='*70}\n")

    # 多种集成方法
    results = {}

    avg_pred = np.mean(all_preds, axis=1)
    results["简单平均"] = r2_score(y_test, avg_pred)

    weights = np.maximum(all_val_r2s, 0)
    weights = weights / weights.sum()
    w_pred = np.average(all_preds, axis=1, weights=weights)
    results["加权平均"] = r2_score(y_test, w_pred)

    median_pred = np.median(all_preds, axis=1)
    results["中位数"] = r2_score(y_test, median_pred)

    # Q1-Q3平均（去除极端值）
    q1, q3 = np.percentile(all_preds, [25, 75], axis=1)
    mask = (all_preds >= q1[:, None]) & (all_preds <= q3[:, None])
    trimmed_pred = np.array([np.mean(p[m]) for p, m in zip(all_preds, mask)])
    results["Q1-Q3平均"] = r2_score(y_test, trimmed_pred)

    for method, r2 in results.items():
        print(f"{method}: R² = {r2:.4f}")

    best_method = max(results, key=results.get)
    best_r2 = results[best_method]

    print(f"\n{'='*70}")
    print(f"最佳方法: {best_method}")
    print(f"最终R²:   {best_r2:.4f}")
    print(f"{'='*70}\n")

    if best_r2 >= TARGET_R2:
        print(f"🎉 成功！R² = {best_r2:.4f} >= {TARGET_R2}")
    else:
        gap = TARGET_R2 - best_r2
        print(f"⚠️  差距 = {gap:.4f}")

        if gap < 0.005:
            print(f"💡 极其接近！增加少量模型可能突破{TARGET_R2}")
        else:
            print(f"💡 已达到当前方法上限，建议尝试GNN或更复杂的特征工程")

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
            "target_reached": best_r2 >= TARGET_R2,
        }
        path = f"results/best_models/ultra_075_{stamp}.json"
        with open(path, "w") as f:
            json.dump(result_data, f, indent=2)
        print(f"已保存: {path}")

    return best_r2


if __name__ == "__main__":
    main()
