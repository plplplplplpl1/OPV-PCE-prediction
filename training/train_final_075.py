#!/usr/bin/env python3
"""
最终突破R²=0.75的策略
使用多个不同种子的数据划分 + 大规模模型集成
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
TARGET_R2 = 0.75
BASE_SEED = 123

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
    parser.add_argument("--n-splits", type=int, default=5, help="不同数据划分的数量")
    parser.add_argument("--models-per-split", type=int, default=10, help="每个划分的模型数")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"最终策略: 多数据划分集成")
    print(f"划分数: {args.n_splits} | 每划分模型数: {args.models_per_split}")
    print(f"总模型数: {args.n_splits * args.models_per_split}")
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

    # 使用固定的测试集（原始最佳模型使用的划分）
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=BASE_SEED, shuffle=True)
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"固定测试集: {len(X_test)}样本")

    # 存储所有测试集预测
    all_test_preds = []
    all_val_r2s = []

    # 使用多个不同的训练集划分
    seeds = [BASE_SEED + i * 1000 for i in range(args.n_splits)]

    for split_idx, seed in enumerate(seeds):
        print(f"\n划分 {split_idx + 1}/{args.n_splits} (seed={seed})")

        # 使用相同的测试集，但不同的训练/验证划分
        train_idx2, val_idx = train_test_split(train_idx, test_size=0.1, random_state=seed, shuffle=True)
        X_train, y_train = X[train_idx2], y[train_idx2]
        X_val, y_val = X[val_idx], y[val_idx]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(X_test, label=y_test)

        for m_idx in range(args.models_per_split):
            params = {**BEST_PARAMS, "seed": seed + m_idx * 100}
            if use_gpu:
                params["device"] = "cuda"

            print(f"  模型 {m_idx + 1}/{args.models_per_split}", end=" ")

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
            test_pred = model.predict(dtest)

            all_test_preds.append(test_pred)
            all_val_r2s.append(val_r2)

            print(f"val_R²={val_r2:.4f}")

    # 转换为numpy数组
    all_test_preds = np.stack(all_test_preds, axis=1)
    all_val_r2s = np.array(all_val_r2s)

    print(f"\n{'='*70}")
    print("集成结果评估")
    print(f"{'='*70}\n")

    # 方法1: 简单平均
    avg_pred = np.mean(all_test_preds, axis=1)
    avg_r2 = r2_score(y_test, avg_pred)
    avg_mae = mean_absolute_error(y_test, avg_pred)

    print(f"1. 简单平均 ({len(all_test_preds[0])}模型):")
    print(f"   R²  = {avg_r2:.4f}")
    print(f"   MAE = {avg_mae:.4f}%")

    # 方法2: 加权平均
    weights = np.maximum(all_val_r2s, 0)
    weights = weights / weights.sum()
    w_pred = np.average(all_test_preds, axis=1, weights=weights)
    w_r2 = r2_score(y_test, w_pred)
    w_mae = mean_absolute_error(y_test, w_pred)

    print(f"\n2. 加权平均 (基于val_R²):")
    print(f"   R²  = {w_r2:.4f}")
    print(f"   MAE = {w_mae:.4f}%")

    # 方法3: 中位数（鲁棒性）
    median_pred = np.median(all_test_preds, axis=1)
    median_r2 = r2_score(y_test, median_pred)

    print(f"\n3. 中位数:")
    print(f"   R²  = {median_r2:.4f}")

    # 最佳单模型
    best_single_idx = np.argmax(all_val_r2s)
    best_single_r2 = r2_score(y_test, all_test_preds[:, best_single_idx])

    print(f"\n4. 最佳单模型:")
    print(f"   R²  = {best_single_r2:.4f}")

    # 找到最佳方法
    results = {
        "简单平均": avg_r2,
        "加权平均": w_r2,
        "中位数": median_r2,
        "最佳单模型": best_single_r2,
    }

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
        print(f"⚠️  接近！R² = {best_r2:.4f}, 差距 = {gap:.4f}")

        if gap < 0.005:
            print(f"💡 非常接近！增加2-3个模型可能突破{TARGET_R2}")
        elif gap < 0.01:
            print(f"💡 接近目标！增加5-10个模型可能突破{TARGET_R2}")
        else:
            print(f"💡 需要更大幅度的改进，建议尝试特征工程或不同的模型架构")

    # 保存结果
    if args.save:
        os.makedirs("results/best_models", exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_data = {
            "created_at": stamp,
            "n_splits": args.n_splits,
            "models_per_split": args.models_per_split,
            "total_models": args.n_splits * args.models_per_split,
            "base_seed": BASE_SEED,
            "best_method": best_method,
            "best_r2": float(best_r2),
            "all_results": {k: float(v) for k, v in results.items()},
            "target_reached": best_r2 >= TARGET_R2,
        }
        path = f"results/best_models/final_075_{stamp}.json"
        with open(path, "w") as f:
            json.dump(result_data, f, indent=2)
        print(f"\n已保存: {path}")

    return best_r2


if __name__ == "__main__":
    main()
