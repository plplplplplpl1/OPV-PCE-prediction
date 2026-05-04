#!/usr/bin/env python3
"""
分析R²瓶颈的具体原因
"""

import os
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import IsolationForest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

    desc_clean = [0.0 if np.isinf(v) or np.isnan(v) or abs(v) > 1e10 else v for v in desc]
    return np.concatenate([fp_arr.astype(np.float64), np.array(desc_clean, dtype=np.float64)])


def load_data_with_smiles():
    df = pd.read_csv("data/data.csv", encoding="latin-1")
    pce_col, smiles_col = df.columns[2], df.columns[-1]

    df[pce_col] = pd.to_numeric(df[pce_col], errors="coerce")
    df[smiles_col] = df[smiles_col].astype(str).str.strip()
    df = df.dropna(subset=[pce_col, smiles_col])
    df = df[df[smiles_col] != "nan"].reset_index(drop=True)
    df = df[df[pce_col] > PCE_THRESHOLD].reset_index(drop=True)

    feats, smiles_list, y = [], [], []
    for _, row in df.iterrows():
        v = rdkit_features(row[smiles_col])
        if v is not None:
            feats.append(v)
            smiles_list.append(row[smiles_col])
            y.append(float(row[pce_col]))

    X = np.stack(feats, axis=0)
    y = np.array(y, dtype=np.float32)
    return X, y, np.array(smiles_list), df


def main():
    print(f"\n{'='*70}")
    print(f"R²瓶颈分析")
    print(f"{'='*70}\n")

    X, y, smiles_list, df = load_data_with_smiles()

    # 固定划分
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=SEED, shuffle=True)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=SEED, shuffle=True)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    test_smiles = smiles_list[test_idx]
    test_pce_true = y_test

    # 训练最佳模型
    params = {**BEST_PARAMS, "seed": SEED}
    try:
        import torch
        if torch.cuda.is_available():
            params["device"] = "cuda"
    except:
        pass

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=8000,
        evals=[(dval, "val")],
        verbose_eval=False,
        early_stopping_rounds=300,
    )

    test_pred = model.predict(dtest)
    r2 = r2_score(y_test, test_pred)

    print(f"测试集R²: {r2:.4f}\n")

    # 分析预测误差
    errors = np.abs(y_test - test_pred)
    residuals = y_test - test_pred

    print(f"{'='*70}")
    print(f"误差分析")
    print(f"{'='*70}\n")

    print(f"平均绝对误差: {np.mean(errors):.4f}%")
    print(f"最大误差: {np.max(errors):.4f}%")
    print(f"误差中位数: {np.median(errors):.4f}%")

    # 找出预测最差的样本
    worst_idx = np.argsort(errors)[-20:]
    print(f"\n预测最差的20个样本:")
    print(f"{'真实PCE':<10} {'预测PCE':<10} {'误差':<10} {'SMILES'}")
    print("-" * 80)

    for idx in worst_idx[::-1]:
        print(f"{y_test[idx]:<10.2f} {test_pred[idx]:<10.2f} {errors[idx]:<10.2f} {test_smiles[idx][:50]}")

    # 按PCE区间分析误差
    print(f"\n{'='*70}")
    print(f"按PCE区间分析")
    print(f"{'='*70}\n")

    pce_ranges = [
        (3, 5, "低PCE (3-5%)"),
        (5, 7, "中低PCE (5-7%)"),
        (7, 9, "中高PCE (7-9%)"),
        (9, 15, "高PCE (>9%)"),
    ]

    for pmin, pmax, label in pce_ranges:
        mask = (y_test >= pmin) & (y_test < pmax)
        if mask.sum() > 0:
            range_r2 = r2_score(y_test[mask], test_pred[mask])
            range_mae = np.mean(errors[mask])
            range_count = mask.sum()
            print(f"{label}: n={range_count:3d}, R²={range_r2:.4f}, MAE={range_mae:.4f}%")

    # 分析是否存在系统性偏差
    print(f"\n{'='*70}")
    print(f"系统性偏差分析")
    print(f"{'='*70}\n")

    # 低估 vs 高估
    underest = residuals > 0  # 预测偏低
    overest = residuals < 0   # 预测偏高

    print(f"低估样本数: {underest.sum()} (平均偏差: {np.mean(residuals[underest]):.4f}%)")
    print(f"高估样本数: {overest.sum()} (平均偏差: {np.mean(residuals[overest]):.4f}%)")

    # 高PCE样本的预测
    high_pce_mask = y_test > 8
    if high_pce_mask.sum() > 0:
        high_pce_r2 = r2_score(y_test[high_pce_mask], test_pred[high_pce_mask])
        high_pce_mae = np.mean(errors[high_pce_mask])
        print(f"\n高PCE样本 (>8%): n={high_pce_mask.sum()}, R²={high_pce_r2:.4f}, MAE={high_pce_mae:.4f}%")

    # 识别离群样本
    print(f"\n{'='*70}")
    print(f"离群样本分析")
    print(f"{'='*70}\n")

    outlier_mask = errors > np.percentile(errors, 90)
    print(f"误差前10%的样本数: {outlier_mask.sum()}")
    print(f"这些样本的平均真实PCE: {np.mean(y_test[outlier_mask]):.2f}%")
    print(f"这些样本的平均预测PCE: {np.mean(test_pred[outlier_mask]):.2f}%")

    # 保存分析结果
    os.makedirs("results/best_models", exist_ok=True)
    analysis = {
        "test_r2": float(r2),
        "mae": float(np.mean(errors)),
        "max_error": float(np.max(errors)),
        "median_error": float(np.median(errors)),
        "worst_predictions": [
            {
                "true_pce": float(y_test[i]),
                "pred_pce": float(test_pred[i]),
                "error": float(errors[i]),
                "smiles": str(test_smiles[i])
            }
            for i in worst_idx[::-1]
        ]
    }

    with open("results/best_models/bottleneck_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)

    print(f"\n分析结果已保存到: results/best_models/bottleneck_analysis.json")

    # 瓶颈结论
    print(f"\n{'='*70}")
    print(f"瓶颈结论")
    print(f"{'='*70}\n")

    print(f"1. 泛化差距: 验证集R²≈0.76 vs 测试集R²={r2:.4f}")
    print(f"2. 误差分布: MAE={np.mean(errors):.4f}%, 最大误差={np.max(errors):.4f}%")
    print(f"3. 样本量: 测试集仅{len(y_test)}个样本，个别异常值影响显著")

    # 计算如果去除最差5%样本后的R²
    sorted_idx = np.argsort(errors)
    trimmed_idx = sorted_idx[:-int(len(y_test) * 0.05)]
    trimmed_r2 = r2_score(y_test[trimmed_idx], test_pred[trimmed_idx])
    print(f"4. 稳健性: 去除最差5%样本后R²={trimmed_r2:.4f}")

    if trimmed_r2 - r2 > 0.01:
        print(f"\n💡 关键发现: 存在少量(~5%)的异常预测样本")
        print(f"   这些样本可能需要特殊处理或额外特征")


if __name__ == "__main__":
    main()
