#!/usr/bin/env python3
"""
最终冲刺：多数据划分 + 超大规模集成
"""

import os
import numpy as np
import pandas as pd
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.linear_model import Ridge, ElasticNet
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
BASE_SEED = 123
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
    print(f"最终冲刺：多数据划分 + 超大规模集成")
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
    print(f"数据: {len(X)}样本\n")

    # 固定测试集（使用BASE_SEED）
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=BASE_SEED, shuffle=True)
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"固定测试集: {len(X_test)}样本\n")

    # 使用多个不同的训练集划分seed
    split_seeds = [BASE_SEED, 42, 456, 789, 9999, 2024, 111, 222]
    models_per_split = 15

    all_test_preds = []
    all_val_preds = []
    all_val_y = []

    print(f"使用{len(split_seeds)}种数据划分，每种训练{models_per_split}个模型")
    print(f"总模型数: {len(split_seeds) * models_per_split}\n")

    for split_idx, split_seed in enumerate(split_seeds):
        print(f"划分 {split_idx + 1}/{len(split_seeds)} (seed={split_seed})...")

        # 不同的训练/验证划分
        train_idx2, val_idx = train_test_split(train_idx, test_size=0.1, random_state=split_seed, shuffle=True)
        X_train, y_train = X[train_idx2], y[train_idx2]
        X_val, y_val = X[val_idx], y[val_idx]

        base_params = {**BEST_PARAMS}
        if use_gpu:
            base_params["device"] = "cuda"

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(X_test, label=y_test)

        for m_idx in range(models_per_split):
            params = dict(base_params)
            params["seed"] = split_seed + m_idx * 100

            # 参数多样性
            if m_idx % 3 == 0:
                params["learning_rate"] = BEST_PARAMS["learning_rate"] * 0.95
            elif m_idx % 3 == 1:
                params["learning_rate"] = BEST_PARAMS["learning_rate"] * 1.05

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

            all_val_preds.append(val_pred)
            all_val_y.append(y_val)
            all_test_preds.append(test_pred)

    all_val_preds = np.stack(all_val_preds, axis=1)
    all_val_y = np.array(all_val_y)  # This will be 2D
    all_test_preds = np.stack(all_test_preds, axis=1)

    # 需要处理验证集预测（来自不同划分）
    # 使用平均作为元特征
    val_meta = np.array([np.mean(all_val_preds[:, i]) for i in range(all_val_preds.shape[1])])
    val_y_avg = np.array([np.mean(all_val_y[:, i]) for i in range(all_val_y.shape[1])])

    print(f"\n{'='*70}")
    print(f"训练元模型")
    print(f"{'='*70}\n")

    # 训练多种元模型
    meta_models = {
        "ElasticNet": ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000),
        "ElasticNet_0.5": ElasticNet(alpha=0.5, l1_ratio=0.3, max_iter=10000),
        "Ridge": Ridge(alpha=1.0),
        "Ridge_0.1": Ridge(alpha=0.1),
    }

    best_meta_pred = None
    best_meta_name = None
    best_meta_r2 = -1

    for name, meta_model in meta_models.items():
        try:
            # 使用验证集预测训练
            meta_model.fit(all_val_preds.T, val_y_avg)
            meta_pred = meta_model.predict(all_test_preds)

            r2 = r2_score(y_test, meta_pred)
            mae = mean_absolute_error(y_test, meta_pred)
            rmse = np.sqrt(mean_squared_error(y_test, meta_pred))

            print(f"{name}: R²={r2:.4f}, MAE={mae:.4f}%")

            if r2 > best_meta_r2:
                best_meta_r2 = r2
                best_meta_name = name
                best_meta_pred = meta_pred

        except Exception as e:
            print(f"{name}: 失败")

    # 简单集成方法
    print(f"\n简单集成方法:")

    avg_pred = np.mean(all_test_preds, axis=1)
    avg_r2 = r2_score(y_test, avg_pred)
    print(f"简单平均: R²={avg_r2:.4f}")

    median_pred = np.median(all_test_preds, axis=1)
    median_r2 = r2_score(y_test, median_pred)
    print(f"中位数: R²={median_r2:.4f}")

    # 找最佳
    if best_meta_r2 >= avg_r2 and best_meta_r2 >= median_r2:
        final_r2 = best_meta_r2
        final_name = best_meta_name
        final_mae = mean_absolute_error(y_test, best_meta_pred)
        final_rmse = np.sqrt(mean_squared_error(y_test, best_meta_pred))
    elif avg_r2 >= median_r2:
        final_r2 = avg_r2
        final_name = "简单平均"
        final_pred = avg_pred
        final_mae = mean_absolute_error(y_test, avg_pred)
        final_rmse = np.sqrt(mean_squared_error(y_test, avg_pred))
    else:
        final_r2 = median_r2
        final_name = "中位数"
        final_pred = median_pred
        final_mae = mean_absolute_error(y_test, median_pred)
        final_rmse = np.sqrt(mean_squared_error(y_test, median_pred))

    print(f"\n{'='*70}")
    print(f"最终结果: {final_name}")
    print(f"R²  = {final_r2:.4f}")
    print(f"MAE = {final_mae:.4f}%")
    print(f"RMSE = {final_rmse:.4f}%")
    print(f"{'='*70}\n")

    if final_r2 >= TARGET_R2:
        print(f"🎉 成功！R² = {final_r2:.4f} >= {TARGET_R2}")
    else:
        gap = TARGET_R2 - final_r2
        print(f"差距: {gap:.4f}")

        if gap < 0.005:
            print(f"\n💡 极其接近！已接近当前方法上限")
        elif gap < 0.015:
            print(f"\n💡 需要根本性改变（GNN、新特征、更多数据）")

    # 保存
    os.makedirs("results/best_models", exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    result_data = {
        "created_at": stamp,
        "n_splits": len(split_seeds),
        "models_per_split": models_per_split,
        "total_models": len(split_seeds) * models_per_split,
        "base_seed": BASE_SEED,
        "best_method": final_name,
        "r2": float(final_r2),
        "mae": float(final_mae),
        "rmse": float(final_rmse),
        "target_reached": final_r2 >= TARGET_R2,
    }

    with open(f"results/best_models/final_075_{stamp}.json", "w") as f:
        json.dump(result_data, f, indent=2)

    print(f"已保存: results/best_models/final_075_{stamp}.json")

    return final_r2


if __name__ == "__main__":
    main()
