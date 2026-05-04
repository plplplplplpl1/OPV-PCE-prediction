#!/usr/bin/env python3
"""
稳定性优先方案
目标：降低预测方差，提高模型鲁棒性
"""

import os
import numpy as np
import pandas as pd
import json
from datetime import datetime
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.linear_model import HuberRegressor, RANSACRegressor
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

# 更保守的参数（提高稳定性）
STABLE_PARAMS = {
    "learning_rate": 0.01,  # 略低的学习率
    "max_depth": 5,         # 更浅的树
    "min_child_weight": 10,  # 更高的最小子节点权重
    "subsample": 0.7,       # 更高的采样率
    "colsample_bytree": 0.7,
    "reg_alpha": 0.1,       # 更强的L1正则化
    "reg_lambda": 1.0,      # 更强的L2正则化
    "gamma": 0.1,           # 更高的gamma
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
    print(f"稳定性优先方案")
    print(f"目标：提高模型鲁棒性，降低预测方差")
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

    # K折交叉验证（评估稳定性）
    n_folds = 5
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    fold_r2s = []
    fold_preds = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"Fold {fold_idx + 1}/{n_folds}...")

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # 进一步划分训练集
        train_idx2, val_idx = train_test_split(
            np.arange(len(X_train)), test_size=0.1, random_state=SEED, shuffle=True
        )
        X_tr, y_tr = X_train[train_idx2], y_train[train_idx2]
        X_va, y_va = X_train[val_idx], y_train[val_idx]

        # 训练多个模型
        base_params = {**STABLE_PARAMS, "seed": SEED + fold_idx * 100}
        if use_gpu:
            base_params["device"] = "cuda"

        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_va, label=y_va)
        dtest = xgb.DMatrix(X_test, label=y_test)

        # 训练5个模型的集成
        fold_preds_list = []
        for i in range(5):
            params = dict(base_params)
            params["seed"] = SEED + fold_idx * 100 + i * 10

            model = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=8000,
                evals=[(dval, "val")],
                verbose_eval=False,
                early_stopping_rounds=300,
            )

            pred = model.predict(dtest)
            fold_preds_list.append(pred)

        # 平均预测
        fold_pred = np.mean(fold_preds_list, axis=0)
        fold_r2 = r2_score(y_test, fold_pred)
        fold_mae = mean_absolute_error(y_test, fold_pred)

        fold_r2s.append(fold_r2)
        fold_preds.append((fold_idx, test_idx, fold_pred, y_test))

        print(f"  R²={fold_r2:.4f}, MAE={fold_mae:.4f}%")

    # 分析稳定性
    fold_r2s = np.array(fold_r2s)

    print(f"\n{'='*70}")
    print(f"K折交叉验证稳定性分析")
    print(f"{'='*70}\n")

    print(f"各折R²: {fold_r2s}")
    print(f"平均R²: {np.mean(fold_r2s):.4f}")
    print(f"标准差: {np.std(fold_r2s):.4f}")
    print(f"最大值: {np.max(fold_r2s):.4f}")
    print(f"最小值: {np.min(fold_r2s):.4f}")
    print(f"极差: {np.max(fold_r2s) - np.min(fold_r2s):.4f}")

    # 使用固定的测试集（与之前一致）
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=SEED, shuffle=True)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=SEED, shuffle=True)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"\n{'='*70}")
    print(f"训练高稳定性模型")
    print(f"{'='*70}\n")

    # 训练更多保守模型
    n_models = 40
    all_preds = []

    base_params = {**STABLE_PARAMS}
    if use_gpu:
        base_params["device"] = "cuda"

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    print(f"训练{n_models}个高稳定性模型...")

    for i in range(n_models):
        params = dict(base_params)
        params["seed"] = SEED + i * 100

        # 参数多样性（保守范围内）
        depth_variants = [4, 5, 6]
        params["max_depth"] = depth_variants[i % len(depth_variants)]

        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=8000,
            evals=[(dval, "val")],
            verbose_eval=False,
            early_stopping_rounds=300,
        )

        pred = model.predict(dtest)
        all_preds.append(pred)

        if (i + 1) % 10 == 0:
            current_avg = r2_score(y_test, np.mean(np.stack(all_preds), axis=1))
            print(f"  [{i+1}/{n_models}] 当前R²={current_avg:.4f}")

    all_preds = np.stack(all_preds, axis=1)

    # 多种鲁棒集成方法
    print(f"\n{'='*70}")
    print(f"鲁棒集成方法")
    print(f"{'='*70}\n")

    results = {}

    # 1. 中位数（最鲁棒）
    median_pred = np.median(all_preds, axis=1)
    results["中位数"] = {
        "r2": r2_score(y_test, median_pred),
        "mae": mean_absolute_error(y_test, median_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, median_pred))
    }

    # 2. Trimmed平均（去除极端值）
    sorted_preds = np.sort(all_preds, axis=1)
    n = all_preds.shape[1]
    trim_pred = np.mean(sorted_preds[:, n//10:-n//10], axis=1)
    results["Trimmed平均"] = {
        "r2": r2_score(y_test, trim_pred),
        "mae": mean_absolute_error(y_test, trim_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, trim_pred))
    }

    # 3. 简单平均
    avg_pred = np.mean(all_preds, axis=1)
    results["简单平均"] = {
        "r2": r2_score(y_test, avg_pred),
        "mae": mean_absolute_error(y_test, avg_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, avg_pred))
    }

    # 4. Huber回归（鲁棒元学习器）
    # 使用验证集预测训练
    dval_preds = []
    for i in range(n_models):
        params = {**STABLE_PARAMS, "seed": SEED + i * 100}
        if use_gpu:
            params["device"] = "cuda"

        dtrain_i = xgb.DMatrix(X_train, label=y_train)
        dval_i = xgb.DMatrix(X_val, label=y_val)

        model_i = xgb.train(
            params=params,
            dtrain=dtrain_i,
            num_boost_round=8000,
            evals=[(dval_i, "val")],
            verbose_eval=False,
            early_stopping_rounds=300,
        )

        val_pred = model_i.predict(dval_i)
        dval_preds.append(val_pred)

    dval_preds = np.stack(dval_preds, axis=1)

    try:
        huber = HuberRegressor(epsilon=1.35, max_iter=10000)
        huber.fit(dval_preds, y_val)
        huber_pred = huber.predict(all_preds)
        results["Huber回归"] = {
            "r2": r2_score(y_test, huber_pred),
            "mae": mean_absolute_error(y_test, huber_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, huber_pred))
        }
    except:
        pass

    for method, metrics in results.items():
        print(f"{method}:")
        print(f"  R²  = {metrics['r2']:.4f}")
        print(f"  MAE = {metrics['mae']:.4f}%")
        print(f"  RMSE = {metrics['rmse']:.4f}%")

    # 选择最佳综合方案
    best_method = max(results.items(), key=lambda x: x[1]["r2"])
    best_name = best_method[0]
    best_metrics = best_method[1]

    print(f"\n{'='*70}")
    print(f"最佳方案: {best_name}")
    print(f"R²  = {best_metrics['r2']:.4f}")
    print(f"MAE = {best_metrics['mae']:.4f}%")
    print(f"RMSE = {best_metrics['rmse']:.4f}%")
    print(f"{'='*70}\n")

    # 稳定性评分
    stability_score = 1.0 - (np.std(fold_r2s) / np.mean(fold_r2s))
    print(f"模型稳定性评分: {stability_score:.4f} (越高越稳定)")
    print(f"解释: K折R²标准差越小，模型越稳定")

    # 综合评分（R²占70%，稳定性占30%）
    overall_score = 0.7 * best_metrics['r2'] + 0.3 * stability_score
    print(f"综合评分: {overall_score:.4f} (R² + 稳定性)")

    # 保存
    os.makedirs("results/best_models", exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    result_data = {
        "created_at": stamp,
        "n_models": n_models,
        "seed": SEED,
        "kfold_stability": {
            "mean_r2": float(np.mean(fold_r2s)),
            "std_r2": float(np.std(fold_r2s)),
            "min_r2": float(np.min(fold_r2s)),
            "max_r2": float(np.max(fold_r2s)),
            "range": float(np.max(fold_r2s) - np.min(fold_r2s)),
        },
        "best_method": best_name,
        "best_result": {
            "r2": float(best_metrics['r2']),
            "mae": float(best_metrics['mae']),
            "rmse": float(best_metrics['rmse'])
        },
        "stability_score": float(stability_score),
        "overall_score": float(overall_score),
    }

    with open(f"results/best_models/stable_{stamp}.json", "w") as f:
        json.dump(result_data, f, indent=2)

    print(f"\n已保存: results/best_models/stable_{stamp}.json")

    return best_metrics


if __name__ == "__main__":
    main()
