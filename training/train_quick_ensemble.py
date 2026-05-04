#!/usr/bin/env python3
"""
快速集成方案 - 使用现有最佳模型+简单加权
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
    raise SystemExit(f"RDKit 未安装或不可用: {e}")

try:
    import xgboost as xgb
except Exception as e:
    raise SystemExit(f"xgboost 未安装或不可用: {e}")

PCE_THRESHOLD = 3.0

# 最佳XGBoost参数
BEST_XGB_PARAMS = {
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
    if df.shape[1] < 3:
        raise ValueError("数据列数不足，无法推断 PCE/SMILES 列")
    return df.columns[2], df.columns[-1]


_ALL_DESC_FUNCS: list = []
try:
    _ALL_DESC_FUNCS = list(Descriptors._descList)
except Exception:
    _ALL_DESC_FUNCS = []


def rdkit_features(smiles: str, fp_bits: int = 4096, fp_radius: int = 2, use_all_descriptors: bool = True) -> np.ndarray | None:
    smiles = str(smiles).strip()
    if not smiles or smiles == "nan":
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=fp_radius, nBits=fp_bits)
    fp_arr = np.frombuffer(fp.ToBitString().encode("ascii"), dtype=np.uint8) - ord("0")

    if use_all_descriptors and _ALL_DESC_FUNCS:
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
    else:
        desc_values = np.array(
            [
                Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol),
                Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol), Descriptors.NumRotatableBonds(mol),
                Descriptors.RingCount(mol), Descriptors.NumAromaticRings(mol), Descriptors.FractionCSP3(mol),
                Descriptors.HeavyAtomCount(mol), Descriptors.NHOHCount(mol), Descriptors.NOCount(mol),
            ],
            dtype=np.float64,
        )

    return np.concatenate([fp_arr.astype(np.float64), desc_values], axis=0)


def load_high_pce_dataset(fp_bits: int = 4096, fp_radius: int = 2, use_all_descriptors: bool = True):
    data_csv = _get_data_path()
    df = pd.read_csv(data_csv, encoding="latin-1")
    pce_col, smiles_col = _infer_columns(df)

    df[pce_col] = pd.to_numeric(df[pce_col], errors="coerce")
    df[smiles_col] = df[smiles_col].astype(str).str.strip()
    df = df.dropna(subset=[pce_col, smiles_col])
    df = df[df[smiles_col] != "nan"].reset_index(drop=True)

    df_high = df[df[pce_col] > PCE_THRESHOLD].reset_index(drop=True)

    feats, y = [], []
    failed = 0
    for smi, pce in zip(df_high[smiles_col].tolist(), df_high[pce_col].tolist()):
        v = rdkit_features(smi, fp_bits=fp_bits, fp_radius=fp_radius, use_all_descriptors=use_all_descriptors)
        if v is None:
            failed += 1
            continue
        feats.append(v)
        y.append(float(pce))

    X = np.stack(feats, axis=0)
    y_arr = np.array(y, dtype=np.float32)
    print(f"数据文件: {data_csv}")
    print(f"高PCE样本 (PCE > {PCE_THRESHOLD}%): {len(df_high)} | 特征成功: {len(y_arr)} | 失败: {failed}")
    print(f"特征维度: {X.shape[1]}")
    return X, y_arr


def train_quick_ensemble(X, y, n_models=10, random_state=42, use_gpu=True):
    print(f"\n{'='*60}")
    print("快速XGBoost集成训练")
    print(f"{'='*60}")

    # 固定数据划分
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=random_state, shuffle=True)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=random_state, shuffle=True)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")

    base_params = {**BEST_XGB_PARAMS}
    if use_gpu:
        base_params["device"] = "cuda"

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # 训练多个模型
    models = []
    val_scores = []
    seeds = [random_state + i * 1000 for i in range(n_models)]

    print(f"\n训练 {n_models} 个XGBoost模型...")
    for i, seed in enumerate(seeds):
        params = {**base_params, "seed": int(seed)}
        print(f"  模型 {i+1}/{n_models} (seed={seed})...", end=" ")

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
        val_scores.append(val_r2)

        models.append(model)
        print(f"val_R²={val_r2:.4f}")

    # 收集预测
    val_predictions = np.stack([m.predict(dval) for m in models], axis=1)
    test_predictions = np.stack([m.predict(dtest) for m in models], axis=1)
    val_scores = np.array(val_scores)

    # 方法1: 简单平均
    avg_pred = np.mean(test_predictions, axis=1)
    avg_mae = mean_absolute_error(y_test, avg_pred)
    avg_rmse = np.sqrt(mean_squared_error(y_test, avg_pred))
    avg_r2 = r2_score(y_test, avg_pred)

    # 方法2: 加权平均（基于验证集R²）
    weights = np.maximum(val_scores, 0)
    weights = weights / weights.sum()
    weighted_pred = np.average(test_predictions, axis=1, weights=weights)
    weighted_mae = mean_absolute_error(y_test, weighted_pred)
    weighted_rmse = np.sqrt(mean_squared_error(y_test, weighted_pred))
    weighted_r2 = r2_score(y_test, weighted_pred)

    # 方法3: 最佳单模型
    best_model_idx = np.argmax(val_scores)
    best_pred = test_predictions[:, best_model_idx]
    best_mae = mean_absolute_error(y_test, best_pred)
    best_rmse = np.sqrt(mean_squared_error(y_test, best_pred))
    best_r2 = r2_score(y_test, best_pred)

    print(f"\n{'='*60}")
    print("结果比较")
    print(f"{'='*60}")

    print(f"\n最佳单模型:")
    print(f"  MAE:  {best_mae:.4f}%")
    print(f"  RMSE: {best_rmse:.4f}%")
    print(f"  R²:   {best_r2:.4f}")

    print(f"\n简单平均 ({n_models}模型):")
    print(f"  MAE:  {avg_mae:.4f}%")
    print(f"  RMSE: {avg_rmse:.4f}%")
    print(f"  R²:   {avg_r2:.4f}")

    print(f"\n加权平均 (基于验证集R²):")
    print(f"  MAE:  {weighted_mae:.4f}%")
    print(f"  RMSE: {weighted_rmse:.4f}%")
    print(f"  R²:   {weighted_r2:.4f}")

    print(f"\n提升幅度:")
    print(f"  加权平均 vs 最佳单模型: ΔR² = {weighted_r2 - best_r2:+.4f}")
    print(f"  简单平均 vs 最佳单模型: ΔR² = {avg_r2 - best_r2:+.4f}")

    return {
        "best_single": {"mae": best_mae, "rmse": best_rmse, "r2": best_r2},
        "average": {"mae": avg_mae, "rmse": avg_rmse, "r2": avg_r2},
        "weighted": {"mae": weighted_mae, "rmse": weighted_rmse, "r2": weighted_r2},
        "n_models": n_models,
        "val_scores": val_scores.tolist(),
        "weights": weights.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-models", type=int, default=15)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--save-results", action="store_true")
    args = parser.parse_args()

    print(f"快速XGBoost集成训练")
    print(f"随机种子: {args.seed}")
    print(f"模型数量: {args.n_models}")

    use_gpu = not args.cpu
    if use_gpu:
        try:
            import torch
            if not torch.cuda.is_available():
                use_gpu = False
        except Exception:
            use_gpu = False

    print(f"使用GPU: {use_gpu}")

    X, y = load_high_pce_dataset(fp_bits=4096, fp_radius=2, use_all_descriptors=True)

    results = train_quick_ensemble(X, y, n_models=args.n_models, random_state=args.seed, use_gpu=use_gpu)

    if args.save_results:
        os.makedirs("results/best_models", exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        result_data = {
            "created_at": stamp,
            "config": {
                "seed": args.seed,
                "n_models": args.n_models,
                "device": "gpu" if use_gpu else "cpu",
            },
            "results": results,
        }

        result_path = f"results/best_models/quick_ensemble_{stamp}.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        print(f"\n已保存结果: {result_path}")

    # 检查目标
    best_r2 = max(results["weighted"]["r2"], results["average"]["r2"])
    if best_r2 >= 0.75:
        print(f"\n🎉 目标达成！R² = {best_r2:.4f} >= 0.75")
    else:
        print(f"\n⚠️  尚未达到目标。当前最佳R² = {best_r2:.4f}, 目标 = 0.75")

    return results


if __name__ == "__main__":
    main()
