#!/usr/bin/env python3
"""
使用K折交叉验证提升泛化性能，目标R²>0.75
"""

import os
import argparse
import numpy as np
import pandas as pd
import json
from datetime import datetime
from sklearn.model_selection import KFold
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
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--models-per-fold", type=int, default=8)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"K折交叉验证集成 | 折数: {args.n_folds} | 每折模型数: {args.models_per_fold}")
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

    # K折划分
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=SEED)

    # 存储每个样本的预测（用于集成）
    oof_preds = np.zeros(len(y))
    oof_weights = np.zeros(len(y))

    models = []

    print(f"训练{args.n_folds * args.models_per_fold}个模型...")
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n折 {fold_idx + 1}/{args.n_folds} (验证集: {len(val_idx)}样本)")

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        for m_idx in range(args.models_per_fold):
            params = {**BEST_PARAMS, "seed": SEED + fold_idx * 1000 + m_idx * 100}
            if use_gpu:
                params["device"] = "cuda"

            print(f"  模型 {m_idx + 1}/{args.models_per_fold}", end=" ")

            model = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=8000,
                evals=[(dval, "val")],
                verbose_eval=False,
                early_stopping_rounds=300,
            )

            pred = model.predict(dval)
            r2 = r2_score(y_val, pred)

            oof_preds[val_idx] += pred
            oof_weights[val_idx] += 1
            models.append(model)

            print(f"val_R²={r2:.4f}")

    # 平均OOF预测
    oof_preds = oof_preds / oof_weights
    oof_r2 = r2_score(y, oof_preds)

    print(f"\n{'='*70}")
    print(f"交叉验证结果")
    print(f"{'='*70}")
    print(f"OOF R²: {oof_r2:.4f}")

    # 评估整体性能
    mae = mean_absolute_error(y, oof_preds)
    rmse = np.sqrt(mean_squared_error(y, oof_preds))

    print(f"MAE: {mae:.4f}%")
    print(f"RMSE: {rmse:.4f}%")

    # 检查是否达到目标
    if oof_r2 >= TARGET_R2:
        print(f"\n🎉 成功！OOF R² = {oof_r2:.4f} >= {TARGET_R2}")
    else:
        gap = TARGET_R2 - oof_r2
        print(f"\n⚠️  接近！OOF R² = {oof_r2:.4f}, 差距 = {gap:.4f}")

        if gap < 0.01:
            print(f"💡 建议: 增加折数或每折模型数可能突破{TARGET_R2}")
        elif gap < 0.02:
            print(f"💡 建议: 需要特征工程或更复杂的模型架构")

    # 保存结果
    if args.save:
        os.makedirs("results/best_models", exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_data = {
            "created_at": stamp,
            "n_folds": args.n_folds,
            "models_per_fold": args.models_per_fold,
            "total_models": args.n_folds * args.models_per_fold,
            "seed": SEED,
            "oof_r2": float(oof_r2),
            "mae": float(mae),
            "rmse": float(rmse),
            "target_reached": oof_r2 >= TARGET_R2,
        }
        path = f"results/best_models/cv_075_{stamp}.json"
        with open(path, "w") as f:
            json.dump(result_data, f, indent=2)
        print(f"\n已保存: {path}")

    return oof_r2


if __name__ == "__main__":
    main()
