#!/usr/bin/env python3
"""
Stacking集成训练脚本
结合XGBoost、CatBoost、LightGBM的多模型Stacking集成
使用K折交叉验证生成元特征，Ridge回归作为元学习器
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import json
from datetime import datetime
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.linear_model import Ridge

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
    print(f"警告: xgboost 未安装 - {e}")
    xgb = None

try:
    from catboost import CatBoostRegressor
except Exception as e:
    print(f"警告: catboost 未安装 - {e}")
    CatBoostRegressor = None

try:
    from lightgbm import LGBMRegressor
except Exception as e:
    print(f"警告: lightgbm 未安装 - {e}")
    LGBMRegressor = None

PCE_THRESHOLD = 3.0


def _get_data_path() -> str:
    return "data/data_merged.csv" if os.path.exists("data/data_merged.csv") else "data/data.csv"


def _infer_columns(df: pd.DataFrame) -> tuple[str, str]:
    if df.shape[1] < 3:
        raise ValueError("数据列数不足，无法推断 PCE/SMILES 列")
    return df.columns[2], df.columns[-1]


_ALL_DESC_FUNCS: list = []
try:
    _ALL_DESC_FUNCS = list(Descriptors._descList)
except Exception:
    _ALL_DESC_FUNCS = []


def rdkit_features(
    smiles: str,
    fp_bits: int = 4096,
    fp_radius: int = 2,
    use_all_descriptors: bool = True,
) -> np.ndarray | None:
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
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.TPSA(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.NumRotatableBonds(mol),
                Descriptors.RingCount(mol),
                Descriptors.NumAromaticRings(mol),
                Descriptors.FractionCSP3(mol),
                Descriptors.HeavyAtomCount(mol),
                Descriptors.NHOHCount(mol),
                Descriptors.NOCount(mol),
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
    print(f"特征维度: {X.shape[1]} (fp_bits={fp_bits} + {len(_ALL_DESC_FUNCS) if use_all_descriptors else 12} descriptors)")
    return X, y_arr


class XGBoostWrapper:
    def __init__(self, **params):
        self.params = params
        self.model = None

    def fit(self, X_train, y_train, X_val, y_val):
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        self.model = xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=8000,
            evals=[(dval, "val")],
            verbose_eval=False,
            early_stopping_rounds=300,
        )
        return self

    def predict(self, X):
        return self.model.predict(xgb.DMatrix(X))


class CatBoostWrapper:
    def __init__(self, **params):
        self.params = params
        self.model = None

    def fit(self, X_train, y_train, X_val, y_val):
        self.model = CatBoostRegressor(**self.params)
        self.model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        return self

    def predict(self, X):
        return self.model.predict(X)


class LightGBMWrapper:
    def __init__(self, **params):
        self.params = params
        self.model = None

    def fit(self, X_train, y_train, X_val, y_val):
        self.model = LGBMRegressor(**self.params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[],
        )
        return self

    def predict(self, X):
        return self.model.predict(X)


def get_base_models(use_gpu=True):
    models = {}

    if xgb is not None:
        base_params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "tree_method": "hist",
        }
        if use_gpu:
            base_params["device"] = "cuda"

        models["xgb_default"] = XGBoostWrapper(**{
            **base_params,
            "learning_rate": 0.02,
            "max_depth": 8,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "gamma": 0.0,
        })

        models["xgb_fit"] = XGBoostWrapper(**{
            **base_params,
            "learning_rate": 0.05,
            "max_depth": 6,
            "min_child_weight": 1,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "gamma": 0.0,
        })

        models["xgb_reg"] = XGBoostWrapper(**{
            **base_params,
            "learning_rate": 0.03,
            "max_depth": 9,
            "min_child_weight": 5,
            "subsample": 0.85,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 5.0,
            "gamma": 0.1,
        })

    if CatBoostRegressor is not None:
        task_type = "GPU" if use_gpu else "CPU"
        models["catboost"] = CatBoostWrapper(
            iterations=5000,
            learning_rate=0.03,
            depth=8,
            l2_leaf_reg=3.0,
            loss_function="RMSE",
            eval_metric="RMSE",
            task_type=task_type,
            od_type="Iter",
            od_wait=300,
            verbose=False,
        )

    if LGBMRegressor is not None:
        models["lightgbm"] = LightGBMWrapper(
            n_estimators=3000,
            learning_rate=0.02,
            max_depth=8,
            num_leaves=64,
            min_child_samples=5,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.0,
            reg_lambda=1.0,
            verbose=-1,
        )

    return models


def train_stacking_ensemble(X, y, test_size=0.2, random_state=42, n_folds=5, use_gpu=True, meta_alpha=1.0):
    print(f"\n{'='*60}")
    print("Stacking集成训练")
    print(f"{'='*60}")

    # 划分测试集
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=random_state, shuffle=True)
    X_train_full, y_train_full = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"训练集大小: {len(X_train_full)}, 测试集大小: {len(X_test)}")

    # 获取基模型
    base_models = get_base_models(use_gpu=use_gpu)
    print(f"\n基模型数量: {len(base_models)}")
    for name in base_models.keys():
        print(f"  - {name}")

    # K折交叉验证生成元特征
    print(f"\n使用{n_folds}折交叉验证生成元特征...")
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    # 训练集元特征
    meta_features_train = np.zeros((len(X_train_full), len(base_models)))
    meta_features_test = np.zeros((len(X_test), len(base_models)))

    model_names = list(base_models.keys())
    individual_results = {}

    for i, model_name in enumerate(model_names):
        print(f"\n训练模型: {model_name}")
        model = base_models[model_name]

        # 训练集的K折预测
        fold_preds = []
        for fold_idx, (train_fold_idx, val_fold_idx) in enumerate(kf.split(X_train_full)):
            X_train_fold = X_train_full[train_fold_idx]
            y_train_fold = y_train_full[train_fold_idx]
            X_val_fold = X_train_full[val_fold_idx]
            y_val_fold = y_train_full[val_fold_idx]

            model.fit(X_train_fold, y_train_fold, X_val_fold, y_val_fold)
            fold_pred = model.predict(X_val_fold)
            fold_preds.append((val_fold_idx, fold_pred))
            print(f"  Fold {fold_idx+1}/{n_folds} 完成")

        # 组装训练集预测
        for val_idx, pred in fold_preds:
            meta_features_train[val_idx, i] = pred

        # 在全部训练数据上重新训练并预测测试集
        train_idx, val_idx = train_test_split(
            np.arange(len(X_train_full)), test_size=0.1, random_state=random_state, shuffle=True
        )
        X_tr, y_tr = X_train_full[train_idx], y_train_full[train_idx]
        X_va, y_va = X_train_full[val_idx], y_train_full[val_idx]

        model.fit(X_tr, y_tr, X_va, y_va)
        test_pred = model.predict(X_test)
        meta_features_test[:, i] = test_pred

        # 评估单个模型性能
        mae = mean_absolute_error(y_test, test_pred)
        rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        r2 = r2_score(y_test, test_pred)
        individual_results[model_name] = {"mae": mae, "rmse": rmse, "r2": r2}

        print(f"  测试集性能: MAE={mae:.4f}%, RMSE={rmse:.4f}%, R²={r2:.4f}")

    # 训练元学习器
    print(f"\n{'='*60}")
    print("训练元学习器 (Ridge回归)")
    print(f"{'='*60}")

    # 使用部分训练数据训练元学习器
    meta_train_idx, meta_val_idx = train_test_split(
        np.arange(len(meta_features_train)), test_size=0.2, random_state=random_state
    )

    meta_model = Ridge(alpha=meta_alpha)
    meta_model.fit(meta_features_train[meta_train_idx], y_train_full[meta_train_idx])

    print(f"元学习器权重:")
    for i, name in enumerate(model_names):
        print(f"  {name}: {meta_model.coef_[i]:.4f}")
    print(f"  截距: {meta_model.intercept_:.4f}")

    # 预测
    print(f"\n{'='*60}")
    print("最终评估")
    print(f"{'='*60}")

    # 元学习器预测
    final_pred = meta_model.predict(meta_features_test)
    mae = mean_absolute_error(y_test, final_pred)
    rmse = np.sqrt(mean_squared_error(y_test, final_pred))
    r2 = r2_score(y_test, final_pred)

    print(f"\nStacking集成性能:")
    print(f"  MAE:  {mae:.4f}%")
    print(f"  RMSE: {rmse:.4f}%")
    print(f"  R²:   {r2:.4f}")

    # 简单平均对比
    avg_pred = np.mean(meta_features_test, axis=1)
    avg_mae = mean_absolute_error(y_test, avg_pred)
    avg_rmse = np.sqrt(mean_squared_error(y_test, avg_pred))
    avg_r2 = r2_score(y_test, avg_pred)

    print(f"\n简单平均性能:")
    print(f"  MAE:  {avg_mae:.4f}%")
    print(f"  RMSE: {avg_rmse:.4f}%")
    print(f"  R²:   {avg_r2:.4f}")

    print(f"\n提升幅度:")
    print(f"  相比简单平均: ΔR² = {r2 - avg_r2:+.4f}")

    # 返回结果
    return {
        "meta_model": meta_model,
        "base_models": base_models,
        "individual_results": individual_results,
        "stacking_results": {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        },
        "average_results": {
            "mae": avg_mae,
            "rmse": avg_rmse,
            "r2": avg_r2,
        },
        "meta_weights": dict(zip(model_names, meta_model.coef_.tolist())),
        "meta_intercept": float(meta_model.intercept_),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--n-folds", type=int, default=5, help="交叉验证折数")
    parser.add_argument("--fp-bits", type=int, default=4096, help="指纹位数")
    parser.add_argument("--use-all-desc", action="store_true", help="使用全量RDKit描述符")
    parser.add_argument("--meta-alpha", type=float, default=1.0, help="Ridge回归正则化系数")
    parser.add_argument("--cpu", action="store_true", help="强制使用CPU")
    parser.add_argument("--save-results", action="store_true", help="保存结果到文件")
    args = parser.parse_args()

    print(f"Stacking集成训练脚本")
    print(f"随机种子: {args.seed}")
    print(f"交叉验证折数: {args.n_folds}")
    print(f"指纹位数: {args.fp_bits}")
    print(f"全量描述符: {args.use_all_desc}")

    use_gpu = not args.cpu
    if use_gpu:
        try:
            import torch
            if not torch.cuda.is_available():
                use_gpu = False
        except Exception:
            use_gpu = False

    print(f"使用GPU: {use_gpu}")

    # 加载数据
    X, y = load_high_pce_dataset(
        fp_bits=args.fp_bits,
        fp_radius=2,
        use_all_descriptors=args.use_all_desc
    )

    # 训练Stacking集成
    results = train_stacking_ensemble(
        X, y,
        test_size=0.2,
        random_state=args.seed,
        n_folds=args.n_folds,
        use_gpu=use_gpu,
        meta_alpha=args.meta_alpha
    )

    # 保存结果
    if args.save_results:
        os.makedirs("results/best_models", exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        result_data = {
            "created_at": stamp,
            "config": {
                "seed": args.seed,
                "n_folds": args.n_folds,
                "fp_bits": args.fp_bits,
                "use_all_descriptors": args.use_all_desc,
                "meta_alpha": args.meta_alpha,
                "device": "gpu" if use_gpu else "cpu",
            },
            "individual_performance": results["individual_results"],
            "stacking_performance": results["stacking_results"],
            "average_performance": results["average_results"],
            "meta_weights": results["meta_weights"],
            "meta_intercept": results["meta_intercept"],
        }

        result_path = f"results/best_models/stacking_results_{stamp}.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        print(f"\n已保存结果: {result_path}")

    # 检查是否达到目标
    if results["stacking_results"]["r2"] >= 0.75:
        print(f"\n🎉 目标达成！R² = {results['stacking_results']['r2']:.4f} >= 0.75")
    else:
        print(f"\n⚠️  尚未达到目标。当前R² = {results['stacking_results']['r2']:.4f}, 目标 = 0.75")

    return results


if __name__ == "__main__":
    main()
