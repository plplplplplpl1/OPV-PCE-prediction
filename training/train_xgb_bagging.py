#!/usr/bin/env python3
"""
XGBoost大规模Bagging集成训练
训练多个不同配置的XGBoost模型并实现智能加权集成
"""

import os
import argparse
import numpy as np
import pandas as pd
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
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
    raise SystemExit(f"xgboost 未安装或不可用: {e}")

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


def get_model_presets(use_gpu=True):
    base = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
    }
    if use_gpu:
        base["device"] = "cuda"

    return {
        "default": {
            **base,
            "learning_rate": 0.02,
            "max_depth": 8,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "gamma": 0.0,
        },
        "fit_v1": {
            **base,
            "learning_rate": 0.05,
            "max_depth": 6,
            "min_child_weight": 1,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "gamma": 0.0,
        },
        "reg_v1": {
            **base,
            "learning_rate": 0.03,
            "max_depth": 9,
            "min_child_weight": 5,
            "subsample": 0.85,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 5.0,
            "gamma": 0.1,
        },
        "deep": {
            **base,
            "learning_rate": 0.015,
            "max_depth": 10,
            "min_child_weight": 3,
            "subsample": 0.75,
            "colsample_bytree": 0.75,
            "reg_alpha": 0.1,
            "reg_lambda": 2.0,
            "gamma": 0.05,
        },
        "shallow": {
            **base,
            "learning_rate": 0.04,
            "max_depth": 5,
            "min_child_weight": 2,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.0,
            "reg_lambda": 0.5,
            "gamma": 0.0,
        },
    }


def train_xgb_bagging(
    X, y,
    split_seed=42,
    model_seeds=None,
    presets=None,
    use_gpu=True,
    weighting_method="performance"
):
    """
    XGBoost大规模Bagging训练

    参数:
        X: 特征矩阵
        y: 目标值
        split_seed: 数据划分种子
        model_seeds: 模型训练种子列表
        presets: 参数预设列表
        use_gpu: 是否使用GPU
        weighting_method: 集成权重方法 ('simple', 'performance', 'ridge')
    """
    print(f"\n{'='*60}")
    print("XGBoost大规模Bagging集成训练")
    print(f"{'='*60}")

    # 默认配置
    if model_seeds is None:
        model_seeds = [42, 123, 456, 789, 9999, 2024, 2026, 111, 222, 333, 444, 555, 666, 777, 888]
    if presets is None:
        presets = list(get_model_presets(use_gpu).keys())

    print(f"数据划分种子: {split_seed}")
    print(f"模型训练种子: {len(model_seeds)}个 - {model_seeds[:5]}...")
    print(f"参数预设: {presets}")
    print(f"权重方法: {weighting_method}")

    # 固定数据划分
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=split_seed, shuffle=True)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=split_seed, shuffle=True)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")

    # 准备数据
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # 获取参数预设
    preset_params = get_model_presets(use_gpu)

    # 训练所有模型
    all_models = []
    model_configs = []

    print(f"\n开始训练 {len(presets)} × {len(model_seeds)} = {len(presets) * len(model_seeds)} 个模型...")

    for preset in presets:
        params = preset_params[preset]
        for seed in model_seeds:
            model_seed = f"{preset}_{seed}"
            print(f"  训练: {model_seed}...", end=" ")

            p = dict(params)
            p["seed"] = int(seed)

            model = xgb.train(
                params=p,
                dtrain=dtrain,
                num_boost_round=8000,
                evals=[(dval, "val")],
                verbose_eval=False,
                early_stopping_rounds=300,
            )

            # 验证集性能
            val_pred = model.predict(dval)
            val_r2 = r2_score(y_val, val_pred)
            val_mae = mean_absolute_error(y_val, val_pred)

            all_models.append(model)
            model_configs.append({
                "preset": preset,
                "seed": seed,
                "val_r2": val_r2,
                "val_mae": val_mae,
                "best_iteration": int(getattr(model, "best_iteration", -1)),
            })
            print(f"R²={val_r2:.4f}, MAE={val_mae:.4f}")

    # 收集所有预测
    print(f"\n收集预测结果...")
    val_predictions = []
    test_predictions = []
    val_r2s = []

    for i, model in enumerate(all_models):
        val_pred = model.predict(dval)
        test_pred = model.predict(dtest)
        val_predictions.append(val_pred)
        test_predictions.append(test_pred)
        val_r2s.append(model_configs[i]["val_r2"])

    val_predictions = np.stack(val_predictions, axis=1)
    test_predictions = np.stack(test_predictions, axis=1)
    val_r2s = np.array(val_r2s)

    # 计算集成权重
    print(f"\n计算集成权重...")

    if weighting_method == "simple":
        # 简单平均
        weights = np.ones(len(all_models)) / len(all_models)
        final_pred = np.mean(test_predictions, axis=1)

    elif weighting_method == "performance":
        # 基于验证集R²的加权
        weights = np.maximum(val_r2s, 0)
        weights = weights / weights.sum()
        final_pred = np.average(test_predictions, axis=1, weights=weights)

    elif weighting_method == "ridge":
        # 使用Ridge学习最优权重
        ridge = Ridge(alpha=1.0)
        ridge.fit(val_predictions, y_val)
        weights = ridge.coef_
        final_pred = ridge.predict(test_predictions)

        # 如果权重有负数，截断为0并重新归一化
        if np.any(weights < 0):
            print("  检测到负权重，进行截断...")
            weights = np.maximum(weights, 0)
            weights = weights / weights.sum()
            final_pred = np.average(test_predictions, axis=1, weights=weights)

    else:
        raise ValueError(f"未知的权重方法: {weighting_method}")

    print(f"权重统计:")
    print(f"  最大权重: {weights.max():.4f}")
    print(f"  最小权重: {weights.min():.4f}")
    print(f"  权重标准差: {weights.std():.4f}")

    # 显示前5个模型权重
    top_indices = np.argsort(weights)[-5:][::-1]
    print(f"  Top 5模型:")
    for idx in top_indices:
        config = model_configs[idx]
        print(f"    {config['preset']}_seed{config['seed']}: weight={weights[idx]:.4f}, val_r2={config['val_r2']:.4f}")

    # 评估
    mae = mean_absolute_error(y_test, final_pred)
    rmse = np.sqrt(mean_squared_error(y_test, final_pred))
    r2 = r2_score(y_test, final_pred)

    print(f"\n{'='*60}")
    print("最终评估")
    print(f"{'='*60}")
    print(f"Bagging集成性能 ({weighting_method}):")
    print(f"  MAE:  {mae:.4f}%")
    print(f"  RMSE: {rmse:.4f}%")
    print(f"  R²:   {r2:.4f}")

    # 单个模型性能对比
    print(f"\n单个模型性能统计:")
    individual_r2s = []
    for i, (model, config) in enumerate(zip(all_models, model_configs)):
        pred = model.predict(dtest)
        r2_i = r2_score(y_test, pred)
        individual_r2s.append(r2_i)

    individual_r2s = np.array(individual_r2s)
    print(f"  最大R²: {individual_r2s.max():.4f}")
    print(f"  最小R²: {individual_r2s.min():.4f}")
    print(f"  平均R²: {individual_r2s.mean():.4f}")
    print(f"  R²标准差: {individual_r2s.std():.4f}")

    # 提升幅度
    improvement = r2 - individual_r2s.mean()
    print(f"\n相比平均单模型提升: ΔR² = {improvement:+.4f}")

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "weights": weights.tolist(),
        "model_configs": model_configs,
        "individual_r2s": individual_r2s.tolist(),
        "weighting_method": weighting_method,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-seed", type=int, default=42, help="数据划分种子")
    parser.add_argument("--model-seeds", type=str, default="42,123,456,789,9999,2024,2026,111,222,333,444,555,666,777,888",
                        help="模型训练种子列表（逗号分隔）")
    parser.add_argument("--presets", type=str, default="default,fit_v1,reg_v1,deep,shallow",
                        help="参数预设列表（逗号分隔）")
    parser.add_argument("--fp-bits", type=int, default=4096, help="指纹位数")
    parser.add_argument("--use-all-desc", action="store_true", help="使用全量RDKit描述符")
    parser.add_argument("--weighting", type=str, default="performance",
                        choices=["simple", "performance", "ridge"],
                        help="集成权重方法")
    parser.add_argument("--cpu", action="store_true", help="强制使用CPU")
    parser.add_argument("--save-results", action="store_true", help="保存结果到文件")
    args = parser.parse_args()

    print(f"XGBoost大规模Bagging集成训练")
    print(f"数据划分种子: {args.split_seed}")

    model_seeds = [int(s.strip()) for s in args.model_seeds.split(",") if s.strip()]
    presets = [p.strip() for p in args.presets.split(",") if p.strip()]

    use_gpu = not args.cpu
    if use_gpu:
        try:
            import torch
            if not torch.cuda.is_available():
                use_gpu = False
        except Exception:
            use_gpu = False

    print(f"使用GPU: {use_gpu}")
    print(f"将训练 {len(presets)} × {len(model_seeds)} = {len(presets) * len(model_seeds)} 个模型")

    # 加载数据
    X, y = load_high_pce_dataset(
        fp_bits=args.fp_bits,
        fp_radius=2,
        use_all_descriptors=args.use_all_desc
    )

    # 训练Bagging集成
    results = train_xgb_bagging(
        X, y,
        split_seed=args.split_seed,
        model_seeds=model_seeds,
        presets=presets,
        use_gpu=use_gpu,
        weighting_method=args.weighting
    )

    # 保存结果
    if args.save_results:
        os.makedirs("results/best_models", exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        result_data = {
            "created_at": stamp,
            "config": {
                "split_seed": args.split_seed,
                "model_seeds": model_seeds,
                "presets": presets,
                "fp_bits": args.fp_bits,
                "use_all_descriptors": args.use_all_desc,
                "weighting_method": args.weighting,
                "n_models": len(presets) * len(model_seeds),
                "device": "gpu" if use_gpu else "cpu",
            },
            "results": {
                "mae": results["mae"],
                "rmse": results["rmse"],
                "r2": results["r2"],
            },
            "individual_stats": {
                "max_r2": max(results["individual_r2s"]),
                "min_r2": min(results["individual_r2s"]),
                "mean_r2": sum(results["individual_r2s"]) / len(results["individual_r2s"]),
                "std_r2": np.std(results["individual_r2s"]),
            },
        }

        result_path = f"results/best_models/xgb_bagging_{args.weighting}_{stamp}.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        print(f"\n已保存结果: {result_path}")

    # 检查是否达到目标
    if results["r2"] >= 0.75:
        print(f"\n🎉 目标达成！R² = {results['r2']:.4f} >= 0.75")
    else:
        print(f"\n⚠️  尚未达到目标。当前R² = {results['r2']:.4f}, 目标 = 0.75")

    return results


if __name__ == "__main__":
    main()
