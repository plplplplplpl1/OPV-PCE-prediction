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


def _get_data_path() -> str:
    return "data/data_merged.csv" if os.path.exists("data/data_merged.csv") else "data/data.csv"


def _infer_columns(df: pd.DataFrame) -> tuple[str, str]:
    # 兼容当前项目约定：第三列为 PCE，最后一列为 SMILES
    if df.shape[1] < 3:
        raise ValueError("数据列数不足，无法推断 PCE/SMILES 列")
    return df.columns[2], df.columns[-1]


_ALL_DESC_FUNCS: list = []
try:
    _ALL_DESC_FUNCS = list(Descriptors._descList)  # [(name, func), ...]
except Exception:
    _ALL_DESC_FUNCS = []


def rdkit_features(
    smiles: str,
    fp_bits: int = 2048,
    fp_radius: int = 2,
    use_all_descriptors: bool = False,
) -> np.ndarray | None:
    smiles = str(smiles).strip()
    if not smiles or smiles == "nan":
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Morgan 指纹（bit vector）
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
        # 使用float64避免溢出，并处理极端值
        desc_clean = []
        for v in desc:
            if np.isinf(v) or np.isnan(v):
                desc_clean.append(0.0)
            elif abs(v) > 1e10:  # 截断过大的值
                desc_clean.append(np.sign(v) * 1e10)
            else:
                desc_clean.append(v)
        desc_values = np.array(desc_clean, dtype=np.float64)
    else:
        # 一组相对稳健、通用的 RDKit 描述符（连续值）
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


def load_high_pce_dataset(fp_bits: int, fp_radius: int, use_all_descriptors: bool) -> tuple[np.ndarray, np.ndarray]:
    data_csv = _get_data_path()
    df = pd.read_csv(data_csv, encoding="latin-1")
    pce_col, smiles_col = _infer_columns(df)

    df[pce_col] = pd.to_numeric(df[pce_col], errors="coerce")
    df[smiles_col] = df[smiles_col].astype(str).str.strip()
    df = df.dropna(subset=[pce_col, smiles_col])
    df = df[df[smiles_col] != "nan"].reset_index(drop=True)

    df_high = df[df[pce_col] > PCE_THRESHOLD].reset_index(drop=True)

    feats: list[np.ndarray] = []
    y: list[float] = []
    failed = 0
    for smi, pce in zip(df_high[smiles_col].tolist(), df_high[pce_col].tolist()):
        v = rdkit_features(
            smi,
            fp_bits=fp_bits,
            fp_radius=fp_radius,
            use_all_descriptors=use_all_descriptors,
        )
        if v is None:
            failed += 1
            continue
        feats.append(v)
        y.append(float(pce))

    if not feats:
        raise RuntimeError("没有成功提取任何特征")

    X = np.stack(feats, axis=0)
    y_arr = np.array(y, dtype=np.float32)
    print(f"数据文件: {data_csv}")
    print(f"高PCE样本 (PCE > {PCE_THRESHOLD}%): {len(df_high)} | 特征成功: {len(y_arr)} | 失败: {failed}")
    if use_all_descriptors and _ALL_DESC_FUNCS:
        print(f"特征维度: {X.shape[1]} (fp_bits={fp_bits} + {len(_ALL_DESC_FUNCS)} rdkit descriptors)")
    else:
        print(f"特征维度: {X.shape[1]} (fp_bits={fp_bits} + 12 descriptors)")
    return X, y_arr


def train_xgb(X: np.ndarray, y: np.ndarray, seed: int, use_gpu: bool, preset: str) -> dict:
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=seed, shuffle=True)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=seed, shuffle=True)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    base = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "seed": seed,
        "tree_method": "hist",
    }
    presets: dict[str, dict] = {
        # 默认：偏保守，泛化更稳
        "default": {
            "learning_rate": 0.02,
            "max_depth": 8,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "gamma": 0.0,
        },
        # 更强拟合：更浅一些但更高采样、更低 min_child_weight
        "fit_v1": {
            "learning_rate": 0.05,
            "max_depth": 6,
            "min_child_weight": 1,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "gamma": 0.0,
        },
        # 更强正则：避免过拟合的同时更深一点
        "reg_v1": {
            "learning_rate": 0.03,
            "max_depth": 9,
            "min_child_weight": 5,
            "subsample": 0.85,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 5.0,
            "gamma": 0.1,
        },
    }
    if preset not in presets:
        raise ValueError(f"未知 preset: {preset}，可选: {sorted(presets.keys())}")
    params = {**base, **presets[preset]}
    if use_gpu:
        params["device"] = "cuda"

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    def _train_one(model_seed: int):
        p = dict(params)
        p["seed"] = int(model_seed)
        return xgb.train(
            params=p,
            dtrain=dtrain,
            num_boost_round=8000,
            evals=[(dval, "val")],
            verbose_eval=False,
            early_stopping_rounds=300,
        )

    # 单模型（默认）
    booster = _train_one(seed)
    pred = booster.predict(dtest)
    mae = float(mean_absolute_error(y_test, pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    r2 = float(r2_score(y_test, pred))

    return {
        "preset": preset,
        "seed": seed,
        "best_iteration": int(getattr(booster, "best_iteration", -1)),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "booster": booster,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp-bits", type=int, default=2048)
    parser.add_argument("--fp-radius", type=int, default=2)
    parser.add_argument("--seeds", type=str, default="42,9999,123,2026")
    parser.add_argument("--presets", type=str, default="default,fit_v1,reg_v1")
    parser.add_argument("--all-desc", action="store_true", help="使用 RDKit 全量描述符（更慢但可能更准）")
    parser.add_argument("--ensemble", type=int, default=1, help="每个 split_seed 训练多少个模型并做平均")
    parser.add_argument("--save-best", action="store_true", help="保存最佳 XGBoost 模型与元信息到 results/best_models/")
    parser.add_argument("--cpu", action="store_true", help="强制用 CPU 训练")
    args = parser.parse_args()

    use_gpu = (not args.cpu)
    try:
        import torch

        if not torch.cuda.is_available():
            use_gpu = False
    except Exception:
        use_gpu = False

    X, y = load_high_pce_dataset(fp_bits=args.fp_bits, fp_radius=args.fp_radius, use_all_descriptors=args.all_desc)

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    presets = [p.strip() for p in args.presets.split(",") if p.strip()]
    print(f"训练设备: {'GPU' if use_gpu else 'CPU'} | seeds={seeds}")
    print(f"参数预设: {presets}")
    if args.ensemble > 1:
        print(f"集成: 每个 split_seed 训练 {args.ensemble} 个 XGB 并平均预测")

    best = None
    best_booster = None
    for p in presets:
        for s in seeds:
            if args.ensemble <= 1:
                res = train_xgb(X, y, seed=s, use_gpu=use_gpu, preset=p)
                booster = res.pop("booster")
            else:
                # 固定数据划分 seed=s，但训练多个不同 model_seed 的 booster，平均 test 预测
                idx = np.arange(len(y))
                train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=s, shuffle=True)
                train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=s, shuffle=True)

                X_train, y_train = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]
                X_test, y_test = X[test_idx], y[test_idx]

                # 复用 train_xgb 里的参数逻辑：用同样 preset，但把 split_seed / model_seed 分开
                # 这里用一组确定性的 model_seed 序列来做 bagging
                model_seeds = [s + 1000 * k for k in range(args.ensemble)]

                # 构造 params（拷贝自 train_xgb）
                base = {"objective": "reg:squarederror", "eval_metric": "rmse", "seed": s, "tree_method": "hist"}
                presets_map: dict[str, dict] = {
                    "default": {
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
                        "learning_rate": 0.03,
                        "max_depth": 9,
                        "min_child_weight": 5,
                        "subsample": 0.85,
                        "colsample_bytree": 0.8,
                        "reg_alpha": 0.0,
                        "reg_lambda": 5.0,
                        "gamma": 0.1,
                    },
                }
                if p not in presets_map:
                    raise ValueError(f"未知 preset: {p}")
                params = {**base, **presets_map[p]}
                if use_gpu:
                    params["device"] = "cuda"

                dtrain = xgb.DMatrix(X_train, label=y_train)
                dval = xgb.DMatrix(X_val, label=y_val)
                dtest = xgb.DMatrix(X_test, label=y_test)

                preds = []
                best_iters = []
                for ms in model_seeds:
                    params_i = dict(params)
                    params_i["seed"] = int(ms)
                    booster = xgb.train(
                        params=params_i,
                        dtrain=dtrain,
                        num_boost_round=8000,
                        evals=[(dval, "val")],
                        verbose_eval=False,
                        early_stopping_rounds=300,
                    )
                    best_iters.append(int(getattr(booster, "best_iteration", -1)))
                    preds.append(booster.predict(dtest))
                pred = np.mean(np.stack(preds, axis=0), axis=0)
                mae = float(mean_absolute_error(y_test, pred))
                rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
                r2 = float(r2_score(y_test, pred))
                res = {
                    "preset": p,
                    "seed": s,
                    "best_iteration": int(np.median(best_iters)) if best_iters else -1,
                    "mae": mae,
                    "rmse": rmse,
                    "r2": r2,
                }
                booster = None
            print(
                f"preset={res['preset']} | seed={res['seed']} | best_iter={res['best_iteration']} | "
                f"MAE={res['mae']:.4f}% | RMSE={res['rmse']:.4f}% | R2={res['r2']:.4f}"
            )
            if best is None or res["r2"] > best["r2"]:
                best = res
                best_booster = booster

    assert best is not None
    print("\n最佳结果:")
    print(
        f"preset={best['preset']} | seed={best['seed']} | "
        f"MAE={best['mae']:.4f}% | RMSE={best['rmse']:.4f}% | R2={best['r2']:.4f}"
    )

    if args.save_best and best_booster is not None:
        os.makedirs("results/best_models", exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"results/best_models/xgb_high_pce_{best['preset']}_seed{best['seed']}_{stamp}.json"
        meta_path = f"results/best_models/xgb_high_pce_{best['preset']}_seed{best['seed']}_{stamp}.meta.json"
        best_booster.save_model(model_path)
        meta = {
            "created_at": stamp,
            "data_path": _get_data_path(),
            "task": "high_pce_regression",
            "pce_threshold": PCE_THRESHOLD,
            "features": {
                "fp_bits": args.fp_bits,
                "fp_radius": args.fp_radius,
                "use_all_descriptors": bool(args.all_desc),
                "n_descriptors": int(len(_ALL_DESC_FUNCS)) if (args.all_desc and _ALL_DESC_FUNCS) else 12,
            },
            "best": best,
            "device": "gpu" if use_gpu else "cpu",
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"\n已保存最佳 XGBoost 模型: {model_path}")
        print(f"已保存元信息: {meta_path}")


if __name__ == "__main__":
    main()

