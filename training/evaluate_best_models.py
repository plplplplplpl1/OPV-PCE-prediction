#!/usr/bin/env python3
"""
评估已有最佳模型并进行集成
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


def load_high_pce_dataset_with_smiles(fp_bits: int = 4096, fp_radius: int = 2, use_all_descriptors: bool = True):
    data_csv = _get_data_path()
    df = pd.read_csv(data_csv, encoding="latin-1")
    pce_col, smiles_col = _infer_columns(df)

    df[pce_col] = pd.to_numeric(df[pce_col], errors="coerce")
    df[smiles_col] = df[smiles_col].astype(str).str.strip()
    df = df.dropna(subset=[pce_col, smiles_col])
    df = df[df[smiles_col] != "nan"].reset_index(drop=True)

    df_high = df[df[pce_col] > PCE_THRESHOLD].reset_index(drop=True)

    feats, smiles_list, y = [], [], []
    failed = 0
    for _, row in df_high.iterrows():
        smi = row[smiles_col]
        v = rdkit_features(smi, fp_bits=fp_bits, fp_radius=fp_radius, use_all_descriptors=use_all_descriptors)
        if v is None:
            failed += 1
            continue
        feats.append(v)
        smiles_list.append(smi)
        y.append(float(row[pce_col]))

    X = np.stack(feats, axis=0)
    y_arr = np.array(y, dtype=np.float32)
    print(f"数据文件: {data_csv}")
    print(f"高PCE样本 (PCE > {PCE_THRESHOLD}%): {len(df_high)} | 特征成功: {len(y_arr)} | 失败: {failed}")
    print(f"特征维度: {X.shape[1]}")
    return X, y_arr, np.array(smiles_list), df_high


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123, help="使用与原始最佳模型相同的seed")
    parser.add_argument("--model-dir", type=str, default="results/best_models", help="模型目录")
    parser.add_argument("--save-results", action="store_true")
    args = parser.parse_args()

    print(f"评估已有最佳模型")
    print(f"使用原始最佳模型的seed: {args.seed}")

    # 加载数据
    X, y, smiles_list, df_high = load_high_pce_dataset_with_smiles(fp_bits=4096, fp_radius=2, use_all_descriptors=True)

    # 使用与原始训练相同的数据划分
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=args.seed, shuffle=True)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=args.seed, shuffle=True)

    X_test, y_test = X[test_idx], y[test_idx]
    test_smiles = smiles_list[test_idx]

    print(f"\n测试集大小: {len(X_test)}")

    # 查找所有XGBoost模型
    model_files = []
    for f in os.listdir(args.model_dir):
        if f.startswith("xgb_") and f.endswith(".json") and not f.endswith(".meta.json"):
            model_files.append(os.path.join(args.model_dir, f))

    print(f"\n找到 {len(model_files)} 个XGBoost模型")

    # 评估每个模型
    results = {}
    predictions = []

    for model_file in model_files:
        try:
            model_name = os.path.basename(model_file)
            meta_file = model_file.replace(".json", ".meta.json")

            # 加载模型
            model = xgb.Booster()
            model.load_model(model_file)

            # 预测
            dtest = xgb.DMatrix(X_test)
            pred = model.predict(dtest)

            # 评估
            mae = mean_absolute_error(y_test, pred)
            rmse = np.sqrt(mean_squared_error(y_test, pred))
            r2 = r2_score(y_test, pred)

            results[model_name] = {"mae": mae, "rmse": rmse, "r2": r2, "model_file": model_file}
            predictions.append(pred)

            print(f"{model_name}:")
            print(f"  MAE:  {mae:.4f}%")
            print(f"  RMSE: {rmse:.4f}%")
            print(f"  R²:   {r2:.4f}")
        except Exception as e:
            print(f"加载模型 {model_file} 失败: {e}")

    if not predictions:
        print("\n没有成功加载任何模型")
        return

    # 集成预测
    predictions = np.stack(predictions, axis=1)

    # 简单平均
    avg_pred = np.mean(predictions, axis=1)
    avg_mae = mean_absolute_error(y_test, avg_pred)
    avg_rmse = np.sqrt(mean_squared_error(y_test, avg_pred))
    avg_r2 = r2_score(y_test, avg_pred)

    print(f"\n{'='*60}")
    print("集成结果")
    print(f"{'='*60}")
    print(f"\n简单平均 ({len(predictions)}个模型):")
    print(f"  MAE:  {avg_mae:.4f}%")
    print(f"  RMSE: {avg_rmse:.4f}%")
    print(f"  R²:   {avg_r2:.4f}")

    # 找到最佳模型
    best_model = max(results.items(), key=lambda x: x[1]["r2"])
    print(f"\n最佳单模型: {best_model[0]}")
    print(f"  R²:   {best_model[1]['r2']:.4f}")

    improvement = avg_r2 - best_model[1]["r2"]
    print(f"\n集成提升: ΔR² = {improvement:+.4f}")

    # 保存结果
    if args.save_results:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_data = {
            "created_at": stamp,
            "seed": args.seed,
            "individual_results": {k: {"mae": v["mae"], "rmse": v["rmse"], "r2": v["r2"]} for k, v in results.items()},
            "ensemble_results": {
                "mae": avg_mae,
                "rmse": avg_rmse,
                "r2": avg_r2,
                "n_models": len(predictions),
            },
        }
        result_path = f"results/best_models/ensemble_evaluation_{stamp}.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        print(f"\n已保存结果: {result_path}")

    # 检查目标
    if avg_r2 >= 0.75:
        print(f"\n🎉 目标达成！R² = {avg_r2:.4f} >= 0.75")
    else:
        print(f"\n⚠️  尚未达到目标。当前R² = {avg_r2:.4f}, 目标 = 0.75")
        print(f"差距 = {0.75 - avg_r2:.4f}")

    return results


if __name__ == "__main__":
    main()
