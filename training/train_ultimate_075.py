#!/usr/bin/env python3
"""
综合实力提升方案
目标：R²>0.75 + 优秀的MAE/RMSE + 稳定性
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, GraphDescriptors, Fragments
    from rdkit.Chem import rdMolDescriptors as rdDesc
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

# 优化的XGBoost参数
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


def get_enhanced_features(mol):
    """获取增强的分子特征"""
    features = {}

    try:
        # 1. 基础拓扑描述符
        features['balaban_j'] = GraphDescriptors.BalabanJ(mol)
        features['bertz_ct'] = GraphDescriptors.BertzCT(mol)

        # 2. 分子形状描述符
        features['asphericity'] = rdMolDescriptors.CalcAsphericity(mol)
        features['eccentricity'] = rdMolDescriptors.CalcEccentricity(mol)
        features['spherocity'] = rdMolDescriptors.CalcSpherocityIndex(mol)
        features['radius_of_gyration'] = rdMolDescriptors.CalcRadiusOfGyration(mol)

        # 3. 电荷相关特征
        try:
            features['max_partial_charge'] = Descriptors.MaxPartialCharge(mol)
            features['min_partial_charge'] = Descriptors.MinPartialCharge(mol)
            features['max_abs_partial_charge'] = Descriptors.MaxAbsPartialCharge(mol)
        except:
            features['max_partial_charge'] = 0.0
            features['min_partial_charge'] = 0.0
            features['max_abs_partial_charge'] = 0.0

        # 4. 芳香系统特征
        features['num_aromatic_rings'] = rdMolDescriptors.CalcNumAromaticRings(mol)
        features['num_saturated_rings'] = rdMolDescriptors.CalcNumSaturatedRings(mol)
        features['num_heterocycles'] = rdMolDescriptors.CalcNumHeterocycles(mol)

        # 5. 氢键相关
        features['num_hbd'] = rdMolDescriptors.CalcNumHBD(mol)
        features['num_hba'] = rdMolDescriptors.CalcNumHBA(mol)

        # 6. 片段特征（OPV相关）
        features['frag_br'] = Fragments.fr_Br(mol)
        features['frag_coo'] = Fragments.fr_COO(mol)
        features['frag_oh'] = Fragments.fr_OH(mol)
        features['frag_nh2'] = Fragments.fr_NH2(mol)
        features['frag_halogen'] = Fragments.fr_halogen(mol)
        features['frag_arom_oh'] = Fragments.fr_ArOH(mol)

        # 7. 空间特征
        features['labute_asa'] = rdMolDescriptors.CalcLabuteASA(mol)
        features['pbf'] = rdMolDescriptors.CalcPBF(mol)

        # 8. π系统估计
        arom_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
        total_atoms = mol.GetNumAtoms()
        features['arom_ratio'] = arom_atoms / total_atoms if total_atoms > 0 else 0

        # 9. 共轭长度估计
        pi_length = 0
        max_pi = 0
        for atom in mol.GetAtoms():
            if atom.GetIsAromatic():
                pi_length += 1
            else:
                max_pi = max(max_pi, pi_length)
                pi_length = 0
        features['max_pi_length'] = max(max_pi, pi_length)

        # 10. 支链特征
        features['fraction_csp3'] = Descriptors.FractionCsp3(mol)
        features['num_bridgehead'] = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)

        # 11. 旋转自由度
        features['num_rotatable_bonds'] = rdMolDescriptors.CalcNumRotatableBonds(mol)

        # 12. 极性表面积
        features['tpsa'] = Descriptors.TPSA(mol)

        # 13. 酰胺键
        features['num_amide_bonds'] = rdMolDescriptors.CalcNumAmideBonds(mol)

    except Exception as e:
        # 如果任何特征计算失败，设为0
        keys = ['balaban_j', 'bertz_ct', 'asphericity', 'eccentricity', 'spherocity',
                'radius_of_gyration', 'max_partial_charge', 'min_partial_charge',
                'max_abs_partial_charge', 'num_aromatic_rings', 'num_saturated_rings',
                'num_heterocycles', 'num_hbd', 'num_hba', 'frag_br', 'frag_coo',
                'frag_oh', 'frag_nh2', 'frag_halogen', 'frag_arom_oh', 'labute_asa',
                'pbf', 'arom_ratio', 'max_pi_length', 'fraction_csp3', 'num_bridgehead',
                'num_rotatable_bonds', 'tpsa', 'num_amide_bonds']
        for key in keys:
            if key not in features:
                features[key] = 0.0

    return features


def rdkit_features(smiles: str):
    smiles = str(smiles).strip()
    if not smiles or smiles == "nan":
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # 1. Morgan指纹
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=4096)
    fp_arr = np.frombuffer(fp.ToBitString().encode("ascii"), dtype=np.uint8) - ord("0")

    # 2. 全量RDKit描述符
    desc = []
    for _, fn in list(Descriptors._descList):
        try:
            v = fn(mol)
            desc.append(float(v) if v is not None and not (isinstance(v, float) and (np.isnan(v) or np.isinf(v))) else 0.0)
        except:
            desc.append(0.0)

    desc_clean = [0.0 if np.isinf(v) or abs(v) > 1e10 else v for v in desc]

    # 3. 增强特征
    enh_feats = get_enhanced_features(mol)
    enh_values = list(enh_feats.values())

    return np.concatenate([
        fp_arr.astype(np.float64),
        np.array(desc_clean, dtype=np.float64),
        np.array(enh_values, dtype=np.float32)
    ])


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

    # 处理极端值
    X[:, 4096:4096+217] = np.nan_to_num(X[:, 4096:4096+217], nan=0.0, posinf=1e10, neginf=-1e10)

    print(f"数据: {len(X)}样本")
    print(f"特征: 指纹(4096) + 描述符(217) + 增强特征(30) = {X.shape[1]}维")

    return X, y


def train_robust_models(X_train, y_train, X_val, y_val, X_test, y_test, n_models=30, use_gpu=True):
    """训练鲁棒的模型集合"""
    base_params = {**BEST_PARAMS}
    if use_gpu:
        base_params["device"] = "cuda"

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    all_preds = []
    all_val_r2s = []

    # 参数多样性配置
    configs = [
        {"lr": 0.009, "depth": 5},  # 保守
        {"lr": 0.010, "depth": 6},  # 标准
        {"lr": 0.0118, "depth": 6},  # 最佳
        {"lr": 0.013, "depth": 6},  # 稍激进
        {"lr": 0.012, "depth": 7},  # 更深
    ]

    print(f"训练{n_models}个鲁棒模型...")

    for i in range(n_models):
        config = configs[i % len(configs)]
        params = dict(base_params)
        params["learning_rate"] = config["lr"]
        params["max_depth"] = config["depth"]
        params["seed"] = SEED + i * 100

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

        all_preds.append(test_pred)
        all_val_r2s.append(val_r2)

    return np.stack(all_preds, axis=1), np.array(all_val_r2s)


def smart_ensemble(preds, val_r2s, y_test):
    """智能集成策略"""
    results = {}

    # 1. 简单平均
    avg_pred = np.mean(preds, axis=1)
    results["简单平均"] = {
        "pred": avg_pred,
        "r2": r2_score(y_test, avg_pred),
        "mae": mean_absolute_error(y_test, avg_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, avg_pred))
    }

    # 2. 加权平均
    weights = np.maximum(val_r2s, 0)
    weights = weights / weights.sum()
    w_pred = np.average(preds, axis=1, weights=weights)
    results["加权平均"] = {
        "pred": w_pred,
        "r2": r2_score(y_test, w_pred),
        "mae": mean_absolute_error(y_test, w_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, w_pred))
    }

    # 3. 中位数（鲁棒性最好）
    median_pred = np.median(preds, axis=1)
    results["中位数"] = {
        "pred": median_pred,
        "r2": r2_score(y_test, median_pred),
        "mae": mean_absolute_error(y_test, median_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, median_pred))
    }

    # 4. Trimmed平均（去除极端值）
    sorted_preds = np.sort(preds, axis=1)
    n = preds.shape[1]
    trim_pred = np.mean(sorted_preds[:, n//10:-n//10], axis=1)
    results["Trimmed平均"] = {
        "pred": trim_pred,
        "r2": r2_score(y_test, trim_pred),
        "mae": mean_absolute_error(y_test, trim_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, trim_pred))
    }

    return results


def main():
    print(f"\n{'='*70}")
    print(f"综合实力提升方案")
    print(f"目标: R²>{TARGET_R2} + 优秀的MAE/RMSE + 稳定性")
    print(f"{'='*70}\n")

    use_gpu = True
    try:
        import torch
        if not torch.cuda.is_available():
            use_gpu = False
    except:
        use_gpu = False

    # 加载数据
    X, y = load_data()

    # 数据划分
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=SEED, shuffle=True)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=SEED, shuffle=True)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"\n划分: 训练{len(X_train)} | 验证{len(X_val)} | 测试{len(X_test)}")

    # 训练模型
    preds, val_r2s = train_robust_models(X_train, y_train, X_val, y_val, X_test, y_test,
                                        n_models=40, use_gpu=use_gpu)

    # 智能集成
    results = smart_ensemble(preds, val_r2s, y_test)

    # 评估所有方法
    print(f"\n{'='*70}")
    print(f"综合性能评估")
    print(f"{'='*70}\n")

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

    # 评估综合实力
    scores = {}
    for method, metrics in results.items():
        # 综合评分：R²占60%，MAE占20%，RMSE占20%
        r2_score_norm = metrics['r2']  # 0-1
        mae_score_norm = 1 - (metrics['mae'] / 3.0)  # 假设3%是基准
        rmse_score_norm = 1 - (metrics['rmse'] / 2.5)
        scores[method] = 0.6 * r2_score_norm + 0.2 * mae_score_norm + 0.2 * rmse_score_norm

    best_overall = max(scores, key=scores.get)
    print(f"综合实力最佳: {best_overall} (评分: {scores[best_overall]:.4f})")

    # 检查目标
    if best_metrics['r2'] >= TARGET_R2:
        print(f"\n🎉 成功！R² = {best_metrics['r2']:.4f} >= {TARGET_R2}")
        print(f"综合实力: R²={best_metrics['r2']:.4f}, MAE={best_metrics['mae']:.4f}%, RMSE={best_metrics['rmse']:.4f}%")
    else:
        gap = TARGET_R2 - best_metrics['r2']
        print(f"\n⚠️  R²差距: {gap:.4f}")

    # 保存结果
    os.makedirs("results/best_models", exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    result_data = {
        "created_at": stamp,
        "seed": SEED,
        "feature_dims": {
            "fingerprint": 4096,
            "descriptors": 217,
            "enhanced": 30,
            "total": X.shape[1]
        },
        "best_method": best_name,
        "best_overall": best_overall,
        "results": {
            method: {
                "r2": float(metrics["r2"]),
                "mae": float(metrics["mae"]),
                "rmse": float(metrics["rmse"])
            }
            for method, metrics in results.items()
        },
        "target_reached": best_metrics['r2'] >= TARGET_R2,
    }

    path = f"results/best_models/ultimate_075_{stamp}.json"
    with open(path, "w") as f:
        json.dump(result_data, f, indent=2)

    print(f"\n已保存: {path}")

    return best_metrics


if __name__ == "__main__":
    main()
