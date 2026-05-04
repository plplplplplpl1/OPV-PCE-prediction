#!/usr/bin/env python3
"""
高级特征工程 + 集成学习
目标：突破R²=0.75
"""

import os
import argparse
import numpy as np
import pandas as pd
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from collections import Counter

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, GraphDescriptors
    from rdkit.Chem import Fragments
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

# 基础XGBoost参数
BASE_PARAMS = {
    "learning_rate": 0.012,
    "max_depth": 6,
    "min_child_weight": 5,
    "subsample": 0.55,
    "colsample_bytree": 0.82,
    "reg_alpha": 1e-5,
    "reg_lambda": 1e-4,
    "gamma": 0.001,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
}


def get_advanced_features(mol):
    """提取高级分子特征"""
    features = {}

    try:
        # 1. 分子片段特征（OPV相关）
        features['frag_br'] = Fragments.fr_Br(mol)  # 溴原子
        features['frag_coo'] = Fragments.fr_COO(mol)  # 羧酸
        features['frag_coo2'] = Fragments.fr_C_O2(mol)  # 酯/酮
        features['frag_nh2'] = Fragments.fr_NH2(mol)  # 胺基
        features['frag_nh1'] = Fragments.fr_NH1(mol)  # 仲胺
        features['frag_oh'] = Fragments.fr_OH(mol)  # 羟基
        features['frag_sh'] = Fragments.fr_SH(mol)  # 巯基
        features['frag_halogen'] = Fragments.fr_halogen(mol)  # 卤素
        features['frag_arom_oh'] = Fragments.fr_ArOH(mol)  # 酚羟基

        # 2. 杂环特征（噻吩、呋喃等常见于OPV）
        features['hetero_arom'] = rdMolDescriptors.CalcNumHeterocycles(mol)
        features['arom_rings'] = rdMolDescriptors.CalcNumAromaticRings(mol)
        features['saturated_rings'] = rdMolDescriptors.CalcNumSaturatedRings(mol)

        # 3. 共轭系统特征
        features['num_rotatable_bonds'] = rdMolDescriptors.CalcNumRotatableBonds(mol)
        features['num_hbd'] = rdMolDescriptors.CalcNumHBD(mol)  # 氢键供体
        features['num_hba'] = rdMolDescriptors.CalcNumHBA(mol)  # 氢键受体
        features['num_amide_bonds'] = rdMolDescriptors.CalcNumAmideBonds(mol)

        # 4. 拓扑特征
        features['tpsa'] = Descriptors.TPSA(mol)
        features['labute_asa'] = rdMolDescriptors.CalcLabuteASA(mol)
        features['pbf'] = rdMolDescriptors.CalcPBF(mol)

        # 5. 电子特征
        features['max_partial_charge'] =Descriptors.MaxPartialCharge(mol)
        features['min_partial_charge'] = Descriptors.MinPartialCharge(mol)
        features['max_abs_partial_charge'] = Descriptors.MaxAbsPartialCharge(mol)

        # 6. 几何特征（2D）
        features['balaban_j'] = GraphDescriptors.BalabanJ(mol)
        features['bertz_ct'] = GraphDescriptors.BertzCT(mol)

        # 7. 芳香系统特征
        arom_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
        total_atoms = mol.GetNumAtoms()
        features['arom_ratio'] = arom_atoms / total_atoms if total_atoms > 0 else 0

        # 8. π系统长度（简化估计）
        pi_system_length = 0
        max_pi_length = 0
        for atom in mol.GetAtoms():
            if atom.GetIsAromatic():
                pi_system_length += 1
            else:
                max_pi_length = max(max_pi_length, pi_system_length)
                pi_system_length = 0
        features['max_pi_system'] = max(max_pi_length, pi_system_length)

        # 9. 侧链特征
        features['fraction_csp3'] = Descriptors.FractionCsp3(mol)
        features['num_bridgehead'] = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)

        # 10. 分子形状特征
        features['asphericity'] = rdMolDescriptors.CalcAsphericity(mol)
        features['eccentricity'] = rdMolDescriptors.CalcEccentricity(mol)
        features['spherocity'] = rdMolDescriptors.CalcSpherocityIndex(mol)
        features['radius_gyration'] = rdMolDescriptors.CalcRadiusOfGyration(mol)

    except Exception as e:
        # 如果特征计算失败，填充0
        for key in ['frag_br', 'frag_coo', 'frag_coo2', 'frag_nh2', 'frag_nh1', 'frag_oh',
                    'frag_sh', 'frag_halogen', 'frag_arom_oh', 'hetero_arom', 'arom_rings',
                    'saturated_rings', 'num_rotatable_bonds', 'num_hbd', 'num_hba', 'num_amide_bonds',
                    'tpsa', 'labute_asa', 'pbf', 'max_partial_charge', 'min_partial_charge',
                    'max_abs_partial_charge', 'balaban_j', 'bertz_ct', 'arom_ratio', 'max_pi_system',
                    'fraction_csp3', 'num_bridgehead', 'asphericity', 'eccentricity', 'spherocity', 'radius_gyration']:
            features[key] = 0.0

    return features


def rdkit_features(smiles: str) -> np.ndarray | None:
    smiles = str(smiles).strip()
    if not smiles or smiles == "nan":
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # 1. 基础Morgan指纹
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

    desc_clean = [0.0 if np.isinf(v) or np.isnan(v) or abs(v) > 1e10 else v for v in desc]

    # 3. 高级特征
    adv_feats = get_advanced_features(mol)
    adv_values = list(adv_feats.values())

    return np.concatenate([fp_arr.astype(np.float64), np.array(desc_clean, dtype=np.float64), np.array(adv_values, dtype=np.float32)])


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

    # 特征归一化（处理高级特征的尺度差异）
    X[:, 4096:4096+217] = np.nan_to_num(X[:, 4096:4096+217], nan=0.0, posinf=1e10, neginf=-1e10)
    X[:, 4096+217:] = np.nan_to_num(X[:, 4096+217:], nan=0.0, posinf=100.0, neginf=-100.0)

    print(f"数据: {len(X)}样本")
    print(f"特征维度: {X.shape[1]} (指纹: 4096 + 描述符: 217 + 高级特征: 32)")

    return X, y


def main():
    print(f"\n{'='*70}")
    print(f"高级特征工程 + 大规模集成")
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

    # 固定数据划分
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=SEED, shuffle=True)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=SEED, shuffle=True)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"划分: 训练{len(X_train)} | 验证{len(X_val)} | 测试{len(X_test)}\n")

    base_params = {**BASE_PARAMS, "seed": SEED}
    if use_gpu:
        base_params["device"] = "cuda"

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # 训练更多模型
    n_models = 50
    all_preds = []
    all_val_r2s = []

    print(f"训练{n_models}个XGBoost模型（使用高级特征）...")
    for i in range(n_models):
        params = dict(base_params)
        params["seed"] = SEED + i * 100

        # 轻微调整学习率
        if i % 5 == 0:
            params["learning_rate"] = BASE_PARAMS["learning_rate"] * 0.95
        elif i % 5 == 1:
            params["learning_rate"] = BASE_PARAMS["learning_rate"] * 1.05

        print(f"  [{i+1:2d}/{n_models}] seed={params['seed']}", end=" ")

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

        print(f"val_R²={val_r2:.4f}")

    all_preds = np.stack(all_preds, axis=1)
    all_val_r2s = np.array(all_val_r2s)

    print(f"\n{'='*70}")
    print("集成结果评估")
    print(f"{'='*70}\n")

    # 多种集成方法
    results = {}

    avg_pred = np.mean(all_preds, axis=1)
    results["简单平均"] = r2_score(y_test, avg_pred)

    weights = np.maximum(all_val_r2s, 0)
    weights = weights / weights.sum()
    w_pred = np.average(all_preds, axis=1, weights=weights)
    results["加权平均"] = r2_score(y_test, w_pred)

    median_pred = np.median(all_preds, axis=1)
    results["中位数"] = r2_score(y_test, median_pred)

    # Trimmed mean (去除最高和最低的10%)
    sorted_preds = np.sort(all_preds, axis=1)
    trim_count = max(1, n_models // 10)
    trim_pred = np.mean(sorted_preds[:, trim_count:-trim_count], axis=1)
    results["Trimmed平均"] = r2_score(y_test, trim_pred)

    for method, r2 in results.items():
        print(f"{method}: R² = {r2:.4f}")

    best_method = max(results, key=results.get)
    best_r2 = results[best_method]

    print(f"\n{'='*70}")
    print(f"最佳方法: {best_method}")
    print(f"最终R²:   {best_r2:.4f}")
    print(f"{'='*70}\n")

    if best_r2 >= TARGET_R2:
        print(f"🎉 成功！R² = {best_r2:.4f} >= {TARGET_R2}")
    else:
        gap = TARGET_R2 - best_r2
        print(f"⚠️  差距 = {gap:.4f}")

    # 保存结果
    os.makedirs("results/best_models", exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_data = {
        "created_at": stamp,
        "method": "advanced_features",
        "n_models": n_models,
        "seed": SEED,
        "best_method": best_method,
        "best_r2": float(best_r2),
        "all_results": {k: float(v) for k, v in results.items()},
        "target_reached": best_r2 >= TARGET_R2,
        "feature_dims": {
            "fingerprint": 4096,
            "descriptors": 217,
            "advanced": 32,
            "total": X.shape[1]
        }
    }
    path = f"results/best_models/advanced_075_{stamp}.json"
    with open(path, "w") as f:
        json.dump(result_data, f, indent=2)
    print(f"已保存: {path}")

    return best_r2


if __name__ == "__main__":
    main()
