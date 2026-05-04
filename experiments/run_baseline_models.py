"""
P1-2: 补充树模型和线性回归基线

比较: XGBoost, CatBoost, LightGBM, Linear/Ridge Regression
在OPV高PCE回归任务上，与GNN对比
"""
import os, sys
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if os.path.basename(project_root) == '实验':
    project_root = os.path.dirname(project_root)
os.chdir(project_root)
print(f"工作目录: {os.getcwd()}")

import json, random, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import catboost as cb
import lightgbm as lgb

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
RDLogger.DisableLog('rdApp.*')

DATA_CSV = 'data/data_merged.csv' if os.path.exists('data/data_merged.csv') else 'data/data.csv'
PCE_THRESHOLD = 3.0
FP_DIM = 2048
SEEDS = [42, 123, 333, 9999]
RESULTS_FILE = 'external_results/baseline_models.json'
EXISTING_XGB_RESULTS = 'external_results/gps_seed9999.json'

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)

def compute_features(smiles_list, fp_dim=FP_DIM):
    """Compute Morgan fingerprints + basic RDKit descriptors"""
    fps, descs = [], []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(np.zeros(fp_dim, dtype=np.float32))
            descs.append(np.zeros(12, dtype=np.float32))
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=fp_dim)
        arr = np.zeros(fp_dim, dtype=np.float32)
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)
        d = np.array([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.RingCount(mol),
            Descriptors.NumAromaticRings(mol),
            Descriptors.NumAliphaticRings(mol),
            Descriptors.FractionCSP3(mol),
            Descriptors.HeavyAtomCount(mol),
            Descriptors.NumHeteroatoms(mol),
        ], dtype=np.float32)
        descs.append(d)
    return np.array(fps), np.array(descs)

def load_data():
    df = pd.read_csv(DATA_CSV, encoding='latin-1')
    pc = df.columns[2]; sc = df.columns[-1]
    df[pc] = pd.to_numeric(df[pc], errors='coerce')
    df[sc] = df[sc].astype(str).str.strip()
    df = df.dropna(subset=[pc, sc])
    df = df[df[sc] != 'nan'].reset_index(drop=True)
    df_h = df[df[pc] > PCE_THRESHOLD].copy().reset_index(drop=True)
    print(f"  高PCE样本: {len(df_h)}")
    fps, descs = compute_features(df_h[sc].values)
    X = np.concatenate([fps, descs], axis=1)
    y = df_h[pc].values.astype(float)
    print(f"  特征维度: {X.shape[1]} ({FP_DIM} FP + 12 desc)")
    return X, y, df_h

def evaluate_model(name, model, X_train, y_train, X_test, y_test, seed):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return {
        'model': name,
        'seed': seed,
        'r2': float(r2_score(y_test, preds)),
        'mae': float(mean_absolute_error(y_test, preds)),
        'rmse': float(np.sqrt(np.mean((preds - y_test) ** 2))),
    }

def main():
    print("=" * 60)
    print("P1-2: Baseline model comparison (XGBoost, CatBoost, LightGBM, Linear/Ridge)")
    print("=" * 60)

    X, y, df = load_data()
    all_results = []

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        set_seed(seed)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed)
        print(f"  训练: {len(X_train)}, 测试: {len(X_test)}")

        # 1) XGBoost — 使用已知的最优超参数
        print("  XGBoost...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.0117, max_depth=6,
            min_child_weight=5, subsample=0.595, colsample_bytree=0.626,
            reg_alpha=0.1, reg_lambda=1.0, random_state=seed, verbosity=0,
        )
        r = evaluate_model('XGBoost', xgb_model, X_train, y_train, X_test, y_test, seed)
        all_results.append(r)
        print(f"    R²={r['r2']:.4f}, MAE={r['mae']:.4f}")

        # 2) CatBoost
        print("  CatBoost...")
        cb_model = cb.CatBoostRegressor(
            iterations=500, learning_rate=0.05, depth=6,
            l2_leaf_reg=3.0, random_seed=seed, verbose=0, allow_writing_files=False,
        )
        r = evaluate_model('CatBoost', cb_model, X_train, y_train, X_test, y_test, seed)
        all_results.append(r)
        print(f"    R²={r['r2']:.4f}, MAE={r['mae']:.4f}")

        # 3) LightGBM
        print("  LightGBM...")
        lgb_model = lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, random_state=seed, verbose=-1,
        )
        r = evaluate_model('LightGBM', lgb_model, X_train, y_train, X_test, y_test, seed)
        all_results.append(r)
        print(f"    R²={r['r2']:.4f}, MAE={r['mae']:.4f}")

        # 4) Linear Regression (需标准化)
        print("  Linear Regression...")
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        lr_model = LinearRegression()
        r = evaluate_model('LinearRegression', lr_model, X_train_s, y_train, X_test_s, y_test, seed)
        all_results.append(r)
        print(f"    R²={r['r2']:.4f}, MAE={r['mae']:.4f}")

        # 5) Ridge Regression
        print("  Ridge Regression...")
        ridge_model = Ridge(alpha=1.0)
        r = evaluate_model('Ridge', ridge_model, X_train_s, y_train, X_test_s, y_test, seed)
        all_results.append(r)
        print(f"    R²={r['r2']:.4f}, MAE={r['mae']:.4f}")

    # 汇总
    df_results = pd.DataFrame(all_results)
    print("\n" + "=" * 60)
    print("汇总 (4-seed mean ± std):")
    print("=" * 60)
    summary = []
    for model_name in ['XGBoost', 'CatBoost', 'LightGBM', 'LinearRegression', 'Ridge']:
        sub = df_results[df_results['model'] == model_name]
        print(f"  {model_name:20s} | "
              f"R²={sub['r2'].mean():.4f}±{sub['r2'].std():.4f} | "
              f"MAE={sub['mae'].mean():.4f}±{sub['mae'].std():.4f} | "
              f"RMSE={sub['rmse'].mean():.4f}±{sub['rmse'].std():.4f}")
        summary.append({
            'model': model_name,
            'r2_mean': float(sub['r2'].mean()),
            'r2_std': float(sub['r2'].std()),
            'mae_mean': float(sub['mae'].mean()),
            'mae_std': float(sub['mae'].std()),
            'rmse_mean': float(sub['rmse'].mean()),
            'rmse_std': float(sub['rmse'].std()),
        })

    # GNN reference (from existing results)
    try:
        gnn_results = json.load(open('external_results/gps_seed9999.json'))
        gnn_r2 = gnn_results.get('test_r2', 'N/A')
        gnn_mae = gnn_results.get('test_mae', 'N/A')
        print(f"\n  GNN参考 (seed=9999): R²={gnn_r2}, MAE={gnn_mae}")
    except:
        print("\n  GNN参考: 未找到gps_seed9999.json")

    output = {
        'method': 'Baseline models: XGBoost, CatBoost, LightGBM, LinearRegression, Ridge',
        'feature': f'{FP_DIM}bit Morgan FP + 12 RDKit descriptors',
        'seeds': SEEDS,
        'results': all_results,
        'summary': summary,
    }
    with open(RESULTS_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n结果已保存: {RESULTS_FILE}")

if __name__ == '__main__':
    main()
