"""
Supplementary Table S5: Multi-seed baseline with seed=444

Runs all 5 baseline models (XGBoost, CatBoost, LightGBM, LinearRegression, Ridge)
with seed=444 and appends results to baseline_models.json for completeness.
"""
import os, sys, json, random, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import catboost as cb
import lightgbm as lgb
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
RDLogger.DisableLog('rdApp.*')

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if os.path.basename(project_root) == '实验':
    project_root = os.path.dirname(project_root)
os.chdir(project_root)

DATA_CSV = 'data/data_merged.csv' if os.path.exists('data/data_merged.csv') else 'data/data.csv'
PCE_THRESHOLD = 3.0
FP_DIM = 2048
SEED = 444
RESULTS_FILE = 'external_results/baseline_models.json'

DESC_FUNCS = [
    Descriptors.MolWt, Descriptors.MolLogP, Descriptors.TPSA,
    Descriptors.NumHDonors, Descriptors.NumHAcceptors,
    Descriptors.NumRotatableBonds, Descriptors.RingCount,
    Descriptors.NumAromaticRings, Descriptors.NumAliphaticRings,
    Descriptors.FractionCSP3, Descriptors.HeavyAtomCount,
    Descriptors.NumHeteroatoms,
]


def compute_features(smiles_list):
    fps, descs = [], []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(np.zeros(FP_DIM, dtype=np.float32))
            descs.append(np.zeros(len(DESC_FUNCS), dtype=np.float32))
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=FP_DIM)
        arr = np.zeros(FP_DIM, dtype=np.float32)
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)
        d = np.array([fn(mol) for fn in DESC_FUNCS], dtype=np.float32)
        descs.append(d)
    return np.concatenate([np.array(fps), np.array(descs)], axis=1)


def load_data():
    df = pd.read_csv(DATA_CSV, encoding='latin-1')
    pc = df.columns[2]
    sc = df.columns[-1]
    df[pc] = pd.to_numeric(df[pc], errors='coerce')
    df[sc] = df[sc].astype(str).str.strip()
    df = df.dropna(subset=[pc, sc])
    df = df[df[sc] != 'nan'].reset_index(drop=True)
    df_h = df[df[pc] > PCE_THRESHOLD].copy().reset_index(drop=True)
    print(f"高PCE样本: {len(df_h)}")
    X = compute_features(df_h[sc].values)
    y = df_h[pc].values.astype(float)
    print(f"特征维度: {X.shape[1]}")
    return X, y


if __name__ == '__main__':
    print("=" * 60)
    print(f"Seed={SEED} baseline experiment (Supplementary Table S5)")
    print("=" * 60)

    X, y = load_data()

    random.seed(SEED)
    np.random.seed(SEED)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED)
    print(f"训练: {len(X_tr)}, 测试: {len(X_te)}")

    results = []

    # XGBoost
    xgb_m = xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.0117, max_depth=6,
        min_child_weight=5, subsample=0.595, colsample_bytree=0.626,
        reg_alpha=0.1, reg_lambda=1.0, random_state=SEED, verbosity=0)
    xgb_m.fit(X_tr, y_tr)
    pred = xgb_m.predict(X_te)
    results.append({'model': 'XGBoost', 'seed': SEED,
                    'r2': float(r2_score(y_te, pred)),
                    'mae': float(mean_absolute_error(y_te, pred)),
                    'rmse': float(np.sqrt(mean_squared_error(y_te, pred)))})
    print(f"  XGBoost: R²={results[-1]['r2']:.4f}")

    # CatBoost
    cb_m = cb.CatBoostRegressor(
        iterations=500, learning_rate=0.05, depth=6,
        l2_leaf_reg=3.0, random_seed=SEED, verbose=0, allow_writing_files=False)
    cb_m.fit(X_tr, y_tr)
    pred = cb_m.predict(X_te)
    results.append({'model': 'CatBoost', 'seed': SEED,
                    'r2': float(r2_score(y_te, pred)),
                    'mae': float(mean_absolute_error(y_te, pred)),
                    'rmse': float(np.sqrt(mean_squared_error(y_te, pred)))})
    print(f"  CatBoost: R²={results[-1]['r2']:.4f}")

    # LightGBM
    lgb_m = lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        num_leaves=31, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, random_state=SEED, verbose=-1)
    lgb_m.fit(X_tr, y_tr)
    pred = lgb_m.predict(X_te)
    results.append({'model': 'LightGBM', 'seed': SEED,
                    'r2': float(r2_score(y_te, pred)),
                    'mae': float(mean_absolute_error(y_te, pred)),
                    'rmse': float(np.sqrt(mean_squared_error(y_te, pred)))})
    print(f"  LightGBM: R²={results[-1]['r2']:.4f}")

    # Linear Regression
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    lr_m = LinearRegression()
    lr_m.fit(X_tr_s, y_tr)
    pred = lr_m.predict(X_te_s)
    results.append({'model': 'LinearRegression', 'seed': SEED,
                    'r2': float(r2_score(y_te, pred)),
                    'mae': float(mean_absolute_error(y_te, pred)),
                    'rmse': float(np.sqrt(mean_squared_error(y_te, pred)))})
    print(f"  LinearRegression: R²={results[-1]['r2']:.4f}")

    # Ridge
    rd_m = Ridge(alpha=1.0)
    rd_m.fit(X_tr_s, y_tr)
    pred = rd_m.predict(X_te_s)
    results.append({'model': 'Ridge', 'seed': SEED,
                    'r2': float(r2_score(y_te, pred)),
                    'mae': float(mean_absolute_error(y_te, pred)),
                    'rmse': float(np.sqrt(mean_squared_error(y_te, pred)))})
    print(f"  Ridge: R²={results[-1]['r2']:.4f}")

    # Append to existing results
    existing = json.load(open(RESULTS_FILE)) if os.path.exists(RESULTS_FILE) else {'results': [], 'summary': [], 'seeds': []}
    existing.setdefault('results', []).extend(results)
    seeds_set = set(existing.get('seeds', []))
    seeds_set.add(SEED)
    existing['seeds'] = sorted(seeds_set)

    # Recompute summary
    df_r = pd.DataFrame(existing['results'])
    summary = []
    for mn in ['XGBoost', 'CatBoost', 'LightGBM', 'LinearRegression', 'Ridge']:
        sub = df_r[df_r['model'] == mn]
        if len(sub) > 0:
            summary.append({
                'model': mn,
                'r2_mean': float(sub['r2'].mean()),
                'r2_std': float(sub['r2'].std()),
                'mae_mean': float(sub['mae'].mean()),
                'mae_std': float(sub['mae'].std()),
                'rmse_mean': float(sub['rmse'].mean()),
                'rmse_std': float(sub['rmse'].std()),
            })
    existing['summary'] = summary

    # Also save standalone seed444 result
    seed444_file = 'external_results/seed444_results.json'
    with open(seed444_file, 'w') as f:
        json.dump({'seed': SEED, 'results': results}, f, indent=2)
    print(f"\nSeed=444 已保存: {seed444_file}")

    with open(RESULTS_FILE, 'w') as f:
        json.dump(existing, f, indent=2)
    print(f"已追加到: {RESULTS_FILE}")

    # Print updated summary
    print(f"\n更新后汇总 ({len(df_r)} 条记录, seeds={existing['seeds']}):")
    for s in summary:
        print(f"  {s['model']:20s}: R²={s['r2_mean']:.4f}±{s['r2_std']:.4f}")
