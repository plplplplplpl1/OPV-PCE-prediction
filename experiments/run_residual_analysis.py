"""
Table 7: Error Analysis by PCE Range

Aggregate XGBoost prediction residuals by PCE intervals:
  3-5%, 5-8%, 8-12%, >12%
Compute MAE, RMSE, bias, and sample count per range.
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
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
SEED = 42
RESULTS_FILE = 'external_results/residual_analysis.json'

PCE_RANGES = [
    (3.0, 5.0, '3-5%'),
    (5.0, 8.0, '5-8%'),
    (8.0, 12.0, '8-12%'),
    (12.0, float('inf'), '>12%'),
]

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


if __name__ == '__main__':
    print("=" * 60)
    print("Residual Analysis by PCE Range (Table 7)")
    print("=" * 60)

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

    # Train/Val/Test split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr, y_tr, test_size=0.1, random_state=SEED)

    model = xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=SEED, verbosity=0)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    preds = model.predict(X_te)

    # Overall metrics
    overall_r2 = float(r2_score(y_te, preds))
    overall_mae = float(mean_absolute_error(y_te, preds))
    overall_rmse = float(np.sqrt(mean_squared_error(y_te, preds)))
    print(f"\n总体: R²={overall_r2:.4f}, MAE={overall_mae:.4f}, RMSE={overall_rmse:.4f}")

    # Per-range analysis
    range_results = {}
    print(f"\n{'Range':>8s} | {'n':>5s} | {'MAE':>8s} | {'RMSE':>8s} | {'Bias':>8s}")
    print("-" * 45)
    for lo, hi, label in PCE_RANGES:
        mask = (y_te >= lo) & (y_te < hi)
        n = mask.sum()
        if n == 0:
            continue
        y_sub = y_te[mask]
        p_sub = preds[mask]
        mae = float(mean_absolute_error(y_sub, p_sub))
        rmse = float(np.sqrt(mean_squared_error(y_sub, p_sub)))
        bias = float(np.mean(p_sub - y_sub))
        print(f"{label:>8s} | {n:>5d} | {mae:>8.4f} | {rmse:>8.4f} | {bias:>+8.4f}")
        range_results[label] = {
            'n_samples': int(n),
            'pce_range': [lo, hi],
            'mae': mae,
            'rmse': rmse,
            'bias': bias,
        }

    # Full per-sample output for raw error analysis
    sample_results = []
    for i in range(len(y_te)):
        sample_results.append({
            'y_true': float(y_te[i]),
            'y_pred': float(preds[i]),
            'residual': float(y_te[i] - preds[i]),
            'abs_error': float(abs(y_te[i] - preds[i])),
        })

    output = {
        'description': 'Residual analysis by PCE range for XGBoost on OPV high-PCE',
        'model': 'XGBoost (Morgan 2048bit + 12 RDKit descriptors)',
        'seed': SEED,
        'overall': {
            'r2': overall_r2,
            'mae': overall_mae,
            'rmse': overall_rmse,
            'n_train': len(X_tr),
            'n_val': len(X_va),
            'n_test': len(X_te),
        },
        'by_pce_range': range_results,
        'n_total': len(sample_results),
    }

    with open(RESULTS_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n结果已保存: {RESULTS_FILE}")
