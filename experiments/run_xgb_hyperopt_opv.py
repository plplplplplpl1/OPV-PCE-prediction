"""
P2-4 supplement: XGBoost Optuna Hyperparameter Search on OPV

Reproduces the paper's XGBoost Optuna optimization result.
Uses Morgan 2048-bit + 12 RDKit descriptors on high-PCE subset.
200 Optuna trials with 3-fold CV, then evaluate on held-out test set.
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
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
N_TRIALS = 200
CV_FOLDS = 3
RESULTS_FILE = 'external_results/xgb_hyperopt_opv.json'

# 12 RDKit descriptors
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
    X = compute_features(df_h[sc].values)
    y = df_h[pc].values.astype(float)
    print(f"高PCE样本: {len(df_h)}, 特征维度: {X.shape[1]}")
    return X, y


def objective(trial, X, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': SEED,
        'verbosity': 0,
    }

    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    scores = []
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        model = xgb.XGBRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_val)
        scores.append(r2_score(y_val, preds))
    return np.mean(scores)


if __name__ == '__main__':
    print("=" * 60)
    print("XGBoost Optuna Hyperparameter Search (OPV)")
    print("=" * 60)

    X, y = load_data()

    # Hold-out test set
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED + 999)
    print(f"Optuna训练: {len(X_tr)}, 测试: {len(X_te)}")

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(lambda trial: objective(trial, X_tr, y_tr),
                   n_trials=N_TRIALS, show_progress_bar=True)

    best_params = study.best_params
    best_cv_r2 = study.best_value
    print(f"\n最优CV R²: {best_cv_r2:.4f}")
    print(f"最优参数: {best_params}")

    # Retrain with best params on full train set
    best_params['random_state'] = SEED
    best_params['verbosity'] = 0
    final_model = xgb.XGBRegressor(**best_params)
    final_model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    preds = final_model.predict(X_te)
    test_r2 = float(r2_score(y_te, preds))
    test_mae = float(mean_absolute_error(y_te, preds))
    test_rmse = float(np.sqrt(mean_squared_error(y_te, preds)))
    print(f"测试 R²={test_r2:.4f}, MAE={test_mae:.4f}, RMSE={test_rmse:.4f}")

    # Also retrain with all non-test data for final best model
    final_model_full = xgb.XGBRegressor(**best_params)
    final_model_full.fit(X, y)
    full_preds = final_model_full.predict(X)
    full_r2 = float(r2_score(y, full_preds))

    output = {
        'task': 'XGBoost Optuna hyperparameter search on OPV high-PCE',
        'dataset': 'OPV high-PCE (PCE > 3%)',
        'features': f'Morgan {FP_DIM}bit + {len(DESC_FUNCS)} RDKit descriptors',
        'optuna_trials': N_TRIALS,
        'cv_folds': CV_FOLDS,
        'best_cv_r2': best_cv_r2,
        'best_params': best_params,
        'test_r2': test_r2,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'n_train': len(X_tr),
        'n_test': len(X_te),
        'n_total': len(X),
    }

    with open(RESULTS_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n结果已保存: {RESULTS_FILE}")
