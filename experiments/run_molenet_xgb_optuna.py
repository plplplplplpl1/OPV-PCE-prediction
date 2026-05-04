"""
XGBoost Hyperparameter Fairness Check on MoleculeNet
=====================================================
目标：检验MoleculeNet实验中XGBoost使用默认超参数是否低估了其性能，
以及Optuna优化后的XGBoost是否会改变"相变"规律的结论。

协议：
- 对ESOL、FreeSolv、Lipophilicity三个数据集
- 每个运行Optuna（50 trials）优化XGBoost超参数
- 比较优化前后XGBoost在不同n下的R²
- 确认相变点是否移动
"""
import os, sys, json, random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_absolute_error

os.environ['RDKIT_SILENCE'] = '1'
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import AllChem

import optuna
import xgboost as xgb

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_FILE = os.path.join(BASE_DIR, 'external_results', 'molenet_xgb_optuna.json')
FP_DIM = 2048
SEED = 42
N_TRIALS = 50
N_TRAIN_VALUES = [100, 250, 500, 1000]  # focus on small-n regime
N_FOLDS = 3
TEST_FRAC = 0.2  # holdout test

def get_fingerprints(smiles_list, fp_dim=FP_DIM):
    fps = []
    valid = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=fp_dim)
            arr = np.zeros(fp_dim, dtype=np.float32)
            AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)
            valid.append(i)
    return np.array(fps), np.array(valid)

def xgb_cv_score(params, X, y, n_folds=3):
    """Return negative MAE from cross-validation."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    scores = []
    for tr_idx, va_idx in kf.split(X):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        model = xgb.XGBRegressor(**params, verbosity=0, n_jobs=-1, random_state=SEED)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        pred = model.predict(X_va)
        scores.append(r2_score(y_va, pred))
    return float(np.mean(scores))

def optuna_optimize(X_train, y_train, n_trials=N_TRIALS):
    """Run Optuna to find best XGBoost hyperparameters."""
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 2000, step=200),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 0.1, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 0.1, log=True),
        }
        return xgb_cv_score(params, X_train, y_train)

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value

# Default XGBoost params used in moleNet experiment
DEFAULT_PARAMS = {'n_estimators': 500, 'max_depth': 8, 'learning_rate': 0.1,
                  'subsample': 0.8, 'colsample_bytree': 0.8}

# Datasets
datasets = {
    'ESOL': 'esol',
    'FreeSolv': 'freesolv',
    'Lipophilicity': 'lipo',
}

all_results = {}

for ds_display, ds_name in datasets.items():
    print(f'\n{"="*50}')
    print(f'Dataset: {ds_display}')
    print(f'{"="*50}')

    # Load
    from torch_geometric.datasets import MoleculeNet
    molnet = MoleculeNet(root='/tmp/molnet', name=ds_name)
    smiles_list = [d['smiles'] for d in molnet]
    y_all = np.array([d.y.item() for d in molnet])

    # Fingerprints
    X_all, valid_idx = get_fingerprints(smiles_list)
    y_all = y_all[valid_idx]
    n_total = len(X_all)
    print(f'  Samples: {n_total}')

    # Holdout test set
    X_tr_hold, X_te, y_tr_hold, y_te = train_test_split(
        X_all, y_all, test_size=TEST_FRAC, random_state=SEED)
    print(f'  Train pool: {len(X_tr_hold)}, Test: {len(X_te)}')

    # Optuna on full training pool
    print(f'  Running Optuna ({N_TRIALS} trials)...')
    best_params, best_cv_r2 = optuna_optimize(X_tr_hold, y_tr_hold)
    print(f'  Best CV R²: {best_cv_r2:.4f}')
    print(f'  Best params: {best_params}')

    # Compare default vs optimized at different n
    ds_n_results = {}
    for n_train in N_TRAIN_VALUES:
        if n_train >= len(X_tr_hold):
            continue
        print(f'  n={n_train}...')
        default_r2_list = []
        optuna_r2_list = []

        for fold in range(N_FOLDS):
            rng = np.random.RandomState(SEED + fold)
            idx = rng.choice(len(X_tr_hold), n_train, replace=False)
            X_sub = X_tr_hold[idx]
            y_sub = y_tr_hold[idx]

            # Default XGBoost
            model_d = xgb.XGBRegressor(**DEFAULT_PARAMS, verbosity=0, n_jobs=-1, random_state=SEED)
            model_d.fit(X_sub, y_sub, verbose=False)
            pred_d = model_d.predict(X_te)
            default_r2_list.append(r2_score(y_te, pred_d))

            # Optimized XGBoost
            model_o = xgb.XGBRegressor(**best_params, verbosity=0, n_jobs=-1, random_state=SEED)
            model_o.fit(X_sub, y_sub, verbose=False)
            pred_o = model_o.predict(X_te)
            optuna_r2_list.append(r2_score(y_te, pred_o))

        ds_n_results[str(n_train)] = {
            'default_r2_mean': float(np.mean(default_r2_list)),
            'default_r2_std': float(np.std(default_r2_list)),
            'optuna_r2_mean': float(np.mean(optuna_r2_list)),
            'optuna_r2_std': float(np.std(optuna_r2_list)),
            'delta': float(np.mean(optuna_r2_list) - np.mean(default_r2_list)),
        }
        print(f'    Default R²={ds_n_results[str(n_train)]["default_r2_mean"]:.4f}')
        print(f'    Optuna  R²={ds_n_results[str(n_train)]["optuna_r2_mean"]:.4f}')
        print(f'    ΔR²     ={ds_n_results[str(n_train)]["delta"]:.4f}')

    # Full training
    model_default_full = xgb.XGBRegressor(**DEFAULT_PARAMS, verbosity=0, n_jobs=-1, random_state=SEED)
    model_default_full.fit(X_tr_hold, y_tr_hold, verbose=False)
    pred_full_d = model_default_full.predict(X_te)
    full_default_r2 = r2_score(y_te, pred_full_d)

    model_optuna_full = xgb.XGBRegressor(**best_params, verbosity=0, n_jobs=-1, random_state=SEED)
    model_optuna_full.fit(X_tr_hold, y_tr_hold, verbose=False)
    pred_full_o = model_optuna_full.predict(X_te)
    full_optuna_r2 = r2_score(y_te, pred_full_o)

    print(f'  Full training: Default R²={full_default_r2:.4f}, Optuna R²={full_optuna_r2:.4f}')

    all_results[ds_display] = {
        'n_total': n_total,
        'best_params': best_params,
        'best_cv_r2': best_cv_r2,
        'full_default_r2': full_default_r2,
        'full_optuna_r2': full_optuna_r2,
        'n_train_results': ds_n_results,
    }

# Save
os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
with open(RESULTS_FILE, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f'\nResults saved to {RESULTS_FILE}')

# Summary
print(f'\n{"="*50}')
print('Summary: Optuna XGBoost vs Default XGBoost on MoleculeNet')
print(f'{"="*50}')
for ds in datasets:
    r = all_results[ds]
    print(f'\n{ds}:')
    print(f'  Full data:   Default={r["full_default_r2"]:.4f}  Optuna={r["full_optuna_r2"]:.4f}  Δ={r["full_optuna_r2"]-r["full_default_r2"]:.4f}')
    for n_str in sorted(r['n_train_results'].keys()):
        nr = r['n_train_results'][n_str]
        print(f'  n={n_str:>4}: Default={nr["default_r2_mean"]:.4f}  Optuna={nr["optuna_r2_mean"]:.4f}  Δ={nr["delta"]:.4f}')

# Compare with GNN performance from manuscript
print(f'\n{"="*50}')
print('Reference GNN performance (from manuscript Table 9):')
print(f'  ESOL: GNN R²=0.779±0.023 (n=250), 0.857±0.011 (n=500), 0.870±0.012 (full)')
print(f'  FreeSolv: GNN R²=0.811±0.055 (n=250), 0.883±0.018 (n=500), 0.871±0.038 (full)')
print(f'  Lipophilicity: GNN R²=0.164±0.029 (n=250), 0.322±0.035 (n=500), 0.658±0.021 (full)')
print('Done.')
