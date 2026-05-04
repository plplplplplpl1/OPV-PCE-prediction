"""
Table 6 / Table S16: Fingerprint Sensitivity Analysis

Compare XGBoost performance across 7 fingerprint/descriptor types
using default (non-Optuna) hyperparameters, seed=9999, single 80/20 split.
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from rdkit.Chem.rdMolDescriptors import GetHashedTopologicalTorsionFingerprint
RDLogger.DisableLog('rdApp.*')

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if os.path.basename(project_root) == '实验':
    project_root = os.path.dirname(project_root)
os.chdir(project_root)

DATA_CSV = 'data/data_merged.csv' if os.path.exists('data/data_merged.csv') else 'data/data.csv'
PCE_THRESHOLD = 3.0
SEED = 9999
GNN_BASELINE_R2 = 0.635

# Default XGBoost (non-Optuna) hyperparams
XGB_DEFAULT = {
    'n_estimators': 500, 'learning_rate': 0.1, 'max_depth': 6,
    'min_child_weight': 1, 'subsample': 1.0, 'colsample_bytree': 1.0,
    'reg_alpha': 0, 'reg_lambda': 1,
    'random_state': SEED, 'verbosity': 0,
}

# 8 basic physicochemical descriptors for the "plus descriptors" variant
BASIC_DESC_FUNCS = [
    ('MolWt', Descriptors.MolWt),
    ('MolLogP', Descriptors.MolLogP),
    ('TPSA', Descriptors.TPSA),
    ('NumHDonors', Descriptors.NumHDonors),
    ('NumHAcceptors', Descriptors.NumHAcceptors),
    ('NumRotatableBonds', Descriptors.NumRotatableBonds),
    ('RingCount', Descriptors.RingCount),
    ('NumAromaticRings', Descriptors.NumAromaticRings),
]


def compute_basic_descriptors(mol):
    d = []
    for _, fn in BASIC_DESC_FUNCS:
        try:
            v = fn(mol)
            if v is None or np.isnan(v) or np.isinf(v):
                v = 0.0
        except:
            v = 0.0
        d.append(float(v))
    return np.array(d, dtype=np.float32)


def morgan_fp(smi, nbits):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.zeros(nbits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nbits)
    arr = np.zeros(nbits, dtype=np.float32)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def maccs_fp(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.zeros(166, dtype=np.float32)
    fp = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros(166, dtype=np.float32)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def topological_fp(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.zeros(2048, dtype=np.float32)
    fp = GetHashedTopologicalTorsionFingerprint(mol, nBits=2048)
    arr = np.zeros(2048, dtype=np.float32)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


FEATURE_FUNCS = {
    'Morgan_512bit': lambda smi: morgan_fp(smi, 512),
    'Morgan_2048bit': lambda smi: morgan_fp(smi, 2048),
    'Morgan_4096bit': lambda smi: morgan_fp(smi, 4096),
    'Morgan_4096bit_plus_8descriptors': lambda smi: np.concatenate([
        morgan_fp(smi, 4096), compute_basic_descriptors(Chem.MolFromSmiles(smi))
    ]) if Chem.MolFromSmiles(smi) is not None else np.zeros(4104, dtype=np.float32),
    'MACCS_keys_166bit': maccs_fp,
    'RDKit_topological_fingerprint': topological_fp,
    'physicochemical_descriptors_only': lambda smi: compute_basic_descriptors(
        Chem.MolFromSmiles(smi)) if Chem.MolFromSmiles(smi) is not None else np.zeros(8, dtype=np.float32),
}


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
    return df_h, pc, sc


if __name__ == '__main__':
    print("=" * 60)
    print("Fingerprint Sensitivity Analysis (Table 6)")
    print("=" * 60)
    print(f"Seed: {SEED}, GNN baseline: R²={GNN_BASELINE_R2}")

    df_h, pc, sc = load_data()
    y_all = df_h[pc].values.astype(float)
    results = {}

    for fname, ffunc in FEATURE_FUNCS.items():
        print(f"\n  [{fname}]")
        X_list = []
        for smi in df_h[sc].values:
            X_list.append(ffunc(smi))
        X_all = np.array(X_list)
        print(f"    特征维度: {X_all.shape[1]}")

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_all, y_all, test_size=0.2, random_state=SEED)
        print(f"    训练: {len(X_tr)}, 测试: {len(X_te)}")

        model = xgb.XGBRegressor(**XGB_DEFAULT)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        r2 = float(r2_score(y_te, preds))
        mae = float(mean_absolute_error(y_te, preds))
        rmse = float(np.sqrt(mean_squared_error(y_te, preds)))
        beats_gnn = r2 > GNN_BASELINE_R2
        print(f"    R²={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}, beats_GNN={beats_gnn}")

        results[fname] = {
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
            'beats_gnn': beats_gnn,
            'n_features': X_all.shape[1],
        }

    output = {
        'description': 'XGBoost regression performance with different fingerprint types (fixed default hyperparams, non-Optuna, seed=9999, single split)',
        'gnn_baseline': {
            'r2_mean': GNN_BASELINE_R2,
            'description': 'HighPCERegressorV3 multi-seed mean R²',
        },
        'results': results,
        'experiment_config': {
            'model': 'XGBoost (default hyperparams, non-Optuna)',
            'data_split_seed': SEED,
            'dataset': 'OPV high-PCE (PCE > 3%, N=1,916)',
            'train_test_split': '80/20',
        },
    }

    out_path = 'external_results/fingerprint_sensitivity.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n结果已保存: {out_path}")
